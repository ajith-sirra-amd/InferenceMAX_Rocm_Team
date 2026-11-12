#!/usr/bin/env python3
"""Bounded GEMM2 owner-wave correctness probe; requires reviewed compiler evidence.

Uses the pinned native API and reviewed FlyDSL cache, with JIT disabled. Tests
synthetic FP4 inputs, not model quality. Timing is skipped unless every eager,
padding, output-bound and changed-input HIP graph check passes. Never installs
or promotes the candidate into serving.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import types

sys.dont_write_bytecode = True
import optimization_moe_owner as compiler
from optimization_moe import assess, error_metrics, paired_graph_timing


H, I, E, TOPK, BM = 3584, 384, 896, 16, 16
SENTINEL = 29.0  # Exactly representable BF16, checked on both sides of output.


def reviewed_sources(review_path, compiled):
    """Inspect without importing Torch, unpickling or launching GPU code.

    Review JSON: status='approved', reviewer, notes, native_sha256,
    candidate_sha256, gpu_sha256={'baseline': [...], 'owner': [...]}.
    The reviewer fills this only after reviewing actual emitted gfx950 ISA.
    """
    review = json.loads(review_path.read_text())
    if review.get('status') != 'approved' or not review.get('reviewer') or not review.get('notes'):
        raise ValueError('Actual emitted ISA must be reviewed before GPU execution')
    original = compiled / 'source/mxmoe_dispatcher.original.py'
    candidate = compiled / 'source/mxmoe_dispatcher.owner.py'
    if compiler.digest(original) != compiler.HASHES[compiler.DISPATCH]:
        raise ValueError('Compile evidence does not contain the pinned native dispatcher')
    expected_candidate = hashlib.sha256(compiler.candidate_source(original.read_text()).encode()).hexdigest()
    if (review.get('native_sha256') != compiler.digest(original)
            or review.get('candidate_sha256') != expected_candidate
            or compiler.digest(candidate) != expected_candidate):
        raise ValueError('Reviewed source differs from this exact candidate')
    if any(not review.get('gpu_sha256', {}).get(arm) for arm in ('baseline', 'owner')):
        raise ValueError('Reviewed GPU executable hashes are required')
    return review


def approved_cache(review_path, compiled, source_cache, target_cache):
    review = reviewed_sources(review_path, compiled)
    records = {}
    pending_copies = []
    for arm in ('baseline', 'owner'):
        files = sorted((source_cache / arm).rglob('*.pkl'))
        if len(files) != 1 or files[0].stat().st_size > 32 * 1024 * 1024:
            raise ValueError(f'{arm}: expected one bounded native GEMM2 cache artifact')
        path = files[0]
        hashes = set()
        for ir in compiler.cached_ir(path.read_bytes()):
            for binary in compiler.mlir_binaries(ir):
                extent = compiler.elf_extent(binary, 0)
                if extent is None:
                    raise ValueError('Invalid cached AMDGPU ELF')
                hashes.add(hashlib.sha256(binary[:extent]).hexdigest())
        expected = review.get('gpu_sha256', {}).get(arm, [])
        if not hashes or hashes != set(expected):
            raise ValueError(f'{arm}: cached GPU executable does not match reviewed ISA')
        relative = path.relative_to(source_cache / arm)
        records[arm] = dict(cache_file=str(path), cache_sha256=compiler.digest(path),
                            gpu_sha256=sorted(hashes), copied_relative=str(relative))
        pending_copies.append((path, target_cache / relative))
    if records['baseline']['gpu_sha256'] == records['owner']['gpu_sha256']:
        raise ValueError('Baseline and owner must be distinct reviewed executables')
    target_cache.mkdir(parents=True, exist_ok=False)
    for source, target in pending_copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError('Reviewed arm caches collide')
        shutil.copyfile(source, target)
    return dict(review=review, compile_evidence=str(compiled), artifacts=records)


def preload_worker(args):
    """Recover a missing disk cache by compilation only in a separate process."""
    reviewed_sources(args.review, args.compiled)
    cache = args.cache / args.preload_worker
    cache.mkdir(parents=True, exist_ok=False)
    os.environ.update(COMPILE_ONLY='1', FLYDSL_GPU_ARCH='gfx950', ARCH='gfx950',
                      GPU_ARCHS='gfx950', CU_NUM='256', AITER_AOT_IMPORT='1',
                      FLYDSL_RUNTIME_CACHE_DIR=str(cache), FLYDSL_RUNTIME_RUN_ONLY='0',
                      FLYDSL_RUNTIME_ENABLE_CACHE='1', FLYDSL_DUMP_IR='0',
                      FLYDSL_DEBUG_DUMP_ASM='0', PYTHONDONTWRITEBYTECODE='1')
    import torch
    from aiter.aot.flydsl.dry_run import dry_run
    from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled
    from aiter.ops.flydsl.kernels import mxmoe_dispatcher as native
    for path, expected in compiler.HASHES.items():
        if compiler.digest(Path(native.__file__).parents[3] / path) != expected:
            raise ValueError('Installed preload source mismatch: ' + path)
    module = native
    config = dict(compiler.CONFIG)
    if args.preload_worker == 'owner':
        name = 'aiter.ops.flydsl.kernels._moe_owner_experiment'
        spec = importlib.util.spec_from_file_location(name, args.compiled / 'source/mxmoe_dispatcher.owner.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        config['_owner_wave'] = True
    with dry_run('gfx950', 256, 'AMD Instinct MI355X'):
        launch = module.compile_gemm2_a4w4_port(**config)
        _preload_compiled(launch, *([0] * 9 + [280, 280, 280, 384, 3584, 0, 0, torch.cuda.current_stream()]))
    return 0


def runtime_cache_source(args, evidence):
    if any(args.approved_cache.rglob('*.pkl')):
        return args.approved_cache
    # Dump-enabled compilation can bypass disk caching. Recreate only in fresh
    # work storage, then approved_cache must match the previously reviewed ELF.
    generated = args.cache.with_name(args.cache.name + '-preload')
    generated.mkdir(parents=True, exist_ok=False)
    evidence.data['preload'] = dict(cache=str(generated), launches=0,
                                  reason='Original compile cache contains no pickle artifacts')
    for arm in ('baseline', 'owner'):
        evidence.save(arm + ': rebuilding missing cache with GPU execution forbidden')
        command = [sys.executable, __file__, '--preload-worker', arm,
                   '--review', str(args.review), '--compiled', str(args.compiled),
                   '--approved-cache', str(args.approved_cache), '--cache', str(generated),
                   '--output', str(args.output)]
        with (args.output / ('preload-' + arm + '.log')).open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                           timeout=75, check=True)
    return generated


def cases(seed):
    """CPU metadata; invalid slots point at guarded rows just beyond M."""
    result = []

    def blocks(name, counts, active=True):
        m = 19
        rows = []
        for block, count in enumerate(counts):
            tokens = [(row * 7) % m for row in range(count)]
            rows.append((block % 4, [(token, block % TOPK) for token in tokens]))
        return dict(name=name, M=m, blocks=rows, active=active, timing=False)

    result.append(blocks('empty_cumsum', [0], active=False))
    result.extend(blocks(f'valid_rows_{count}', [count]) for count in range(17))
    result.append(blocks('padded_experts_atomic_overlap', [1, 7, 16, 3, 16]))
    for m in (280, 336):
        rng = random.Random(seed + m)
        experts = [[] for _ in range(E)]
        for token in range(m):
            for slot, expert in enumerate(rng.sample(range(E), TOPK)):
                experts[expert].append((token, slot))
        rows = [(expert, tokens[start:start + BM])
                for expert, tokens in enumerate(experts)
                for start in range(0, len(tokens), BM)]
        result.append(dict(name=f'synthetic_M{m}', M=m, blocks=rows, active=True, timing=True))
    return result


def prepare_case(torch, spec, w2, s2):
    m = spec['M']
    nblocks = len(spec['blocks'])
    capacity = (nblocks + 2) * BM
    ids = []
    eids = []
    used = set()
    for expert, tokens in spec['blocks']:
        eids.append(expert)
        ids.extend(token | (slot << 24) for token, slot in tokens)
        used.update(token for token, _ in tokens)
        ids.extend((m + row % 2) | (TOPK << 24) for row in range(BM - len(tokens)))
    ids.extend([m] * (capacity - len(ids)))
    eids.extend([0] * (capacity // BM - len(eids)))
    device = dict(device='cuda')
    data = dict(
        inter_sorted_quant=torch.randint(0, 256, (capacity, I // 2), dtype=torch.uint8, **device),
        # BM16 native A-scale layout stores a full 32-row scale chunk/block.
        inter_sorted_shuffled_scale=torch.full((capacity // BM, 512), 124, dtype=torch.uint8, **device),
        w2_u8=w2, w2_scale_u8=s2,
        sorted_token_ids=torch.tensor(ids, dtype=torch.int32, **device),
        sorted_expert_ids=torch.tensor(eids, dtype=torch.int32, **device),
        cumsum_tensor=torch.tensor([nblocks * BM if spec['active'] else 0, m], dtype=torch.int32, **device),
        # Nonzero padding weights ensure invalid-token masking is exercised.
        sorted_weights=torch.full((capacity,), 1 / TOPK, dtype=torch.float32, **device),
        M_logical=m, max_sorted=capacity, NE=E, D_HIDDEN=H, D_INTER=I, topk=TOPK,
        BM=BM, BN=128, BK=128, use_nt=True, a_dtype='fp4', b_dtype='fp4',
        epilog='atomic', SBM=BM, persist=False, n_sorted_padded=capacity,
        g2_bf16_lds=True, g2_spart=402, out_dtype='bf16', HIDDEN_MAX=8192,
    )
    if not spec['active']:
        used.clear()
    guards = {arm: torch.full((m + 4, H), SENTINEL, dtype=torch.bfloat16, **device)
              for arm in ('baseline', 'owner')}
    unused = sorted(set(range(m)) - used)
    return data, guards, unused


def check_bounds(torch, guard, unused):
    edges = bool((guard[:2] == SENTINEL).all().item() and (guard[-2:] == SENTINEL).all().item())
    untouched = not unused or bool((guard[2:-2][unused] == 0).all().item())
    return dict(guard_rows_unchanged=edges, unaddressed_rows_zero=untouched, passed=edges and untouched)


class Evidence:
    def __init__(self, folder):
        self.folder = folder
        self.started = time.monotonic()
        self.data = dict(status='starting', correctness_passed=False, timing_started=False,
                         timing_scope='Isolated native GEMM2 plus output zeroing; excludes GEMM1, sorting, serving and TP communication',
                         limitations=['Synthetic physical-layout FP4 bytes; initially constant E8M0 scales, patterned A-scale changes during replay checks',
                                      'Comparison against repeated native BF16 atomics; no dense reference or model-quality proof',
                                      'Overlapping-route numerical envelope uses a provisional one-BF16-unit floor; single-block cases require exact equality',
                                      'Guard rows detect output overruns, not all possible input overreads'], cases=[])

    def save(self, message):
        self.data['elapsed_seconds'] = round(time.monotonic() - self.started, 3)
        self.data['progress'] = message
        compiler.save(self.folder / 'result.json', self.data)
        print(message, flush=True)


def run(args, evidence):
    reviewed_sources(args.review, args.compiled)
    source_cache = runtime_cache_source(args, evidence)
    evidence.data['approval'] = approved_cache(args.review, args.compiled, source_cache, args.cache)
    evidence.save('Reviewed source/executables checked; candidate execution now permitted')
    sys.dont_write_bytecode = True
    os.environ.update(FLYDSL_RUNTIME_CACHE_DIR=str(args.cache), FLYDSL_RUNTIME_RUN_ONLY='1',
                      FLYDSL_RUNTIME_ENABLE_CACHE='1', FLYDSL_GPU_ARCH='gfx950', ARCH='gfx950',
                      GPU_ARCHS='gfx950', CU_NUM='256', AITER_AOT_IMPORT='1',
                      PYTHONDONTWRITEBYTECODE='1', COMPILE_ONLY='0')
    import torch
    from flydsl.utils import env
    from aiter.ops.flydsl.kernels import mxmoe_dispatcher as native
    if not env.runtime.run_only or Path(env.runtime.cache_dir).resolve() != args.cache.resolve():
        raise RuntimeError('Native FlyDSL run-only cache contract is unavailable')
    installed = Path(native.__file__).parents[3]
    for name, expected in compiler.HASHES.items():
        if compiler.digest(installed / name) != expected:
            raise ValueError('Installed source differs from reviewed compiler input: ' + name)
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if 'gfx950' not in str(props.gcnArchName):
        raise RuntimeError('This experiment requires MI355X/gfx950')
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    evidence.data['environment'] = dict(torch=torch.__version__, hip=torch.version.hip,
                                        device=props.name, arch=props.gcnArchName, run_only=True)
    name = 'aiter.ops.flydsl.kernels._moe_owner_experiment'
    spec = importlib.util.spec_from_file_location(name, args.compiled / 'source/mxmoe_dispatcher.owner.py')
    owner = importlib.util.module_from_spec(spec)
    sys.modules[name] = owner
    spec.loader.exec_module(owner)
    apis = {}
    for arm, module in (('baseline', native), ('owner', owner)):
        config = dict(compiler.CONFIG)
        if arm == 'owner':
            config['_owner_wave'] = True
        launch = module.compile_gemm2_a4w4_port(**config)
        # Clone the pinned API's globals; never patch installed get_g2/cache.
        globals_copy = dict(module.__dict__, get_g2=lambda *a, _launch=launch, **k: _launch)
        apis[arm] = types.FunctionType(module.mxfp4_moe_gemm2.__code__, globals_copy)
        apis[arm].__kwdefaults__ = module.mxfp4_moe_gemm2.__kwdefaults__
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device='cuda')
    s2 = torch.full((E, H, 16), 121, dtype=torch.uint8, device='cuda')
    timing_cases = []
    for case_spec in cases(args.seed):
        result = dict(name=case_spec['name'], M=case_spec['M'], checks=[],
                      changed_input='In-place FP4 sign-bit XOR 0x88, route weights multiplied by 0.75, every seventh stored A-scale byte incremented by one')
        evidence.data['cases'].append(result)
        evidence.save(case_spec['name'] + ': checking eager and graph correctness')
        data, guards, unused = prepare_case(torch, case_spec, w2, s2)
        outputs = {arm: guard[2:-2] for arm, guard in guards.items()}

        def operation(arm, data=data, outputs=outputs):
            outputs[arm].zero_()
            apis[arm](out=outputs[arm], **data)
            return outputs[arm]

        for arm in apis:
            operation(arm)
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        graphs = {}
        for arm in apis:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                operation(arm)
            graphs[arm] = graph
        torch.cuda.current_stream().wait_stream(side)
        previous = None
        for state in ('original', 'changed_input'):
            if state == 'changed_input':
                data['inter_sorted_quant'].bitwise_xor_(0x88)
                data['sorted_weights'].mul_(0.75)
                data['inter_sorted_shuffled_scale'].view(-1)[::7].add_(1)
            reference = operation('baseline').clone()
            torch.cuda.synchronize()
            repeats = []
            for _ in range(3):
                actual = operation('baseline')
                torch.cuda.synchronize()
                repeats.append(error_metrics(torch, actual, reference))
            changed = error_metrics(torch, reference, previous) if previous is not None else None
            expected_nonzero = bool(case_spec['active'] and any(tokens for _, tokens in case_spec['blocks']))
            if expected_nonzero and (not bool((reference != 0).any().item())
                                     or (changed is not None and changed.get('max_abs_error', 0) == 0)):
                raise RuntimeError('Vacuous baseline or changed-input test')
            for mode in ('eager', 'graph'):
                for arm in apis:
                    if mode == 'graph':
                        graphs[arm].replay()
                    else:
                        operation(arm)
                    torch.cuda.synchronize()
                    metrics = error_metrics(torch, outputs[arm], reference)
                    numerical = assess(metrics, repeats)
                    exact_required = len(case_spec['blocks']) == 1
                    numerical['exact_required'] = exact_required
                    numerical['exact_passed'] = metrics.get('unequal_elements') == 0
                    bounds = check_bounds(torch, guards[arm], unused)
                    check = dict(state=state, mode=mode, arm=arm, error=metrics,
                                 numerical=numerical, bounds=bounds, baseline_repeats=repeats)
                    if changed is not None:
                        check['reference_change'] = changed
                    result['checks'].append(check)
                    if (numerical['status'] != 'within_baseline_noise_envelope'
                            or (exact_required and not numerical['exact_passed']) or not bounds['passed']):
                        evidence.save(case_spec['name'] + ': failed; no timing will run')
                        raise RuntimeError('Correctness/output-bound gate failed')
            previous = reference
        result['correctness_passed'] = True
        if case_spec['timing']:
            # Retain every buffer referenced by the captured graph.
            timing_cases.append((result, graphs, data, guards))
        evidence.save(case_spec['name'] + ': all correctness checks passed')
    evidence.data['correctness_passed'] = True
    evidence.save('Every correctness/graph case passed; starting bounded paired timing')
    evidence.data['timing_started'] = True
    for result, graphs, data, guards in timing_cases:
        result['paired_graph_timing'] = paired_graph_timing(torch, graphs['baseline'], graphs['owner'], args.iterations)
        evidence.save(result['name'] + ': paired timing saved')
    evidence.data['status'] = 'completed'
    evidence.data['max_memory_allocated_bytes'] = torch.cuda.max_memory_allocated()
    evidence.save('Completed isolated GEMM2 experiment; no serving changes')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True, type=Path)
    parser.add_argument('--compiled', required=True, type=Path, help='Reviewed compile-evidence directory')
    parser.add_argument('--approved-cache', required=True, type=Path, help='Original compile cache containing baseline/owner')
    parser.add_argument('--cache', required=True, type=Path, help='Fresh work cache; only approved artifacts copied')
    parser.add_argument('--output', required=True, type=Path, help='New evidence directory')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--seed', type=int, default=355)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--timeout-seconds', type=int, default=180)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--preload-worker', choices=('baseline', 'owner'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 5 <= args.iterations <= 100 or not 30 <= args.timeout_seconds <= 240:
        parser.error('Use 5..100 timing iterations and a 30..240 second deadline')
    return args


def main(args):
    if args.preload_worker:
        return preload_worker(args)
    if not args.worker:
        args.output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, __file__, *sys.argv[1:], '--worker']
        with (args.output / 'probe.log').open('w') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                return child.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                path = args.output / 'result.json'
                result = json.loads(path.read_text()) if path.exists() else {}
                result.update(status='timed_out', timeout_seconds=args.timeout_seconds)
                compiler.save(path, result)
                return 124
    evidence = Evidence(args.output)
    try:
        run(args, evidence)
        return 0
    except Exception as exc:
        evidence.data.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        evidence.save('Failed; saved evidence and stopped')
        return 1


if __name__ == '__main__':
    raise SystemExit(main(arguments()))
