#!/usr/bin/env python3
"""Bounded unit diagnostic of frozen native rejection_sample; no model load."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / 'manifest.json').read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def event(output, **record):
    record['elapsed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with (output / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate(source_root):
    actual = {name: sha(source_root / name) for name in MANIFEST['source_files']}
    if actual != MANIFEST['source_files']:
        changed = [p for p, expected in MANIFEST['source_files'].items() if actual[p] != expected]
        raise RuntimeError(f'Frozen source identity mismatch: {changed}')
    return {'source_files': actual, 'manifest_sha256': sha(HERE / 'manifest.json'),
            'harness_sha256': sha(__file__), 'source_root': str(source_root),
            'expected_image': MANIFEST['image']}


def token_layout(concurrency, vocab_size):
    """Distinct request/step IDs exercise multiple vocabulary blocks."""
    drafts = [[(r * 1009 + step * 17011 + 7) % vocab_size for step in range(6)]
              for r in range(concurrency)]
    different = [[(token + vocab_size // 2 + 17) % vocab_size for token in row]
                 for row in drafts]
    bonus = [(r * 31337 + vocab_size - 1) % vocab_size for r in range(concurrency)]
    if any(a == b for row, other in zip(drafts, different) for a, b in zip(row, other)):
        raise ValueError('Vocabulary is too small for disjoint token fixture')
    return drafts, different, bonus


def target_tokens(pattern, drafts, different, bonus):
    if pattern in ('equal', 'draft_disjoint'):
        rows = [list(row) for row in drafts]
    elif pattern in ('disjoint', 'nan_draft'):
        rows = [list(row) for row in different]
    elif pattern == 'prefix3':
        rows = [row[:3] + other[3:] for row, other in zip(drafts, different)]
    else:
        raise ValueError(pattern)
    return [row + [last] for row, last in zip(rows, bonus)]


def expected_outputs(pattern, drafts, different, bonus):
    if pattern == 'equal':
        return [row + [last] for row, last in zip(drafts, bonus)], 7
    if pattern == 'disjoint':
        return [[row[0]] for row in different], 1
    if pattern == 'draft_disjoint':
        return [[row[0]] for row in drafts], 1
    if pattern == 'prefix3':
        return [row[:3] + [other[3]] for row, other in zip(drafts, different)], 4
    raise ValueError('Invalid inputs have no correctness expectation')


def worker(args):
    provenance = validate(args.source_root)
    # Import the unchanged installed implementation only in the execution child.
    import torch
    import triton
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as native
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
    expected_path = args.source_root / 'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py'
    if Path(native.__file__).resolve() != expected_path:
        raise RuntimeError(f'Imported native source from unexpected path: {native.__file__}')
    if not torch.cuda.is_available():
        raise RuntimeError('ROCm GPU unavailable')
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    provenance.update(torch_version=str(torch.__version__), triton_version=str(triton.__version__),
                      hip_version=torch.version.hip, gpu_name=torch.cuda.get_device_name(args.device),
                      imported_native_file=str(native.__file__),
                      environment={k: v for k, v in os.environ.items() if k.startswith(('VLLM_', 'AITER_')) or k in ('HSA_NO_SCRATCH_RECLAIM', 'THP_MEM_ALLOC_ENABLE', 'PYTHONHASHSEED')})
    save(args.output / 'provenance.json', provenance)
    event(args.output, stage='worker_ready', gpu=provenance['gpu_name'])
    (args.output / 'cases').mkdir(exist_ok=True)
    results = []
    groups = []

    for concurrency in args.concurrencies:
        for dtype_name in args.dtypes:
            dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[dtype_name]
            vocab = args.vocab_size
            drafts, different, bonus = token_layout(concurrency, vocab)
            group_base = f'c{concurrency}-{dtype_name}-v{vocab}'
            event(args.output, stage='input_allocation', group=group_base)
            # Native layout: seven target rows per request; row zero is prior
            # context token, then six draft IDs in draft_sampled[1:7].
            target = torch.full((concurrency * 7, vocab), -80.0, dtype=dtype, device=device)
            cache = DraftLogitsCache(concurrency, 6, vocab, dtype=dtype, fill=-80.0,
                                     device=device, sharded=False)
            draft_ids = torch.tensor(drafts, dtype=torch.int64, device=device)
            different_ids = torch.tensor(different, dtype=torch.int64, device=device)
            target_ids = torch.empty((concurrency * 7, 1), dtype=torch.int64, device=device)
            sampled_drafts = torch.tensor([[0] + row for row in drafts], dtype=torch.int64, device=device).flatten()
            request_ids = torch.arange(concurrency, dtype=torch.int32, device=device)
            offsets = torch.arange(concurrency + 1, dtype=torch.int32, device=device) * 7
            expanded_ids = request_ids.repeat_interleave(7)
            local_positions = torch.arange(7, dtype=torch.int32, device=device).repeat(concurrency)
            positions = local_positions.to(torch.int64) + 128
            seeds = request_ids.to(torch.int64) + 20261112
            temperature = torch.empty(concurrency, dtype=torch.float32, device=device)
            pointers = {'target': target.data_ptr(), 'draft_cache': cache.values.data_ptr(),
                        'draft_sampled': sampled_drafts.data_ptr()}

            def fill(pattern):
                rows = target_tokens(pattern, drafts, different, bonus)
                target_ids.copy_(torch.tensor(rows, dtype=torch.int64, device=device).reshape(-1, 1))
                target.fill_(-80.0).scatter_(1, target_ids, 80.0)
                current_draft_ids = different_ids if pattern == 'draft_disjoint' else draft_ids
                cache.values.fill_(-80.0).scatter_(2, current_draft_ids.unsqueeze(-1), 80.0)
                sampled_drafts.view(concurrency, 7)[:, 1:].copy_(current_draft_ids)
                if pattern == 'nan_draft':
                    cache.values.fill_(float('nan'))
                torch.cuda.synchronize()
                if target.data_ptr() != pointers['target'] or cache.values.data_ptr() != pointers['draft_cache']:
                    raise RuntimeError('Fixture mutation replaced graph input storage')

            for temp in args.temperatures:
                temperature.fill_(temp)
                for mode in args.modes:
                    group = f'{group_base}-t{int(temp)}-{mode}'
                    event(args.output, stage='group_started', group=group)

                    def invoke():
                        return native.rejection_sample(
                            target, cache, sampled_drafts, offsets, positions, request_ids,
                            expanded_ids, local_positions, temperature, seeds, 6,
                            synthetic_conditional_rates=None, use_fp64=False,
                            use_block_verification=mode == 'block')

                    def record_case(pattern, lane, action):
                        case_id = f'{len(results):03d}-{group}-{lane}-{pattern}'
                        path = args.output / 'cases' / (case_id + '.json')
                        invalid = pattern == 'nan_draft'
                        row = {'id': case_id, 'status': 'running', 'group': group,
                               'concurrency': concurrency, 'dtype': dtype_name,
                               'vocab_size': vocab, 'temperature': temp, 'mode': mode,
                               'pattern': pattern, 'lane': lane, 'num_speculative_steps': 6,
                               'input_classification': 'invalid_nan_draft_logits' if invalid else 'finite_deterministic',
                               'input_pointers': pointers, 'synthetic_acceptance': False,
                               'source_manifest_sha256': provenance['manifest_sha256']}
                        save(path, row)
                        event(args.output, stage='case_started', case=case_id)
                        started = time.monotonic()
                        try:
                            fill(pattern)
                            sampled, counts = action()
                            torch.cuda.synchronize()
                            counts_cpu = counts.cpu().tolist()
                            tokens_cpu = sampled.cpu().tolist()
                            # The tail after num_sampled is uninitialized by contract.
                            valid_tokens = [tokens[:max(0, min(int(count), 7))]
                                            for tokens, count in zip(tokens_cpu, counts_cpu)]
                            row.update(num_sampled=counts_cpu, accepted_drafts=[int(n) - 1 for n in counts_cpu],
                                       valid_sampled_tokens=valid_tokens,
                                       returned_count_in_bounds=all(1 <= n <= 7 for n in counts_cpu),
                                       returned_token_ids_in_bounds=all(0 <= token < vocab for tokens in valid_tokens for token in tokens))
                            if invalid:
                                row.update(status='invalid_input_observation', correctness_qualified=False,
                                           all_six_drafts_reported_accepted=all(n == 7 for n in counts_cpu),
                                           note='NaN logits violate the probability input contract. Observed all-accept behavior is not a passing correctness result.')
                            else:
                                expected, count = expected_outputs(pattern, drafts, different, bonus)
                                row.update(expected_num_sampled=count, expected_valid_sampled_tokens=expected)
                                passed = counts_cpu == [count] * concurrency and valid_tokens == expected
                                row.update(status='passed' if passed else 'failed', correctness_qualified=passed)
                        except BaseException as error:
                            row.update(status='invalid_input_error' if invalid else 'failed', correctness_qualified=False,
                                       error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
                        row['elapsed_seconds_including_compile'] = time.monotonic() - started
                        save(path, row)
                        results.append(row)
                        event(args.output, stage='case_finished', case=case_id, status=row['status'])
                        return row

                    # Eager calls also compile native kernels before graph capture.
                    for pattern in ('equal', 'disjoint', 'prefix3'):
                        record_case(pattern, 'eager', invoke)
                    graph = torch.cuda.CUDAGraph()
                    fill('equal')
                    event(args.output, stage='graph_capture_started', group=group)
                    try:
                        # Warm on a side stream before capture, per Torch's graph contract.
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            warm_output = invoke()
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        with torch.cuda.graph(graph, stream=stream):
                            graph_output = invoke()

                        def replay():
                            graph.replay()
                            return graph_output

                        for index, pattern in enumerate(('equal', 'disjoint', 'draft_disjoint', 'prefix3', 'equal')):
                            record_case(pattern, f'graph{index}', replay)
                        groups.append({'group': group, 'graph_capture': 'passed', 'input_pointers': pointers})
                    except BaseException as error:
                        failure = {'group': group, 'graph_capture': 'failed', 'error': str(error), 'traceback': traceback.format_exc()}
                        groups.append(failure)
                        save(args.output / ('graph-failure-' + group + '.json'), failure)
                        event(args.output, stage='graph_capture_failed', group=group, error=str(error))
                    # Invalid-input observations follow all valid checks for this group.
                    if args.nan_probe and temp == 1.0:
                        record_case('nan_draft', 'exploratory_eager', invoke)
                    save(args.output / 'groups.json', groups)
            del target, cache
            torch.cuda.synchronize()

    valid = [r for r in results if r['input_classification'] == 'finite_deterministic']
    invalid = [r for r in results if r['input_classification'] != 'finite_deterministic']
    failures = [r['id'] for r in valid if r['status'] != 'passed']
    graph_failures = [g for g in groups if g['graph_capture'] != 'passed']
    expected_groups = len(args.concurrencies) * len(args.dtypes) * len(args.temperatures) * len(args.modes)
    complete = len(valid) == expected_groups * 8 and len(groups) == expected_groups
    result = {'status': 'passed' if complete and not failures and not graph_failures else 'failed',
              'qualification_scope': 'finite deterministic native rejection-sampler unit cases only',
              'valid_case_count': len(valid), 'expected_valid_case_count': expected_groups * 8,
              'failed_cases': failures, 'graph_failures': graph_failures,
              'invalid_input_observations': [{'id': r['id'], 'status': r['status'],
                    'all_six_drafts_reported_accepted': r.get('all_six_drafts_reported_accepted')} for r in invalid],
              'model_quality_qualified': False, 'model_run_reproduced': False,
              'provenance': provenance, 'limits': MANIFEST['limits']}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--timeout', type=int, default=150)
    parser.add_argument('--runtime-dir', type=Path, default=Path('/logs/work/rejection-smoke'))
    parser.add_argument('--concurrencies', type=int, nargs='+', choices=(1, 48), default=[1, 48])
    parser.add_argument('--dtypes', nargs='+', choices=('bf16', 'fp32'), default=['bf16', 'fp32'])
    parser.add_argument('--temperatures', type=float, nargs='+', choices=(0.0, 1.0), default=[0.0, 1.0])
    parser.add_argument('--modes', nargs='+', choices=('standard', 'block'), default=['standard', 'block'])
    parser.add_argument('--vocab-size', type=int, default=163840)
    parser.add_argument('--nan-probe', action='store_true')
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.output = args.output.resolve()
    args.runtime_dir = args.runtime_dir.resolve()
    if not 1 <= args.timeout <= 150 or args.vocab_size not in (1024, 163840):
        parser.error('Worker limit must be <=150 seconds; vocab must be 1024 or 163840')
    if args.worker:
        try:
            result = worker(args)
        except BaseException as error:
            result = {'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc()}
        save(args.output / 'result.json', result)
        print(json.dumps(result, allow_nan=False), flush=True)
        raise SystemExit(0 if result['status'] == 'passed' else 1)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        source = validate(args.source_root)
        save(args.output / 'source-validation.json', source)
    except BaseException as error:
        save(args.output / 'result.json', {'status': 'failed_source_validation', 'error': str(error), 'traceback': traceback.format_exc()})
        raise
    if not args.execute:
        print('Source validation passed; GPU not executed.', flush=True)
        return
    if args.runtime_dir.is_relative_to(args.output):
        raise RuntimeError('Runtime cache must stay outside the artifact output directory')
    args.runtime_dir.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    for name, directory in (('TRITON_CACHE_DIR', 'triton'), ('XDG_CACHE_HOME', 'xdg'), ('TMPDIR', 'tmp')):
        path = args.runtime_dir / directory
        path.mkdir(parents=True)
        environment[name] = str(path)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    argv = [sys.executable, '-B', str(HERE / 'diagnose.py'), '--worker', '--source-root', str(args.source_root),
            '--output', str(args.output), '--device', str(args.device), '--vocab-size', str(args.vocab_size),
            '--concurrencies', *map(str, args.concurrencies), '--dtypes', *args.dtypes,
            '--temperatures', *map(str, args.temperatures), '--modes', *args.modes]
    if args.nan_probe:
        argv.append('--nan-probe')
    save(args.output / 'worker-command.json', {'argv': argv, 'timeout_seconds': args.timeout})
    with (args.output / 'worker.log').open('w') as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        def interrupted(signum, frame):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            save(args.output / 'result.json', {'status': 'interrupted', 'signal': signum,
                  'partial_cases_preserved': True, 'correctness_qualified': False})
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            save(args.output / 'result.json', {'status': 'timed_out', 'deadline_seconds': args.timeout,
                  'partial_cases_preserved': True, 'correctness_qualified': False})
            raise SystemExit(124)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
