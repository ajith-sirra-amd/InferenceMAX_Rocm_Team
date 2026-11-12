"""Bounded TP8/DCP8 medium Int32 gather qualification; no model or E2E claim.

Uses native vLLM communicators and their shared AITER pool. Only run-local
method bindings change; installed packages and the original research patch do
not. Supply the frozen serving environment before starting this interpreter.
"""
import argparse
import ast
from contextlib import contextmanager
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
from types import MethodType, SimpleNamespace


SOURCE_HASHES = {
    'vllm/distributed/device_communicators/cuda_communicator.py':
        'b73bdd09bd11c9d5ecf58ffeda62bac559601c492f30e6ae249182cb7b789792',
    'vllm/distributed/device_communicators/aiter_custom_all_reduce.py':
        'cd9857aecdd186c3064a5fa78fd9e9be69558d9dc1d8a0c68ee1bc615f031f87',
    'aiter/dist/device_communicators/custom_all_reduce.py':
        '70b3a8dd977e718461b0288bf736b98f6674a05c7528ecb06bb8d3fcbbcf7478',
    'vllm/distributed/parallel_state.py':
        'f37b5ca1fc25c2a872aa0d3c4cc0288abc0a781b2affa7842f3e731fff1bc7ec',
    'vllm/device_allocator/alloc_conf.py':
        'ef11ab3c5cb1fc66c707d94edd11bb1be2ede84d80bcf690ff7981da5e2881bd',
}
CANDIDATE_SHA256 = '2e524068d8ea894c7da00fd87f8efccf83e83a0637e07156884c0cb36a76e4a2'
RESEARCH_PATCH_SHA256 = '6d527ed05a634ab7ca319bb1723e39cf5ba04a5904a2aca515764a35fcdebdc7'
SHAPE_HELPER = '''

def _medium_dcp_topk_gather_shape_ok(shape, dim, world_size, pool_bytes):
    # Research scope: observed Int32 selected-ID rows, legacy TP8/DCP8 only.
    if dim != 0 or world_size != 8 or len(shape) != 2 or shape[1] != 2048:
        return False
    input_bytes = shape[0] * shape[1] * 4
    return (
        8 * 1024**2 < input_bytes <= 12 * 1024**2
        and input_bytes <= pool_bytes
        # The legacy C++ size/offset ABI uses signed 32-bit indices.
        and input_bytes * world_size < 2**31
    )
'''
ALLOW_METHOD = '''
    def _allow_medium_dcp_topk_gather(self, comm, input_, dim):
        # Existing candidate/resolve guards retain same ordered TP/DCP ranks,
        # no ubatching, no cuMem allocator, and a live TP-owned communicator.
        if not self._dcp_ag_medium_topk or input_.dtype != torch.int32:
            return False
        impl = comm._impl
        if (
            not getattr(impl, "_same_node", False)
            or not getattr(impl, "fully_connected", False)
            or getattr(impl, "disabled", True)
            or getattr(impl, "_is_gfx1250", True)
            or getattr(impl, "_use_vmm", True)
            or getattr(impl, "_use_symm_mem", True)
            or getattr(impl, "_ops_all_gather", True) is not None
        ):
            return False
        return _medium_dcp_topk_gather_shape_ok(
            input_.shape, dim, self.world_size, impl._pool["input"].max_size
        )
'''


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_source(source_root, output):
    identities = {}
    for name, expected in SOURCE_HASHES.items():
        identities[name] = digest(source_root / name)
        require(identities[name] == expected, 'Pinned source mismatch: ' + name)
    source = (source_root / next(iter(SOURCE_HASHES))).read_text()
    replacements = [
        ('AITER_ALL_GATHER_PACK_BYTES = 16\n', 'AITER_ALL_GATHER_PACK_BYTES = 16\n' + SHAPE_HELPER),
        ('        # Log the first AITER gather and the first fall back to RCCL.\n',
         '        self._dcp_ag_medium_topk = False\n'
         '        if self._dcp_ag_candidate and envs.VLLM_ROCM_DCP_AITER_MEDIUM_TOPK:\n'
         '            from vllm.platforms.rocm import on_gfx950\n\n'
         '            self._dcp_ag_medium_topk = on_gfx950()\n'
         '        # Log the first AITER gather and the first fall back to RCCL.\n'),
        ('\n    def _dcp_aiter_all_gather(\n', ALLOW_METHOD + '\n    def _dcp_aiter_all_gather(\n'),
        ('        if not comm.should_custom_ag(payload):\n',
         '        if not comm.should_custom_ag(payload) and not self._allow_medium_dcp_topk_gather(\n'
         '            comm, input_, dim\n'
         '        ):\n'),
    ]
    for old, new in replacements:
        require(source.count(old) == 1, 'Ambiguous candidate source edit')
        source = source.replace(old, new, 1)
    require(hashlib.sha256(source.encode()).hexdigest() == CANDIDATE_SHA256,
            'Candidate differs from original research patch')
    target = output / 'candidate_cuda_communicator.py'
    target.write_text(source)
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == '_medium_dcp_topk_gather_shape_ok':
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == 'CudaCommunicator':
            nodes.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and
                         n.name in ('_allow_medium_dcp_topk_gather', '_dcp_aiter_all_gather'))
    require(len(nodes) == 3, 'Missing exact candidate functions')
    code = compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(target), 'exec')
    return code, dict(baseline=identities, candidate_sha256=CANDIDATE_SHA256,
                      original_research_patch_sha256=RESEARCH_PATCH_SHA256)


def validate_guards(code):
    ns = {'torch': SimpleNamespace(Tensor=object, int32='int32')}
    exec(code, ns)
    shape_ok, allow = ns['_medium_dcp_topk_gather_shape_ok'], ns['_allow_medium_dcp_topk_gather']
    pool = 128 * 1024**2
    for rows in (1024, 1025, 1152, 1536, 1537):
        require(shape_ok((rows, 2048), 0, 8, pool) == (1024 < rows <= 1536), 'Row boundary')
    for shape, dim, world, capacity in [((1152, 1024), 0, 8, pool), ((1152, 2048), 1, 8, pool),
            ((1152, 2048), 0, 4, pool), ((2359296,), 0, 8, pool),
            ((1152, 2048), 0, 8, 9 * 1024**2 - 1)]:
        require(not shape_ok(shape, dim, world, capacity), 'Unsupported shape admitted')
    attrs = dict(_same_node=True, fully_connected=True, disabled=False, _is_gfx1250=False,
                 _use_vmm=False, _use_symm_mem=False, _ops_all_gather=None,
                 _pool={'input': SimpleNamespace(max_size=pool)})
    owner = SimpleNamespace(_dcp_ag_medium_topk=True, world_size=8)
    tensor = SimpleNamespace(dtype='int32', shape=(1152, 2048))
    require(allow(owner, SimpleNamespace(_impl=SimpleNamespace(**attrs)), tensor, 0), 'Eligible guard')
    for key in ('_same_node', 'fully_connected', 'disabled', '_is_gfx1250', '_use_vmm', '_use_symm_mem', '_ops_all_gather'):
        for missing in (False, True):
            changed = dict(attrs)
            if missing:
                changed.pop(key)
            else:
                changed[key] = not changed[key] if isinstance(changed[key], bool) else object()
            require(not allow(owner, SimpleNamespace(_impl=SimpleNamespace(**changed)), tensor, 0), key)
    owner._dcp_ag_medium_topk = False
    require(not allow(owner, SimpleNamespace(_impl=SimpleNamespace(**attrs)), tensor, 0), 'Disabled flag')
    owner._dcp_ag_medium_topk = True
    tensor.dtype = 'int64'
    require(not allow(owner, SimpleNamespace(_impl=SimpleNamespace(**attrs)), tensor, 0), 'Dtype guard')
    return 'passed: exact source hash, boundaries, disabled flag, dtype and fail-closed transport guards'


def distribution(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values), std=statistics.pstdev(values), n=len(values))


@contextmanager
def observe(shared, dcp, dist):
    """Temporary dispatch counters; removed before timing or graph replay."""
    calls = dict(aiter_gather=0, rccl_gather=0, aiter_all_reduce=0)
    original_ag, original_ar = shared.custom_all_gather, shared.custom_all_reduce
    original_rccl = dist.all_gather_into_tensor
    def ag(*a, **kw):
        calls['aiter_gather'] += 1
        return original_ag(*a, **kw)
    def ar(*a, **kw):
        calls['aiter_all_reduce'] += 1
        return original_ar(*a, **kw)
    def rccl(*a, **kw):
        if kw.get('group') is dcp.device_group:
            calls['rccl_gather'] += 1
        return original_rccl(*a, **kw)
    shared.custom_all_gather, shared.custom_all_reduce = ag, ar
    dist.all_gather_into_tensor = rccl
    try:
        yield calls
    finally:
        shared.custom_all_gather, shared.custom_all_reduce = original_ag, original_ar
        dist.all_gather_into_tensor = original_rccl


def worker(args):
    import torch
    import torch.distributed as dist
    import vllm.distributed.device_communicators.cuda_communicator as native
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.distributed import (init_distributed_environment, initialize_model_parallel,
                                 destroy_model_parallel, destroy_distributed_environment)
    from vllm.distributed.parallel_state import get_tp_group, get_dcp_group, graph_capture
    from vllm.device_allocator.alloc_conf import non_expandable_allocations
    from aiter.dist.device_communicators.custom_all_reduce import _ipc_exportable
    from vllm.platforms.rocm import on_gfx950
    torch.set_num_threads(1)
    torch.cuda.set_device(args.rank)
    require('MI355X' in torch.cuda.get_device_name(args.rank).upper(), 'Requires MI355X')
    source_root = Path(inspect.getsourcefile(native)).parents[3]
    code, identities = prepare_source(source_root, args.output)
    record = dict(rank=args.rank, status='initializing', sources=identities, cases=[],
                  guard_validation=validate_guards(code), probe_sha256=digest(Path(__file__)),
                  torch=torch.__version__, hip=torch.version.hip, device=torch.cuda.get_device_name(),
                  runtime_environment={k: v for k, v in os.environ.items() if k.startswith(
                      ('VLLM_', 'AITER_', 'HIP_', 'HSA_', 'ROCPROF', 'PYTORCH_'))})
    save = lambda: write(args.output / 'result.json', record)
    save()
    config = VllmConfig(compilation_config=CompilationConfig(
        mode=0, cudagraph_mode='FULL_AND_PIECEWISE', custom_ops=['+fused_rms_norm_gated']))
    config.kernel_config.set_platform_defaults(config)
    config.kernel_config.ir_op_priority.set_default()
    with set_current_vllm_config(config), torch.inference_mode():
        init_distributed_environment(8, args.rank, args.rendezvous, args.rank,
                                     timeout=datetime.timedelta(seconds=60))
        initialize_model_parallel(8, decode_context_model_parallel_size=8)
        tp, group = get_tp_group(), get_dcp_group()
        dcp, shared = group.device_communicator, tp.device_communicator.aiter_ar_comm
        require(dcp._dcp_ag_candidate and on_gfx950(), 'Native DCP AITER/gfx950 eligibility absent')
        require(shared is not None and dcp._resolve_dcp_ag_comm() is shared, 'DCP must borrow native TP communicator')
        require(list(dcp.ranks) == list(tp.device_communicator.ranks) == list(range(8)), 'Ordered TP/DCP ranks differ')
        ns = dict(vars(native))
        exec(code, ns)
        original_method = dcp._dcp_aiter_all_gather
        candidate_method = MethodType(ns['_dcp_aiter_all_gather'], dcp)
        dcp._allow_medium_dcp_topk_gather = MethodType(ns['_allow_medium_dcp_topk_gather'], dcp)
        dcp._dcp_ag_medium_topk = True
        probe = torch.empty((1152, 2048), device='cuda', dtype=torch.int32)
        require(dcp._allow_medium_dcp_topk_gather(shared, probe, 0), 'Legacy transport guard rejected; do not force backend')
        del probe
        impl = shared._impl
        pool_pointer = impl._pool['input'].data_ptr
        record['transport'] = dict(tp_ranks=list(tp.ranks), dcp_ranks=list(group.ranks),
            shared_identity=True, pool_bytes=impl._pool['input'].max_size,
            flags={k: getattr(impl, k) for k in ('_same_node', 'fully_connected', 'disabled',
                                               '_is_gfx1250', '_use_vmm', '_use_symm_mem')},
            constructor_scope='Native initialization; exact candidate gather methods bound to DCP only. '
                              'Candidate opt-in is per-instance; env registration and full model are not exercised.')
        record['effective_config'] = dict(compilation_mode=str(config.compilation_config.mode),
            cudagraph_mode=str(config.compilation_config.cudagraph_mode),
            use_ubatching=config.parallel_config.use_ubatching,
            cumem_graph_pool=config.use_cumem_cudagraph_pool)
        with non_expandable_allocations():
            ar_before = torch.empty((16, 7168), device='cuda', dtype=torch.bfloat16)
            ar_after = torch.empty_like(ar_before)
        require(shared.should_custom_ar(ar_before), 'Surrounding TP reductions must use shared AITER pool')
        cases = [dict(shape=[rows, 2048], dim=0, dtype='int32', layout='contiguous', timing=True)
                 for rows in args.rows]
        cases += [dict(shape=shape, dim=dim, dtype=dtype, layout=layout, timing=layout != 'contiguous')
                  for shape, dim, dtype, layout in [
                      ([1152, 2048], 0, 'int32', 'rank-offset'),
                      ([1152, 2048], 0, 'int32', 'strided'),
                      ([576, 4096], 0, 'int32', 'contiguous'),
                      ([1152, 2048], 1, 'int32', 'contiguous'),
                      ([576, 2048], 0, 'int64', 'contiguous'),
                      ([2359296], 0, 'int32', 'contiguous')]]
        for spec in cases:
            shape, dim, dtype = spec['shape'], spec['dim'], getattr(torch, spec['dtype'])
            count = 1
            for n in shape:
                count *= n
            # Persistent replay inputs must also use exportable allocations;
            # switching allocator mode only during capture cannot repair them.
            with non_expandable_allocations():
                if spec['layout'] == 'strided':
                    storage = torch.empty((*shape[:-1], shape[-1] * 2), device='cuda', dtype=dtype)
                    x = storage[..., ::2]
                else:
                    offset = args.rank % 4 if spec['layout'] == 'rank-offset' else 0
                    storage = torch.empty(count + 4, device='cuda', dtype=dtype)
                    x = storage[offset:offset + count].view(shape)
            case = dict(**spec, status='correctness', checks=[], samples=[],
                        input_bytes=x.numel() * x.element_size(), strides=list(x.stride()),
                        alignment_mod16=x.data_ptr() % 16)
            case['native_should_custom_ag'] = shared.should_custom_ag(x.contiguous().view(-1).view(torch.int32))
            require(case['native_should_custom_ag'] == (case['input_bytes'] <= 8 * 1024**2),
                    'Native inclusive 8 MiB eligibility differs from pinned expectation')
            record['cases'].append(case)
            save()
            ipc_checks = {name: _ipc_exportable(tensor.data_ptr()) for name, tensor in
                          (('gather_input', x), ('pre_reduce_input', ar_before), ('post_reduce_input', ar_after))}
            all_ipc = [None] * 8
            dist.all_gather_object(all_ipc, ipc_checks, group=tp.cpu_group)
            case['persistent_input_ipc_checks'] = all_ipc
            save()
            require(all(all(checks.values()) for checks in all_ipc),
                    'Persistent inputs are not IPC-exportable on every rank; no backend fallback attempted')
            def payload(rank, epoch):
                index = torch.arange(count, device='cuda', dtype=torch.int32)
                value = index * 1664525 + rank * 123457 + epoch * 33407
                # Raw integer words include +/- extrema, negative IDs and float NaNs.
                special = torch.tensor([0, -1, -2147483648, 2147483647, 2143289345,
                                        2139095041, -4194303, -8388607], device='cuda', dtype=torch.int32)
                value[:8].copy_(special.roll(rank + epoch))
                return value.to(dtype).view(shape)
            def update(epoch):
                x.copy_(payload(args.rank, epoch))
                ar_before.fill_(args.rank + 1 + epoch)
                ar_after.fill_(2 * args.rank + epoch)
            def select(name):
                dcp._dcp_ag_candidate = name != 'rccl'
                dcp._dcp_ag_medium_topk = name == 'candidate'
                dcp._dcp_aiter_all_gather = original_method if name == 'native' else candidate_method
            def run(name, scope='surrounded'):
                before = tp.all_reduce(ar_before) if scope == 'surrounded' else None
                # The native RCCL base requires contiguous input. Include this
                # necessary normalization in the measured complete operation.
                inp = x.contiguous() if name == 'rccl' else x
                gathered = group.all_gather(inp, dim=dim)
                after = tp.all_reduce(ar_after) if scope == 'surrounded' else None
                return before, gathered, after
            def expected_backend(name):
                if name == 'rccl':
                    return 'rccl'
                if x.numel() * x.element_size() <= 8 * 1024**2:
                    return 'aiter'
                eligible = (dtype == torch.int32 and dim == 0 and len(shape) == 2 and
                            shape[1] == 2048 and 1024 < shape[0] <= 1536)
                return 'aiter' if name == 'candidate' and eligible else 'rccl'
            def check(outputs, epoch, label, calls=None):
                before, gathered, after = outputs
                expected = torch.cat([payload(rank, epoch) for rank in range(8)], dim=dim)
                result = dict(label=label, epoch=epoch, gather_bitwise_exact=bool(torch.equal(gathered, expected)),
                    dtype_exact=gathered.dtype == expected.dtype,
                    pre_reduce_exact=before is None or bool(torch.equal(before, torch.full_like(before, 36 + 8 * epoch))),
                    post_reduce_exact=after is None or bool(torch.equal(after, torch.full_like(after, 56 + 8 * epoch))))
                if calls is not None:
                    result['observed_dispatches'] = dict(calls)
                case['checks'].append(result)
                save()
                require(all(result[k] for k in ('gather_bitwise_exact', 'dtype_exact', 'pre_reduce_exact', 'post_reduce_exact')),
                        'Exact gather/shared-signal parity failed: ' + label)
            def attest(name, calls, scope='surrounded'):
                backend = expected_backend(name)
                require(calls == dict(aiter_gather=int(backend == 'aiter'), rccl_gather=int(backend == 'rccl'),
                                      aiter_all_reduce=2 if scope == 'surrounded' else 0),
                        'Unexpected native dispatch: ' + str(calls))
                require(impl._pool['input'].data_ptr == pool_pointer, 'Shared pool replaced')
            # Native legacy fallback does not normalize strided RCCL input;
            # only compare the supported repaired candidate with contiguous RCCL.
            variants = ('rccl', 'candidate') if spec['layout'] == 'strided' else ('rccl', 'native', 'disabled', 'candidate')
            update(0)
            for name in variants:
                select(name)
                with observe(shared, dcp, dist) as calls:
                    outputs = run(name)
                torch.cuda.synchronize()
                attest(name, calls)
                check(outputs, 0, name + '-eager', calls)
                del outputs
            graphs, graph_outputs = {}, {}
            scopes = ('gather', 'surrounded') if spec['timing'] else ('surrounded',)
            for scope in scopes:
                for name in ('rccl', 'candidate'):
                    select(name)
                    for _ in range(2):
                        run(name, scope)
                    torch.cuda.synchronize()
                    dist.barrier()
                    key = scope + ':' + name
                    with graph_capture(torch.device('cuda', args.rank)) as capture:
                        graph = torch.cuda.CUDAGraph()
                        with observe(shared, dcp, dist) as calls:
                            with torch.cuda.graph(graph, stream=capture.stream):
                                graph_outputs[key] = run(name, scope)
                        case.setdefault('graph_capture', {})[key] = dict(
                            registered=impl.enable_register_for_capturing, calls=dict(calls))
                        attest(name, calls, scope)
                    graphs[key] = graph
                    torch.cuda.synchronize()
            for epoch in (1, 2, 3):
                update(epoch)
                for key, graph in graphs.items():
                    graph.replay()
                    torch.cuda.synchronize()
                    check(graph_outputs[key], epoch, key + '-changed-input-graph')
            if spec['timing']:
                case['status'] = 'timing'
                save()
                for scope in scopes:
                    for mode in ('eager', 'graph'):
                        for iteration in range(args.iterations):
                            order = ('rccl', 'candidate') if iteration % 2 == 0 else ('candidate', 'rccl')
                            for name in order:
                                select(name)
                                dist.barrier()
                                torch.cuda.synchronize()
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record()
                                began = time.perf_counter_ns()
                                if mode == 'graph':
                                    graphs[scope + ':' + name].replay()
                                else:
                                    outputs = run(name, scope)
                                end.record()
                                end.synchronize()
                                wall_us = (time.perf_counter_ns() - began) / 1000
                                case['samples'].append(dict(scope=scope, mode=mode, variant=name, iteration=iteration,
                                    gpu_us=start.elapsed_time(end) * 1000, host_synchronized_us=wall_us))
                                if mode == 'eager':
                                    del outputs
                case['distributions_gpu_us'] = {scope: {mode: {name: distribution([s['gpu_us'] for s in case['samples']
                    if s['scope'] == scope and s['mode'] == mode and s['variant'] == name]) for name in ('rccl', 'candidate')}
                    for mode in ('eager', 'graph')} for scope in scopes}
            case['status'] = 'passed'
            save()
            del graphs, graph_outputs, graph, x, storage
        record['status'] = 'passed'
        record['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['timing_scope'] = ('Complete DCP gather wrapper alone and separately between two native TP AITER reductions; '
            'eager includes allocation/staging; graph measures replay of captured operations. '
            'Host time includes event recording/completion wait. Synthetic payloads, warm allocator, no E2E claim.')
        record['graph_allocation_policy'] = ('Native parallel_state.graph_capture manages capture allocation policy; '
            'persistent inputs use native non_expandable_allocations before capture and pass all-rank IPC checks. '
            'TP owns the shared AITER communicator; the borrowed DCP communicator is not captured a second time.')
        save()
        destroy_model_parallel()
        destroy_distributed_environment()


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    if args.validate_only:
        code, sources = prepare_source(args.source_root, args.output)
        write(args.output / 'result.json', dict(status=validate_guards(code), sources=sources, gpu_tested=False))
        return 0
    if args.worker:
        try:
            worker(args)
            return 0
        except Exception:
            (args.output / 'error.txt').write_text(traceback.format_exc())
            raise
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        rendezvous = 'tcp://127.0.0.1:' + str(sock.getsockname()[1])
    children, logs = [], []
    reason = None
    try:
        for rank in range(8):
            log = (args.output / f'rank-{rank}.log').open('w')
            logs.append(log)
            argv = [sys.executable, __file__, '--worker', '--rank', str(rank), '--rendezvous', rendezvous,
                    '--output', str(args.output / f'rank-{rank}'), '--iterations', str(args.iterations),
                    '--rows', *map(str, args.rows)]
            children.append(subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + args.timeout_seconds
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                reason = 'worker_failed'
                break
            if time.monotonic() >= deadline:
                reason = 'worker_timeout'
                break
            time.sleep(0.5)
    finally:
        alive = [p for p in children if p.poll() is None]
        for p in alive:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        stop = time.monotonic() + 5
        for p in alive:
            try:
                p.wait(timeout=max(0.01, stop - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait(timeout=5)
        for log in logs:
            log.close()
    workers, errors = [], []
    for rank in range(8):
        try:
            workers.append(json.loads((args.output / f'rank-{rank}/result.json').read_text()))
        except (OSError, ValueError) as exc:
            errors.append(dict(rank=rank, error=str(exc)))
    success = len(workers) == 8 and all(p.returncode == 0 for p in children) and all(w['status'] == 'passed' for w in workers)
    aggregate = []
    if success:
        for index, case in enumerate(workers[0]['cases']):
            if not case['timing']:
                continue
            ranks = [w['cases'][index] for w in workers]
            max_rank = []
            for sample in case['samples']:
                key = (sample['scope'], sample['mode'], sample['variant'], sample['iteration'])
                matching = [[s for s in rank['samples'] if (s['scope'], s['mode'], s['variant'], s['iteration']) == key] for rank in ranks]
                require(all(len(m) == 1 for m in matching), 'Rank timing populations differ')
                max_rank.append(dict(scope=key[0], mode=key[1], variant=key[2], iteration=key[3],
                                     gpu_us=max(m[0]['gpu_us'] for m in matching)))
            aggregate.append(dict(shape=case['shape'], dtype=case['dtype'], dim=case['dim'], layout=case['layout'],
                max_rank_samples=max_rank,
                max_rank_gpu_us={scope: {mode: {name: distribution([s['gpu_us'] for s in max_rank
                    if s['scope'] == scope and s['mode'] == mode and s['variant'] == name]) for name in ('rccl', 'candidate')}
                    for mode in ('eager', 'graph')} for scope in ('gather', 'surrounded')}))
    write(args.output / 'result.json', dict(success=success, reason=reason, workers=workers, read_errors=errors,
        returncodes=[p.returncode for p in children], aggregate=aggregate, scope=__doc__))
    print(json.dumps(dict(success=success, reason=reason, output=str(args.output))), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=[1024, 1025, 1152, 1536, 1537])
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--timeout-seconds', type=int, default=480)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--rendezvous')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    if any(n not in (1024, 1025, 1152, 1536, 1537) for n in args.rows) or len(set(args.rows)) != len(args.rows):
        parser.error('Use unique rows from 1024,1025,1152,1536,1537')
    if not 4 <= args.iterations <= 32 or not 30 <= args.timeout_seconds <= 540:
        parser.error('Iterations must be 4..32; subprocess deadline 30..540 seconds')
    if args.validate_only and args.source_root is None:
        parser.error('--validate-only requires --source-root')
    raise SystemExit(main(args))
