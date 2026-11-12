"""Short TP8 context-projection comparison using native vLLM linear/norm/gather.

Synthetic weights and replicated inputs, no Kimi load or E2E claim. Installed
packages stay untouched. Candidate is ColumnParallelLinear(gather_output=True).
"""
import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import traceback


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def distribution(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values), std=statistics.pstdev(values), n=len(values))


def worker(args):
    import torch
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.distributed import (init_distributed_environment, initialize_model_parallel,
                                  get_tp_group, destroy_model_parallel, destroy_distributed_environment)
    from vllm.model_executor.layers.linear import ReplicatedLinear, ColumnParallelLinear
    from vllm.model_executor.layers.layernorm import RMSNorm
    torch.set_num_threads(1)
    torch.cuda.set_device(args.rank)
    assert 'MI355X' in torch.cuda.get_device_name(args.rank).upper()
    result = dict(rank=args.rank, status='initializing', cases=[], scope=__doc__,
                  torch=torch.__version__, hip=torch.version.hip,
                  device=torch.cuda.get_device_name(args.rank), sources={})
    for obj, expected in ((ReplicatedLinear, 'd4b97d850eb8bdee5fd2459e7ac688d00623b54f5ec0ea590c08da0173f8b4d4'),
                          (RMSNorm, '45a29c6f8a78c6a76fb6abf4dbe68ff3ab21b73293cd0fe63d612322f098e0c6'),
                          (initialize_model_parallel, 'f37b5ca1fc25c2a872aa0d3c4cc0288abc0a781b2affa7842f3e731fff1bc7ec')):
        path = Path(inspect.getsourcefile(obj))
        result['sources'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert result['sources'][str(path)] == expected, f'Pinned source mismatch: {path}'
    save = lambda: write(args.output / 'result.json', result)
    save()
    # Breakable graphs force mode NONE in the saved SA run; graph mode stays
    # FULL_AND_PIECEWISE. Install platform IR priorities like worker startup.
    # No model path: initialization must not download/load model weights.
    config = VllmConfig(compilation_config=CompilationConfig(
        mode=0, cudagraph_mode='FULL_AND_PIECEWISE', custom_ops=['+fused_rms_norm_gated']))
    config.kernel_config.set_platform_defaults(config)
    config.kernel_config.ir_op_priority.set_default()
    result['effective_config'] = dict(compilation_mode=str(config.compilation_config.mode),
        cudagraph_mode=str(config.compilation_config.cudagraph_mode),
        ir_op_priority=str(config.kernel_config.ir_op_priority))
    result['runtime_environment'] = {key: value for key, value in os.environ.items()
        if key.startswith(('VLLM_', 'AITER_', 'ROCPROF', 'HSA_', 'HIP_', 'ROCBLAS_', 'HIPBLAS_'))
        or key in ('LD_PRELOAD', 'LD_AUDIT', 'ROCTRACER_DOMAIN', 'HSA_TOOLS_LIB', 'HSA_TOOLS_REPORT_LOAD_FAILURE')}
    result['loaded_profile_libraries'] = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if any(s in line.lower() for s in ('rocprof', 'roctrace', 'kineto')) and '/' in line})
    save()
    with set_current_vllm_config(config), torch.inference_mode():
        init_distributed_environment(8, args.rank, args.rendezvous, args.rank,
                                     timeout=datetime.timedelta(seconds=90))
        initialize_model_parallel(8, decode_context_model_parallel_size=8)
        torch.manual_seed(441)
        with torch.device('cuda'):
            baseline = ReplicatedLinear(35840, 7168, bias=False, return_bias=False,
                                        params_dtype=torch.bfloat16)
            candidate = ColumnParallelLinear(35840, 7168, bias=False, return_bias=False,
                                             params_dtype=torch.bfloat16, gather_output=True)
            norm = RMSNorm(7168, eps=1e-5, dtype=torch.bfloat16)
        baseline.weight.normal_(0, 0.005)
        candidate.weight.weight_loader(candidate.weight, baseline.weight)
        assert torch.equal(candidate.weight, baseline.weight[args.rank * 896:(args.rank + 1) * 896])
        for layer in (baseline, candidate):
            layer.quant_method.process_weights_after_loading(layer)
        norm.weight.uniform_(0.75, 1.25)
        result['weight_shapes'] = [list(layer.weight.shape) for layer in (baseline, candidate)]
        result['weight_strides'] = [list(layer.weight.stride()) for layer in (baseline, candidate)]
        result['loader'] = str(candidate.weight.weight_loader)
        result['gemm_impl'] = str(baseline.quant_method._gemm_impl)
        result['status'] = 'testing'
        save()
        for rows in args.rows:
            torch.manual_seed(570 + rows)
            x = torch.randn((rows, 35840), device='cuda', dtype=torch.bfloat16)
            record = dict(rows=rows, state='correctness', samples=[], correctness=[])
            result['cases'].append(record)
            save()
            def run(name):
                return norm((baseline if name == 'baseline' else candidate)(x))
            def check(a, b, label):
                error = (a.float() - b.float()).abs()
                entry = dict(label=label, finite=bool(torch.isfinite(b).all()),
                             max_abs=float(error.max()), rms=float(error.square().mean().sqrt()),
                             unequal_fraction=float((a != b).float().mean()),
                             valid=bool(torch.allclose(a, b, atol=0.03125, rtol=0.015625)))
                record['correctness'].append(entry)
                save()
                assert entry['finite'] and entry['valid'], entry
            a = run('baseline')
            check(a, run('baseline'), 'baseline-repeat')
            check(a, run('candidate'), 'candidate-eager')
            # Independent full-K FP32 dot checks expose shard/layout errors.
            spots = []
            for row, col in ((0, 0), (rows // 2, 3590), (rows - 1, 7167)):
                # Check pre-norm linear against FP32. Full normalized results
                # above retain native RMSNorm in both comparison arms.
                expected = (x[row].float() * baseline.weight[col].float()).sum()
                actual = baseline(x)[row, col].float()
                ok = bool(torch.isclose(actual, expected, atol=0.01, rtol=0.02))
                spots.append(dict(row=row, col=col, expected=float(expected), actual=float(actual), valid=ok))
                assert ok
            record['scalar_checks'] = spots
            graphs, outputs = {}, {}
            for name in ('baseline', 'candidate'):
                for _ in range(3):
                    run(name)
                torch.cuda.synchronize()
                dist.barrier()
                with get_tp_group().graph_capture() as capture:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture.stream):
                        outputs[name] = run(name)
                torch.cuda.synchronize()
                graphs[name] = graph
            x.mul_(0.75).add_(0.125)
            reference = run('baseline')
            for name, graph in graphs.items():
                graph.replay()
                torch.cuda.synchronize()
                check(reference, outputs[name], name + '-changed-input-graph')
            record['state'] = 'timing'
            save()
            for mode in ('eager', 'graph'):
                for i in range(args.iterations):
                    order = ('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline')
                    for name in order:
                        # Equal rank entry; event includes native gather and RMSNorm.
                        dist.barrier()
                        torch.cuda.synchronize()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        if mode == 'graph':
                            graphs[name].replay()
                        else:
                            y = run(name)
                        end.record()
                        end.synchronize()
                        record['samples'].append(dict(mode=mode, iteration=i, variant=name,
                                                       duration_us=start.elapsed_time(end)*1000))
            record['distributions_us'] = {mode: {name: distribution([s['duration_us'] for s in record['samples']
                if s['mode'] == mode and s['variant'] == name]) for name in ('baseline', 'candidate')}
                for mode in ('eager', 'graph')}
            record['state'] = 'passed'
            save()
            del graphs, outputs, reference, a, x
        result['status'] = 'passed'
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['timing_scope'] = 'Complete native linear + gather (candidate) + RMSNorm; isolated rank-synchronized GPU events; warm buffers; no serving/E2E measurement'
        save()
        destroy_model_parallel()
        destroy_distributed_environment()


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
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
    try:
        for rank in range(8):
            log = (args.output / f'rank-{rank}.log').open('w')
            logs.append(log)
            argv = [sys.executable, __file__, '--worker', '--rank', str(rank),
                    '--rendezvous', rendezvous, '--output', str(args.output / f'rank-{rank}'),
                    '--iterations', str(args.iterations), '--rows', *map(str, args.rows)]
            children.append(subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 480
        while time.monotonic() < deadline and any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                break
            time.sleep(1)
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for log in logs:
            log.close()
    results = [json.loads(p.read_text()) for p in sorted(args.output.glob('rank-*/result.json'))]
    success = len(results) == 8 and all(p.returncode == 0 for p in children) and all(r['status'] == 'passed' for r in results)
    write(args.output / 'result.json', dict(success=success, workers=results,
          returncodes=[p.returncode for p in children], scope=__doc__))
    print(json.dumps(dict(success=success, output=str(args.output))), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=[280, 336])
    parser.add_argument('--iterations', type=int, default=16)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--rendezvous')
    args = parser.parse_args()
    if any(n < 1 or n > 1024 for n in args.rows) or not 4 <= args.iterations <= 64:
        parser.error('Bound rows to 1..1024 and iterations to 4..64')
    raise SystemExit(main(args))
