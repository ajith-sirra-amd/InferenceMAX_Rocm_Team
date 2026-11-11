"""Bounded pinned-image checks for counter collection and native trace metadata.

Synthetic evidence only: this does not identify Kimi shapes or measure its roofline.
Every subprocess owns its process group and has an export-inclusive deadline.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def workload(mode, output):
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    x = torch.ones((2048, 2048), device='cuda', dtype=torch.bfloat16)
    y = torch.empty_like(x)
    for _ in range(3):
        torch.mm(x, x, out=y)
    torch.cuda.synchronize()
    if mode == 'graph':
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.mm(x, x, out=y)
        for _ in range(4):
            graph.replay()
            torch.cuda.synchronize()
    elif mode == 'metadata':
        from vllm.config import ProfilerConfig
        from vllm.profiler.wrapper import TorchProfilerWrapper
        config = ProfilerConfig(profiler='torch', torch_profiler_dir=str(output),
            torch_profiler_activities=['CPU', 'CUDA'], ignore_frontend=True,
            torch_profiler_record_shapes=True, torch_profiler_with_stack=True,
            torch_profiler_with_memory=True, torch_profiler_with_flops=True,
            torch_profiler_use_gzip=True, torch_profiler_dump_cuda_time_total=False)
        profile = TorchProfilerWrapper(config, 'metadata-probe', 0, ['CPU', 'CUDA'])
        profile.start()
        for _ in range(4):
            torch.mm(x, x, out=y)
            torch.cuda.synchronize()
            profile.step()
        profile.stop()
        write(output / 'native-profiler-config.json', config.model_dump(mode='json'))
        trace = list(output.glob('*.pt.trace.json.gz'))
        if len(trace) != 1:
            raise RuntimeError('Native metadata probe did not export exactly one trace')
        with gzip.open(trace[0], 'rt') as stream:
            events = json.load(stream)['traceEvents']
        categories = collections.Counter(e.get('cat', '') for e in events)
        shapes = [e for e in events if e.get('args', {}).get('Input Dims')]
        memory = [e for e in events if e.get('name') == '[memory]']
        write(output / 'metadata-summary.json', dict(categories=categories,
            shape_event_count=len(shapes), shape_examples=shapes[:6],
            memory_event_count=len(memory), memory_examples=memory[:3],
            trace_sha256=hashlib.sha256(trace[0].read_bytes()).hexdigest()))
        assert categories['kernel'] > 0 and shapes, 'Missing GPU/shape metadata'
    else:
        for _ in range(4):
            torch.mm(x, x, out=y)
            torch.cuda.synchronize()
    assert float(y[0, 0]) == 2048, 'Synthetic GEMM returned wrong value'
    print(json.dumps(dict(status='passed', mode=mode, torch=torch.__version__,
        hip=torch.version.hip, device=torch.cuda.get_device_name(0))), flush=True)


def command(root, name, argv, timeout=100):
    record = dict(argv=argv, timeout_seconds=timeout, started_utc_ns=time.time_ns())
    with (root / (name + '.log')).open('w') as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            record['returncode'] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            record['error'] = 'Export-inclusive command deadline exceeded'
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    record['finished_utc_ns'] = time.time_ns()
    write(root / (name + '-command.json'), record)
    print(name + ': ' + json.dumps(record), flush=True)
    return record


def probe(root):
    root.mkdir(parents=True, exist_ok=False)
    result = dict(scope='Synthetic pinned-image capability; not Kimi performance', checks={})
    result['checks']['rocprof-version'] = command(root, 'rocprof-version', ['rocprofv3', '--version'], 30)
    result['checks']['rocprof-counters'] = command(root, 'rocprof-counters', ['rocprofv3', '--list-avail'], 30)
    result['checks']['metadata'] = command(root, 'metadata',
        [sys.executable, __file__, '--output', str(root / 'metadata'), '--worker', 'metadata'], 120)
    counters = ['FETCH_SIZE', 'WRITE_SIZE', 'SQ_INSTS_VALU_MFMA_MOPS_BF16']
    for mode in ('eager', 'graph'):
        folder = root / ('counters-' + mode)
        folder.mkdir()
        argv = ['rocprofv3', '--pmc', *counters, '--kernel-trace',
            '--disable-signal-handlers', 'true', '-f', 'csv', 'json', '-d', str(folder),
            '--', sys.executable, __file__, '--output', str(folder), '--worker', mode]
        record = command(root, 'counters-' + mode, argv, 120)
        csvs = {}
        for file in folder.rglob('*.csv'):
            with file.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            csvs[str(file.relative_to(root))] = dict(rows=len(rows), examples=rows[:4])
        record['csvs'] = csvs
        result['checks']['counters-' + mode] = record
    result['status'] = 'completed'
    write(root / 'capability-result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['checks']['metadata'].get('returncode') == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=['eager', 'graph', 'metadata'])
    args = parser.parse_args()
    if args.worker:
        args.output.mkdir(parents=True, exist_ok=True)
        workload(args.worker, args.output)
    else:
        raise SystemExit(probe(args.output))
