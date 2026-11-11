"""Bounded pinned-image checks for counter collection and native trace metadata.

Synthetic evidence only: this does not identify Kimi shapes or measure its roofline.
Every subprocess owns its process group and has an export-inclusive deadline.
"""
import argparse
import collections
import csv
import dataclasses
import gzip
import hashlib
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
        write(output / 'native-profiler-config.json', dataclasses.asdict(config))
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
    elif mode == 'proton':
        import ctypes
        import triton
        from vllm.config import ProfilerConfig
        from vllm.profiler.wrapper import ProtonProfilerWrapper
        config = ProfilerConfig(profiler='proton', proton_profiler_dir=str(output),
            proton_graph_attribution=True, proton_context='python')
        write(output / 'native-profiler-config.json', dataclasses.asdict(config))
        library = os.environ.get('TRITON_ROCPROFILER_SDK_LIBRARY')
        if library:
            try:
                ctypes.CDLL(library)
            except OSError as error:
                write(output / 'sdk-loader-error.json', dict(library=library, error=str(error)))
                raise
        profile = ProtonProfilerWrapper(config, 'proton-probe')
        graph = torch.cuda.CUDAGraph()
        with profile.capture_cuda_graphs():
            with torch.cuda.graph(graph):
                torch.mm(x, x, out=y)
        profile.start()
        for _ in range(4):
            graph.replay()
            torch.cuda.synchronize()
        profile.stop()
        profile.shutdown()
        files = sorted(output.glob('*.hatchet'))
        assert files, 'No Proton graph-attribution output'
        write(output / 'proton-summary.json', dict(triton=triton.__version__,
            files=[dict(name=f.name, bytes=f.stat().st_size) for f in files]))
    else:
        for _ in range(4):
            torch.mm(x, x, out=y)
            torch.cuda.synchronize()
    assert float(y[0, 0]) == 2048, 'Synthetic GEMM returned wrong value'
    print(json.dumps(dict(status='passed', mode=mode, torch=torch.__version__,
        hip=torch.version.hip, device=torch.cuda.get_device_name(0))), flush=True)


def command(root, name, argv, timeout=100, env=None):
    record = dict(argv=argv, timeout_seconds=timeout, started_utc_ns=time.time_ns())
    if env:
        record['environment_additions'] = env
    with (root / (name + '.log')).open('w') as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, **(env or {})),
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


def probe(root, proton_only=False):
    root.mkdir(parents=True, exist_ok=False)
    result = dict(scope='Synthetic pinned-image capability; not Kimi performance', checks={})
    result['checks']['rocprof-version'] = command(root, 'rocprof-version', ['rocprofv3', '--version'], 30)
    result['checks']['rocprof-counters'] = command(root, 'rocprof-counters', ['rocprofv3', '--list-avail'], 30)
    if not proton_only:
        result['checks']['metadata'] = command(root, 'metadata',
            [sys.executable, __file__, '--output', str(root / 'metadata'), '--worker', 'metadata'], 120)
    search = [Path(shutil.which('rocprofv3')).resolve().parent.parent, Path('/opt/rocm')]
    search += [Path(p) / package for p in sys.path if p
               for package in ('_rocm_sdk_devel', '_rocm_sdk_core')]
    libraries = sorted({str(f.resolve()) for base in search if base.exists()
                        for f in base.rglob('librocprofiler-sdk.so*') if f.is_file()})
    write(root / 'rocprofiler-libraries.json', libraries)
    aliases = root / 'library-aliases'
    aliases.mkdir()
    if len(libraries) == 1:
        # This image ships only the versioned SDK SONAME. Proton dlopens the
        # development name; alias the same bytes inside the new output tree.
        (aliases / 'librocprofiler-sdk.so').symlink_to(libraries[0])
    library_path = ':'.join([str(aliases), *sorted({str(Path(f).parent) for f in libraries})])
    library_path += ':' + os.environ.get('LD_LIBRARY_PATH', '')
    proton_env = {'LD_LIBRARY_PATH': library_path}
    if len(libraries) == 1:
        # Proton's explicit loader directory can override LD_LIBRARY_PATH.
        # Select the already-installed binary; never substitute another SDK.
        proton_env.update(TRITON_ROCPROFILER_SDK_LIBRARY=libraries[0],
                          TRITON_ROCPROFILER_SDK_LIB_PATH=str(Path(libraries[0]).parent))
    result['checks']['proton'] = command(root, 'proton',
        [sys.executable, __file__, '--output', str(root / 'proton'), '--worker', 'proton'], 90,
        env=proton_env)
    groups = {'read-write': ['FETCH_SIZE', 'WRITE_SIZE'],
              'read': ['FETCH_SIZE'], 'write': ['WRITE_SIZE'],
              'math': ['SQ_INSTS_VALU_MFMA_MOPS_BF16', 'SQ_INSTS_VALU_MFMA_MOPS_F8',
                       'SQ_INSTS_VALU_MFMA_MOPS_F6F4']}
    if proton_only:
        groups = {}
    for group, counters in groups.items():
        folder = root / ('counters-graph-' + group)
        folder.mkdir()
        argv = ['rocprofv3', '--pmc', *counters, '--kernel-trace',
            '--disable-signal-handlers', 'true', '-f', 'csv', 'json', '-d', str(folder),
            '--', sys.executable, __file__, '--output', str(folder), '--worker', 'graph']
        record = command(root, 'counters-graph-' + group, argv, 90)
        csvs = {}
        for file in folder.rglob('*.csv'):
            with file.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            csvs[str(file.relative_to(root))] = dict(rows=len(rows), examples=rows[:4])
        record['csvs'] = csvs
        result['checks']['counters-graph-' + group] = record
    result['status'] = 'completed'
    write(root / 'capability-result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    required = 'proton' if proton_only else 'metadata'
    return 0 if result['checks'][required].get('returncode') == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=['eager', 'graph', 'metadata', 'proton'])
    parser.add_argument('--proton-only', action='store_true')
    args = parser.parse_args()
    if args.worker:
        args.output.mkdir(parents=True, exist_ok=True)
        workload(args.worker, args.output)
    else:
        raise SystemExit(probe(args.output, args.proton_only))
