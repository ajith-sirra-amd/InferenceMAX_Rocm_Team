"""Short eight-GPU graph/export check; synthetic batch 8, not a Kimi benchmark."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_exports import validate_exports


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def worker(device, root):
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    x = torch.ones((8, 128, 128), dtype=torch.bfloat16, device=f'cuda:{device}')
    y = torch.empty_like(x)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.bmm(x, x, out=y)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.bmm(x, x, out=y)
    write(root / f'ready-{device}.json', dict(pid=os.getpid(), device=device,
          gpu=torch.cuda.get_device_name(device), batch=8, dtype='bfloat16'))
    deadline = time.monotonic() + 120
    while not (root / 'go.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Timed out waiting for all GPU workers')
        time.sleep(.1)
    stop = json.loads((root / 'go.json').read_text())['stop_ns']
    calls = 0
    while time.clock_gettime_ns(time.CLOCK_BOOTTIME) < stop:
        graph.replay()
        torch.cuda.synchronize()
        calls += 1
        time.sleep(.01)
    if not bool(torch.all(y == 128).item()):
        raise RuntimeError(f'GPU {device}: wrong matrix result')
    write(root / f'done-{device}.json', dict(replays=calls, status='passed'))


def workers(root):
    jobs = [subprocess.Popen([sys.executable, __file__, '--output', str(root), '--worker', str(i)])
            for i in range(8)]
    try:
        deadline = time.monotonic() + 120
        while len(list(root.glob('ready-*.json'))) != 8:
            if any(p.poll() is not None for p in jobs) or time.monotonic() > deadline:
                raise RuntimeError('Eight GPU workers did not become ready')
            time.sleep(.2)
        start = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        write(root / 'go.tmp', dict(start_ns=start, stop_ns=start + 15 * 10**9))
        (root / 'go.tmp').rename(root / 'go.json')
        print('All eight GPUs ready; replaying batch-8 graphs for 15 seconds.', flush=True)
        codes = [p.wait(timeout=300) for p in jobs]
        if any(codes):
            raise RuntimeError(f'GPU worker exit codes: {codes}')
    finally:
        for p in jobs:
            if p.poll() is None:
                p.terminate()
        for p in jobs:
            p.wait(timeout=60)


def check(root):
    root.mkdir(parents=True, exist_ok=False)
    command = ['rocprofv3', '--kernel-trace', '--hip-graph-trace', '--memory-copy-trace',
               '--marker-trace', '--stats', '--disable-signal-handlers', 'true',
               '--collection-period', '0:60:1', '--collection-period-unit', 'sec',
               '-f', 'csv', 'json', '-d', str(root / 'traces'), '--',
               sys.executable, __file__, '--output', str(root), '--workers']
    result = dict(status='failed', scope='Synthetic batch-8 BF16 graphs on eight GPUs; not Kimi or SA replication',
                  command=command, steady_seconds=15, required_common_seconds=10)
    try:
        # The child returns only after its GPU workers and profiler exporters exit.
        with (root / 'profiler.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=480)
        print('Workers exited; validating completed CSV/JSON exports.', flush=True)
        result['exports'] = validate_exports(root / 'traces')
        phase = json.loads((root / 'go.json').read_text())
        spans = {}
        for path in (root / 'traces').rglob('*kernel_trace.csv'):
            with path.open(newline='') as stream:
                for row in csv.DictReader(stream):
                    start, end = int(row['Start_Timestamp']), int(row['End_Timestamp'])
                    if phase['start_ns'] <= start <= end <= phase['stop_ns']:
                        d = spans.setdefault(row['Agent_Id'], dict(first_ns=start, last_ns=end, calls=0))
                        d.update(first_ns=min(d['first_ns'], start), last_ns=max(d['last_ns'], end), calls=d['calls']+1)
        if len(spans) != 8:
            raise RuntimeError(f'Expected eight steady GPU traces, got {len(spans)}')
        common = (min(d['last_ns'] for d in spans.values()) - max(d['first_ns'] for d in spans.values())) / 1e9
        result.update(devices=spans, common_steady_seconds=common)
        if common < 10:
            raise RuntimeError(f'Only {common:.3f}s of common steady GPU coverage')
        if len(list(root.glob('done-*.json'))) != 8:
            raise RuntimeError('Missing successful worker completion records')
        result['status'] = 'passed'
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError) as error:
        result['error'] = str(error)
    write(root / 'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', type=int)
    p.add_argument('--workers', action='store_true')
    a = p.parse_args()
    if a.worker is not None:
        worker(a.worker, a.output)
    elif a.workers:
        workers(a.output)
    else:
        raise SystemExit(check(a.output))
