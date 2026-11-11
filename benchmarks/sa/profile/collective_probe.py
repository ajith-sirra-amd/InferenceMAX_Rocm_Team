"""Bounded isolated replay of a trace-bound all-gather payload.

READ and WRITE counters run separately. This does not recreate model arrival,
cache state, request critical path or communicator/channel choices.
"""
import argparse
import csv
import ctypes
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from evidence_probe import command, write


def worker(args):
    import torch
    import torch.distributed as dist
    case = json.loads(args.case.read_text())
    rank, world, count = args.rank, case['world_size'], case['input_elements']
    assert case['dtype'] == 'int32' and case['output_elements'] == count * world
    assert case['launch_mode'] == 'eager' and case['collective'] == '_allgather_base'
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=args.rendezvous, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=60))
    source = torch.full((count,), rank, dtype=torch.int32, device='cuda')
    target = torch.empty((count * world,), dtype=torch.int32, device='cuda')
    for _ in range(3):
        dist.all_gather_into_tensor(target, source)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[rank])
    torch.cuda.synchronize()
    roctx = None
    if args.roctx:
        roctx = ctypes.CDLL(args.roctx)
        for name in ('roctxProfilerResume', 'roctxProfilerPause'):
            function = getattr(roctx, name)
            function.argtypes = [ctypes.c_uint64]
            function.restype = ctypes.c_int
    start = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    if roctx:
        assert roctx.roctxProfilerResume(0) == 0
    for i in range(4):
        start[i].record()
        dist.all_gather_into_tensor(target, source)
        end[i].record()
        torch.cuda.synchronize()
    if roctx:
        assert roctx.roctxProfilerPause(0) == 0
    durations = [s.elapsed_time(e) * 1000 for s, e in zip(start, end)]
    valid = all(bool(torch.all(target[j * count:(j + 1) * count] == j).item()) for j in range(world))
    assert valid, 'All-gather output mismatch'
    record = dict(rank=rank, pid=os.getpid(), device=torch.cuda.get_device_name(rank),
                  case=case, payload_valid=valid, iterations=4,
                  isolated_event_duration_us=durations, counters_enabled=bool(roctx),
                  event_duration_scope='Enclosing event interval, not request E2E',
                  torch=torch.__version__, hip=torch.version.hip,
                  nccl_version=torch.cuda.nccl.version(),
                  replay_group='Fresh default NCCL/RCCL communicator; global ranks 0 through 7',
                  visible_devices={k: os.environ.get(k) for k in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')})
    loaded = sorted({line.split(maxsplit=5)[5] for line in Path('/proc/self/maps').read_text().splitlines()
                     if len(line.split(maxsplit=5)) == 6 and ('librccl' in line or 'libnccl' in line)})
    record['loaded_collective_libraries'] = []
    for name in loaded:
        path = Path(name)
        if path.is_file():
            with path.open('rb') as stream:
                record['loaded_collective_libraries'].append(dict(path=name,
                    sha256=hashlib.file_digest(stream, 'sha256').hexdigest()))
    dist.destroy_process_group()
    write(args.output / 'worker-result.json', record)


def launch(output, case, counter, roctx):
    output.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    workers, logs, records = [], [], []
    try:
        for rank in range(8):
            rank_output = output / ('rank-' + str(rank))
            rank_output.mkdir()
            task = [sys.executable, __file__, '--worker', '--rank', str(rank),
                    '--rendezvous', 'tcp://127.0.0.1:' + str(port),
                    '--case', str(case), '--output', str(rank_output)]
            if counter:
                task += ['--roctx', roctx]
                task = ['rocprofv3', '--pmc', counter, '--kernel-trace', '--selected-regions',
                        '--disable-signal-handlers', 'true', '-f', 'csv', 'json',
                        '-d', str(rank_output / 'counters'), '--', *task]
            log = (rank_output / 'worker.log').open('w')
            logs.append(log)
            env = dict(os.environ, NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,GRAPH,TUNING,COLL',
                       NCCL_DEBUG_FILE=str(rank_output / 'rccl.%h.%p.log'))
            records.append(dict(rank=rank, command=task))
            workers.append(subprocess.Popen(task, stdout=log, stderr=subprocess.STDOUT, env=env))
        deadline = time.monotonic() + 150
        while any(p.poll() is None for p in workers) and time.monotonic() < deadline:
            if any(p.poll() not in (None, 0) for p in workers):
                break
            time.sleep(0.2)
    finally:
        for p in workers:
            if p.poll() is None:
                p.terminate()
        for p in workers:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
        for log in logs:
            log.close()
    for record, process in zip(records, workers):
        record['returncode'] = process.returncode
    write(output / 'commands.json', records)
    summary = dict(counter=counter, workers=records, rank_results=[], counter_rows=[])
    for rank in range(8):
        rank_output = output / ('rank-' + str(rank))
        result = rank_output / 'worker-result.json'
        if result.exists():
            summary['rank_results'].append(json.loads(result.read_text()))
        for path in rank_output.rglob('*counter_collection.csv'):
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            summary['counter_rows'].append(dict(rank=rank, path=str(path.relative_to(output)), rows=rows))
    summary['success'] = len(summary['rank_results']) == 8 and all(p.returncode == 0 for p in workers)
    if counter:
        summary['success'] = summary['success'] and {v['rank'] for v in summary['counter_rows'] if v['rows']} == set(range(8))
        validation = []
        for rank in range(8):
            rows = [row for entry in summary['counter_rows'] if entry['rank'] == rank for row in entry['rows']]
            names = sorted({row['Kernel_Name'] for row in rows})
            valid = len(rows) == 4 and all(
                row['Counter_Name'] == counter and math.isfinite(float(row['Counter_Value']))
                and float(row['Counter_Value']) >= 0
                and int(row['End_Timestamp']) > int(row['Start_Timestamp'])
                and 'ncclDevKernel' in row['Kernel_Name'] for row in rows)
            validation.append(dict(rank=rank, selected_collective_dispatches=len(rows),
                                   kernel_symbols=names, valid=valid))
        summary['selected_region_validation'] = validation
        summary['success'] = summary['success'] and all(v['valid'] for v in validation)
    write(output / 'result.json', summary)
    return summary['success']


def probe(args):
    args.output.mkdir(parents=True, exist_ok=False)
    case = json.loads(args.case.read_text())
    assert case['world_size'] == 8 and case['dtype'] == 'int32'
    env = case['server_environment']
    os.environ.update(env)
    write(args.output / 'case.json', case)
    write(args.output / 'environment.json', {k: v for k, v in os.environ.items()
        if k.startswith(('NCCL_', 'RCCL_', 'VLLM_', 'AITER_', 'HSA_'))})
    source = command(args.output, 'source-inventory', [sys.executable,
        str(Path(__file__).with_name('source_inventory.py')), '--output', str(args.output / 'installed-source')], 60)
    roots = [Path(p) / '_rocm_sdk_devel' / 'lib' for p in sys.path if p]
    roots += [Path('/opt/rocm/lib')]
    roctx = sorted({str(p.resolve()) for root in roots if root.is_dir()
                    for p in root.glob('librocprofiler-sdk-roctx.so*') if p.is_file()})
    write(args.output / 'roctx-libraries.json', roctx)
    if len(roctx) != 1:
        raise RuntimeError('Expected exactly one installed ROCTX library')
    outcomes = {'source_inventory': source.get('returncode') == 0}
    for name, counter in (('baseline', None), ('read', 'FETCH_SIZE'), ('write', 'WRITE_SIZE')):
        outcomes[name] = launch(args.output / name, args.case, counter, roctx[0])
        if not outcomes[name]:
            break
    write(args.output / 'result.json', dict(scope=__doc__, outcomes=outcomes))
    return 0 if all(outcomes.values()) and 'write' in outcomes else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--rendezvous')
    parser.add_argument('--roctx')
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        raise SystemExit(probe(args))
