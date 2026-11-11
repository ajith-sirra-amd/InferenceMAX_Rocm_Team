"""Bounded isolated BF16 bmm.out replay using directly observed tensor layouts.

Exact kernel-symbol agreement is required before accepting hardware counters.
Input values, allocation addresses, cache state and request scheduling differ
from the original model. This is not a model roofline or throughput benchmark.
"""
import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from evidence_probe import command, write


def validate_case(case):
    observed = case['observed']
    assert observed['operator'] == 'aten::bmm'
    assert observed['input_types'] == ['c10::BFloat16'] * 3
    assert len(observed['input_dims']) == len(observed['input_strides']) == 3
    assert observed['concrete_inputs'] == ['', '', '']
    shapes, strides = observed['input_dims'], observed['input_strides']
    for shape, stride in zip(shapes, strides):
        assert len(shape) == len(stride) == 3
        assert all(isinstance(v, int) and v > 0 for v in shape + stride)
        # Positive, non-overlapping (possibly padded/interleaved) layout.
        span = 1
        for step, size in sorted(zip(stride, shape)):
            if size > 1:
                assert step >= span
                span = step * (size - 1) + span
        assert span * 2 < 1024**3, 'Keep each probe tensor under 1 GiB'
    a, b, c = shapes
    assert a[0] == b[0] == c[0] and a[2] == b[1]
    assert c[1:] == [a[1], b[2]]
    assert len(case['kernel_symbols']) == 1
    return shapes, strides


def worker(args):
    import torch
    case = json.loads(args.case.read_text())
    shapes, strides = validate_case(case)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.cuda.get_device_name(0)
    assert 'MI355X' in device.upper()
    tensors = [torch.empty_strided(shape, stride, dtype=torch.bfloat16,
                                  device='cuda') for shape, stride in zip(shapes, strides)]
    a, b, out = tensors
    a.fill_(1)
    b.fill_(1)
    for _ in range(3):
        torch.bmm(a, b, out=out)
    torch.cuda.synchronize()
    roctx = None
    if args.roctx:
        roctx = ctypes.CDLL(args.roctx)
        for name in ('roctxProfilerResume', 'roctxProfilerPause'):
            f = getattr(roctx, name)
            f.argtypes = [ctypes.c_uint64]
            f.restype = ctypes.c_int
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    if roctx:
        assert roctx.roctxProfilerResume(0) == 0
    for start, end in zip(starts, ends):
        start.record()
        torch.bmm(a, b, out=out)
        end.record()
        torch.cuda.synchronize()
    if roctx:
        assert roctx.roctxProfilerPause(0) == 0
    assert bool(torch.all(out == shapes[0][2]).item()), 'Incorrect bmm output'
    write(args.output / 'worker-result.json', dict(
        payload_valid=True, device=device, torch=torch.__version__, hip=torch.version.hip,
        blas_preference=str(torch.backends.cuda.preferred_blas_library()),
        shapes=[list(t.shape) for t in tensors], strides=[list(t.stride()) for t in tensors],
        input_output_precision='BF16', intermediate_precision='Not measured by this probe',
        iterations=4, warmup_iterations=3,
        isolated_event_duration_us=[s.elapsed_time(e)*1000 for s, e in zip(starts, ends)],
        mathematical_flops_per_call=2*shapes[0][0]*shapes[0][1]*shapes[0][2]*shapes[1][2],
        scope='Repeated-buffer, deterministic all-ones inputs; no model request/cache behavior',
        case_sha256=hashlib.sha256(args.case.read_bytes()).hexdigest()))


def normalize(name):
    return name.removesuffix(' [clone .kd]').removesuffix('.kd')


def validate_rows(rows, case, counter):
    expected = normalize(next(iter(case['kernel_symbols'])))
    match = len(rows) == 4 and all(normalize(r['Kernel_Name']) == expected for r in rows)
    valid = match and all(r['Counter_Name'] == counter
        and math.isfinite(float(r['Counter_Value'])) and float(r['Counter_Value']) >= 0
        and int(r['End_Timestamp']) > int(r['Start_Timestamp']) for r in rows)
    return dict(valid=valid, exact_symbol_match=match, rows=len(rows),
                kernel_symbols=sorted({r['Kernel_Name'] for r in rows}))


def probe(args):
    case = json.loads(args.case.read_text())
    validate_case(case)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'case.json', case)
    os.environ.update(case['server_environment'])
    write(args.output / 'environment.json', case['server_environment'])
    roots = [Path(p) / '_rocm_sdk_devel' / 'lib' for p in sys.path if p]
    roots.append(Path('/opt/rocm/lib'))
    libs = sorted({str(p.resolve()) for root in roots if root.is_dir()
                   for p in root.glob('librocprofiler-sdk-roctx.so.1') if p.is_file()})
    assert len(libs) == 1, 'Installed ROCTX SONAME must be unambiguous'
    write(args.output / 'selected-roctx.json', dict(path=libs[0]))
    result = dict(scope=__doc__, passes={})
    groups = [('baseline', None), ('dram-read', 'TCC_EA0_RDREQ_DRAM_32B_sum'),
              ('dram-write', 'TCC_EA0_WRREQ_WRITE_DRAM_32B_sum'),
              ('bf16-math', 'SQ_INSTS_VALU_MFMA_MOPS_BF16')]
    for group, counter in groups:
        folder = args.output / group
        folder.mkdir()
        task = [sys.executable, __file__, '--worker', '--case', str(args.case), '--output', str(folder)]
        if counter:
            task += ['--roctx', libs[0]]
            task = ['rocprofv3', '--pmc', counter, '--kernel-trace', '--selected-regions',
                    '--disable-signal-handlers', 'true', '-f', 'csv', 'json',
                    '-d', str(folder / 'counters'), '--', *task]
        record = command(args.output, group, task, 100)
        r = dict(counter=counter, command=record, valid=record.get('returncode') == 0)
        worker_path = folder / 'worker-result.json'
        if worker_path.exists():
            r['worker'] = json.loads(worker_path.read_text())
            r['valid'] = r['valid'] and r['worker']['payload_valid']
        else:
            r['valid'] = False
        if counter:
            rows = []
            for file in folder.rglob('*counter_collection.csv'):
                with file.open(newline='') as stream:
                    rows.extend(csv.DictReader(stream))
            r.update(validation=validate_rows(rows, case, counter), counter_rows=rows)
            r['valid'] = r['valid'] and r['validation']['valid']
        result['passes'][group] = r
        write(args.output / 'result.json', result)
        if not r['valid']:
            return 1
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--roctx')
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        raise SystemExit(probe(args))
