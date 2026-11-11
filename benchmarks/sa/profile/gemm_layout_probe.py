"""Bounded observed-BMM layout comparison, including the cost of per-call packing.

This tests one isolated pipeline hypothesis; it does not measure model E2E.
Original B/output layouts, BF16 dtype and installed math policy stay unchanged.
"""
import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from evidence_probe import command, write
from gemm_probe import validate_case, validate_rows

VARIANTS = ('original', 'prepacked-a', 'copy-plus-bmm')


def distribution(values):
    ordered = sorted(values)
    return dict(count=len(values), mean=statistics.mean(values),
                median=statistics.median(values), min=min(values), max=max(values),
                std=statistics.pstdev(values), p95=ordered[math.ceil(len(values)*.95)-1])


def worker(args):
    import torch
    case = json.loads(args.case.read_text())
    shapes, strides = validate_case(case)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    assert 'MI355X' in torch.cuda.get_device_name(0).upper()
    a, b, out = [torch.empty_strided(shape, stride, dtype=torch.bfloat16, device='cuda')
                 for shape, stride in zip(shapes, strides)]
    # Small nonuniform integer values are exact in both BF16 and FP32.
    a.copy_((torch.arange(a.numel(), device='cuda', dtype=torch.int32).reshape(a.shape) % 7) - 3)
    b.copy_((torch.arange(b.numel(), device='cuda', dtype=torch.int32).reshape(b.shape) % 5) - 2)
    variants = ('original',) if args.original_only else VARIANTS
    packed = None if args.original_only else a.contiguous()
    copied = None if args.original_only else torch.empty_like(packed)
    reference = torch.bmm(a.float(), b.float()).to(torch.bfloat16)
    # Also independently check sampled FP32-reference results with host scalar dots.
    samples = []
    for batch in (0, shapes[0][0] - 1):
        for row in (0, shapes[0][1] // 2, shapes[0][1] - 1):
            col = (row * 17 + batch * 13) % shapes[1][2]
            aa = a[batch, row, :].cpu().tolist()
            bb = b[batch, :, col].cpu().tolist()
            exact = sum(x*y for x, y in zip(aa, bb))
            rounded = float(torch.tensor(exact, dtype=torch.bfloat16))
            assert float(reference[batch, row, col]) == rounded
            samples.append(dict(batch=batch, row=row, col=col, exact_dot=exact, rounded_bf16=rounded))

    def invoke(variant):
        operand = a
        if variant == 'prepacked-a':
            operand = packed
        elif variant == 'copy-plus-bmm':
            copied.copy_(a)
            operand = copied
        torch.bmm(operand, b, out=out)

    checks = {}
    for variant in variants:
        for _ in range(5):
            invoke(variant)
        torch.cuda.synchronize()
        checks[variant] = bool(torch.equal(out, reference))
        assert checks[variant], 'BF16 result differs from independently computed FP32 reference'
    roctx = None
    if args.roctx:
        roctx = ctypes.CDLL(args.roctx)
        for name in ('roctxProfilerResume', 'roctxProfilerPause'):
            f = getattr(roctx, name)
            f.argtypes, f.restype = [ctypes.c_uint64], ctypes.c_int
    samples_timing = []
    plan = []
    if args.counter_variant:
        # Warm the measured variant last, after all correctness/reference work.
        for _ in range(5):
            invoke(args.counter_variant)
        torch.cuda.synchronize()
        plan = [(i, 0, args.counter_variant) for i in range(args.iterations or 4)]
    else:
        for round_id in range(args.iterations or 48):
            offset = round_id % len(variants)
            order = variants[offset:] + variants[:offset]
            plan.extend((round_id, position, variant) for position, variant in enumerate(order))
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in plan]
    if roctx:
        assert roctx.roctxProfilerResume(0) == 0
    for (round_id, position, variant), (start, end) in zip(plan, events):
        start.record()
        invoke(variant)
        end.record()
        torch.cuda.synchronize()
        samples_timing.append(dict(round=round_id, position=position, variant=variant,
                                   duration_us=start.elapsed_time(end)*1000))
    if roctx:
        assert roctx.roctxProfilerPause(0) == 0
    assert bool(torch.equal(out, reference))
    actual_strides = [list(t.stride()) for t in (a, b, out)]
    assert actual_strides == strides
    record = dict(payload_valid=True, correctness_by_variant=checks,
        reference='Separate full FP32 BMM rounded to BF16, plus six host scalar-dot checks',
        scalar_reference_samples=samples, device=torch.cuda.get_device_name(0),
        torch=torch.__version__, hip=torch.version.hip,
        blas_preference=str(torch.backends.cuda.preferred_blas_library()),
        math_policy=dict(allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        shapes=shapes, original_strides=strides, packed_a_stride=list(packed.stride()) if packed is not None else None,
        actual_input_output_strides=actual_strides, input_output_precision='BF16',
        mathematical_flops_per_call=2*shapes[0][0]*shapes[0][1]*shapes[0][2]*shapes[1][2],
        warmups_per_variant=5, samples=samples_timing,
        timing_scope='Enclosing GPU events; local isolated pipeline, no model E2E',
        prepacked_scope='Packing and allocation performed once outside timings; requires producer to supply contiguous A',
        copy_scope='Preallocated contiguous buffer; each timed call includes A copy plus BMM',
        input_values='A: linear_index % 7 - 3; B: linear_index % 5 - 2',
        case_sha256=hashlib.sha256(args.case.read_bytes()).hexdigest())
    record['distributions_us'] = {variant: distribution([s['duration_us'] for s in samples_timing
                                  if s['variant'] == variant]) for variant in VARIANTS
                                  if any(s['variant'] == variant for s in samples_timing)}
    if not args.counter_variant and not args.original_only:
        by_round = {i: {s['variant']: s['duration_us'] for s in samples_timing if s['round'] == i}
                    for i in range(args.iterations or 48)}
        record['paired_difference_us'] = {variant: distribution([
            row[variant] - row['original'] for row in by_round.values()]) for variant in VARIANTS[1:]}
        record['paired_speedup_ratio'] = {variant: distribution([
            row['original'] / row[variant] for row in by_round.values()]) for variant in VARIANTS[1:]}
    write(args.output / 'worker-result.json', record)


def probe(args):
    case = json.loads(args.case.read_text())
    validate_case(case)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'case.json', case)
    os.environ.update(case['server_environment'])
    write(args.output / 'environment.json', case['server_environment'])
    roots = [Path(p) / '_rocm_sdk_devel' / 'lib' for p in sys.path if p] + [Path('/opt/rocm/lib')]
    libs = sorted({str(p.resolve()) for root in roots if root.is_dir()
                   for p in root.glob('librocprofiler-sdk-roctx.so.1') if p.is_file()})
    assert len(libs) == 1
    write(args.output / 'selected-roctx.json', dict(path=libs[0]))
    groups = [('timing', None, None)] + [(variant + '-' + kind, counter, variant)
        for variant in VARIANTS[:2] for kind, counter in (
            ('read', 'TCC_EA0_RDREQ_DRAM_32B_sum'), ('write', 'TCC_EA0_WRREQ_WRITE_DRAM_32B_sum'))]
    result = dict(scope=__doc__, passes={})
    for name, counter, variant in groups:
        folder = args.output / name
        folder.mkdir()
        task = [sys.executable, __file__, '--worker', '--case', str(args.case), '--output', str(folder)]
        if counter:
            task += ['--counter-variant', variant, '--roctx', libs[0]]
            task = ['rocprofv3', '--pmc', counter, '--kernel-trace', '--selected-regions',
                    '--disable-signal-handlers', 'true', '-f', 'csv', 'json',
                    '-d', str(folder / 'counters'), '--', *task]
        record = command(args.output, name, task, 90)
        entry = dict(command=record, counter=counter, valid=record.get('returncode') == 0)
        worker_path = folder / 'worker-result.json'
        if worker_path.exists():
            entry['worker'] = json.loads(worker_path.read_text())
            entry['valid'] = entry['valid'] and entry['worker']['payload_valid']
        else:
            entry['valid'] = False
        if counter:
            rows = []
            for file in folder.rglob('*counter_collection.csv'):
                with file.open(newline='') as stream:
                    rows.extend(csv.DictReader(stream))
            entry['counter_rows'] = rows
            entry['symbols'] = sorted({r['Kernel_Name'] for r in rows})
            if variant == 'original':
                entry['validation'] = validate_rows(rows, case, counter)
            else:
                # Changing the layout can legitimately select another GEMM variant.
                entry['validation'] = dict(valid=len(rows) == 4 and len(entry['symbols']) == 1
                    and all(r['Counter_Name'] == counter and math.isfinite(float(r['Counter_Value']))
                            and float(r['Counter_Value']) >= 0
                            and int(r['End_Timestamp']) > int(r['Start_Timestamp']) for r in rows),
                    scope='Changed-layout kernel; original symbol need not match')
            entry['valid'] = entry['valid'] and entry['validation']['valid']
        result['passes'][name] = entry
        write(args.output / 'result.json', result)
        if not entry['valid']:
            return 1
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--counter-variant', choices=VARIANTS[:2])
    parser.add_argument('--roctx')
    parser.add_argument('--original-only', action='store_true')
    parser.add_argument('--iterations', type=int, choices=(4, 8, 48))
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        raise SystemExit(probe(args))
