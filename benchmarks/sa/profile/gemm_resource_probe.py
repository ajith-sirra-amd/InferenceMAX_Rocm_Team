"""Bounded exact-layout BMM counters with dispatch-paired math and traffic.

READ and WRITE remain separate passes. Utilization comes from the installed
SDK expression and retained raw counters, not from launch geometry alone.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import sys

from evidence_probe import command, write
from gemm_probe import normalize, validate_case

GROUPS = [
    ('math-read', ['SQ_INSTS_VALU_MFMA_MOPS_BF16', 'TCC_EA0_RDREQ_DRAM_32B_sum']),
    ('math-write', ['SQ_INSTS_VALU_MFMA_MOPS_BF16', 'TCC_EA0_WRREQ_WRITE_DRAM_32B_sum']),
    ('utilization', ['MfmaUtil', 'SQ_WAVES_sum', 'SQ_VALU_MFMA_BUSY_CYCLES',
                     'GRBM_GUI_ACTIVE', 'SIMD_NUM']),
]


def validate_group(rows, case, names):
    expected = normalize(next(iter(case['kernel_symbols'])))
    paired = defaultdict(list)
    for row in rows:
        paired[(row['Process_Id'], row['Agent_Id'], row['Dispatch_Id'])].append(row)
    valid = len(paired) == 8 and all(Counter(row['Counter_Name'] for row in values)
        == Counter(names) for values in paired.values())
    valid = valid and all(normalize(row['Kernel_Name']) == expected
        and math.isfinite(float(row['Counter_Value'])) and float(row['Counter_Value']) >= 0
        and int(row['End_Timestamp']) > int(row['Start_Timestamp']) for row in rows)
    valid = valid and all(len({(row['Start_Timestamp'], row['End_Timestamp']) for row in values}) == 1
                          for values in paired.values())
    return dict(valid=valid, paired_dispatches=len(paired), expected_counters=names,
                pairing_keys=['Process_Id', 'Agent_Id', 'Dispatch_Id'],
                kernel_symbols=sorted({row['Kernel_Name'] for row in rows}))


def probe(args):
    case = json.loads(args.case.read_text())
    validate_case(case)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'case.json', case)
    os.environ.update(case['server_environment'])
    write(args.output / 'environment.json', case['server_environment'])
    available = command(args.output, 'available-counters', ['rocprofv3', '--list-avail'], 30)
    if available.get('returncode') != 0:
        return 1
    text = (args.output / 'available-counters.log').read_text()
    definitions = {}
    for name in sorted({n for _, names in GROUPS for n in names}):
        match = re.search(r'Counter_Name\s*:\s*' + re.escape(name) + r'\s*\n(.*?)(?=\n\s*\n)', text, re.S)
        assert match, 'Counter is absent from installed SDK: ' + name
        definitions[name] = match.group(1)
    assert 'Unit: percent' in definitions['MfmaUtil']
    assert 'reduce(SQ_VALU_MFMA_BUSY_CYCLES,sum)/(reduce(GRBM_GUI_ACTIVE,max)*SIMD_NUM)*100' in definitions['MfmaUtil']
    write(args.output / 'installed-counter-definitions.json', definitions)
    roots = [Path(p) / '_rocm_sdk_devel' / 'lib' for p in sys.path if p] + [Path('/opt/rocm/lib')]
    libs = sorted({str(p.resolve()) for root in roots if root.is_dir()
                   for p in root.glob('librocprofiler-sdk-roctx.so.1') if p.is_file()})
    assert len(libs) == 1
    write(args.output / 'selected-roctx.json', dict(path=libs[0]))
    result = dict(scope=__doc__, passes={})
    for name, names in [('timing', []), *GROUPS]:
        folder = args.output / name
        folder.mkdir()
        task = [sys.executable, str(Path(__file__).with_name('gemm_layout_probe.py')),
                '--worker', '--original-only', '--iterations', '8' if names else '48',
                '--case', str(args.case), '--output', str(folder)]
        if names:
            task += ['--counter-variant', 'original', '--roctx', libs[0]]
            task = ['rocprofv3', '--pmc', *names, '--kernel-trace', '--selected-regions',
                    '--disable-signal-handlers', 'true', '-f', 'csv', 'json',
                    '-d', str(folder / 'counters'), '--', *task]
        record = command(args.output, name, task, 90)
        entry = dict(command=record, counters=names, valid=record.get('returncode') == 0)
        worker_path = folder / 'worker-result.json'
        if worker_path.exists():
            entry['worker'] = json.loads(worker_path.read_text())
            entry['valid'] = entry['valid'] and entry['worker']['payload_valid']
        else:
            entry['valid'] = False
        if names:
            rows = []
            for file in folder.rglob('*counter_collection.csv'):
                with file.open(newline='') as stream:
                    rows.extend(csv.DictReader(stream))
            entry['counter_rows'] = rows
            entry['validation'] = validate_group(rows, case, names)
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
    raise SystemExit(probe(parser.parse_args()))
