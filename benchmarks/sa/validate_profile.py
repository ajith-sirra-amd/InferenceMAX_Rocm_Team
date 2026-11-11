"""Reject missing or out-of-window native AgentX GPU captures."""
import argparse
import csv
import json
from pathlib import Path
import re

from validate_exports import validate_exports


def validate(logs, seconds, devices=8, output=None, require_attachment=False, minimum_seconds=None,
             require_exports=False):
    root = Path(logs) / 'profiles/rocprof'
    result = {'status': 'failed', 'requested_seconds': seconds,
              'stop_grace_seconds': 15, 'devices': {}, 'errors': []}
    try:
        if minimum_seconds is not None and (type(minimum_seconds) is not int or not 1 <= minimum_seconds <= seconds):
            raise ValueError('minimum_seconds must be a positive integer no greater than seconds')
        for log in Path(logs).glob('*_agg_w*.out'):
            for line in log.read_text(errors='replace').splitlines():
                if re.search(r'Segfault encountered|Segmentation fault|buffer\s+overrun|'
                             r'dropped\s+[1-9]\d*\s+(?:events|records)|'
                             r'workers still running after (?:grace period|SIGTERM)', line, re.I):
                    result['errors'].append(log.name + ': ' + line.strip()[:300])
        if require_attachment:
            attachment = json.loads((root / 'attachment-status.json').read_text())
            if attachment.get('status') != 'passed' or attachment.get('exported_while_workers_alive') is not True:
                raise ValueError('Bounded attachment did not export while all workers remained alive')
            result['attachment'] = attachment
        clock = json.loads((root / 'clock-sync.json').read_text())
        low = clock['utc_before_ns'] - clock['boottime_ns']
        high = clock['utc_after_ns'] - clock['boottime_ns']
        phases = re.findall(r'PhaseRecordsStats\(phase=CreditPhase.PROFILING,.*?start_ns=(\d+), sent_end_ns=(\d+)',
                            (Path(logs) / 'benchmark.out').read_text(errors='replace'))
        if len(phases) != 1:
            raise ValueError('Expected exactly one completed AgentX measured phase')
        begin, end = map(int, phases[0])
        result['measurement_utc_ns'] = [begin, end]
        for path in sorted((root / 'traces').rglob('*kernel_trace.csv')):
            with path.open(newline='', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    start = int(row.get('Start_Timestamp') or row['Start_Timestamp_ns'])
                    stop = int(row.get('End_Timestamp') or row['End_Timestamp_ns'])
                    device = row.get('Agent_Id') or row.get('GPU_ID')
                    if not device or stop < start:
                        raise ValueError('Invalid device or kernel timestamps in ' + path.name)
                    d = result['devices'].setdefault(device, dict(calls=0, first_ns=start, last_ns=stop))
                    d['calls'] += 1
                    d['first_ns'] = min(start, d['first_ns'])
                    d['last_ns'] = max(stop, d['last_ns'])
        if len(result['devices']) != devices:
            result['errors'].append(f"Expected {devices} captured GPU devices, found {len(result['devices'])}")
        for device, d in result['devices'].items():
            d['span_seconds'] = (d['last_ns'] - d['first_ns']) / 1e9
            d['utc_first_ns'] = d['first_ns'] + low
            d['utc_last_ns'] = d['last_ns'] + high
            if not begin <= d['utc_first_ns'] <= d['utc_last_ns'] <= end:
                result['errors'].append(device + ': capture falls outside measured sending period')
            # Recorded graph dispatches can extend beyond the timer: the last
            # smoke requested 60s and exported 70.76s. Keep that allowance
            # explicit and bounded; it never relaxes measured-phase containment.
            if not 0.8 * seconds <= d['span_seconds'] <= seconds + result['stop_grace_seconds']:
                result['errors'].append(device + ': kernel span does not cover the requested window')
        if minimum_seconds is not None and result['devices']:
            # Use the inner clock bracket: every device must cover the same
            # minimum interval inside measured sending, not separate intervals.
            common_begin = max(begin, max(d['first_ns'] + high for d in result['devices'].values()))
            common_end = min(end, min(d['last_ns'] + low for d in result['devices'].values()))
            result['minimum_common_seconds'] = minimum_seconds
            result['common_coverage_utc_ns'] = [common_begin, common_end]
            result['common_coverage_seconds'] = max(0, common_end - common_begin) / 1e9
            if common_end - common_begin < minimum_seconds * 10**9:
                result['errors'].append(f'Common measured GPU coverage is below {minimum_seconds} seconds')
        if require_exports:
            result['exports'] = validate_exports(root / 'traces')
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        result['errors'].append(str(error))
    if not result['errors']:
        result['status'] = 'passed'
    result['limits'] = 'Device/window extent check only; not proof of gap-free capture. Inspect dropped events, rank mapping and profiler perturbation separately.'
    destination = Path(output) if output is not None else root / 'capture-validation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', type=Path)
    p.add_argument('--seconds', type=int, required=True)
    p.add_argument('--devices', type=int, default=8)
    p.add_argument('--output', type=Path, help='Writable report path outside container-owned raw traces')
    p.add_argument('--require-attachment', action='store_true')
    p.add_argument('--minimum-seconds', type=int, help='Required common measured interval across every captured device')
    p.add_argument('--require-exports', action='store_true', help='Require complete, stable JSON matching every worker CSV')
    args = p.parse_args()
    result = validate(args.logs, args.seconds, args.devices, args.output, args.require_attachment,
                      args.minimum_seconds, args.require_exports)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
