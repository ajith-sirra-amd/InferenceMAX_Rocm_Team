"""Reject missing or out-of-window native AgentX GPU captures."""
import argparse
import csv
import json
from pathlib import Path
import re


def validate(logs, seconds, devices=8):
    root = Path(logs) / 'profiles/rocprof'
    result = {'status': 'failed', 'devices': {}, 'errors': []}
    try:
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
            if not 0.8 * seconds <= d['span_seconds'] <= seconds + 5:
                result['errors'].append(device + ': kernel span does not cover the requested window')
    except (OSError, ValueError, KeyError, csv.Error) as error:
        result['errors'].append(str(error))
    if not result['errors']:
        result['status'] = 'passed'
    result['limits'] = 'Device/window check only; inspect dropped events, rank mapping and profiler perturbation separately.'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'capture-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', type=Path)
    p.add_argument('--seconds', type=int, required=True)
    p.add_argument('--devices', type=int, default=8)
    args = p.parse_args()
    result = validate(args.logs, args.seconds, args.devices)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
