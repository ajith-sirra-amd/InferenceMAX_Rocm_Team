"""Validate complete native Torch exports against the AgentX measured interval."""
import argparse
import gzip
import json
from pathlib import Path
import re


def validate(logs, minimum_seconds):
    root = logs / 'profiles/torch'
    control = json.loads((root / 'profile-control.json').read_text())
    if control['status'] != 'exported':
        raise ValueError('Controller did not complete profile export')
    phases = re.findall(r'PhaseRecordsStats\(phase=CreditPhase.PROFILING,.*?start_ns=(\d+), sent_end_ns=(\d+)',
                        (logs / 'agentic/benchmark.log').read_text(errors='replace'))
    if len(phases) != 1:
        raise ValueError('Expected one completed measured phase')
    begin, end = map(int, phases[0])
    paths = sorted(root.rglob('*.trace.json*'))
    if len(paths) != 8:
        raise ValueError(f'Expected eight worker traces; found {len(paths)}')
    devices, files = {}, []
    for path in paths:
        before = path.stat()
        with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
            trace = json.load(stream)
        base = int(trace.get('baseTimeNanoseconds', 0))
        seen = set()
        count = 0
        for event in trace['traceEvents']:
            if event.get('cat') != 'kernel' or event.get('dur', 0) <= 0:
                continue
            device = str(event['args']['device'])
            seen.add(device)
            start = base + round(event['ts'] * 1000)
            stop = start + round(event['dur'] * 1000)
            d = devices.setdefault(device, dict(calls=0, first_ns=start, last_ns=stop))
            d.update(calls=d['calls'] + 1, first_ns=min(start, d['first_ns']), last_ns=max(stop, d['last_ns']))
            count += 1
        if len(seen) != 1 or count == 0:
            raise ValueError(f'{path.name}: expected one worker GPU and nonzero kernels')
        if (before.st_size, before.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError(f'{path.name}: export still changing')
        files.append(dict(file=path.name, bytes=before.st_size, kernels=count, devices=sorted(seen)))
        print(json.dumps(files[-1]), flush=True)
        del trace
    if set(devices) != set(map(str, range(8))):
        raise ValueError('Expected GPU devices 0–7')
    for d in devices.values():
        if not begin <= d['first_ns'] <= d['last_ns'] <= end:
            raise ValueError('GPU capture is outside measured sending')
    common = (min(d['last_ns'] for d in devices.values()) - max(d['first_ns'] for d in devices.values())) / 1e9
    if common < minimum_seconds:
        raise ValueError(f'Only {common:.3f}s common GPU extent; require {minimum_seconds}s')
    return dict(status='passed', tool='torch', common_coverage_seconds=common,
                measurement_utc_ns=[begin, end], devices=devices, files=files,
                limits='Extent is not proof of gap-free recording, decode attribution or profiler overhead.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', type=Path)
    p.add_argument('--minimum-seconds', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        result = validate(a.logs, a.minimum_seconds)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(status='failed', error=str(error))
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 1)
