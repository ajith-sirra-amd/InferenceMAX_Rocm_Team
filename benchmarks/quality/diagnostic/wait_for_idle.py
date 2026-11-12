"""Observe eight idle GPUs before launching; write a sibling availability log."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time


def snapshot(sysfs=Path('/sys')):
    record = dict(at=datetime.now(timezone.utc).isoformat(),
                  hostname=socket.gethostname(), state='error', devices=[], pids=[])
    try:
        cards = sorted(p for p in (sysfs / 'class/drm').iterdir()
                       if p.name.startswith('card') and p.name[4:].isdigit())
        for card in cards:
            device = card / 'device'
            total_path = device / 'mem_info_vram_total'
            try:
                total_path.stat()
            except FileNotFoundError:
                continue
            total = int(total_path.read_text().strip())
            used = int((device / 'mem_info_vram_used').read_text().strip())
            if total <= 0 or not 0 <= used <= total:
                raise ValueError(f'Invalid VRAM counters for {device}')
            required = (91 * total + 99) // 100 + 2 * 1024**3
            record['devices'].append(dict(device=str(device), total_bytes=total,
                                          used_bytes=used, free_bytes=total - used,
                                          required_free_bytes=required))
        record['pids'] = sorted(int(p.name) for p in
                                (sysfs / 'class/kfd/kfd/proc').iterdir()
                                if p.name.isdigit())
        if len(record['devices']) != 8:
            raise ValueError(f"Expected 8 VRAM devices, found {len(record['devices'])}")
        enough_memory = all(d['free_bytes'] >= d['required_free_bytes']
                            for d in record['devices'])
        record['state'] = 'ready' if enough_memory and not record['pids'] else 'busy'
    except (OSError, ValueError) as error:
        record['error'] = f'{type(error).__name__}: {error}'
    return record


def wait_for_idle(log_path, *, sysfs=Path('/sys'), timeout=7200, interval=300,
                  monotonic=time.monotonic, sleep=time.sleep):
    if timeout <= 0 or interval <= 0:
        raise ValueError('Timeout and polling interval must be positive')
    start = monotonic()
    while True:
        record = snapshot(sysfs)
        elapsed = monotonic() - start
        record.update(elapsed_seconds=elapsed, timeout_seconds=timeout,
                      poll_interval_seconds=interval)
        if elapsed >= timeout and record['state'] != 'error':
            record['observed_state'] = record['state']
            record['state'] = 'timeout'
        line = json.dumps(record)
        with log_path.open('a', encoding='utf-8') as sink:
            sink.write(line + '\n')
            sink.flush()
        print(line, flush=True)
        if record['state'] == 'ready':
            return 0
        if record['state'] == 'error':
            return 2
        if record['state'] == 'timeout':
            return 1
        sleep(min(interval, timeout - elapsed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('availability_log', type=Path)
    args = parser.parse_args()
    log_path = args.availability_log.resolve()
    base = Path('/share_nfs/ajith/benchmarks')
    if (not log_path.is_relative_to(base)
            or not log_path.name.endswith('.availability.jsonl')):
        parser.error('Use /share_nfs/ajith/benchmarks/...availability.jsonl')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return wait_for_idle(log_path)


if __name__ == '__main__':
    raise SystemExit(main())
