"""Check completed native JSON against each worker's CSV after server shutdown."""
from collections import Counter
import csv
import json
from pathlib import Path


def validate_exports(traces):
    paths = sorted(Path(traces).rglob('*kernel_trace.csv'))
    if not paths:
        raise ValueError('No worker kernel CSV exports found')
    exports = []
    # Read one worker at a time; never load all GPU exports into memory together.
    for csv_path in paths:
        path = csv_path.with_name(csv_path.name.replace('_kernel_trace.csv', '_results.json'))
        before = path.stat()
        with path.open() as stream:
            document = json.load(stream)
        entries = document['rocprofiler-sdk-tool']
        if len(entries) != 1:
            raise ValueError(f'{path.name}: expected one process per export')
        entry = entries[0]
        metadata = entry['metadata']
        if metadata['fini_time'] <= metadata['init_time']:
            raise ValueError(f'{path.name}: missing profiler finalization timestamp')
        kernels = entry['buffer_records']['kernel_dispatch']
        agents = {a['id']['handle']: str(a['logical_node_id']) for a in entry['agents']}
        json_counts = Counter(agents[event['dispatch_info']['agent_id']['handle']] for event in kernels)
        symbols = {s['kernel_id'] for s in entry['kernel_symbols']}
        if not kernels or any(event['dispatch_info']['kernel_id'] not in symbols for event in kernels):
            raise ValueError(f'{path.name}: empty dispatches or missing symbols')
        csv_before = csv_path.stat()
        with csv_path.open(newline='', encoding='utf-8-sig') as stream:
            csv_counts = Counter(row['Agent_Id'].removeprefix('Agent ') for row in csv.DictReader(stream))
        if json_counts != csv_counts:
            raise ValueError(f'{path.name}: CSV and JSON device/dispatch counts differ')
        for summary in entry.get('summary', []):
            if summary['domain'] == 'KERNEL_DISPATCH' and summary['stats']['count'] != len(kernels):
                raise ValueError(f'{path.name}: native summary and dispatch counts differ')
        for file, old in ((path, before), (csv_path, csv_before)):
            new = file.stat()
            if (new.st_size, new.st_mtime_ns) != (old.st_size, old.st_mtime_ns):
                raise ValueError(f'{file.name}: export changed during validation; writer not finished')
        exports.append(dict(file=str(path.relative_to(traces)), bytes=before.st_size,
                            pid=metadata['pid'], fini_time=metadata['fini_time'],
                            kernel_calls=len(kernels)))
        print(f'Validated completed export: {path.name}, {len(kernels)} kernels', flush=True)
        del document, entries, entry, kernels
    return exports
