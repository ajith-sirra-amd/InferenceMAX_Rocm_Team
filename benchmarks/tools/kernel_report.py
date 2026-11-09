"""Streaming kernel aggregation and evidence-bound top-ten reports (Python 3.10+)."""
import argparse
import csv
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile
import zipfile
from contextlib import nullcontext as _null_context
from decimal import Decimal

NOTE = ('GPU time % uses summed kernel dispatch durations across captured devices. '
        'Concurrent dispatches overlap; this is not wall time, request latency, or an end-to-end speedup estimate. '
        'Std is population standard deviation. Iteration totals include only observed kernel/iteration groups, '
        'scoped to a trace, process and device; absent markers remain unknown.')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


class Stats:
    def __init__(self):
        self.calls = 0
        self.total = self.mean = self.m2 = 0.0
        self.low = self.high = None

    def add(self, value):
        self.calls += 1
        self.total += value
        delta = value - self.mean
        self.mean += delta / self.calls
        self.m2 += delta * (value - self.mean)
        self.low = value if self.low is None else min(self.low, value)
        self.high = value if self.high is None else max(self.high, value)

    def record(self):
        return dict(calls=self.calls, total_us=self.total, avg_us=self.mean if self.calls else None,
                    min_us=self.low, max_us=self.high,
                    std_us=math.sqrt(max(0, self.m2) / self.calls) if self.calls else None)


class JSONStream:
    """Decode one trace event at a time, including from gzip; no whole-trace load."""
    def __init__(self, handle):
        self.handle, self.buffer, self.eof = handle, '', False
        self.decoder = json.JSONDecoder()

    def fill(self):
        chunk = self.handle.read(65536)
        self.eof = not chunk
        self.buffer += chunk

    def peek(self):
        self.buffer = self.buffer.lstrip()
        while not self.buffer and not self.eof:
            self.fill()
            self.buffer = self.buffer.lstrip()
        return self.buffer[:1]

    def take(self, token):
        if self.peek() != token:
            raise ValueError('Invalid trace JSON; expected ' + token)
        self.buffer = self.buffer[1:]

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
                # A scalar may end at the chunk boundary and continue in the next chunk.
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                self.buffer = self.buffer[end:]
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise ValueError('Truncated or invalid trace JSON')
                self.fill()

    def array(self):
        self.take('[')
        if self.peek() != ']':
            while True:
                yield self.value()
                if self.peek() == ']':
                    break
                self.take(',')
        self.take(']')


def trace_events(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig') as handle:
        stream = JSONStream(handle)
        if stream.peek() == '[':
            yield from stream.array()
        else:
            stream.take('{')
            found = False
            if stream.peek() != '}':
                while True:
                    key = stream.value()
                    stream.take(':')
                    if key == 'traceEvents':
                        if found:
                            raise ValueError('Duplicate traceEvents array')
                        found = True
                        yield from stream.array()
                    else:
                        stream.value()
                    if stream.peek() == '}':
                        break
                    stream.take(',')
            stream.take('}')
            if not found:
                raise ValueError('No Chrome traceEvents array; convert this native format first')
        if stream.peek():
            raise ValueError('Trailing data after trace JSON')


def first(row, *keys):
    return next((row[key] for key in keys if row.get(key) is not None and row[key] != ''), None)


def ns(value, scale=1):
    result = Decimal(str(value)) * scale
    if not result.is_finite() or result != result.to_integral_value():
        raise ValueError('Timestamp must resolve to finite integer nanoseconds')
    return int(result)


def normalize(event, family):
    if family == 'rocprof':
        start = ns(first(event, 'Start_Timestamp', 'Start_Timestamp_ns'))
        end = ns(first(event, 'End_Timestamp', 'End_Timestamp_ns'))
        return ('kernel', first(event, 'Kernel_Name', 'Name'), start, end, event,
                first(event, 'Process_Id', 'PID'), first(event, 'Agent_Id', 'GPU_ID'),
                first(event, 'Queue_Id', 'Stream_Id'), None)
    if not isinstance(event, dict) or event.get('ph') != 'X':
        return None
    args = event.get('args') or {}
    categories = str(event.get('cat', '')).split(',')
    kind = 'kernel' if 'kernel' in categories else 'operator' if 'cpu_op' in categories else None
    if str(event.get('name', '')).startswith('ProfilerStep#'):
        kind = 'iteration'
    if kind is None:
        return None
    duration = ns(event.get('dur'), 1000)
    if duration < 0:
        raise ValueError('Negative event duration')
    start = ns(event['ts'], 1000) if 'ts' in event else None
    # Preserve a duration-only event, but never invent its position on a timeline.
    end = start + duration if start is not None else None
    args = dict(args, _duration_ns=duration, _thread=str(event.get('tid', 'unknown')))
    return (kind, event.get('name'), start, end, args, event.get('pid'),
            args.get('device', event.get('pid') if kind == 'kernel' else None),
            args.get('stream', event.get('tid')), first(args, 'External id', 'External Id', 'external_id'))


def metadata(args, operator):
    def dimensions(short, native):
        value = first(args, short, native)
        if value is not None:
            return value
        values = [args.get(native + '_' + axis) for axis in 'XYZ']
        return values if all(v is not None and v != '' for v in values) else None

    # CPU operator input metadata is kept explicitly separate from proven kernel tensors.
    return {key: value for key, value in {
        'operator': operator.get('_name'),
        'operator_input_shapes': first(operator, 'Input Dims', 'Input Shapes'),
        'operator_input_dtypes': first(operator, 'Input type', 'Input Types'),
        'operator_input_strides': operator.get('Input Strides'),
        'grid': dimensions('grid', 'Grid_Size'), 'block': dimensions('block', 'Workgroup_Size'),
        'registers_per_thread': first(args, 'registers per thread', 'VGPR_Count'),
        'scalar_registers': args.get('SGPR_Count'),
        'accumulator_registers': args.get('Accum_VGPR_Count'),
        'scratch_size': args.get('Scratch_Size'),
        'shared_memory': first(args, 'shared memory', 'LDS_Block_Size'),
        'code_object': first(args, 'Code_Object_Id', 'code_object'),
        'phase': first(args, 'phase', 'Phase') or first(operator, 'phase', 'Phase'),
    }.items() if value is not None}


def source_ref(root, ref):
    if not isinstance(ref, dict) or not isinstance(ref.get('path'), str):
        raise ValueError('Evidence needs a relative file path and SHA256')
    path = (root / ref['path']).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Evidence file is missing or outside this profile directory')
    if sha(path) != ref.get('sha256'):
        raise ValueError('Evidence file hash differs: ' + ref['path'])
    return ref


def positive(row, key):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Expected finite positive ' + key)
    return value


def analyze_evidence(root, evidence, capture, variant):
    result = {'roofline': {'status': 'unavailable'}, 'tensors': {'status': 'unavailable'},
              'path': {'status': 'insufficient_evidence'}, 'resources': {'status': 'unavailable'}, 'errors': []}
    if evidence is None:
        return result
    try:
        if evidence.get('capture_sha256') != capture or evidence.get('variant_id') != variant['variant_id']:
            raise ValueError('Evidence belongs to a different capture or kernel variant')
        binding = evidence['binding']
        for key in ('image_digest', 'code_object_sha256', 'runtime_backend', 'shape_signature', 'device', 'phase', 'graph_mode'):
            if not isinstance(binding.get(key), str) or not binding[key].strip():
                raise ValueError('Missing runtime binding: ' + key)
        if binding['shape_signature'] != variant['shape_signature']:
            raise ValueError('Evidence shape signature differs from the observed variant')
        if binding['device'] != variant['device']:
            raise ValueError('Evidence device differs from timing variant')
        source_ref(root, binding['source'])
    except (KeyError, ValueError, TypeError) as error:
        result['errors'].append(str(error))
        return result
    for section in ('tensors', 'roofline', 'path', 'resources'):
        if section not in evidence:
            continue
        try:
            row = evidence[section]
            source_ref(root, row['source'])
            if section == 'tensors':
                for key in ('inputs', 'outputs', 'intermediates', 'compute_dtype', 'accumulation_dtype'):
                    if key not in row:
                        raise ValueError('Missing tensor/precision field: ' + key)
                if row.get('method') not in ('runtime_metadata_and_source', 'runtime_metadata_and_disassembly'):
                    raise ValueError('Precision requires bound runtime metadata and implementation evidence')
                for role in ('inputs', 'outputs', 'intermediates'):
                    if not isinstance(row[role], list) or (role != 'intermediates' and not row[role]):
                        raise ValueError('Tensor roles must contain explicit tensor metadata')
                    for tensor in row[role]:
                        if not isinstance(tensor.get('shape'), list) or any(type(d) is not int or d < 0 for d in tensor['shape']):
                            raise ValueError('Tensor shape must contain nonnegative dimensions')
                        if not isinstance(tensor.get('dtype'), str) or not tensor['dtype']:
                            raise ValueError('Tensor dtype is missing')
                if not row['intermediates'] and not row.get('intermediate_explanation'):
                    raise ValueError('Explain fused/register-only intermediates when no materialized tensor exists')
                for key in ('compute_dtype', 'accumulation_dtype'):
                    if not isinstance(row[key], str) or not row[key]:
                        raise ValueError('Compute/accumulation precision needs explicit evidence')
                result[section] = dict(row, status='evidence_provided')
            elif section == 'roofline':
                for key in ('flop_definition', 'hbm_byte_definition', 'compute_precision', 'scope', 'measurement_id'):
                    if not row.get(key):
                        raise ValueError('Missing roofline scope/definition: ' + key)
                if row.get('scope') != 'single_device_matched_dispatch_population':
                    raise ValueError('Roofline requires matching FLOPs, HBM bytes and GPU times on one device')
                source_ref(root, row['calibration_source'])
                duration = positive(row, 'duration_us') * 1e-6
                flops, size = positive(row, 'flops'), positive(row, 'hbm_bytes')
                peak, bandwidth = positive(row, 'compute_ceiling_tflops'), positive(row, 'hbm_ceiling_gbs')
                calls = positive(row, 'calls')
                if int(calls) != calls:
                    raise ValueError('Counter population calls must be integral')
                if row.get('ceiling_kind') not in ('measured', 'theoretical'):
                    raise ValueError('Label reference ceilings as measured or theoretical')
                intensity = flops / size
                achieved = flops / duration / 1e12
                memory_roof = bandwidth * intensity / 1000
                roof = min(peak, memory_roof)
                if achieved > roof * 1.05:
                    raise ValueError('Achieved throughput exceeds the supplied roof; check scope, units and calibration')
                result[section] = dict(row, status='computed_from_supplied_evidence',
                    arithmetic_intensity_flops_per_byte=intensity, achieved_tflops=achieved,
                    achieved_hbm_gbs=size / duration / 1e9, roof_tflops=roof,
                    roof_utilization_pct=100 * achieved / roof,
                    limiting_roof='HBM' if memory_roof < peak else 'compute',
                    interpretation='Roofline headroom is not a predicted optimization gain; communication/collective kernels also require xGMI/topology/rank-arrival analysis.')
            elif section == 'resources':
                if not isinstance(row.get('metrics'), list) or not row['metrics']:
                    raise ValueError('Supply measured resource metrics with units, scope and definitions')
                for metric in row['metrics']:
                    if any(not metric.get(key) for key in ('name', 'unit', 'scope', 'definition')):
                        raise ValueError('Each resource metric needs a name, unit, scope and definition')
                    value = metric.get('value')
                    if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                        raise ValueError('Resource metric values must be finite and nonnegative')
                    if metric['unit'] == '%' and value > 100:
                        raise ValueError('Percent resource metric exceeds 100%')
                result[section] = dict(row, status='evidence_provided')
            else:
                if not row.get('selected_implementation') or not row.get('tested_candidates'):
                    raise ValueError('Path review needs the selected implementation and matched candidate tests')
                source_ref(root, row['dispatch_source'])
                accepted = []
                for candidate in row['tested_candidates']:
                    source_ref(root, candidate['source'])
                    if candidate.get('binding') != binding:
                        raise ValueError('Candidate comparison conditions differ from the selected kernel')
                    if candidate.get('correctness_passed') is not True or candidate.get('compatible') is not True:
                        continue
                    if candidate.get('repeats', 0) < 3:
                        raise ValueError('Candidate timings need at least three matched repetitions')
                    positive(candidate, 'avg_us')
                    positive(candidate, 'max_us')
                    positive(candidate, 'min_us')
                    if not candidate['min_us'] <= candidate['avg_us'] <= candidate['max_us']:
                        raise ValueError('Invalid candidate timing range')
                    accepted.append(candidate)
                selected = [c for c in accepted if c.get('implementation') == row['selected_implementation']]
                alternatives = [c for c in accepted if c.get('implementation') != row['selected_implementation']]
                if len(selected) != 1 or not alternatives:
                    raise ValueError('Need the selected path and at least one compatible, correct alternative')
                current = selected[0]
                faster = [c for c in alternatives if c['max_us'] < current['min_us']]
                slower = all(current['max_us'] < c['min_us'] for c in alternatives)
                status = 'faster_tested_path_available' if faster else 'best_among_tested_paths' if slower else 'inconclusive_timing_overlap'
                result[section] = dict(row, status=status,
                    interpretation='This compares supplied matched tests; it does not prove global optimality or end-to-end benefit.')
        except (KeyError, ValueError, TypeError) as error:
            result['errors'].append(section + ': ' + str(error))
    result['binding'] = binding
    return result


def aggregate(directory, output=None, family=None, expected_devices=None, window_ns=None):
    directory = Path(directory)
    output = Path(output) if output is not None else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    candidates = sorted(p for p in directory.rglob('*') if p.is_file() and
                        (p.name.endswith('kernel_trace.csv') or p.name.endswith('.json.gz') or p.name.endswith('.trace.json')))
    family = family or ('rocprof' if any(p.name.endswith('kernel_trace.csv') for p in candidates) else 'torch')
    if window_ns is not None and (family != 'rocprof' or len(window_ns) != 2
            or any(type(n) is not int for n in window_ns) or window_ns[0] >= window_ns[1]):
        raise ValueError('A native rocprof window requires integer start/end timestamps with start < end')
    excluded = 0
    paths = [p for p in candidates if p.name.endswith('kernel_trace.csv') == (family == 'rocprof')]
    errors, files, warnings, seen = [], [], [], set()
    if len(paths) != len(candidates):
        warnings.append('Only ' + family + ' traces are aggregated; other formats are excluded to avoid double counting.')
    # SQLite spills correlations and iteration groups to local disk instead of retaining all dispatches in RAM.
    with tempfile.TemporaryDirectory(prefix='benchmark-kernels-', dir=output or directory.parent) as temporary:
        db = sqlite3.connect(str(Path(temporary) / 'events.sqlite'))
        try:
            db.execute('PRAGMA cache_size=-8192')
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('CREATE TABLE events (kind, name, source, record, start_ns INTEGER, end_ns INTEGER, duration_us REAL, args, pid, device, stream, external, thread)')
            db.execute('CREATE TABLE iterations (name, source, pid, device, iteration, calls INTEGER, total_us REAL, min_us REAL, max_us REAL, m2 REAL, PRIMARY KEY(name,source,pid,device,iteration))')
            for path in paths:
                relative = str(path.relative_to(directory))
                digest = sha(path)
                if digest in seen:
                    warnings.append('Identical trace copy excluded: ' + relative)
                    continue
                seen.add(digest)
                files.append({'path': relative, 'sha256': digest, 'bytes': path.stat().st_size})
                try:
                    with path.open(newline='', encoding='utf-8-sig') if family == 'rocprof' else _null_context() as handle:
                        rows = csv.DictReader(handle) if family == 'rocprof' else trace_events(path)
                        for index, event in enumerate(rows, 1):
                            try:
                                row = normalize(event, family)
                                if row is None:
                                    continue
                                kind, name, start, end, args, pid, device, stream, external = row
                                duration = (end - start) / 1000 if start is not None else args['_duration_ns'] / 1000
                                if not name or not math.isfinite(duration) or duration < 0:
                                    raise ValueError('Invalid kernel/operator name or duration')
                                if window_ns is not None and (start < window_ns[0] or end > window_ns[1]):
                                    excluded += 1
                                    continue  # Retain whole dispatches; never clip their durations.
                                db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                    (kind, name, relative, index, start, end, duration, encode(args),
                                     str(pid) if pid is not None else 'unknown', str(device) if device is not None else 'unknown',
                                     str(stream) if stream is not None else 'unknown', str(external) if external is not None else None,
                                     str(args.get('_thread', 'unknown'))))
                            except (ValueError, TypeError, KeyError, ArithmeticError) as error:
                                errors.append({'file': relative, 'record': index, 'error': str(error)})
                                if len(errors) >= 100:
                                    raise ValueError('Stopped after 100 malformed records')
                except (OSError, ValueError, csv.Error) as error:
                    errors.append({'file': relative, 'error': str(error)})
                db.commit()
            db.execute('CREATE INDEX correlations ON events(kind,source,external)')
            db.execute('CREATE INDEX steps ON events(kind,source,pid,thread,start_ns,end_ns)')
            identity = files if window_ns is None else {'files': files, 'window_ns': list(window_ns)}
            capture = hashlib.sha256(encode(identity).encode()).hexdigest()
            evidence = {}
            if output and (output / 'kernel-evidence.json').exists():
                evidence = json.loads((output / 'kernel-evidence.json').read_text())
                if not isinstance(evidence, dict):
                    raise ValueError('kernel-evidence.json must map variant IDs to evidence objects')
            kernels, variants, per_device, operators = {}, {}, {}, {}
            devices, processes = set(), set()
            linked = timed = with_iteration = 0
            dispatch_path = output / 'kernel-dispatches.csv.gz' if output else Path(temporary) / 'dispatches.csv.gz'
            with gzip.open(dispatch_path, 'wt', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['trace_file', 'record', 'kernel', 'variant_id', 'device', 'process', 'stream', 'start_ns', 'end_ns', 'duration_us', 'iteration_id', 'operator', 'operator_input_shapes', 'operator_input_dtypes'])
                query = '''SELECT k.*, o.name, o.args, o.start_ns, o.end_ns, o.pid, o.thread FROM events k
                    LEFT JOIN events o ON o.kind='operator' AND o.source=k.source AND o.external=k.external
                    AND (SELECT COUNT(*) FROM events x WHERE x.kind='operator' AND x.source=k.source AND x.external=k.external)=1
                    WHERE k.kind='kernel' ORDER BY k.source,k.record'''
                for row in db.execute(query):
                    _, name, source, index, start, end, duration, raw, pid, device, stream, external, thread, op_name, op_raw, op_start, op_end, op_pid, op_thread = row
                    args = json.loads(raw)
                    op = dict(json.loads(op_raw), _name=op_name) if op_raw else {}
                    linked += bool(op_raw)
                    timed += start is not None
                    meta = metadata(args, op)
                    signature = {'name': name, 'device': device, **meta}
                    signature['shape_signature'] = hashlib.sha256(encode({k: v for k, v in meta.items() if 'shape' in k or 'stride' in k or 'dtype' in k or k in ('grid', 'block')}).encode()).hexdigest()
                    variant_id = hashlib.sha256(encode(signature).encode()).hexdigest()[:24]
                    kernel = kernels.setdefault(name, Stats()); kernel.add(duration)
                    variant = variants.setdefault(variant_id, {'variant_id': variant_id, **signature, 'stats': Stats()})
                    variant['stats'].add(duration)
                    per_device.setdefault((name, device), Stats()).add(duration)
                    devices.add(device); processes.add(source + ':' + pid)
                    iteration = first(args, 'iteration_id', 'Iteration_Id') or first(op, 'iteration_id', 'Iteration_Id')
                    if iteration is None and op_start is not None:
                        steps = list(db.execute("SELECT name,start_ns FROM events WHERE kind='iteration' AND source=? AND pid=? AND thread=? AND start_ns<=? AND end_ns>=? LIMIT 2",
                                                (source, op_pid, op_thread, op_start, op_end)))
                        if len(steps) == 1:
                            iteration = steps[0][0] + '@' + str(steps[0][1])
                    if iteration is not None:
                        with_iteration += 1
                        db.execute('''INSERT INTO iterations VALUES (?,?,?,?,?,1,?,?,?,0) ON CONFLICT(name,source,pid,device,iteration)
                                      DO UPDATE SET m2=m2+(excluded.total_us-total_us/calls)*(excluded.total_us-(total_us+excluded.total_us)/(calls+1)),
                                      calls=calls+1,total_us=total_us+excluded.total_us,min_us=min(min_us,excluded.min_us),max_us=max(max_us,excluded.max_us)''',
                                   (name, source, str(op_pid or pid), device, str(iteration), duration, duration, duration))
                    writer.writerow([source, index, name, variant_id, device, pid, stream, start, end, duration, iteration,
                                     op_name, encode(meta.get('operator_input_shapes')), encode(meta.get('operator_input_dtypes'))])
            for name, raw, duration in db.execute("SELECT name,args,duration_us FROM events WHERE kind='operator'"):
                record = operators.setdefault(name, {'name': name, 'stats': Stats()})
                record['stats'].add(duration)
                for key in ('Input Dims', 'Input type', 'Input Strides'):
                    if key in json.loads(raw):
                        record.setdefault(key, json.loads(raw)[key])
            iteration_stats = {}
            iteration_path = output / 'kernel-iterations.csv.gz' if output else Path(temporary) / 'iterations.csv.gz'
            with gzip.open(iteration_path, 'wt', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['kernel', 'trace_file', 'process', 'device', 'iteration_id', 'calls', 'summed_gpu_us', 'avg_us', 'min_us', 'max_us', 'std_us'])
                for row in db.execute('SELECT * FROM iterations ORDER BY source,pid,device,iteration,name'):
                    calls, total, low, high, m2 = row[5:]
                    writer.writerow([*row[:7], total/calls, low, high, math.sqrt(max(0,m2)/calls)])
                    iteration_stats.setdefault(row[0], Stats()).add(total)
            total = sum(s.total for s in kernels.values())
            records = [dict(name=name, **s.record(), time_pct=100*s.total/total if total else None,
                            iteration_totals=iteration_stats[name].record() if name in iteration_stats else None)
                       for name, s in kernels.items()]
            records.sort(key=lambda r: (-r['total_us'], r['name']))
            variant_records = [dict((k, v) for k, v in row.items() if k != 'stats') | row['stats'].record() for row in variants.values()]
            variant_records.sort(key=lambda r: (-r['total_us'], r['variant_id']))
            top = []
            for kernel in records[:10]:
                selected = []
                for variant in variant_records:
                    if variant['name'] == kernel['name']:
                        selected.append(dict(variant, analysis=analyze_evidence(output, evidence.get(variant['variant_id']), capture, variant)))
                top.append(dict(kernel, variants=selected))
            summary = {'schema_version': 1, 'kernel_count': sum(s.calls for s in kernels.values()),
                'unique_kernels': len(records), 'total_gpu_us': total, 'devices': sorted(devices), 'processes': sorted(processes),
                'files': [f['path'] for f in files], 'trace_manifest': files, 'capture_sha256': capture,
                'parser_sha256': sha(Path(__file__)), 'format': family, 'errors': errors, 'warnings': warnings,
                'kernels': records, 'top10': top,
                'per_device': [dict(name=n, device=d, **s.record()) for (n, d), s in per_device.items()],
                'operators': [dict((k, v) for k, v in row.items() if k != 'stats') | row['stats'].record() for row in operators.values()],
                'coverage': {'dispatches_with_timestamps': timed, 'dispatches_with_operator_link': linked,
                             'selected_window_ns': list(window_ns) if window_ns is not None else None,
                             'dispatches_outside_window': excluded,
                             'dispatches_with_iteration': with_iteration, 'dropped_events': 'unknown',
                             'rank_mapping': 'trace-local process/device identities; global rank mapping requires runtime evidence'},
                'note': NOTE}
            if expected_devices is not None:
                summary.update(expected_devices=expected_devices,
                               device_coverage_complete=len(devices) == expected_devices and 'unknown' not in devices)
            if output:
                write_json(output / 'trace-summary.json', summary)
                export_report(output, summary, variant_records)
            return summary
        finally:
            db.close()


def escape(value):
    return html.escape(str(value), quote=True).replace('|', '&#124;').replace('\n', ' ')


def display(value):
    return 'Unknown' if value is None else f'{value:.4f}' if isinstance(value, float) else escape(value)


def export_report(output, summary, variants):
    fields = ['name', 'calls', 'time_pct', 'total_us', 'avg_us', 'min_us', 'max_us', 'std_us']
    with (output / 'kernel-summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in summary['kernels'])
    write_json(output / 'kernel-variants.json', variants)
    plan = {'capture_sha256': summary['capture_sha256'], 'status': 'needs_targeted_collection', 'kernels': []}
    lines = ['# BENCHMARK aggregated kernel profile', '',
        f"Captured dispatches: {summary['kernel_count']}; distinct kernel symbols: {summary['unique_kernels']}.", '', NOTE, '',
        'All timing units below are microseconds. Statistics describe profiler-observed dispatches, not unprofiled latency.', '',
        '## Coverage', '', '```json', json.dumps(summary['coverage'], indent=2), '```', '',
        'Capture completeness, dropped events and profiling perturbation require the run manifests and a matched baseline.', '',
        '## All captured kernels, highest GPU time first', '',
        '| Kernel | Calls | GPU time % | Total (us) | Avg (us) | Min (us) | Max (us) | Std (us) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    if summary['coverage'].get('selected_window_ns') is not None:
        lines[2:2] = ['Selected timestamp window only: ' +
            str(summary['coverage']['selected_window_ns']) +
            ' ns in the native profiler clock. Crossing dispatches are excluded. '
            'This selection does not validate the original capture or exporter exit.', '']
    for row in summary['kernels']:
        lines.append('| ' + ' | '.join(display(row[key]) for key in fields) + ' |')
    lines += ['', 'Individual dispatches: [kernel-dispatches.csv.gz](kernel-dispatches.csv.gz).',
              'Explicit iteration groups: [kernel-iterations.csv.gz](kernel-iterations.csv.gz).',
              'All shape/device/launch variants: [kernel-variants.json](kernel-variants.json).', '',
              '## Top 10 kernel investigations', '']
    for rank, kernel in enumerate(summary['top10'], 1):
        lines += [f"### {rank}. {escape(kernel['name'])}", '',
                  f"GPU time share: {display(kernel['time_pct'])}%; calls: {kernel['calls']}; variants: {len(kernel['variants'])}.", '']
        if kernel['iteration_totals']:
            lines += ['Observed per-iteration summed kernel time (us): ' + escape(encode(kernel['iteration_totals'])), '']
        else:
            lines += ['Per-model-iteration timing: unavailable; add worker step markers and dispatch correlation.', '']
        for variant in kernel['variants']:
            analysis = variant['analysis']
            lines += [f"#### Variant {variant['variant_id']} / device {escape(variant['device'])}", '',
                f"Calls: {variant['calls']}; avg/min/max/std (us): {variant['avg_us']:.4f} / {variant['min_us']:.4f} / {variant['max_us']:.4f} / {variant['std_us']:.4f}.", '',
                'Correlated operator inputs / launch metadata (operator metadata does not prove fused kernel inputs or internal precision):', '',
                '```json', json.dumps({k: v for k, v in variant.items() if k not in ('analysis', 'name')}, indent=2), '```', '',
                f"Roofline: **{analysis['roofline']['status']}**. Tensor/precision evidence: **{analysis['tensors']['status']}**.", '',
                f"Optimized-path assessment: **{analysis['path']['status']}**.", '',
                '```json', json.dumps(analysis, indent=2), '```', '']
            required = []
            if analysis['roofline']['status'] == 'unavailable':
                required.append('Matched per-device FLOPs, actual HBM bytes and GPU durations; precision-specific compute and HBM ceilings with source hashes.')
            if analysis['tensors']['status'] == 'unavailable':
                required.append('Runtime-correlated input/output shapes, strides and dtypes; implementation/ISA evidence for intermediate, compute and accumulation precision.')
            if analysis['path']['status'] == 'insufficient_evidence':
                required.append('Selected backend/source/code object and dispatch proof; shape eligibility; correct compatible candidate timings under matched conditions.')
            required += ['Inspect occupancy, VGPR/SGPR use, spills, LDS, cache hit rates, memory/coalescing, MFMA/VALU mix, stalls and synchronization where applicable.']
            plan['kernels'].append({'name': kernel['name'], 'variant_id': variant['variant_id'],
                                    'device': variant['device'], 'shape_signature': variant['shape_signature'], 'required_evidence': required,
                                    'counter_command_template': ['rocprof-compute', 'profile', '--name', 'benchmark-' + variant['variant_id'],
                                        '--output-directory', '<fresh-counter-pass>', '--kernel', kernel['name'], '--roof-only', '--', '<verified-matched-reproducer>'],
                                    'command_status': 'template_only; verify installed help, counter support, exact dispatch selection and reproducer equivalence before execution'})
            lines += ['Required follow-up:'] + ['- ' + item for item in required] + ['']
    lines += ['## Interpretation limits', '',
              'A selected fast backend or high roofline utilization does not prove the most optimized implementation. '
              'Only matched, correctness-checked alternatives can support a best-among-tested verdict. '
              'Internal precision and tensor shapes are never inferred from kernel symbols. '
              'Counter/replay timings stay separate from the timeline timing population.', '']
    if summary['errors'] or summary['warnings']:
        lines += ['## Capture/parser issues', '', '```json', json.dumps({'errors': summary['errors'], 'warnings': summary['warnings']}, indent=2), '```']
    (output / 'kernel-report.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'top10-collection-plan.json', plan)
    export_xlsx(output / 'kernel-report.xlsx', summary)
    export_html(output / 'kernel-report.html', summary)



def export_xlsx(path, summary):
    """Small Excel workbook; millions of individual dispatches stay in compressed CSV."""
    headings = ['Kernel', 'Precision Evidence', 'Calls', 'Mean (us)', 'Min (us)', 'Max (us)',
                'Total GPU (ms)', 'GPU Duration Share %', 'Std (us)']
    rows = [headings] + [[r['name'], 'See Top10 details; symbol alone is not precision proof', r['calls'], r['avg_us'],
                         r['min_us'], r['max_us'], r['total_us']/1000, r['time_pct'], r['std_us']] for r in summary['kernels']]
    details = [['Kernel', 'Variant', 'Device', 'Calls', 'Operator input shapes', 'Operator input dtypes',
                'Tensor evidence', 'Roofline status', 'AI (FLOPs/byte)', 'Achieved TFLOP/s', 'HBM GB/s',
                'Roof TFLOP/s', 'Roof utilization %', 'Optimized path', 'Evidence and limits']]
    for kernel in summary['top10']:
        for v in kernel['variants']:
            a = v['analysis']; roof = a['roofline']
            details.append([kernel['name'], v['variant_id'], v['device'], v['calls'], encode(v.get('operator_input_shapes')),
                encode(v.get('operator_input_dtypes')), a['tensors']['status'], roof['status'],
                roof.get('arithmetic_intensity_flops_per_byte'), roof.get('achieved_tflops'), roof.get('achieved_hbm_gbs'),
                roof.get('roof_tflops'), roof.get('roof_utilization_pct'), a['path']['status'], encode(a)])
    coverage = [['Field', 'Value'], ['Interpretation', NOTE], ['Capture SHA256', summary['capture_sha256']],
                ['Coverage', encode(summary['coverage'])], ['Errors', encode(summary['errors'])], ['Warnings', encode(summary['warnings'])]]
    sheets = [('kernels_aggregate', rows), ('Top10 details', details), ('Coverage', coverage)]
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' +
            ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,4)) + '</Types>')
        archive.writestr('_rels/.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>' +
            ''.join(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i,(name,_) in enumerate(sheets,1)) + '</sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
            ''.join(f'<Relationship Id="rId{i}" Type="{rel}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,4)) + '</Relationships>')
        for index, (_, table) in enumerate(sheets,1):
            with archive.open(f'xl/worksheets/sheet{index}.xml', 'w') as handle:
                handle.write(f'<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews><sheetData>'.encode())
                for n, values in enumerate(table,1):
                    cells = []
                    for column, value in enumerate(values):
                        ref = chr(65+column)+str(n)
                        if isinstance(value, (int,float)):
                            cells.append(f'<c r="{ref}"><v>{value}</v></c>')
                        else:
                            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(value) if value is not None else 'Unknown')[:32000]
                            cells.append(f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{html.escape(text)}</t></is></c>')
                    handle.write((f'<row r="{n}">' + ''.join(cells) + '</row>').encode())
                handle.write(f'</sheetData><autoFilter ref="A1:{chr(64+len(table[0]))}{len(table)}"/></worksheet>'.encode())


def export_html(path, summary):
    parts = ['<!doctype html><meta charset="utf-8"><title>BENCHMARK kernel report</title>',
             '<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:6px;text-align:right}td:first-child{text-align:left;overflow-wrap:anywhere;max-width:500px}pre{white-space:pre-wrap;overflow-wrap:anywhere}svg{max-width:640px;width:100%}</style>',
             '<h1>Aggregated kernel profile</h1><p>' + html.escape(NOTE) + '</p>',
             '<p><a href="kernel-report.xlsx">Excel workbook</a> · <a href="kernel-summary.csv">CSV</a> · <a href="kernel-dispatches.csv.gz">Every dispatch</a> · <a href="kernel-iterations.csv.gz">Iteration groups</a> · <a href="kernel-report.md">Full report</a> · <a href="trace-summary.json">JSON</a></p>',
             '<h2>All captured kernels</h2><table><tr>' + ''.join('<th>'+v+'</th>' for v in ['Kernel','Calls','GPU time %','Total us','Avg us','Min us','Max us','Std us']) + '</tr>']
    if summary['coverage'].get('selected_window_ns') is not None:
        parts.insert(3, '<p><strong>Selected timestamp window only:</strong> ' +
            html.escape(str(summary['coverage']['selected_window_ns'])) +
            ' ns in the native profiler clock; crossing dispatches are excluded. '
            'This selection does not validate the original capture or exporter exit.</p>')
    for row in summary['kernels']:
        parts.append('<tr>' + ''.join('<td>'+display(row[k])+'</td>' for k in ['name','calls','time_pct','total_us','avg_us','min_us','max_us','std_us']) + '</tr>')
    parts.append('</table><h2>Top 10 investigations</h2>')
    for kernel in summary['top10']:
        parts.append('<h3>'+html.escape(kernel['name'])+'</h3>')
        for variant in kernel['variants']:
            parts.append('<details><summary>Device '+html.escape(variant['device'])+' · '+variant['variant_id']+' · '+variant['analysis']['path']['status']+'</summary>')
            roof = variant['analysis']['roofline']
            if roof['status'] == 'computed_from_supplied_evidence':
                ai = roof['arithmetic_intensity_flops_per_byte']; ridge = roof['compute_ceiling_tflops']*1000/roof['hbm_ceiling_gbs']
                lo, hi = math.log10(min(ai,ridge))-2, math.log10(max(ai,ridge))+2
                yl, yh = math.log10(min(roof['achieved_tflops'],roof['roof_tflops']))-2, math.log10(roof['compute_ceiling_tflops'])+1
                x = lambda n: 65+500*(math.log10(n)-lo)/(hi-lo)
                y = lambda n: 280-240*(math.log10(n)-yl)/(yh-yl)
                points=[]
                for i in range(101):
                    value=10**(lo+(hi-lo)*i/100)
                    ceiling=min(roof['compute_ceiling_tflops'],roof['hbm_ceiling_gbs']*value/1000)
                    if math.log10(ceiling)>=yl:points.append(f'{x(value):.2f},{y(ceiling):.2f}')
                parts.append('<svg viewBox="0 0 640 330" role="img" aria-label="Per-variant roofline"><path d="M65 30V280H580" fill="none" stroke="black"/>'+
                    '<polyline points="'+' '.join(points)+'" fill="none" stroke="blue"/>'+
                    f'<circle cx="{x(ai):.2f}" cy="{y(roof["achieved_tflops"]):.2f}" r="5" fill="red"/>'+
                    '<text x="100" y="320">Arithmetic intensity (FLOPs/HBM byte), log scale</text><text x="70" y="20">TFLOP/s, log scale</text></svg>')
            parts.append('<pre>'+html.escape(json.dumps(variant,indent=2))+'</pre></details>')
    parts.append('<h2>Coverage and issues</h2><pre>'+html.escape(json.dumps({k:summary[k] for k in ['coverage','errors','warnings','capture_sha256']},indent=2))+'</pre>')
    path.write_text(''.join(parts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--format', choices=('torch', 'rocprof'))
    parser.add_argument('--expected-devices', type=int)
    parser.add_argument('--start-ns', type=int, help='Inclusive native rocprof timestamp; use with --end-ns')
    parser.add_argument('--end-ns', type=int, help='Inclusive native rocprof timestamp; exclude crossing dispatches')
    args = parser.parse_args()
    if (args.start_ns is None) != (args.end_ns is None):
        parser.error('--start-ns and --end-ns must be supplied together')
    window = None if args.start_ns is None else (args.start_ns, args.end_ns)
    result = aggregate(args.traces, args.output, args.format, args.expected_devices, window)
    print(json.dumps({'kernels': result['unique_kernels'], 'dispatches': result['kernel_count'], 'errors': result['errors']}))
    raise SystemExit(1 if result['errors'] or not result['kernel_count'] or result.get('device_coverage_complete') is False else 0)
