"""Extract observed direct-launch operator arguments for reviewed kernel replays.

Graph-launch correlation identifies an outer replay, not an inner operator. Such
events are deliberately excluded. Output is evidence, not executable replay code.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re


def extract(trace, operator_pattern):
    events = trace['traceEvents']
    operators, launches = defaultdict(list), defaultdict(list)
    for event in events:
        args = event.get('args', {})
        if event.get('cat') == 'cpu_op' and args.get('External id') is not None:
            operators[(event.get('pid'), str(args['External id']))].append(event)
        if event.get('cat') in ('cuda_runtime', 'hip_runtime') and args.get('correlation') is not None:
            launches[str(args['correlation'])].append(event)
    cases, invocations = {}, defaultdict(set)
    skipped = dict(graph_launches=0, missing_launch=0, ambiguous_launch=0,
                   missing_operator=0, ambiguous_operator=0, excluded_operator=0, missing_shape=0)
    for kernel in events:
        if kernel.get('cat') != 'kernel':
            continue
        candidates = launches.get(str(kernel.get('args', {}).get('correlation')), [])
        if not candidates:
            skipped['missing_launch'] += 1
            continue
        if len(candidates) != 1:
            skipped['ambiguous_launch'] += 1
            continue
        launch = candidates[0]
        if 'GraphLaunch' in launch.get('name', ''):
            skipped['graph_launches'] += 1
            continue
        operator_key = (launch.get('pid'), str(launch.get('args', {}).get('External id')))
        candidates = operators.get(operator_key, [])
        if not candidates or operator_key[0] is None:
            skipped['missing_operator'] += 1
            continue
        if len(candidates) != 1:
            skipped['ambiguous_operator'] += 1
            continue
        operator = candidates[0]
        if not re.search(operator_pattern, operator.get('name', '')):
            skipped['excluded_operator'] += 1
            continue
        args = operator.get('args', {})
        if not args.get('Input Dims') or not args.get('Input type'):
            skipped['missing_shape'] += 1
            continue
        observed = dict(operator=operator['name'], input_dims=args['Input Dims'],
            input_types=args['Input type'], input_strides=args.get('Input Strides'),
            concrete_inputs=args.get('Concrete Inputs'))
        signature = hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).hexdigest()
        case = cases.setdefault(signature, dict(id=signature, observed=observed,
            kernel_dispatches=0, operator_invocations=0, summed_kernel_us=0.0,
            kernel_symbols={}, launch_apis={}, example_events=[]))
        case['kernel_dispatches'] += 1
        invocations[signature].add(operator_key)
        case['operator_invocations'] = len(invocations[signature])
        case['summed_kernel_us'] += kernel.get('dur', 0)
        for key, name in (('kernel_symbols', kernel['name']), ('launch_apis', launch['name'])):
            case[key][name] = case[key].get(name, 0) + 1
        if len(case['example_events']) < 2:
            case['example_events'].append(dict(kernel=kernel, launch=launch, operator=operator))
    return dict(cases=sorted(cases.values(), key=lambda c: -c['summed_kernel_us']), skipped=skipped,
        binding_audit=dict(cpu_operator_keys=len(operators), launch_correlation_keys=len(launches),
            ambiguous_operator_keys=sum(len(v) != 1 for v in operators.values()),
            ambiguous_launch_keys=sum(len(v) != 1 for v in launches.values())),
        limits=['Direct unique CPU (PID, external-id) and globally unique HIP launch correlation only; ambiguous and graph-internal arguments excluded.',
                'Empty concrete values can be unrecorded arguments, not necessarily defaults or null.',
                'Review the installed operator schema and all scalar/optional arguments before replay.',
                'A replay must verify exact kernel symbol, tensor strides, dtype and call configuration; '
                'it cannot establish original model cache state or exposed request latency.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--operator-regex', default=r'^aiter::_gemm_a16w16_asm$|^aten::(?:mm|bmm)$')
    args = parser.parse_args()
    with (gzip.open(args.trace, 'rt') if args.trace.suffix == '.gz' else args.trace.open()) as stream:
        trace = json.load(stream)
    report = extract(trace, args.operator_regex)
    digest = hashlib.sha256()
    with args.trace.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    report.update(trace=str(args.trace), trace_sha256=digest.hexdigest(),
                  operator_regex=args.operator_regex)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), cases=len(report['cases']), skipped=report['skipped'])))
