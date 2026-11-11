"""Extract observed direct-launch operator arguments for reviewed kernel replays.

Graph-launch correlation identifies an outer replay, not an inner operator. Such
events are deliberately excluded. Output is evidence, not executable replay code.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


def extract(trace, operator_pattern):
    events = trace['traceEvents']
    operators = {str(e['args']['External id']): e for e in events
                 if e.get('cat') == 'cpu_op' and e.get('args', {}).get('External id') is not None
                 and re.search(operator_pattern, e.get('name', ''))}
    launches = {str(e['args']['correlation']): e for e in events
                if e.get('cat') in ('cuda_runtime', 'hip_runtime')
                and e.get('args', {}).get('correlation') is not None}
    cases, skipped = {}, dict(graph_launches=0, missing_launch=0, missing_operator=0, missing_shape=0)
    for kernel in events:
        if kernel.get('cat') != 'kernel':
            continue
        launch = launches.get(str(kernel.get('args', {}).get('correlation')))
        if not launch:
            skipped['missing_launch'] += 1
            continue
        if 'GraphLaunch' in launch.get('name', ''):
            skipped['graph_launches'] += 1
            continue
        operator = operators.get(str(launch.get('args', {}).get('External id')))
        if not operator:
            skipped['missing_operator'] += 1
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
            calls=0, summed_kernel_us=0.0, kernel_symbols={}, launch_apis={}, example_events=[]))
        case['calls'] += 1
        case['summed_kernel_us'] += kernel.get('dur', 0)
        for key, name in (('kernel_symbols', kernel['name']), ('launch_apis', launch['name'])):
            case[key][name] = case[key].get(name, 0) + 1
        if len(case['example_events']) < 2:
            case['example_events'].append(dict(kernel=kernel, launch=launch, operator=operator))
    return dict(cases=sorted(cases.values(), key=lambda c: -c['summed_kernel_us']), skipped=skipped,
        limits=['Direct CPU external-id and HIP launch correlation only; graph-internal arguments excluded.',
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
