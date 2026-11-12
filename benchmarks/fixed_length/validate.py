#!/usr/bin/env python3
"""Validate the existing InferenceX run_benchmark_serving JSON artifact."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def statistics_ms(values):
    samples = sorted(value * 1000 for value in values)
    if not samples:
        return None
    def percentile(p):
        position = (len(samples) - 1) * p / 100
        left, right = math.floor(position), math.ceil(position)
        return samples[left] + (samples[right] - samples[left]) * (position - left)
    return dict(count=len(samples), unit='ms', avg=statistics.mean(samples),
                min=samples[0], max=samples[-1], std=statistics.pstdev(samples),
                **{f'p{p}': percentile(p) for p in (50, 90, 95, 99)})


def finite_number(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def validate_result(data, input_tokens=8192, output_tokens=1024, prompts=192, concurrency=48):
    errors = []
    expected = dict(input_tokens=input_tokens, output_tokens=output_tokens, prompts=prompts, concurrency=concurrency)
    if not isinstance(data, dict):
        return dict(status='failed', expected=expected, errors=['Result must be a JSON object'])
    for key, value in (('num_prompts', prompts), ('completed', prompts), ('max_concurrency', concurrency),
                       ('total_input_tokens', prompts * input_tokens), ('total_output_tokens', prompts * output_tokens)):
        if type(data.get(key)) is not int or data[key] != value:
            errors.append(f'{key}={data.get(key)!r}; expected {value}')
    outcome = data.get('benchmark_outcome', {})
    if not isinstance(outcome, dict) or outcome.get('status') != 'passed' or outcome.get('failed') != 0:
        errors.append('Native benchmark_outcome must pass with zero failed requests')
    if not finite_number(data.get('duration'), positive=True):
        errors.append('Missing or invalid benchmark duration')
    for key in ('input_lens', 'output_lens', 'ttfts', 'itls', 'errors'):
        if not isinstance(data.get(key), list) or len(data[key]) != prompts:
            errors.append(f'{key}: expected {prompts} entries; use --save-detailed')
    if errors:
        return dict(status='failed', expected=expected, errors=errors)
    for index in range(prompts):
        for key, required in (('input_lens', input_tokens), ('output_lens', output_tokens)):
            actual = data[key][index]
            if type(actual) is not int or actual != required:
                errors.append(f'request {index}: {key}={actual!r}, expected {required}')
        if data['errors'][index]:
            errors.append(f'request {index}: {data["errors"][index]}')
        if not finite_number(data['ttfts'][index], positive=True):
            errors.append(f'request {index}: invalid TTFT')
        intervals = data['itls'][index]
        if not isinstance(intervals, list) or any(not finite_number(value) for value in intervals):
            errors.append(f'request {index}: invalid streaming intervals')
    if errors:
        return dict(status='failed', expected=expected, errors=errors)
    e2e = [ttft + sum(intervals) for ttft, intervals in zip(data['ttfts'], data['itls'])]
    intervals = [value for request in data['itls'] for value in request]
    return dict(status='passed', expected=expected, completed=data['completed'],
                benchmark_duration_seconds=data['duration'], request_throughput=prompts / data['duration'],
                output_tokens_per_second=prompts * output_tokens / data['duration'],
                e2e_reconstructed=statistics_ms(e2e), ttft=statistics_ms(data['ttfts']),
                tpot=statistics_ms([sum(value) / (output_tokens - 1) for value in data['itls']]) if output_tokens > 1 else None,
                streaming_interval=statistics_ms(intervals),
                native_e2e={key: value for key, value in data.items() if key.endswith('_e2el_ms')},
                evidence_notes=[
                    'Input lengths are native client tokenizer counts, including its chat template.',
                    'Native output lengths prefer server usage but can fall back to re-tokenization; '
                    'the artifact does not identify which source was used.',
                    'Per-request E2E is reconstructed from TTFT plus streaming intervals. Native '
                    'aggregate E2E is retained separately; timestamps can differ slightly.',
                    'Streaming intervals measure response chunks, which can contain multiple tokens.',
                    'Native warmup outcomes are not saved in this measured-result artifact.',
                    'This fixed-length throughput test does not establish semantic output quality.',
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--input-tokens', type=int, default=8192)
    parser.add_argument('--output-tokens', type=int, default=1024)
    parser.add_argument('--prompts', type=int, default=192)
    parser.add_argument('--concurrency', type=int, default=48)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.input_tokens, args.output_tokens, args.prompts, args.concurrency) < 1:
        parser.error('Expected counts and lengths must be positive')
    try:
        raw = args.result.read_bytes()
        result = validate_result(json.loads(raw), args.input_tokens, args.output_tokens, args.prompts, args.concurrency)
        result.update(source=str(args.result), source_sha256=hashlib.sha256(raw).hexdigest())
    except (OSError, ValueError, TypeError) as error:
        result = dict(status='failed', errors=[str(error)])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
