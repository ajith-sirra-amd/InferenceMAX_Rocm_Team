#!/usr/bin/env python3
"""
extract_agg_bmk.py — Build agg_bmk.json from raw aiperf artifact directories.

Reads, for each results directory:
  - vllm_command.txt or sglang_command.txt  → tp, conc, model, offloading, ep, framework
  - benchmark_command.txt                   → concurrency (cross-check), duration, dataset
  - aiperf_artifacts/profile_export_aiperf.json → all latency / throughput metrics
  - aiperf_artifacts/server_metrics_export.json → prompt token cache source breakdown

Usage (single run):
  python gg_agentic/extract_agg_bmk.py \\
      --results-dir results/ \\
      --hw mi355x \\
      --model-prefix kimik2.7-code

Usage (multiple runs → single combined json):
  python gg_agentic/extract_agg_bmk.py \\
      --results-dir results_none/ results_lmcache/ \\
      --hw mi355x \\
      --model-prefix kimik2.7-code \\
      --output combined/results_bmk/agg_bmk.json

Required arguments:
  --results-dir DIR [DIR ...]  One or more raw artifact directories.
  --hw          NAME           Hardware label (mi355x / h200 / b200 / …).
  --model-prefix NAME          Short model prefix used in bench filenames
                               (e.g. kimik2.7-code, glm5.1, dsv4).

Optional arguments:
  --output PATH   Output json path.
                  Default: <first results-dir>/results_bmk/agg_bmk.json
  --precision STR Override precision (fp4 / fp8 / int4 / bf16).
                  When omitted the script infers it from the model name.
  --image STR     Docker image tag (informational only).
"""

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arg(tokens, *flags, default=None):
    """Return the value that follows any of *flags in a token list."""
    for i, t in enumerate(tokens):
        for f in flags:
            if t == f and i + 1 < len(tokens):
                return tokens[i + 1]
            # handle --flag=value
            if t.startswith(f + '='):
                return t[len(f) + 1:]
    return default


def _has_flag(tokens, *flags):
    return any(t in flags for t in tokens)


def _infer_precision(model_name: str) -> str:
    mn = model_name.upper()
    if 'MXFP4' in mn or 'FP4' in mn:
        return 'fp4'
    if 'FP8' in mn:
        return 'fp8'
    if 'INT4' in mn or 'GPTQ' in mn or 'AWQ' in mn:
        return 'int4'
    if 'BF16' in mn or 'BF16' in mn:
        return 'bf16'
    return 'unknown'


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_server_command(results_dir: Path) -> dict:
    """
    Parse vllm_command.txt or sglang_command.txt.
    Returns a dict with: framework, model, tp, conc, ep, offloading, image.
    """
    info = {
        'framework': 'vllm',
        'model': '',
        'tp': 8,
        'conc': 16,
        'ep': 1,
        'offloading': 'none',
        'image': '',
    }

    for fname in ('vllm_command.txt', 'sglang_command.txt'):
        cmd_path = results_dir / fname
        if not cmd_path.exists():
            continue

        raw = cmd_path.read_text(encoding='utf-8').strip()
        tokens = shlex.split(raw)

        if 'sglang' in fname or any('sglang' in t for t in tokens[:3]):
            info['framework'] = 'sglang'

        # model (first positional after 'serve' / 'launch')
        for i, t in enumerate(tokens):
            if t in ('serve', 'launch') and i + 1 < len(tokens):
                info['model'] = tokens[i + 1]
                break

        # --served-model-name overrides if present
        v = _arg(tokens, '--served-model-name')
        if v:
            info['model'] = v

        # --model (sglang)
        v = _arg(tokens, '--model')
        if v:
            info['model'] = v

        # tensor parallelism
        v = _arg(tokens, '--tensor-parallel-size', '--tensor-parallel-size', '-tp')
        if v:
            info['tp'] = int(v)

        # max sequences / concurrency
        v = _arg(tokens, '--max-num-seqs', '--max-running-requests')
        if v:
            info['conc'] = int(v)

        # expert parallelism
        ep_size = _arg(tokens, '--ep-size', '--expert-parallel-size')
        if ep_size:
            info['ep'] = int(ep_size)
        elif _has_flag(tokens, '--enable-expert-parallel'):
            info['ep'] = info['tp']  # best guess when EP_SIZE not written

        # offloading detection
        if any('LMCacheMPConnector' in t or 'LMCache' in t
               for t in tokens):
            info['offloading'] = 'lmcache'
        elif _has_flag(tokens, '--kv_offloading_backend', '--kv-offloading-backend'):
            info['offloading'] = 'cpu'
        elif any('hicache' in t.lower() for t in tokens):
            info['offloading'] = 'hicache'

        break  # found one file, stop

    return info


def parse_benchmark_command(results_dir: Path) -> dict:
    """Parse benchmark_command.txt → conc, duration, dataset."""
    info = {'conc': None, 'duration': None, 'dataset': ''}

    cmd_path = results_dir / 'benchmark_command.txt'
    if not cmd_path.exists():
        return info

    raw = cmd_path.read_text(encoding='utf-8').strip()
    tokens = shlex.split(raw)

    v = _arg(tokens, '--concurrency')
    if v:
        info['conc'] = int(v)

    v = _arg(tokens, '--benchmark-duration')
    if v:
        info['duration'] = float(v)

    v = _arg(tokens, '--public-dataset')
    if v:
        info['dataset'] = v

    return info


def parse_aiperf_json(results_dir: Path) -> dict:
    """Read profile_export_aiperf.json and return a flat metrics dict."""
    path = results_dir / 'aiperf_artifacts' / 'profile_export_aiperf.json'
    if not path.exists():
        print(f"  WARNING: {path} not found", file=sys.stderr)
        return {}

    with open(path) as f:
        d = json.load(f)

    def _p(key, stat):
        return (d.get(key) or {}).get(stat)

    return {
        'num_requests_successful': int(_p('request_count', 'avg') or 0),
        'mean_qps': _p('request_throughput', 'avg'),

        'mean_ttft':  _p('time_to_first_token', 'avg'),
        'p75_ttft':   _p('time_to_first_token', 'p75'),
        'p90_ttft':   _p('time_to_first_token', 'p90'),
        'p95_ttft':   _p('time_to_first_token', 'p95'),
        'std_ttft':   _p('time_to_first_token', 'std'),

        'mean_e2el':  _p('request_latency', 'avg'),
        'p75_e2el':   _p('request_latency', 'p75'),
        'p90_e2el':   _p('request_latency', 'p90'),
        'p95_e2el':   _p('request_latency', 'p95'),
        'std_e2el':   _p('request_latency', 'std'),

        'mean_itl':   _p('inter_token_latency', 'avg'),
        'p75_itl':    _p('inter_token_latency', 'p75'),
        'p90_itl':    _p('inter_token_latency', 'p90'),
        'p95_itl':    _p('inter_token_latency', 'p95'),
        'std_itl':    _p('inter_token_latency', 'std'),

        'mean_input_tokens':        _p('input_sequence_length', 'avg'),
        'p75_input_tokens':         _p('input_sequence_length', 'p75'),
        'p90_input_tokens':         _p('input_sequence_length', 'p90'),
        'p95_input_tokens':         _p('input_sequence_length', 'p95'),
        'std_input_tokens':         _p('input_sequence_length', 'std'),

        'mean_output_tokens_actual': _p('output_sequence_length', 'avg'),
        'p75_output_tokens_actual':  _p('output_sequence_length', 'p75'),
        'p90_output_tokens_actual':  _p('output_sequence_length', 'p90'),
        'p95_output_tokens_actual':  _p('output_sequence_length', 'p95'),
        'std_output_tokens_actual':  _p('output_sequence_length', 'std'),

        'theoretical_cache_hit_rate': (d.get('theoretical_prefix_cache_hit') or {}).get('avg'),

        'total_prompt_tokens':      _p('total_isl', 'avg'),
        'total_generation_tokens':  _p('total_osl', 'avg'),
        'total_requests_completed': int(_p('request_count', 'avg') or 0),

        'input_tput_tps':   _p('input_token_throughput', 'avg'),
        'output_tput_tps':  _p('output_token_throughput', 'avg'),
        'total_tput_tps':   _p('total_token_throughput', 'avg'),
        'duration_seconds': _p('benchmark_duration', 'avg'),

        # convert ms → s where needed
        '_ttft_unit': 'ms',   # signal for post-processing
    }


def parse_server_metrics(results_dir: Path, framework: str) -> dict:
    """Read server_metrics_export.json → cache token source breakdown."""
    path = results_dir / 'aiperf_artifacts' / 'server_metrics_export.json'
    result = {
        'cache_local_compute': None,
        'cache_local_cache_hit': None,
        'cache_external_kv_transfer': None,
    }
    if not path.exists():
        return result

    with open(path) as f:
        data = json.load(f)

    metrics = data.get('metrics', {})

    if framework == 'vllm':
        src_map = {}
        for s in metrics.get('vllm:prompt_tokens_by_source', {}).get('series', []):
            src = s.get('labels', {}).get('source', '')
            src_map[src] = s.get('stats', {}).get('total', 0.0) or 0.0
        result['cache_local_compute'] = src_map.get('local_compute', 0.0)
        result['cache_local_cache_hit'] = src_map.get('local_cache_hit', 0.0)
        result['cache_external_kv_transfer'] = src_map.get('external_kv_transfer', 0.0)
    else:
        # sglang: cached_tokens by cache_source (device = GPU, host = CPU/ext)
        prompt_total = sum(
            (s.get('stats', {}).get('total') or 0.0)
            for s in metrics.get('sglang:prompt_tokens', {}).get('series', [])
        )
        device_total = 0.0
        host_total = 0.0
        for s in metrics.get('sglang:cached_tokens', {}).get('series', []):
            cs = s.get('labels', {}).get('cache_source', '')
            t = s.get('stats', {}).get('total') or 0.0
            if cs == 'device':
                device_total = t
            elif cs == 'host':
                host_total = t
        result['cache_local_cache_hit'] = device_total
        result['cache_external_kv_transfer'] = host_total
        result['cache_local_compute'] = max(0.0, prompt_total - device_total - host_total)

    return result


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_one(results_dir: Path, hw: str, model_prefix: str,
                precision_override: str | None, image_override: str) -> dict:
    print(f"  Extracting: {results_dir}")

    srv = parse_server_command(results_dir)
    bmk = parse_benchmark_command(results_dir)
    aip = parse_aiperf_json(results_dir)
    cac = parse_server_metrics(results_dir, srv['framework'])

    # conc: benchmark_command.txt takes priority (it's what the loadgen used)
    conc = bmk['conc'] or srv['conc']

    # model
    model = srv['model'] or ''

    # precision
    precision = precision_override or _infer_precision(model)

    # ms → s conversion for latency fields
    ttft_scale = 1e-3 if aip.get('_ttft_unit') == 'ms' else 1.0
    e2el_scale = 1e-3  # request_latency is always in ms in aiperf json

    def ms2s(v):
        return v * 1e-3 if v is not None else None

    # tput per GPU
    tp = srv['tp']
    total_tput = aip.get('total_tput_tps')
    tput_per_gpu = (total_tput / tp) if total_tput and tp else None
    out_tput = aip.get('output_tput_tps')
    inp_tput = aip.get('input_tput_tps')

    entry = {
        'hw':                hw,
        'conc':              conc,
        'image':             image_override or '',
        'model':             model,
        'infmax_model_prefix': model_prefix,
        'framework':         srv['framework'],
        'precision':         precision,
        'spec_decoding':     'none',
        'disagg':            False,
        'scenario_type':     'agentic-coding',
        'is_multinode':      False,
        'tp':                tp,
        'ep':                srv['ep'],
        'dp_attention':      'false',
        'offloading':        srv['offloading'],

        # request stats
        'num_requests_total':      aip.get('num_requests_successful'),
        'num_requests_successful': aip.get('num_requests_successful'),

        # throughput
        'mean_qps': aip.get('mean_qps'),

        # TTFT (convert ms → s)
        'mean_ttft': ms2s(aip.get('mean_ttft')),
        'p75_ttft':  ms2s(aip.get('p75_ttft')),
        'p90_ttft':  ms2s(aip.get('p90_ttft')),
        'p95_ttft':  ms2s(aip.get('p95_ttft')),
        'std_ttft':  ms2s(aip.get('std_ttft')),

        # E2E latency (convert ms → s)
        'mean_e2el': ms2s(aip.get('mean_e2el')),
        'p75_e2el':  ms2s(aip.get('p75_e2el')),
        'p90_e2el':  ms2s(aip.get('p90_e2el')),
        'p95_e2el':  ms2s(aip.get('p95_e2el')),
        'std_e2el':  ms2s(aip.get('std_e2el')),

        # ITL (convert ms → s)
        'mean_itl': ms2s(aip.get('mean_itl')),
        'p75_itl':  ms2s(aip.get('p75_itl')),
        'p90_itl':  ms2s(aip.get('p90_itl')),
        'p95_itl':  ms2s(aip.get('p95_itl')),
        'std_itl':  ms2s(aip.get('std_itl')),

        # sequence lengths
        'mean_input_tokens':        aip.get('mean_input_tokens'),
        'p75_input_tokens':         aip.get('p75_input_tokens'),
        'p90_input_tokens':         aip.get('p90_input_tokens'),
        'p95_input_tokens':         aip.get('p95_input_tokens'),
        'std_input_tokens':         aip.get('std_input_tokens'),
        'mean_output_tokens_actual': aip.get('mean_output_tokens_actual'),
        'p75_output_tokens_actual':  aip.get('p75_output_tokens_actual'),
        'p90_output_tokens_actual':  aip.get('p90_output_tokens_actual'),
        'p95_output_tokens_actual':  aip.get('p95_output_tokens_actual'),
        'std_output_tokens_actual':  aip.get('std_output_tokens_actual'),

        # cache
        'theoretical_cache_hit_rate': aip.get('theoretical_cache_hit_rate'),
        'server_gpu_cache_hit_rate':   None,
        'server_cpu_cache_hit_rate':   None,

        # totals
        'total_prompt_tokens':      aip.get('total_prompt_tokens'),
        'total_generation_tokens':  aip.get('total_generation_tokens'),
        'total_requests_completed': aip.get('total_requests_completed'),

        # throughput (tok/s)
        'input_tput_tps':  inp_tput,
        'output_tput_tps': out_tput,
        'total_tput_tps':  total_tput,
        'duration_seconds': aip.get('duration_seconds'),

        # derived: per-GPU
        'tput_per_gpu':        tput_per_gpu,
        'output_tput_per_gpu': (out_tput / tp) if out_tput and tp else None,
        'input_tput_per_gpu':  (inp_tput / tp) if inp_tput and tp else None,

        # cache source breakdown (for stacked bar)
        **cac,
    }

    return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build agg_bmk.json from raw aiperf artifact directories.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--results-dir', nargs='+', required=True, metavar='DIR',
                        help='One or more raw results directories (each has aiperf_artifacts/).')
    parser.add_argument('--hw', required=True,
                        help='Hardware label, e.g. mi355x, h200, b200.')
    parser.add_argument('--model-prefix', required=True,
                        help='Short model prefix, e.g. kimik2.7-code, glm5.1.')
    parser.add_argument('--precision', default=None,
                        help='Precision override (fp4/fp8/int4/bf16). Auto-detected if omitted.')
    parser.add_argument('--image', default='',
                        help='Docker image tag (informational).')
    parser.add_argument('--output', default=None,
                        help='Output json path. Default: <first results-dir>/results_bmk/agg_bmk.json')
    args = parser.parse_args()

    entries = []
    for d in args.results_dir:
        p = Path(d)
        if not p.exists():
            print(f"ERROR: directory not found: {p}", file=sys.stderr)
            sys.exit(1)
        entry = extract_one(p, args.hw, args.model_prefix, args.precision, args.image)
        entries.append(entry)
        print(f"    offloading={entry['offloading']}  "
              f"conc={entry['conc']}  tp={entry['tp']}  "
              f"p90_e2el={entry['p90_e2el']:.1f}s  "
              f"tput/gpu={entry['tput_per_gpu']:.1f} tok/s")

    # determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.results_dir[0]) / 'results_bmk' / 'agg_bmk.json'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(entries, f, indent=2)

    print(f"\nWrote {len(entries)} entry/entries -> {out_path}")
    print("\nNext step - generate the chart:")
    print(f"  python gg_agentic/pareto_chart_kimik27.py")


if __name__ == '__main__':
    main()
