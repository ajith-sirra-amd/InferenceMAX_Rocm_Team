#!/usr/bin/env python3
"""
bmk_table.py — Append benchmark results to a summary CSV table.

Supports two input modes (auto-detected per directory):

  RAW ARTIFACTS  — directory contains vllm_command.txt / sglang_command.txt
                   and aiperf_artifacts/.  Same structure as extract_agg_bmk.py.

  AGG_BMK JSON   — directory contains an already-extracted agg_bmk.json
                   (one or more entries; one CSV row is written per entry).
                   Supports both the upstream pipeline format and our own format.

Columns
-------
Model | HW | Framework | Precision | TP | CONC | Offloading
TPUT/GPU (tok/s) | GPU Cache Hit % | Ext KV Hit % | GPU KV Usage %
mean TTFT (s) | p90 TTFT (s) | mean TPOT (ms) | p90 E2EL (s) | Requests OK

Usage (single run — raw artifacts):
  python gg_agentic/bmk_table.py \\
      --results-dir results/ \\
      --hw mi355x \\
      --model-prefix kimik2.7-code

Usage (directories with agg_bmk.json):
  python gg_agentic/bmk_table.py \\
      --results-dir results/results_bmk_offload_none results/results_bmk_offload_lmcache \\
      --hw mi355x \\
      --model-prefix kimik2.7-code \\
      --output results_combined/results_bmk/benchmark_table.csv

Multiple raw artifact dirs at once (one row each, appended in order):
  python gg_agentic/bmk_table.py \\
      --results-dir results_none/ results_lmcache/ results_cpu/ \\
      --hw mi355x \\
      --model-prefix kimik2.7-code
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Reuse parsers from extract_agg_bmk (same package directory)
sys.path.insert(0, str(Path(__file__).parent))
from extract_agg_bmk import (
    parse_server_command,
    parse_benchmark_command,
    parse_aiperf_json,
    _infer_precision,
)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

COLUMNS = [
    'model_prefix', 'hw', 'framework', 'precision', 'tp', 'conc', 'offloading',
    'tput_per_gpu', 'gpu_cache_hit_pct', 'ext_kv_hit_pct', 'gpu_kv_usage_pct',
    'mean_ttft_s', 'p90_ttft_s', 'mean_tpot_ms', 'p90_e2el_s', 'requests_ok',
]

HEADERS = [
    'Model', 'HW', 'Framework', 'Precision', 'TP', 'CONC', 'Offloading',
    'TPUT/GPU (tok/s)', 'GPU Cache Hit %', 'Ext KV Hit %', 'GPU KV Usage %',
    'mean TTFT (s)', 'p90 TTFT (s)', 'mean TPOT (ms)', 'p90 E2EL (s)', 'Requests OK',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms2s(v):
    """Convert milliseconds to seconds, or return None."""
    return round(v * 1e-3, 3) if v is not None else None


def _pct(numerator, denominator, decimals=1):
    """Return percentage rounded to *decimals*, or None."""
    if denominator and denominator > 0 and numerator is not None:
        return round(numerator / denominator * 100, decimals)
    return None


def _frac_to_pct(v, decimals=1):
    """Convert a 0-1 fraction to a rounded percentage, or None."""
    return round(v * 100, decimals) if v is not None else None


# ---------------------------------------------------------------------------
# Extended server-metrics parser for raw artifact directories
# ---------------------------------------------------------------------------

def _parse_server_metrics_extended(results_dir: Path, framework: str) -> dict:
    """
    Returns cache token source totals + GPU KV cache average utilisation.

    vllm  -> vllm:prompt_tokens_by_source  +  vllm:kv_cache_usage_perc
    sglang -> sglang:prompt_tokens / sglang:cached_tokens  +  sglang:gpu_cache_usage
    """
    result = {
        'cache_local_compute':        0.0,
        'cache_local_cache_hit':      0.0,
        'cache_external_kv_transfer': 0.0,
        'kv_usage_avg':               None,
    }

    path = results_dir / 'aiperf_artifacts' / 'server_metrics_export.json'
    if not path.exists():
        print(f"  WARNING: {path} not found", file=sys.stderr)
        return result

    with open(path) as f:
        data = json.load(f)

    metrics = data.get('metrics', {})

    if framework == 'vllm':
        src_map = {}
        for s in metrics.get('vllm:prompt_tokens_by_source', {}).get('series', []):
            src = s.get('labels', {}).get('source', '')
            src_map[src] = s.get('stats', {}).get('total', 0.0) or 0.0
        result['cache_local_compute']        = src_map.get('local_compute', 0.0)
        result['cache_local_cache_hit']      = src_map.get('local_cache_hit', 0.0)
        result['cache_external_kv_transfer'] = src_map.get('external_kv_transfer', 0.0)

        for s in metrics.get('vllm:kv_cache_usage_perc', {}).get('series', []):
            avg = s.get('stats', {}).get('avg')
            if avg is not None:
                result['kv_usage_avg'] = avg
                break

    else:  # sglang
        prompt_total = sum(
            (s.get('stats', {}).get('total') or 0.0)
            for s in metrics.get('sglang:prompt_tokens', {}).get('series', [])
        )
        device_total = 0.0
        host_total   = 0.0
        for s in metrics.get('sglang:cached_tokens', {}).get('series', []):
            cs = s.get('labels', {}).get('cache_source', '')
            t  = s.get('stats', {}).get('total') or 0.0
            if cs == 'device':
                device_total = t
            elif cs == 'host':
                host_total = t
        result['cache_local_cache_hit']      = device_total
        result['cache_external_kv_transfer'] = host_total
        result['cache_local_compute']        = max(0.0, prompt_total - device_total - host_total)

        for s in metrics.get('sglang:gpu_cache_usage', {}).get('series', []):
            avg = s.get('stats', {}).get('avg')
            if avg is not None:
                result['kv_usage_avg'] = avg
                break

    return result


# ---------------------------------------------------------------------------
# Row extraction — raw artifact directory
# ---------------------------------------------------------------------------

def _extract_from_artifacts(results_dir: Path, hw: str, model_prefix: str,
                             precision_override: str | None) -> list[dict]:
    """Read raw artifact files and return a list with one row dict."""
    srv = parse_server_command(results_dir)
    bmk = parse_benchmark_command(results_dir)
    aip = parse_aiperf_json(results_dir)
    met = _parse_server_metrics_extended(results_dir, srv['framework'])

    conc = bmk['conc'] or srv['conc']
    tp   = srv['tp']
    precision = precision_override or _infer_precision(srv['model'])

    total_tput   = aip.get('total_tput_tps')
    tput_per_gpu = round(total_tput / tp, 1) if total_tput and tp else None

    lc  = met['cache_local_compute']
    lh  = met['cache_local_cache_hit']
    ek  = met['cache_external_kv_transfer']
    tok_total         = (lc or 0.0) + (lh or 0.0) + (ek or 0.0)
    gpu_cache_hit_pct = _pct(lh, tok_total)
    ext_kv_hit_pct    = _pct(ek, tok_total)

    kv_raw           = met.get('kv_usage_avg')
    gpu_kv_usage_pct = round(kv_raw * 100, 1) if kv_raw is not None else None

    # aiperf latency is in ms — convert to s
    mean_ttft_s  = _ms2s(aip.get('mean_ttft'))
    p90_ttft_s   = _ms2s(aip.get('p90_ttft'))
    mean_tpot_ms = round(aip.get('mean_itl'), 2) if aip.get('mean_itl') else None
    p90_e2el_s   = _ms2s(aip.get('p90_e2el'))

    row = {
        'model_prefix':      model_prefix,
        'hw':                hw,
        'framework':         srv['framework'],
        'precision':         precision,
        'tp':                tp,
        'conc':              conc,
        'offloading':        srv['offloading'],
        'tput_per_gpu':      tput_per_gpu,
        'gpu_cache_hit_pct': gpu_cache_hit_pct,
        'ext_kv_hit_pct':    ext_kv_hit_pct,
        'gpu_kv_usage_pct':  gpu_kv_usage_pct,
        'mean_ttft_s':       mean_ttft_s,
        'p90_ttft_s':        p90_ttft_s,
        'mean_tpot_ms':      mean_tpot_ms,
        'p90_e2el_s':        p90_e2el_s,
        'requests_ok':       aip.get('num_requests_successful'),
    }
    return [row]


# ---------------------------------------------------------------------------
# Row extraction — agg_bmk.json (upstream or our own format)
# ---------------------------------------------------------------------------

def _extract_from_agg_bmk(results_dir: Path, hw: str, model_prefix: str,
                           precision_override: str | None) -> list[dict]:
    """
    Read agg_bmk.json and return one row dict per entry.

    Handles two variants:
      - 'upstream' format: has gpu_kv_cache_usage_pct / server_gpu_cache_hit_rate
        (all latency values in seconds, cache rates as 0-1 fractions)
      - 'our' format (extract_agg_bmk.py): has cache_local_compute /
        cache_local_cache_hit / cache_external_kv_transfer
        (all latency values in seconds, mean_itl in seconds)
    """
    agg_path = results_dir / 'agg_bmk.json'
    with open(agg_path) as f:
        entries = json.load(f)

    rows = []
    for e in entries:
        # --- identity fields (fall back to CLI args if empty) ---
        hw_val        = e.get('hw') or hw
        mp_val        = e.get('infmax_model_prefix') or model_prefix
        framework_val = e.get('framework') or 'vllm'
        model_name    = e.get('model', '')
        precision_val = precision_override or e.get('precision') or _infer_precision(model_name)
        tp            = e.get('tp') or 8
        conc          = e.get('conc')
        offloading    = e.get('offloading', 'none')

        # --- throughput ---
        tput_per_gpu = e.get('tput_per_gpu')
        if tput_per_gpu is not None:
            tput_per_gpu = round(tput_per_gpu, 1)

        # --- cache rates ---
        # Detect format: upstream has 'gpu_kv_cache_usage_pct',
        #                our format has 'cache_local_compute'.
        if 'gpu_kv_cache_usage_pct' in e or 'server_gpu_cache_hit_rate' in e:
            # upstream format — rates stored as fractions (0-1)
            gpu_cache_hit_pct = _frac_to_pct(e.get('server_gpu_cache_hit_rate'))
            ext_kv_hit_pct    = _frac_to_pct(e.get('server_external_cache_hit_rate'))
            gpu_kv_usage_pct  = _frac_to_pct(e.get('gpu_kv_cache_usage_pct'))
        else:
            # our format — compute from token counts
            lc  = e.get('cache_local_compute', 0.0) or 0.0
            lh  = e.get('cache_local_cache_hit', 0.0) or 0.0
            ek  = e.get('cache_external_kv_transfer', 0.0) or 0.0
            tok_total         = lc + lh + ek
            gpu_cache_hit_pct = _pct(lh, tok_total)
            ext_kv_hit_pct    = _pct(ek, tok_total)
            gpu_kv_usage_pct  = None   # not stored in our agg_bmk.json

        # --- latency: both formats store values in seconds ---
        mean_ttft_s  = round(e['mean_ttft'],  3) if e.get('mean_ttft')  is not None else None
        p90_ttft_s   = round(e['p90_ttft'],   3) if e.get('p90_ttft')   is not None else None
        p90_e2el_s   = round(e['p90_e2el'],   3) if e.get('p90_e2el')   is not None else None
        # mean_itl is in seconds in both formats -> convert to ms for the table
        itl_s        = e.get('mean_itl') or e.get('mean_tpot')
        mean_tpot_ms = round(itl_s * 1e3, 2) if itl_s is not None else None

        row = {
            'model_prefix':      mp_val,
            'hw':                hw_val,
            'framework':         framework_val,
            'precision':         precision_val,
            'tp':                tp,
            'conc':              conc,
            'offloading':        offloading,
            'tput_per_gpu':      tput_per_gpu,
            'gpu_cache_hit_pct': gpu_cache_hit_pct,
            'ext_kv_hit_pct':    ext_kv_hit_pct,
            'gpu_kv_usage_pct':  gpu_kv_usage_pct,
            'mean_ttft_s':       mean_ttft_s,
            'p90_ttft_s':        p90_ttft_s,
            'mean_tpot_ms':      mean_tpot_ms,
            'p90_e2el_s':        p90_e2el_s,
            'requests_ok':       e.get('num_requests_successful'),
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extract_rows(results_dir: Path, hw: str, model_prefix: str,
                 precision_override: str | None) -> list[dict]:
    """
    Auto-detect whether *results_dir* has raw artifacts or an agg_bmk.json
    and call the appropriate extractor.
    """
    has_artifacts = (
        (results_dir / 'vllm_command.txt').exists()
        or (results_dir / 'sglang_command.txt').exists()
        or (results_dir / 'aiperf_artifacts').is_dir()
    )
    has_agg = (results_dir / 'agg_bmk.json').exists()

    if has_artifacts:
        print(f"  [{results_dir.name}] mode: raw artifacts")
        rows = _extract_from_artifacts(results_dir, hw, model_prefix, precision_override)
    elif has_agg:
        print(f"  [{results_dir.name}] mode: agg_bmk.json")
        rows = _extract_from_agg_bmk(results_dir, hw, model_prefix, precision_override)
    else:
        print(f"  WARNING: [{results_dir.name}] no artifacts and no agg_bmk.json found",
              file=sys.stderr)
        return []

    for r in rows:
        print(f"    offloading={r['offloading']}  conc={r['conc']}  tp={r.get('tp')}"
              f"  tput/gpu={r['tput_per_gpu']}  gpu_cache={r['gpu_cache_hit_pct']}%"
              f"  ext_kv={r['ext_kv_hit_pct']}%  kv_usage={r['gpu_kv_usage_pct']}%"
              f"  p90_e2el={r['p90_e2el_s']}s")
    return rows


# ---------------------------------------------------------------------------
# CSV writer (append or create)
# ---------------------------------------------------------------------------

def append_to_csv(out_path: Path, rows: list[dict]) -> None:
    """Write *rows* to *out_path* (CSV).  Creates header if file is new."""
    is_new = not out_path.exists() or out_path.stat().st_size == 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        if is_new:
            f.write('# ' + ','.join(HEADERS) + '\n')
            writer.writeheader()
        writer.writerows(rows)

    print(f"\nAppended {len(rows)} row(s) -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Append benchmark results to a summary CSV table.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--results-dir', nargs='+', required=True, metavar='DIR',
                        help='One or more results directories '
                             '(raw artifacts or with agg_bmk.json).')
    parser.add_argument('--hw', required=True,
                        help='Hardware label (mi355x / mi325x / h200 / b200 / ...).')
    parser.add_argument('--model-prefix', required=True,
                        help='Short model prefix, e.g. kimik2.7-code.')
    parser.add_argument('--precision', default=None,
                        help='Precision override (fp4/fp8/int4/bf16). Auto-detected if omitted.')
    parser.add_argument('--output', default=None,
                        help='CSV output path.  '
                             'Default: <first results-dir>/results_bmk/benchmark_table.csv')
    args = parser.parse_args()

    all_rows = []
    for d in args.results_dir:
        p = Path(d)
        if not p.exists():
            print(f"ERROR: directory not found: {p}", file=sys.stderr)
            sys.exit(1)
        all_rows.extend(extract_rows(p, args.hw, args.model_prefix, args.precision))

    if not all_rows:
        print("No rows extracted. Check your --results-dir paths.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else \
               Path(args.results_dir[0]) / 'results_bmk' / 'benchmark_table.csv'

    append_to_csv(out_path, all_rows)
    print("\nDone. Open in Excel / LibreOffice Calc or import into your dashboard.")


if __name__ == '__main__':
    main()
