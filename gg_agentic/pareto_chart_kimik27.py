"""
Pareto / TTFT / Cache-hit chart for the Kimi-K2.7-Code agentic runs.

Reads one or more agg_bmk.json files and produces, per file, a row of 3 panels:
  left   : Pareto frontier  -> x = p90 E2E latency (s), y = throughput/GPU (tok/s)
  middle : TTFT bar         -> x = concurrency,          y = p90 TTFT (s)
  right  : Cache-hit stack  -> x = concurrency,          y = prompt tokens by source

Conventions follow pareto_chart_lat.py (the skill reference).
Cache source totals are read from the aggregated json itself
(cache_local_compute / cache_local_cache_hit / cache_external_kv_transfer),
falling back to ../aiperf_artifacts/server_metrics_export.json when absent.
"""
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILES = [
    'results/results_bmk/agg_bmk.json',
]

OUTPUT_FILE = os.path.join(BASE_DIR, 'gg_agentic', 'pareto_chart_kimik27.png')

AMD_HW = {'mi355x', 'mi325x', 'mi300x'}

OFFLOAD_COLORS_AMD = {'none': '#A9A9A9', 'lmcache': '#DC143C', 'hicache': '#DC143C'}
OFFLOAD_COLORS_NV  = {'none': '#A9A9A9', 'lmcache': '#006400', 'hicache': '#006400'}
OFFLOAD_MARKERS = {'none': '^', 'lmcache': 'o', 'hicache': 's'}
OFFLOAD_LABELS  = {'none': 'No offloading', 'lmcache': 'LMCache', 'hicache': 'HiCache'}

CACHE_COLORS_AMD = {'local_compute': '#DC143C', 'local_cache_hit': '#FF6699', 'external_kv_transfer': '#000000'}
CACHE_COLORS_NV  = {'local_compute': '#006400', 'local_cache_hit': '#7FFF00', 'external_kv_transfer': '#000000'}
CACHE_LABELS = {'local_compute': 'Local Compute', 'local_cache_hit': 'Cache Hit', 'external_kv_transfer': 'Ext KV Transfer'}


def get_color(hw, off):
    return (OFFLOAD_COLORS_AMD if hw in AMD_HW else OFFLOAD_COLORS_NV).get(off, '#999')


def get_cache_color(hw, src):
    return (CACHE_COLORS_AMD if hw in AMD_HW else CACHE_COLORS_NV).get(src, '#999')


def compute_pareto(points):
    if not points:
        return []
    pts = sorted(points, key=lambda p: p[0])
    frontier = [pts[0]]
    for pt in pts[1:]:
        if pt[1] > frontier[-1][1]:
            frontier.append(pt)
    return frontier


def extract_cache(entry, parent_dir):
    # Prefer cache totals embedded in the aggregated json.
    keys = ('cache_local_compute', 'cache_local_cache_hit', 'cache_external_kv_transfer')
    if any(k in entry for k in keys):
        return {
            'local_compute': entry.get('cache_local_compute', 0.0) or 0.0,
            'local_cache_hit': entry.get('cache_local_cache_hit', 0.0) or 0.0,
            'external_kv_transfer': entry.get('cache_external_kv_transfer', 0.0) or 0.0,
        }
    # Fallback: read the raw server metrics export (vllm only here).
    result = {'local_compute': 0.0, 'local_cache_hit': 0.0, 'external_kv_transfer': 0.0}
    candidates = glob.glob(os.path.join(parent_dir, '..', 'aiperf_artifacts', 'server_metrics_export.json'))
    candidates += glob.glob(os.path.join(parent_dir, 'aiperf_artifacts', 'server_metrics_export.json'))
    for path in candidates:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        for s in data.get('metrics', {}).get('vllm:prompt_tokens_by_source', {}).get('series', []):
            src = s.get('labels', {}).get('source', '')
            if src in result:
                result[src] = s.get('stats', {}).get('total', 0.0) or 0.0
        break
    return result


def plot_row(axes, data, parent_dir, title_str):
    ax_pareto, ax_ttft, ax_cache = axes
    hw = data[0]['hw']

    groups = {}
    for d in data:
        groups.setdefault(d.get('offloading', 'none'), []).append(d)

    # === Pareto chart ===
    for off in sorted(groups):
        entries = groups[off]
        color = get_color(hw, off)
        marker = OFFLOAD_MARKERS.get(off, 'o')
        xs = [e['p90_e2el'] for e in entries]
        ys = [e['tput_per_gpu'] for e in entries]
        ax_pareto.scatter(xs, ys, c=color, marker=marker, label=OFFLOAD_LABELS.get(off, off),
                          s=80, edgecolors='black', linewidths=0.5, zorder=5)
        for x, y, e in zip(xs, ys, entries):
            ax_pareto.annotate(f"[c={e['conc']},gpus={e['tp']}]", (x, y),
                               fontsize=6, ha='left', va='bottom',
                               xytext=(4, 4), textcoords='offset points')
        frontier = compute_pareto(list(zip(xs, ys)))
        if len(frontier) > 1:
            ax_pareto.plot([p[0] for p in frontier], [p[1] for p in frontier],
                           '--', color=color, linewidth=1.5, alpha=0.6, zorder=3)

    ax_pareto.set_xlabel('p90 E2E Latency (s)', fontsize=9)
    ax_pareto.set_ylabel('Throughput / GPU (tok/s)', fontsize=9)
    ax_pareto.set_title(f'Pareto: {title_str}', fontsize=10, fontweight='bold')
    ax_pareto.legend(fontsize=8, loc='best')
    ax_pareto.grid(True, alpha=0.3)
    ax_pareto.tick_params(labelsize=8)

    # === TTFT bar chart ===
    all_concs = sorted(set(d['conc'] for d in data))
    off_keys = sorted(groups)
    bw = 0.8 / max(len(off_keys), 1)
    for i, off in enumerate(off_keys):
        c2t = {e['conc']: e['p90_ttft'] for e in groups[off]}
        vals = [c2t.get(c, 0) for c in all_concs]
        pos = np.arange(len(all_concs)) + i * bw
        ax_ttft.bar(pos, vals, bw, color=get_color(hw, off), label=OFFLOAD_LABELS.get(off, off),
                    alpha=0.85, edgecolor='black', linewidth=0.5,
                    hatch='//' if off == 'none' else '')
    ax_ttft.set_xlabel('Concurrency', fontsize=9)
    ax_ttft.set_ylabel('p90 TTFT (s)', fontsize=9)
    ax_ttft.set_title(f'TTFT: {title_str}', fontsize=10, fontweight='bold')
    ax_ttft.set_xticks(np.arange(len(all_concs)) + bw * (len(off_keys) - 1) / 2)
    ax_ttft.set_xticklabels([str(c) for c in all_concs], fontsize=8)
    ax_ttft.legend(fontsize=8, loc='best')
    ax_ttft.grid(True, alpha=0.3, axis='y')
    ax_ttft.tick_params(labelsize=8)

    # === Cache hit stacked bar ===
    cache_data = {}
    for d in data:
        cache_data[(d['conc'], d.get('offloading', 'none'))] = extract_cache(d, parent_dir)

    sources = ['local_compute', 'local_cache_hit', 'external_kv_transfer']
    for i, off in enumerate(off_keys):
        pos = np.arange(len(all_concs)) + i * bw
        bottoms = np.zeros(len(all_concs))
        for src in sources:
            vals = np.array([cache_data.get((c, off), {}).get(src, 0) for c in all_concs], dtype=float)
            if vals.sum() == 0:
                continue
            ax_cache.bar(pos, vals, bw, bottom=bottoms,
                         color=get_cache_color(hw, src), edgecolor='black', linewidth=0.3,
                         hatch='//' if off == 'none' else '',
                         label=f"{CACHE_LABELS[src]} ({OFFLOAD_LABELS.get(off, off)})")
            bottoms += vals

    ax_cache.set_xlabel('Concurrency', fontsize=9)
    ax_cache.set_ylabel('Prompt Tokens by Source', fontsize=9)
    ax_cache.set_title(f'Cache: {title_str}', fontsize=10, fontweight='bold')
    ax_cache.set_xticks(np.arange(len(all_concs)) + bw * (len(off_keys) - 1) / 2)
    ax_cache.set_xticklabels([str(c) for c in all_concs], fontsize=8)
    handles, labels = ax_cache.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_cache.legend(by_label.values(), by_label.keys(), fontsize=6.5, loc='best')
    ax_cache.grid(True, alpha=0.3, axis='y')
    ax_cache.tick_params(labelsize=8)


def main():
    n = len(INPUT_FILES)
    fig, axes = plt.subplots(n, 3, figsize=(22, 6.0 * n))
    if n == 1:
        axes = [axes]

    for i, rel in enumerate(INPUT_FILES):
        full = os.path.join(BASE_DIR, rel)
        print(f"Processing: {rel}")
        with open(full) as f:
            data = json.load(f)
        parent_dir = os.path.dirname(full)
        d0 = data[0]
        title = f"{d0['hw']}, {d0['infmax_model_prefix']}, {d0['precision']}, {d0['framework']}"
        plot_row(axes[i], data, parent_dir, title)

    fig.suptitle('Kimi-K2.7-Code Agentic - Pareto (E2EL vs Tput/GPU), TTFT & Cache Hit',
                 fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved {OUTPUT_FILE}")
    plt.close()


if __name__ == '__main__':
    main()
