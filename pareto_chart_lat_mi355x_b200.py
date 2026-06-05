import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = '/home/srok/InferenceMAX_rocm_fresh/InferenceMAX_rocm'

INPUT_FILES = [
    'results/0604/kimik2.5-fp4-b200-vllm-agentic-lmcache/results_bmk/agg_bmk.json',
    'results/0604/kimik2.5-fp4-mi355x-vllm-agentic-lmcache/results_bmk/agg_bmk.json',
]

# Fallback metadata inferred from directory name when JSON fields are empty
FALLBACK_META = {
    'kimik2.5-fp4-b200-vllm-agentic-lmcache': {
        'hw': 'b200', 'infmax_model_prefix': 'kimik2.5',
        'framework': 'vllm', 'precision': 'fp4',
    },
}

# Remove offloading='none' entries
EXCLUDE_OFFLOADING = {'none'}

OUTPUT_FILE = os.path.join(BASE_DIR, 'pareto_chart_lat.png')

AMD_HW = {'mi355x', 'mi325x', 'mi300x'}
NV_HW = {'h100', 'h200', 'b200'}

# Colors per HW family
COLOR_AMD = '#DC143C'
COLOR_NV = '#006400'

MARKER_HW = {'mi355x': 'o', 'b200': 's', 'mi300x': '^', 'mi325x': 'D', 'h100': 'v', 'h200': 'P'}

CACHE_COLORS_AMD = {'local_compute': '#DC143C', 'local_cache_hit': '#FF6699', 'external_kv_transfer': '#000000'}
CACHE_COLORS_NV  = {'local_compute': '#006400', 'local_cache_hit': '#7FFF00', 'external_kv_transfer': '#000000'}
CACHE_LABELS = {'local_compute': 'Local Compute', 'local_cache_hit': 'Cache Hit', 'external_kv_transfer': 'Ext KV Transfer'}


def hw_color(hw):
    return COLOR_AMD if hw in AMD_HW else COLOR_NV


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


def fix_metadata(entry, rel_path):
    """Fill empty metadata fields from directory name fallback."""
    for dir_key, meta in FALLBACK_META.items():
        if dir_key in rel_path:
            for k, v in meta.items():
                if not entry.get(k):
                    entry[k] = v
            break
    return entry


def find_metrics_file(parent_dir, entry):
    pattern = os.path.join(
        parent_dir,
        f"agentic_{entry['infmax_model_prefix']}_tp{entry['tp']}_conc{entry['conc']}"
        f"_offload{entry.get('offloading','none')}_{entry['precision']}_{entry['framework']}"
        f"_tp{entry['tp']}-ep1-dpafalse_disagg-false_spec-none_conc{entry['conc']}_{entry['hw']}-*",
        'aiperf_artifacts', 'server_metrics_export.json')
    m = glob.glob(pattern)
    return m[0] if m else None


def extract_cache(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {}
    metrics = data.get('metrics', {})
    result = {'local_compute': 0.0, 'local_cache_hit': 0.0, 'external_kv_transfer': 0.0}
    for s in metrics.get('vllm:prompt_tokens_by_source', {}).get('series', []):
        src = s.get('labels', {}).get('source', '')
        if src in result:
            result[src] = s.get('stats', {}).get('total', 0.0) or 0.0
    return result


def load_all_data():
    """Load and merge data from all input files, applying filters and metadata fixes."""
    all_entries = []
    parent_dirs = {}  # hw -> parent_dir for cache lookup
    for rel in INPUT_FILES:
        full = os.path.join(BASE_DIR, rel)
        print(f"Processing: {rel}")
        with open(full) as f:
            data = json.load(f)
        parent_dir = os.path.dirname(os.path.dirname(full))
        for entry in data:
            fix_metadata(entry, rel)
            off = entry.get('offloading', 'none')
            if off in EXCLUDE_OFFLOADING:
                continue
            entry['_parent_dir'] = parent_dir
            all_entries.append(entry)
    return all_entries


def main():
    entries = load_all_data()

    # Group by hw
    hw_groups = {}
    for e in entries:
        hw_groups.setdefault(e['hw'], []).append(e)

    hw_list = sorted(hw_groups.keys())

    # Build title from first entry
    d0 = entries[0]
    model_name = d0.get('infmax_model_prefix', d0.get('model', ''))
    precision = d0.get('precision', '')
    framework = d0.get('framework', '')
    hw_label = ' vs '.join(hw_list)
    title_str = f"{hw_label} | {model_name} | {precision} | {framework}"

    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    ax_pareto, ax_ttft, ax_cache = axes

    # ==================== Pareto chart ====================
    for hw in hw_list:
        hw_entries = hw_groups[hw]
        color = hw_color(hw)
        marker = MARKER_HW.get(hw, 'o')
        xs = [e['p90_e2el'] for e in hw_entries]
        ys = [e['tput_per_gpu'] for e in hw_entries]
        ax_pareto.scatter(xs, ys, c=color, marker=marker, label=hw, s=70,
                          edgecolors='black', linewidths=0.5, zorder=5)
        for x, y, e in zip(xs, ys, hw_entries):
            ax_pareto.annotate(f"[c={e['conc']},gpus={e['tp']}]", (x, y),
                               fontsize=5, ha='left', va='bottom',
                               xytext=(3, 3), textcoords='offset points')
        frontier = compute_pareto(list(zip(xs, ys)))
        if len(frontier) > 1:
            ax_pareto.plot([p[0] for p in frontier], [p[1] for p in frontier],
                           '--', color=color, linewidth=1.5, alpha=0.6, zorder=3)

    ax_pareto.set_xlabel('p90 E2E Latency (s)', fontsize=8)
    ax_pareto.set_ylabel('Throughput/GPU (tok/s)', fontsize=8)
    ax_pareto.set_title(f'Pareto: {title_str}', fontsize=9, fontweight='bold')
    ax_pareto.legend(fontsize=7, loc='best')
    ax_pareto.grid(True, alpha=0.3)
    ax_pareto.tick_params(labelsize=7)

    # ==================== TTFT bar chart ====================
    all_concs = sorted(set(e['conc'] for e in entries))
    bw = 0.8 / max(len(hw_list), 1)

    for i, hw in enumerate(hw_list):
        hw_entries = hw_groups[hw]
        color = hw_color(hw)
        c2t = {e['conc']: e['p90_ttft'] for e in hw_entries}
        vals = [c2t.get(c, 0) for c in all_concs]
        pos = np.arange(len(all_concs)) + i * bw
        ax_ttft.bar(pos, vals, bw, color=color, label=hw, alpha=0.85,
                    edgecolor='black', linewidth=0.5)

    ax_ttft.set_xlabel('Concurrency', fontsize=8)
    ax_ttft.set_ylabel('p90 TTFT (s)', fontsize=8)
    ax_ttft.set_title(f'TTFT: {title_str}', fontsize=9, fontweight='bold')
    ax_ttft.set_xticks(np.arange(len(all_concs)) + bw * (len(hw_list) - 1) / 2)
    ax_ttft.set_xticklabels([str(c) for c in all_concs], fontsize=7)
    ax_ttft.legend(fontsize=7, loc='best')
    ax_ttft.grid(True, alpha=0.3, axis='y')
    ax_ttft.tick_params(labelsize=7)

    # ==================== Cache hit stacked bar ====================
    sources = ['local_compute', 'local_cache_hit', 'external_kv_transfer']

    for i, hw in enumerate(hw_list):
        hw_entries = hw_groups[hw]
        pos = np.arange(len(all_concs)) + i * bw
        bottoms = np.zeros(len(all_concs))

        # Build cache data
        cache_data = {}
        for e in hw_entries:
            mpath = find_metrics_file(e['_parent_dir'], e)
            if mpath:
                cache_data[e['conc']] = extract_cache(mpath)
            else:
                cache_data[e['conc']] = {}

        for src in sources:
            vals = np.array([cache_data.get(c, {}).get(src, 0) for c in all_concs], dtype=float)
            if vals.sum() == 0:
                continue
            ax_cache.bar(pos, vals, bw, bottom=bottoms,
                         color=get_cache_color(hw, src), edgecolor='black', linewidth=0.3,
                         label=f"{CACHE_LABELS[src]} ({hw})")
            bottoms += vals

    ax_cache.set_xlabel('Concurrency', fontsize=8)
    ax_cache.set_ylabel('Prompt Tokens by Source', fontsize=8)
    ax_cache.set_title(f'Cache: {title_str}', fontsize=9, fontweight='bold')
    ax_cache.set_xticks(np.arange(len(all_concs)) + bw * (len(hw_list) - 1) / 2)
    ax_cache.set_xticklabels([str(c) for c in all_concs], fontsize=7)
    handles, labels = ax_cache.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_cache.legend(by_label.values(), by_label.keys(), fontsize=5.5, loc='best')
    ax_cache.grid(True, alpha=0.3, axis='y')
    ax_cache.tick_params(labelsize=7)

    fig.suptitle('Agentic Coding - Pareto (E2EL vs Tput) & TTFT & Cache Hit',
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved {OUTPUT_FILE}")
    plt.close()


if __name__ == '__main__':
    main()
