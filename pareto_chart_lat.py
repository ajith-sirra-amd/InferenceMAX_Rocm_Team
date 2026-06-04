import json
import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np

files = [
    '/home/srok/InferenceMAX_rocm/results/0603_sa/kimik2.5-fp4-mi355x-vllm-agentic-lmcache/results_bmk/agg_bmk.json',
    '/home/srok/InferenceMAX_rocm/results/0603_sa/qwen3.5-fp8-mi355x-sglang-agentic-hicache/results_bmk/agg_bmk.json',
    '/home/srok/InferenceMAX_rocm/results/0603_sa/glm5.1-fp4-mi355x-sglang-agentic-hicache/results_bmk/agg_bmk.json',
    '/home/srok/InferenceMAX_rocm/results/0603_sa/minimaxm2.5-fp4-mi355x-vllm-agentic-lmcache/results_bmk/agg_bmk.json',
    '/home/srok/InferenceMAX_rocm/results/0603_sa/minimaxm2.5-fp8-mi355x-vllm-agentic-lmcache/results_bmk/agg_bmk.json',
]

offload_colors = {
    'none': '#1f77b4',
    'hicache': '#ff7f0e',
    'lmcache': '#2ca02c',
}

offload_markers = {
    'none': 'o',
    'hicache': 's',
    'lmcache': '^',
}


def compute_pareto(x_vals, y_vals):
    """Return mask of pareto-optimal points (minimize x, maximize y)."""
    points = list(zip(x_vals, y_vals, range(len(x_vals))))
    points.sort(key=lambda p: p[0])
    mask = [False] * len(x_vals)
    max_y = -float('inf')
    for x, y, idx in points:
        if y >= max_y:
            mask[idx] = True
            max_y = y
    return mask


def load_cache_data(json_path):
    """Load prompt_tokens_by_source from server_metrics_export.json files
    in sibling agentic_* directories."""
    results_bmk_dir = os.path.dirname(json_path)
    parent_dir = os.path.dirname(results_bmk_dir)

    pattern = os.path.join(parent_dir, 'agentic_*/aiperf_artifacts/server_metrics_export.json')
    metric_files = glob.glob(pattern)

    cache_data = {}  # (conc, offload) -> {local_compute, local_cache_hit, external_kv_transfer}
    for mf in metric_files:
        dirname = mf.split(os.sep)[-3]
        # Extract conc and offload from dirname like:
        # agentic_kimik2.5_tp4_conc8_offloadlmcache_fp4_vllm_...
        m = re.search(r'_conc(\d+)_offload([a-z]+)_', dirname)
        if not m:
            continue
        conc = int(m.group(1))
        offload = m.group(2)

        try:
            with open(mf) as fh:
                d = json.load(fh)
            metrics = d.get('metrics', {})
            sources = {}

            # Try vllm format first
            metric = metrics.get('vllm:prompt_tokens_by_source', {})
            if metric and metric.get('series'):
                for item in metric.get('series', []):
                    src = item.get('labels', {}).get('source', '')
                    total = item.get('stats', {}).get('total', 0)
                    sources[src] = total
            else:
                # sglang format: derive from prompt_tokens and cached_tokens
                pt_metric = metrics.get('sglang:prompt_tokens', {})
                ct_metric = metrics.get('sglang:cached_tokens', {})
                if pt_metric and pt_metric.get('series'):
                    total_prompt = pt_metric['series'][0].get('stats', {}).get('total', 0)
                    device_cached = 0
                    host_cached = 0
                    for item in ct_metric.get('series', []):
                        cs = item.get('labels', {}).get('cache_source', '')
                        t = item.get('stats', {}).get('total', 0)
                        if cs == 'device':
                            device_cached = t
                        elif cs == 'host':
                            host_cached = t
                    sources['local_cache_hit'] = device_cached
                    sources['external_kv_transfer'] = host_cached
                    sources['local_compute'] = max(0, total_prompt - device_cached - host_cached)

            if sources:
                cache_data[(conc, offload)] = sources
        except Exception:
            continue

    return cache_data


datasets = []
cache_datasets = []
for f in files:
    with open(f) as fh:
        datasets.append(json.load(fh))
    cache_datasets.append(load_cache_data(f))

fig, axes = plt.subplots(len(datasets), 3, figsize=(27, 6 * len(datasets)))
if len(datasets) == 1:
    axes = [axes]

for row_idx, data in enumerate(datasets):
    ax_pareto = axes[row_idx][0]
    ax_ttft = axes[row_idx][1]
    ax_cache = axes[row_idx][2]

    sample = data[0]
    hw = sample['hw']
    model = sample['model'].split('/')[-1]
    precision = sample['precision']
    framework = sample['framework']
    title = f"{hw} | {model} | {precision} | {framework}"

    offloads = sorted(set(e['offloading'] for e in data))

    # --- Pareto chart ---
    for offload in offloads:
        subset = [e for e in data if e['offloading'] == offload]
        subset.sort(key=lambda e: e['p90_e2el'])
        x = [e['p90_e2el'] for e in subset]
        y = [e['tput_per_gpu'] for e in subset]
        labels = [f"c={e['conc']},gpus={e['tp']}" for e in subset]
        color = offload_colors.get(offload, '#999999')
        marker = offload_markers.get(offload, 'o')

        ax_pareto.scatter(x, y, color=color, marker=marker, s=80, label=offload, zorder=5)

        for xi, yi, lbl in zip(x, y, labels):
            ax_pareto.annotate(lbl, (xi, yi), textcoords="offset points",
                               xytext=(5, 5), fontsize=7)

        # Pareto frontier line
        pareto_mask = compute_pareto(x, y)
        px = [xi for xi, m in zip(x, pareto_mask) if m]
        py = [yi for yi, m in zip(y, pareto_mask) if m]
        sorted_pairs = sorted(zip(px, py))
        if sorted_pairs:
            px_s, py_s = zip(*sorted_pairs)
            ax_pareto.plot(px_s, py_s, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

    ax_pareto.set_xlabel('p90_e2el (s)')
    ax_pareto.set_ylabel('tput_per_gpu (tok/sec)')
    ax_pareto.set_title(f'{title}\nPareto Frontier')
    ax_pareto.legend(title='offloading')
    ax_pareto.grid(True, alpha=0.3)

    # --- TTFT bar chart ---
    concs_all = sorted(set(e['conc'] for e in data))
    bar_width = 0.8 / max(len(offloads), 1)
    x_pos = np.arange(len(concs_all))

    for i, offload in enumerate(offloads):
        subset = {e['conc']: e for e in data if e['offloading'] == offload}
        ttft_vals = [subset[c]['p90_ttft'] if c in subset else 0 for c in concs_all]
        color = offload_colors.get(offload, '#999999')
        ax_ttft.bar(x_pos + i * bar_width, ttft_vals, bar_width,
                    color=color, label=offload, alpha=0.85)

    ax_ttft.set_xlabel('conc (size)')
    ax_ttft.set_ylabel('p90_ttft (sec)')
    ax_ttft.set_title(f'{title}\nTTFT by Concurrency')
    ax_ttft.set_xticks(x_pos + bar_width * (len(offloads) - 1) / 2)
    ax_ttft.set_xticklabels([str(c) for c in concs_all])
    ax_ttft.legend(title='offloading')
    ax_ttft.grid(True, alpha=0.3, axis='y')

    # --- Cache hit stacked bar chart ---
    cache_data = cache_datasets[row_idx]
    source_names = ['local_compute', 'local_cache_hit', 'external_kv_transfer']
    source_colors = {'local_compute': '#d62728', 'local_cache_hit': '#2ca02c', 'external_kv_transfer': '#9467bd'}

    bar_width_cache = 0.8 / max(len(offloads), 1)
    x_pos_cache = np.arange(len(concs_all))

    for i, offload in enumerate(offloads):
        bottoms = np.zeros(len(concs_all))
        for src in source_names:
            vals = []
            for c in concs_all:
                sources = cache_data.get((c, offload), {})
                vals.append(sources.get(src, 0))
            vals = np.array(vals)
            label_str = f"{offload}:{src}" if i == 0 or True else None
            ax_cache.bar(x_pos_cache + i * bar_width_cache, vals, bar_width_cache,
                         bottom=bottoms, color=source_colors.get(src, '#999999'),
                         label=f"{offload} - {src}" if True else None, alpha=0.85,
                         edgecolor='white', linewidth=0.5)
            bottoms += vals

    ax_cache.set_xlabel('conc (size)')
    ax_cache.set_ylabel('prompt tokens (total)')
    ax_cache.set_title(f'{title}\nCache Hit by Source')
    ax_cache.set_xticks(x_pos_cache + bar_width_cache * (len(offloads) - 1) / 2)
    ax_cache.set_xticklabels([str(c) for c in concs_all])
    # Deduplicate legend
    handles, labels_leg = ax_cache.get_legend_handles_labels()
    seen = {}
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)
    ax_cache.legend(unique_handles, unique_labels, title='source', fontsize=7, loc='upper left')
    ax_cache.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = '/home/srok/InferenceMAX_rocm/results/0603_sa/pareto_chart_lat.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path}')
plt.close()
