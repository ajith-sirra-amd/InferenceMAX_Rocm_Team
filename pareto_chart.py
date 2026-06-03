import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')

def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_pareto_front(x, y):
    """Get pareto frontier indices with decreasing tput_per_gpu.
    Sort by x ascending, then greedily pick the upper envelope
    where y is non-increasing (the trade-off frontier)."""
    points = sorted(zip(x, y, range(len(x))), key=lambda p: p[0])
    pareto = [points[0][2]]
    current_y = points[0][1]
    # Find max y first, then trace decreasing from there
    max_y_val = max(p[1] for p in points)
    # Start from point with max y, trace rightward keeping only decreasing y
    max_idx = max(range(len(points)), key=lambda i: points[i][1])
    pareto = [points[max_idx][2]]
    current_y = points[max_idx][1]
    for i in range(max_idx + 1, len(points)):
        if points[i][1] < current_y:
            pareto.append(points[i][2])
            current_y = points[i][1]
    return pareto

def plot_dataset(axes, data, row_idx):
    hw = data[0].get('hw') or 'unknown'
    model = data[0].get('model') or 'unknown'
    precision = data[0].get('precision') or 'unknown'
    framework = data[0].get('framework') or 'unknown'
    title = f"{hw} | {model} | {precision} | {framework}"

    # Color map for offloading
    color_map = {'none': '#1f77b4', 'lmcache': '#ff7f0e', 'hicache': '#2ca02c'}
    marker_map = {'none': 'o', 'lmcache': 's', 'hicache': '^'}

    ax_pareto = axes[0]
    ax_ttft = axes[1]

    # Group by offloading (skip entries missing required fields)
    groups = {}
    for entry in data:
        if 'mean_intvty' not in entry or 'tput_per_gpu' not in entry:
            continue
        off = entry['offloading']
        if off not in groups:
            groups[off] = []
        groups[off].append(entry)

    # --- Pareto Frontier Chart (left) ---
    for off in sorted(groups.keys()):
        entries = groups[off]
        x = [e['mean_intvty'] for e in entries]
        y = [e['tput_per_gpu'] for e in entries]
        labels = [f"c={e['conc']},gpus={e['tp']}" for e in entries]
        color = color_map.get(off, 'gray')
        marker = marker_map.get(off, 'o')

        ax_pareto.scatter(x, y, c=color, marker=marker, s=80, label=f"offloading={off}", zorder=5, edgecolors='black', linewidth=0.5)

        for xi, yi, lab in zip(x, y, labels):
            ax_pareto.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=6)

        # Pareto frontier line - connect nearby points on the frontier
        pareto_idx = get_pareto_front(x, y)
        if len(pareto_idx) >= 2:
            px = [x[i] for i in pareto_idx]
            py = [y[i] for i in pareto_idx]
            ax_pareto.plot(px, py, c=color, linestyle='--', linewidth=1.2, alpha=0.7)

    ax_pareto.set_xlabel('mean_intvty (tok/s/user)', fontsize=9)
    ax_pareto.set_ylabel('tput_per_gpu (tok/sec)', fontsize=9)
    ax_pareto.set_title(f"Pareto Frontier - {title}", fontsize=10)
    ax_pareto.legend(fontsize=7, loc='best')
    ax_pareto.grid(True, alpha=0.3)

    # --- TTFT Bar Chart (right) ---
    offloading_types = sorted(groups.keys())
    all_concs = sorted(set(e['conc'] for e in data))
    bar_width = 0.8 / max(len(offloading_types), 1)

    for i, off in enumerate(offloading_types):
        entries = sorted(groups[off], key=lambda e: e['conc'])
        concs = [e['conc'] for e in entries]
        ttfts = [e['mean_ttft'] for e in entries]
        color = color_map.get(off, 'gray')

        x_positions = [all_concs.index(c) + i * bar_width for c in concs]
        ax_ttft.bar(x_positions, ttfts, width=bar_width, color=color, label=f"offloading={off}", edgecolor='black', linewidth=0.5)

    ax_ttft.set_xlabel('conc (size)', fontsize=9)
    ax_ttft.set_ylabel('mean_ttft (sec)', fontsize=9)
    ax_ttft.set_title(f"TTFT - {title}", fontsize=10)
    ax_ttft.set_xticks([j + bar_width * (len(offloading_types) - 1) / 2 for j in range(len(all_concs))])
    ax_ttft.set_xticklabels(all_concs, fontsize=8)
    ax_ttft.legend(fontsize=7, loc='best')
    ax_ttft.grid(True, alpha=0.3, axis='y')


# Load data
files = [
    "/home/srok/InferenceMAX_rocm/results/0603/results_bmk (15)/agg_bmk.json",
]

datasets = [load_json(f) for f in files]

fig, axes = plt.subplots(len(datasets), 2, figsize=(18, 6 * len(datasets)))
if len(datasets) == 1:
    axes = [axes]

for i, data in enumerate(datasets):
    plot_dataset(axes[i], data, i)

plt.tight_layout()
plt.savefig('pareto_chart.png', dpi=150, bbox_inches='tight')
print("Chart saved to pareto_chart.png")
