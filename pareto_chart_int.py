import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')

# JSON file paths
json_files = [
    "results/0603_sa/kimik2.5-fp4-mi355x-vllm-agentic-lmcache/results_bmk/agg_bmk.json",
    "results/0603_sa/qwen3.5-fp8-mi355x-sglang-agentic-hicache/results_bmk/agg_bmk.json",
    "results/0603_sa/glm5.1-fp4-mi355x-sglang-agentic-hicache/results_bmk/agg_bmk.json",
    "results/0603_sa/minimaxm2.5-fp4-mi355x-vllm-agentic-lmcache/results_bmk/agg_bmk.json",
    "results/0603_sa/minimaxm2.5-fp8-mi355x-vllm-agentic-lmcache/results_bmk/agg_bmk.json",
]

# Colors for offloading types
offloading_colors = {
    "none": "#1f77b4",
    "lmcache": "#ff7f0e",
    "hicache": "#2ca02c",
}
offloading_markers = {
    "none": "o",
    "lmcache": "s",
    "hicache": "^",
}
offloading_labels = {
    "none": "No offloading",
    "lmcache": "LMCache",
    "hicache": "HiCache",
}


def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def get_title(entry):
    hw = entry.get("hw", "")
    model = entry.get("infmax_model_prefix", entry.get("model", ""))
    precision = entry.get("precision", "")
    framework = entry.get("framework", "")
    return f"{hw} | {model} | {precision} | {framework}"


def compute_pareto_frontier(points):
    """Given list of (x, y) sorted by x ascending,
    return subset where y is strictly increasing."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: p[0])
    pareto = [sorted_pts[0]]
    for pt in sorted_pts[1:]:
        if pt[1] > pareto[-1][1]:
            pareto.append(pt)
    return pareto


def plot_model(ax_pareto, ax_ttft, data, title):
    # Group by offloading
    groups = {}
    for entry in data:
        off = entry.get("offloading", "none")
        if off not in groups:
            groups[off] = []
        groups[off].append(entry)

    # Pareto chart: x=p90_intvty, y=tput_per_gpu
    all_points_for_pareto = []
    for off in sorted(groups.keys()):
        entries = groups[off]
        color = offloading_colors.get(off, "#999999")
        marker = offloading_markers.get(off, "o")
        label = offloading_labels.get(off, off)
        xs = [e["p90_intvty"] for e in entries]
        ys = [e["tput_per_gpu"] for e in entries]
        labels_txt = [f"[c={e['conc']},gpus={e['tp']}]" for e in entries]
        ax_pareto.scatter(xs, ys, c=color, marker=marker, label=label, s=60, zorder=5,
                          edgecolors='black', linewidths=0.5)
        for x, y, lbl in zip(xs, ys, labels_txt):
            ax_pareto.annotate(lbl, (x, y), fontsize=5.5, ha='left', va='bottom',
                               xytext=(3, 3), textcoords='offset points')
            all_points_for_pareto.append((x, y))

    # Draw pareto frontier line
    pareto_pts = compute_pareto_frontier(all_points_for_pareto)
    if len(pareto_pts) > 1:
        px = [p[0] for p in pareto_pts]
        py = [p[1] for p in pareto_pts]
        ax_pareto.plot(px, py, 'k--', linewidth=1, alpha=0.6, zorder=3)

    ax_pareto.set_xscale('log')
    ax_pareto.set_xlabel("p90 Interactivity (tok/s/user)", fontsize=8)
    ax_pareto.set_ylabel("Throughput per GPU (tok/sec)", fontsize=8)
    ax_pareto.set_title(title, fontsize=9, fontweight='bold')
    ax_pareto.legend(fontsize=7, loc='best')
    ax_pareto.grid(True, alpha=0.3)
    ax_pareto.tick_params(labelsize=7)

    # TTFT bar chart: x=conc, y=p90_ttft, grouped by offloading
    off_keys = sorted(groups.keys())
    all_concs = sorted(set(e["conc"] for e in data))
    bar_width = 0.8 / max(len(off_keys), 1)

    for i, off in enumerate(off_keys):
        entries = groups[off]
        color = offloading_colors.get(off, "#999999")
        label = offloading_labels.get(off, off)
        conc_to_ttft = {e["conc"]: e["p90_ttft"] / 1000.0 for e in entries}
        vals = [conc_to_ttft.get(c, 0) for c in all_concs]
        positions = np.arange(len(all_concs)) + i * bar_width
        ax_ttft.bar(positions, vals, bar_width, color=color, label=label, alpha=0.8,
                    edgecolor='black', linewidth=0.5)

    ax_ttft.set_xlabel("Concurrency (size)", fontsize=8)
    ax_ttft.set_ylabel("p90 TTFT (sec)", fontsize=8)
    ax_ttft.set_title(f"{title} - TTFT", fontsize=9, fontweight='bold')
    ax_ttft.set_xticks(np.arange(len(all_concs)) + bar_width * (len(off_keys) - 1) / 2)
    ax_ttft.set_xticklabels([str(c) for c in all_concs], fontsize=7)
    ax_ttft.legend(fontsize=7, loc='best')
    ax_ttft.grid(True, alpha=0.3, axis='y')
    ax_ttft.tick_params(labelsize=7)


num_files = len(json_files)
fig, axes = plt.subplots(num_files, 2, figsize=(18, 5.5 * num_files))

if num_files == 1:
    axes = axes.reshape(1, -1)

for i, filepath in enumerate(json_files):
    data = load_data(filepath)
    title = get_title(data[0])
    plot_model(axes[i, 0], axes[i, 1], data, title)

fig.suptitle("Agentic Coding - Pareto Frontier (Interactivity vs Throughput) & TTFT",
             fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig("pareto_chart_int.png", dpi=200, bbox_inches='tight')
print("Saved pareto_chart_int.png")
