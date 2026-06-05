import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load data
with open("results/0601/results_bmk (1)/agg_bmk.json") as f:
    data = json.load(f)

# Split by offloading type
none_pts = [(d["mean_e2el"], d["tput_per_gpu"], d["conc"]) for d in data if d["offloading"] == "none"]
lmc_pts  = [(d["mean_e2el"], d["tput_per_gpu"], d["conc"]) for d in data if d["offloading"] == "lmcache"]

def pareto_frontier(points):
    """Return Pareto-optimal points (minimize x, maximize y), sorted by x."""
    sorted_pts = sorted(points, key=lambda p: p[0])  # sort by mean_e2el ascending
    frontier = []
    best_y = -np.inf
    for p in sorted_pts:
        if p[1] > best_y:
            frontier.append(p)
            best_y = p[1]
    return frontier

none_frontier = pareto_frontier(none_pts)
lmc_frontier  = pareto_frontier(lmc_pts)

# Colors
COLOR_NONE = "#E05A2B"   # orange-red
COLOR_LMC  = "#2B7BE0"   # blue

fig, ax = plt.subplots(figsize=(11, 7))

# --- scatter all points ---
for x, y, c in none_pts:
    ax.scatter(x, y, color=COLOR_NONE, s=80, zorder=3, alpha=0.85, edgecolors="white", linewidths=0.6)
    ax.annotate(f"conc={c}", (x, y), textcoords="offset points", xytext=(6, 4),
                fontsize=7.5, color=COLOR_NONE, alpha=0.9)

for x, y, c in lmc_pts:
    ax.scatter(x, y, color=COLOR_LMC, s=80, zorder=3, alpha=0.85, edgecolors="white", linewidths=0.6)
    ax.annotate(f"conc={c}", (x, y), textcoords="offset points", xytext=(6, 4),
                fontsize=7.5, color=COLOR_LMC, alpha=0.9)

# --- draw Pareto frontiers as step lines ---
def draw_frontier(ax, frontier, color, label):
    xs = [p[0] for p in frontier]
    ys = [p[1] for p in frontier]
    ax.plot(xs, ys, color=color, linewidth=2.2, linestyle="--",
            marker="D", markersize=8, markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=1.0, zorder=4, label=label)

draw_frontier(ax, none_frontier, COLOR_NONE, "offloading: none  (Pareto frontier)")
draw_frontier(ax, lmc_frontier,  COLOR_LMC,  "offloading: lmcache  (Pareto frontier)")

# --- labels & formatting ---
hw    = data[0]["hw"]
model = data[0]["model"]

ax.set_xlabel("Mean E2E Latency  (ms)", fontsize=12)
ax.set_ylabel("Throughput per GPU  (tokens/s)", fontsize=12)
ax.set_title(f"Pareto Frontier  |  {hw}  ·  {model}\nagentic-coding scenario  ·  tp=4", fontsize=13)

ax.legend(fontsize=10, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xscale("log")   # log-scale clarifies the wide latency range
ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()
out = "results/0601/pareto_frontier.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()
