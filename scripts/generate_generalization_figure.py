"""Generate Figure 15: Cross-Region Generalization & Zero-Shot Transfer Study.
Plots safe success rate, evacuation time, per-step Dijkstra agreement, and the generalization gap
across the In-Distribution Multi-District network, George Town Dense Grid, and OMR Sparse Suburban.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 13


def main():
    print("Generating Figure 15: Cross-Region Generalization Study...")

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    regions = ["In-Distribution\n(Multi-District, 336N)", "Dense Urban Grid\n(George Town, 366N)", "Sparse Suburban\n(OMR Corridor, 336N)"]
    x = np.arange(len(regions))
    width = 0.22

    # Panel A: Success Rate Comparison across Regions
    ax1 = fig.add_subplot(gs[0, 0])
    ppo_succ = [96.4, 97.2, 94.8]
    dqn_succ = [91.8, 94.7, 93.6]
    dijk_succ = [97.0, 98.4, 97.2]
    rand_succ = [9.2, 4.8, 7.2]

    b1 = ax1.bar(x - 1.5 * width, ppo_succ, width, label="MaskablePPO (Zero-Shot)", color="#1E88E5", edgecolor="black")
    b2 = ax1.bar(x - 0.5 * width, dqn_succ, width, label="Masked DQN (Zero-Shot)", color="#FB8C00", edgecolor="black")
    b3 = ax1.bar(x + 0.5 * width, dijk_succ, width, label="Dynamic Dijkstra (Oracle)", color="#43A047", edgecolor="black")
    b4 = ax1.bar(x + 1.5 * width, rand_succ, width, label="Random Valid", color="#E53935", edgecolor="black")

    ax1.set_title("(a) Safe Evacuation Success Rate across Unseen Topologies", fontweight="bold")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions)
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper right", frameon=True)

    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0D47A1")

    # Panel B: Evacuation Time Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ppo_time = [133.13, 137.93, 147.75]
    dqn_time = [161.97, 151.75, 153.79]
    dijk_time = [137.27, 136.14, 141.61]

    ax2.bar(x - width, ppo_time, width, label="MaskablePPO", color="#1E88E5", edgecolor="black")
    ax2.bar(x, dqn_time, width, label="Masked DQN", color="#FB8C00", edgecolor="black")
    ax2.bar(x + width, dijk_time, width, label="Dynamic Dijkstra", color="#43A047", edgecolor="black")

    ax2.set_title("(b) Mean Evacuation Time (s) across Network Archetypes", fontweight="bold")
    ax2.set_ylabel("Evacuation Time (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions)
    ax2.set_ylim(0, 200)
    ax2.legend(loc="upper left", frameon=True)

    for i, t in enumerate(ppo_time):
        ax2.text(x[i] - width, t + 3, f"{t:.1f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for i, t in enumerate(dijk_time):
        ax2.text(x[i] + width, t + 3, f"{t:.1f}s", ha="center", va="bottom", fontsize=8.5)

    # Panel C: Per-Step Decision Agreement vs Dynamic Dijkstra
    ax3 = fig.add_subplot(gs[1, 0])
    accs = [94.70, 90.20, 91.43]
    f1s = [0.9287, 0.8610, 0.8802]
    colors = ["#3949AB", "#00897B", "#5E35B1"]

    bars = ax3.bar(regions, accs, color=colors, edgecolor="black", width=0.45)
    ax3.set_title("(c) Per-Step Routing Decision Agreement vs. Dijkstra", fontweight="bold")
    ax3.set_ylabel("Per-Step Action Accuracy (%)")
    ax3.set_ylim(0, 110)

    for bar, acc, f1 in zip(bars, accs, f1s):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{acc:.2f}%\n(F1: {f1:.4f})", ha="center", va="bottom", fontweight="bold")

    ax3.text(0.5, 0.25, "Architectural Invariance:\nRelative candidate-slot embeddings & action masking\nenable >90% decision fidelity on unseen topologies.",
             transform=ax3.transAxes, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#E0F2F1", alpha=0.9, edgecolor="#00897B"))

    # Panel D: Generalization Gap Breakdown
    ax4 = fig.add_subplot(gs[1, 1])
    gap_categories = ["Success Rate\nGap (%)", "Travel Time\nDelta (s)", "Hazard Exp.\nDelta (HEI)"]
    dense_deltas = [+0.8, +4.80, -3.39]
    sparse_deltas = [-1.6, +14.62, +4.05]
    x_gap = np.arange(len(gap_categories))
    w_gap = 0.3

    ax4.bar(x_gap - w_gap / 2, dense_deltas, w_gap, label="Dense Grid (George Town)", color="#00897B", edgecolor="black")
    ax4.bar(x_gap + w_gap / 2, sparse_deltas, w_gap, label="Sparse Suburban (OMR)", color="#5E35B1", edgecolor="black")
    ax4.axhline(0.0, color="gray", linestyle="--", lw=1.2)

    ax4.set_title("(d) Zero-Shot Generalization Gap ($\Delta = \mathrm{Metric}_{\mathrm{Zero}} - \mathrm{Metric}_{\mathrm{InDist}}$)", fontweight="bold")
    ax4.set_ylabel("Metric Delta")
    ax4.set_xticks(x_gap)
    ax4.set_xticklabels(gap_categories)
    ax4.set_ylim(-8, 20)
    ax4.legend(loc="upper right", frameon=True)

    for i, (d, s) in enumerate(zip(dense_deltas, sparse_deltas)):
        ax4.text(x_gap[i] - w_gap / 2, d + (0.8 if d >= 0 else -1.8), f"{d:+.1f}", ha="center", va="bottom" if d >= 0 else "top", fontweight="bold")
        ax4.text(x_gap[i] + w_gap / 2, s + (0.8 if s >= 0 else -1.8), f"{s:+.1f}", ha="center", va="bottom" if s >= 0 else "top", fontweight="bold")

    out_fig = "results/figures/fig15_cross_region_generalization.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f">> Saved Figure 15 to {out_fig}")


if __name__ == "__main__":
    main()
