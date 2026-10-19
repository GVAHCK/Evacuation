"""Generates publication-quality IEEE-ready research figures from experiment outputs."""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger

logger = setup_logger("GenerateFigures")

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_training_curves():
    """Figure 1 & 2: PPO & DQN Training reward curves and lengths."""
    monitor_files = glob.glob("logs/**/monitor.csv", recursive=True)
    if not monitor_files:
        logger.warning("No monitor.csv files found for training curves.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mf in monitor_files:
        try:
            df = pd.read_csv(mf, skiprows=1)
            if "r" in df.columns:
                # Rolling smoothed reward
                roll_r = df["r"].rolling(window=min(20, len(df)), min_periods=1).mean()
                label = "PPO" if "ppo" in mf else "DQN"
                ax.plot(roll_r, label=label, linewidth=2.0)
        except Exception:
            continue

    ax.set_title("Training Reward Convergence Across Timesteps")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return (R)")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig1_training_reward_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Generated {out_path}")


def plot_model_comparisons():
    """Figures 3, 4, 5, 6: Model success rate, evacuation time, and HEI comparisons."""
    summary_path = "results/tables/model_comparison.csv"
    raw_path = "results/tables/raw_evaluation_records.csv"

    if os.path.isfile(raw_path):
        raw_df = pd.read_csv(raw_path)

        # 1. Success Rate Bar Chart
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        succ_rates = raw_df.groupby("model")["success"].mean() * 100
        bars = ax.bar(succ_rates.index, succ_rates.values, color=["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"])
        ax.set_ylabel("Safe Evacuation Success Rate (%)")
        ax.set_title("Evacuation Success Rate Comparison Across Unseen Scenarios")
        ax.set_ylim(0, 105)
        for b in bars:
            yval = b.get_height()
            ax.text(b.get_x() + b.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig3_success_rate_comparison.png"), dpi=300)
        plt.close()

        # 2. Evacuation Time Box Plot
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(data=raw_df, x="model", y="evacuation_time", ax=ax, palette="Set2")
        ax.set_ylabel("Evacuation Travel Time (s)")
        ax.set_title("Evacuation Travel Time Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig4_evacuation_time_boxplot.png"), dpi=300)
        plt.close()

        # 3. Hazard Exposure Index (HEI)
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        mean_hei = raw_df.groupby("model")["hazard_exposure"].mean()
        bars = ax.bar(mean_hei.index, mean_hei.values, color=["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])
        ax.set_ylabel("Mean Hazard Exposure Index (HEI)")
        ax.set_title("Cumulative Hazard Contact (Lower = Safer Route)")
        for b in bars:
            yval = b.get_height()
            ax.text(b.get_x() + b.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig5_hazard_exposure_comparison.png"), dpi=300)
        plt.close()

        # 4. Success Rate vs Traffic Level
        fig, ax = plt.subplots(figsize=(7, 4.5))
        traffic_perf = raw_df.groupby(["traffic_level", "model"])["success"].mean().unstack() * 100
        traffic_perf.plot(kind="bar", ax=ax)
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Algorithm Robustness Across Traffic Congestion Regimes")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig8_success_vs_traffic.png"), dpi=300)
        plt.close()


def plot_ablations():
    """Figures 9 & 10: State and Reward Ablation charts."""
    state_path = "results/tables/state_ablation.csv"
    if os.path.isfile(state_path):
        df_st = pd.read_csv(state_path)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.barh(df_st["Ablation_Condition"], df_st["Success_Rate"], color="#3182bd")
        ax.set_xlabel("Safe Evacuation Success Rate (%)")
        ax.set_title("Ablation Study: Contribution of State Feature Modalities")
        for b in bars:
            w = b.get_width()
            ax.text(w + 1.0, b.get_y() + b.get_height()/2.0, f"{w:.1f}%", ha='left', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig10_state_ablation.png"), dpi=300)
        plt.close()

    rew_path = "results/tables/reward_ablation.csv"
    if os.path.isfile(rew_path):
        df_rw = pd.read_csv(rew_path)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=df_rw, x="Reward_Formulation", y="Mean_HEI", ax=ax, palette="Blues_r")
        ax.set_ylabel("Hazard Exposure Index (HEI)")
        ax.set_title("Reward Function Ablation: Impact on Safety vs Distance")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig9_reward_ablation.png"), dpi=300)
        plt.close()


def main():
    logger.info("Generating all research figures from logged experiment data...")
    plot_training_curves()
    plot_model_comparisons()
    plot_ablations()
    logger.info("All figures generated successfully in results/figures/!")


if __name__ == "__main__":
    main()
