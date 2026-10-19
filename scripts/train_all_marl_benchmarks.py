"""Batch Runner for All MARL Benchmark Trainings.
Trains all remaining models and seeds with resume capability (skips already-trained checkpoints).
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from scripts.train_marl_six import train_single_model, MODEL_REGISTRY, DEFAULT_SEEDS

logger = setup_logger("BatchTrainMARL")


def main():
    parser = argparse.ArgumentParser(description="Batch Train all MARL benchmarks")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument("--network", type=str, default="small")
    args = parser.parse_args()

    models = list(MODEL_REGISTRY.keys())
    seeds = DEFAULT_SEEDS

    logger.info(f"Starting batch training across {len(models)} models and {len(seeds)} seeds: {models}")

    total_runs = len(models) * len(seeds)
    run_idx = 0

    for m in models:
        for s in seeds:
            run_idx += 1
            ckpt_path = os.path.join("models", "marl_six", m, f"seed_{s}", "best_model.pt")
            if os.path.isfile(ckpt_path):
                logger.info(f"[{run_idx}/{total_runs}] Model {m} seed {s} already exists at {ckpt_path}. Skipping.")
                continue

            logger.info(f"[{run_idx}/{total_runs}] Training Model {m} (Seed {s})...")
            train_single_model(
                model_key=m,
                seed=s,
                mode=args.mode,
                network_size=args.network,
                offline_data_dir="data/offline_marl"
            )

    logger.info("All batch MARL training runs completed successfully!")


if __name__ == "__main__":
    main()
