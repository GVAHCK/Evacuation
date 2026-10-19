"""Benchmark Evaluation Script running models against 100+ unseen test scenarios."""

import os
import sys
import argparse
import pandas as pd
from stable_baselines3 import PPO, DQN

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.metrics.evaluator import EvacuationEvaluator

logger = setup_logger("Evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL models & Baselines on unseen scenarios")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--test-scenarios", type=str, default="data/generated/test_scenarios.json")
    parser.add_argument("--ppo-model", type=str, default="models/ppo/best_model.zip")
    parser.add_argument("--dqn-model", type=str, default="models/dqn/best_model.zip")
    parser.add_argument("--max-scenarios", type=int, default=100)
    parser.add_argument("--use-traci", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting Benchmark Evaluation Suite on Unseen Test Scenarios...")

    ppo_model = None
    if os.path.isfile(args.ppo_model):
        logger.info(f"Loading trained PPO model from {args.ppo_model}")
        try:
            from sb3_contrib import MaskablePPO
            ppo_model = MaskablePPO.load(args.ppo_model)
        except Exception:
            ppo_model = PPO.load(args.ppo_model)
    else:
        logger.warning(f"PPO model not found at {args.ppo_model}. Will evaluate without PPO.")

    dqn_model = None
    if os.path.isfile(args.dqn_model):
        logger.info(f"Loading trained DQN model from {args.dqn_model}")
        dqn_model = DQN.load(args.dqn_model)
    else:
        logger.warning(f"DQN model not found at {args.dqn_model}. Will evaluate without DQN.")

    evaluator = EvacuationEvaluator(
        config_path=args.config,
        test_scenarios_path=args.test_scenarios,
        use_traci=args.use_traci
    )

    summary_df = evaluator.run_benchmark_suite(
        ppo_model=ppo_model,
        dqn_model=dqn_model,
        max_scenarios=args.max_scenarios
    )

    print("\n" + "=" * 80)
    print("           BENCHMARK EVALUATION RESULTS ACROSS UNSEEN SCENARIOS")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
