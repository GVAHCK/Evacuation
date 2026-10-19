"""DQN Training Script."""

import os
import sys
import argparse

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.agents.dqn_agent import DQNTrainer

logger = setup_logger("TrainDQN")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on Evacuation Digital Twin")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to default config")
    parser.add_argument("--dqn-config", type=str, default="configs/dqn.yaml", help="Path to DQN config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=25000, help="Total training timesteps")
    parser.add_argument("--use-traci", action="store_true", help="Enable live SUMO TraCI synchronization")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Starting DQN Training with seed={args.seed}, timesteps={args.timesteps}...")

    trainer = DQNTrainer(
        config_path=args.config,
        dqn_config_path=args.dqn_config,
        seed=args.seed,
        use_traci=args.use_traci
    )
    model_path = trainer.train(total_timesteps=args.timesteps)
    logger.info(f"DQN Training completed successfully! Model at {model_path}")


if __name__ == "__main__":
    main()
