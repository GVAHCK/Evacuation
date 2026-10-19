"""PPO Curriculum Training Script."""

import os
import sys
import argparse

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.agents.ppo_agent import PPOCurriculumTrainer

logger = setup_logger("TrainPPO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Evacuation Digital Twin")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to default config")
    parser.add_argument("--ppo-config", type=str, default="configs/ppo.yaml", help="Path to PPO config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stage-steps", type=int, default=5000, help="Timesteps per curriculum stage")
    parser.add_argument("--use-traci", action="store_true", help="Enable live SUMO TraCI synchronization")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Starting PPO Curriculum Training with seed={args.seed}, stage_steps={args.stage_steps}...")

    trainer = PPOCurriculumTrainer(
        config_path=args.config,
        ppo_config_path=args.ppo_config,
        seed=args.seed,
        use_traci=args.use_traci
    )
    summary = trainer.train_curriculum(stage_timesteps=args.stage_steps)
    logger.info("PPO Curriculum Training completed successfully!")
    for stg, data in summary.items():
        logger.info(f"  {stg}: {data['name']} -> Success Rate: {data['recent_success_rate']*100:.1f}%, HEI: {data['mean_hazard_exposure']:.2f}")


if __name__ == "__main__":
    main()
