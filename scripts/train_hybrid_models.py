"""Automated Hybrid Models Training Pipeline
Trains:
  1. GNN-MaskablePPO
  2. RecurrentMaskablePPO (LSTM)
  3. GNN-QRDQN
across 5 seeds [42, 101, 2024, 777, 999] on the Multi-District Urban Network.
Supports --mode [quick | full] and --model [gnn_ppo | lstm_ppo | gnn_qrdqn | all].
"""

import os
import sys
import json
import time
import argparse
import yaml
import torch
from typing import List, Dict, Any

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.agents.gnn_ppo_agent import GNNPPOTrainer
from src.agents.recurrent_ppo_agent import RecurrentPPOTrainer
from src.agents.gnn_qrdqn_agent import GNNQRDQNTrainer

logger = setup_logger("TrainHybridModels")

DEFAULT_SEEDS = [42, 101, 2024, 777, 999]


def get_experiment_id(model_name: str, seed: int) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{model_name.upper()}_seed{seed}_{timestamp}"


def train_single_model_seed(
    model_type: str,
    seed: int,
    mode: str = "quick",
    config_path: str = "configs/large.yaml",
    train_scenarios_path: str = "data/generated/train_large_scenarios.json",
    val_scenarios_path: str = "data/generated/val_large_scenarios.json"
) -> Dict[str, Any]:
    """Trains a specific model architecture for one random seed."""
    exp_id = get_experiment_id(model_type, seed)
    logger.info(f"=== Starting Experiment: {exp_id} (Mode: {mode}, Seed: {seed}) ===")

    # Load scenarios
    with open(train_scenarios_path, "r", encoding="utf-8") as f:
        train_scenarios = json.load(f)
    with open(val_scenarios_path, "r", encoding="utf-8") as f:
        val_scenarios = json.load(f)

    if mode == "quick":
        train_scenarios = train_scenarios[:200]
        val_scenarios = val_scenarios[:20]
        timesteps = 5000
    else:
        timesteps = 50000

    start_time = time.time()
    checkpoint_path = ""

    if model_type == "gnn_ppo":
        trainer = GNNPPOTrainer(
            config_path=config_path,
            gnn_config_path="configs/hybrid/gnn_ppo.yaml",
            seed=seed,
            use_traci=False,
            n_envs=4 if mode == "full" else 2
        )
        checkpoint_path = trainer.train(
            train_scenarios=train_scenarios,
            val_scenarios=val_scenarios,
            total_timesteps=timesteps
        )
    elif model_type == "lstm_ppo":
        trainer = RecurrentPPOTrainer(
            config_path=config_path,
            lstm_config_path="configs/hybrid/lstm_ppo.yaml",
            seed=seed,
            use_traci=False
        )
        checkpoint_path = trainer.train(
            train_scenarios=train_scenarios,
            val_scenarios=val_scenarios,
            total_timesteps=timesteps,
            rollout_steps=128 if mode == "quick" else 256
        )
    elif model_type == "gnn_qrdqn":
        trainer = GNNQRDQNTrainer(
            config_path=config_path,
            qrdqn_config_path="configs/hybrid/gnn_qrdqn.yaml",
            seed=seed,
            use_traci=False
        )
        checkpoint_path = trainer.train(
            train_scenarios=train_scenarios,
            val_scenarios=val_scenarios,
            total_timesteps=timesteps
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    duration = time.time() - start_time
    logger.info(f"Finished {exp_id} in {duration:.1f}s. Checkpoint: {checkpoint_path}")

    summary = {
        "experiment_id": exp_id,
        "model_type": model_type,
        "seed": seed,
        "mode": mode,
        "timesteps": timesteps,
        "training_duration_s": round(duration, 2),
        "checkpoint_path": checkpoint_path,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }

    # Save summary metadata in seed directory
    seed_dir = os.path.dirname(checkpoint_path)
    with open(os.path.join(seed_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid RL Evacuation Models")
    parser.add_argument("--model", type=str, default="all", choices=["gnn_ppo", "lstm_ppo", "gnn_qrdqn", "all"],
                        help="Model architecture to train")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"],
                        help="Training mode: quick (smoke test) or full (50k steps)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Seeds to train (default: 42 101 2024 777 999)")
    parser.add_argument("--network", type=str, default="large", choices=["small", "large"],
                        help="Road network configuration to train on")
    args = parser.parse_args()

    config_path = "configs/large.yaml" if args.network == "large" else "configs/default.yaml"
    train_path = "data/generated/train_large_scenarios.json" if args.network == "large" else "data/generated/train_scenarios.json"
    val_path = "data/generated/val_large_scenarios.json" if args.network == "large" else "data/generated/val_scenarios.json"

    models_to_train = ["gnn_ppo", "lstm_ppo", "gnn_qrdqn"] if args.model == "all" else [args.model]

    print("=" * 80)
    print(f"HYBRID RL TRAINING PIPELINE: Models={models_to_train}, Mode={args.mode}, Seeds={args.seeds}")
    print("=" * 80)

    summaries = []
    for model_type in models_to_train:
        for seed in args.seeds:
            summary = train_single_model_seed(
                model_type=model_type,
                seed=seed,
                mode=args.mode,
                config_path=config_path,
                train_scenarios_path=train_path,
                val_scenarios_path=val_path
            )
            summaries.append(summary)

    out_file = f"models/hybrid/training_run_{args.mode}_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("=" * 80)
    print(f"TRAINING COMPLETE. Successfully trained {len(summaries)} model checkpoints. Log: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
