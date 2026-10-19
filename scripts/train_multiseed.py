"""Multi-seed Scaled Training Pipeline for MaskablePPO and DQN with periodic validation tracking."""

import os
import sys
import json
import time
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.environment.evacuation_env import EvacuationEnv

logger = setup_logger("MultiSeedTrainer")

SEEDS = [42, 101, 2024]
PPO_STAGE_STEPS = 5120   # 5 stages * 5120 = 25,600 steps per seed
DQN_TIMESTEPS = 25000    # 25,000 steps per seed


def evaluate_on_validation(model, is_ppo: bool = True, max_scens: int = 50) -> Dict[str, float]:
    """Evaluates a model checkpoint on validation scenarios."""
    val_scenarios = json.load(open("data/generated/val_scenarios.json"))[:max_scens]
    env = EvacuationEnv(scenarios=val_scenarios, use_traci=False, curriculum_stage=4)

    successes = []
    heis = []
    times = []
    returns = []

    for scen in val_scenarios:
        obs, info = env.reset(options={"scenario": scen})
        done = False
        ep_ret = 0.0

        while not done:
            mask = env.action_masks()
            if is_ppo:
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)
            else:
                obs_t = torch.as_tensor(obs, device=model.device).unsqueeze(0)
                with torch.no_grad():
                    q = model.q_net(obs_t).squeeze(0).cpu().numpy()
                q[~mask] = -1e9
                action = int(np.argmax(q))

            obs, r, term, trunc, step_info = env.step(action)
            ep_ret += r
            done = term or trunc

        successes.append(step_info.get("success", False))
        heis.append(step_info.get("hazard_exposure_index", 0.0))
        times.append(step_info.get("cumulative_travel_time", 0.0))
        returns.append(ep_ret)

    env.close()
    return {
        "val_success_rate": float(np.mean(successes)) * 100.0,
        "val_mean_hei": float(np.mean(heis)),
        "val_mean_time": float(np.mean(times)),
        "val_mean_return": float(np.mean(returns))
    }


def train_ppo_seed(seed: int) -> Dict[str, Any]:
    """Trains MaskablePPO for one seed across 5 curriculum stages."""
    set_seed(seed)
    seed_dir = f"models/ppo/seed_{seed}"
    log_dir = f"logs/ppo/seed_{seed}"
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_scenarios = json.load(open("data/generated/train_scenarios.json"))

    raw_env = EvacuationEnv(scenarios=train_scenarios, use_traci=False, curriculum_stage=1)
    masked_env = ActionMasker(raw_env, lambda env: env.unwrapped.action_masks())
    env = Monitor(masked_env, filename=os.path.join(log_dir, "monitor.csv"))

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        tensorboard_log=log_dir,
        verbose=0
    )

    stage_logs = []
    t0 = time.time()
    for stg in range(1, 6):
        env.unwrapped.curriculum_stage = stg
        model.learn(total_timesteps=PPO_STAGE_STEPS, reset_num_timesteps=False)

        val_metrics = evaluate_on_validation(model, is_ppo=True, max_scens=50)
        logger.info(f"PPO Seed {seed} Stage {stg}/5 -> Val Success: {val_metrics['val_success_rate']:.1f}%, Val Return: {val_metrics['val_mean_return']:.2f}")

        stg_path = os.path.join(seed_dir, f"stage_{stg}.zip")
        model.save(stg_path)
        stage_logs.append({
            "stage": stg,
            **val_metrics
        })

    elapsed = time.time() - t0
    throughput = (PPO_STAGE_STEPS * 5) / elapsed
    logger.info(f"PPO Seed {seed} completed in {elapsed:.1f}s ({throughput:.1f} steps/s)")

    best_path = os.path.join(seed_dir, "best_model.zip")
    model.save(best_path)
    env.close()

    return {
        "model": "PPO",
        "seed": seed,
        "timesteps": PPO_STAGE_STEPS * 5,
        "elapsed_sec": round(elapsed, 1),
        "throughput_fps": round(throughput, 1),
        "final_val_success": stage_logs[-1]["val_success_rate"],
        "final_val_hei": stage_logs[-1]["val_mean_hei"],
        "final_val_return": stage_logs[-1]["val_mean_return"],
        "model_path": best_path
    }


def train_dqn_seed(seed: int) -> Dict[str, Any]:
    """Trains DQN for one seed with periodic validation evaluation."""
    set_seed(seed)
    seed_dir = f"models/dqn/seed_{seed}"
    log_dir = f"logs/dqn/seed_{seed}"
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_scenarios = json.load(open("data/generated/train_scenarios.json"))

    raw_env = EvacuationEnv(scenarios=train_scenarios, use_traci=False, curriculum_stage=4)
    env = Monitor(raw_env, filename=os.path.join(log_dir, "monitor.csv"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        seed=seed,
        device=device,
        verbose=0
    )

    t0 = time.time()
    # Train in 5 chunks of 5000 steps to log validation progress
    chunk_size = DQN_TIMESTEPS // 5
    for chunk in range(1, 6):
        model.learn(total_timesteps=chunk_size, reset_num_timesteps=False)
        val_metrics = evaluate_on_validation(model, is_ppo=False, max_scens=50)
        logger.info(f"DQN Seed {seed} Chunk {chunk}/5 ({chunk*chunk_size} steps) -> Val Success: {val_metrics['val_success_rate']:.1f}%, Val Return: {val_metrics['val_mean_return']:.2f}")

    elapsed = time.time() - t0
    throughput = DQN_TIMESTEPS / elapsed
    logger.info(f"DQN Seed {seed} completed in {elapsed:.1f}s ({throughput:.1f} steps/s)")

    best_path = os.path.join(seed_dir, "best_model.zip")
    model.save(best_path)
    env.close()

    return {
        "model": "DQN",
        "seed": seed,
        "timesteps": DQN_TIMESTEPS,
        "elapsed_sec": round(elapsed, 1),
        "throughput_fps": round(throughput, 1),
        "final_val_success": val_metrics["val_success_rate"],
        "final_val_hei": val_metrics["val_mean_hei"],
        "final_val_return": val_metrics["val_mean_return"],
        "model_path": best_path
    }


def main():
    logger.info("=" * 80)
    logger.info("PHASE C: MULTI-SEED SCALED TRAINING (PPO & DQN x 3 SEEDS)")
    logger.info("=" * 80)

    summary_records = []

    for seed in SEEDS:
        logger.info(f"\n>>> Starting PPO Seed {seed} <<<")
        ppo_res = train_ppo_seed(seed)
        summary_records.append(ppo_res)

    for seed in SEEDS:
        logger.info(f"\n>>> Starting DQN Seed {seed} <<<")
        dqn_res = train_dqn_seed(seed)
        summary_records.append(dqn_res)

    # Save summary table
    os.makedirs("results/tables", exist_ok=True)
    df = pd.DataFrame(summary_records)
    df.to_csv("results/tables/multiseed_training_summary.csv", index=False)

    print("\n" + "=" * 90)
    print("MULTI-SEED TRAINING SUMMARY ACROSS 3 SEEDS")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")

    # Update top-level best_model.zip to point to seed_42 best models for compatibility
    import shutil
    shutil.copyfile("models/ppo/seed_42/best_model.zip", "models/ppo/best_model.zip")
    shutil.copyfile("models/dqn/seed_42/best_model.zip", "models/dqn/best_model.zip")
    logger.info("Phase C multi-seed training completed successfully!")


if __name__ == "__main__":
    main()
