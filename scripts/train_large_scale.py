"""Large-Scale Multi-Seed Training Pipeline (5 Seeds PPO + 5 Seeds DQN)
on the Multi-District Urban Road Network (336 nodes, 736 edges, 5000 scenarios).
Utilizes vectorized environment workers to saturate GPU training throughput.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.environment.evacuation_env import EvacuationEnv

logger = setup_logger("LargeScaleTrainer")

SEEDS = [42, 101, 2024, 777, 999]
CONFIG_PATH = "configs/large.yaml"
TRAIN_SCENARIOS_PATH = "data/generated/train_large_scenarios.json"
VAL_SCENARIOS_PATH = "data/generated/val_large_scenarios.json"

PPO_STAGE_STEPS = 10000  # 5 stages * 10,000 = 50,000 steps per seed
DQN_TIMESTEPS = 50000   # 50,000 steps per seed
N_VEC_ENVS = 4


def evaluate_on_validation(model, is_ppo: bool = True, max_scens: int = 40) -> Dict[str, float]:
    """Evaluates a model checkpoint on validation scenarios from the large network."""
    val_scenarios = json.load(open(VAL_SCENARIOS_PATH))[:max_scens]
    env = EvacuationEnv(config_path=CONFIG_PATH, scenarios=val_scenarios, use_traci=False, curriculum_stage=4)

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


def train_ppo_seed(seed: int, train_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Trains MaskablePPO for one seed across 5 curriculum stages using vectorized envs."""
    set_seed(seed)
    seed_dir = f"models/large/ppo/seed_{seed}"
    log_dir = f"logs/large/ppo/seed_{seed}"
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    def make_single_env(curriculum_stg: int = 1):
        def _thunk():
            raw_env = EvacuationEnv(config_path=CONFIG_PATH, scenarios=train_scenarios, use_traci=False, curriculum_stage=curriculum_stg)
            masked_env = ActionMasker(raw_env, lambda env: env.unwrapped.action_masks())
            return Monitor(masked_env)
        return _thunk

    vec_env = DummyVecEnv([make_single_env(1) for _ in range(N_VEC_ENVS)])

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        verbose=0
    )

    t0 = time.time()
    stage_logs = []

    for stg in range(1, 6):
        # Update curriculum stage across vector envs
        for sub_env in vec_env.envs:
            sub_env.unwrapped.curriculum_stage = stg

        model.learn(total_timesteps=PPO_STAGE_STEPS, reset_num_timesteps=False)
        val_metrics = evaluate_on_validation(model, is_ppo=True, max_scens=40)
        logger.info(f"PPO Seed {seed} Stage {stg}/5 -> Val Success: {val_metrics['val_success_rate']:.1f}%, Val Return: {val_metrics['val_mean_return']:.2f}")

        stg_path = os.path.join(seed_dir, f"stage_{stg}.zip")
        model.save(stg_path)
        stage_logs.append({"stage": stg, **val_metrics})

    elapsed = time.time() - t0
    total_steps = PPO_STAGE_STEPS * 5
    throughput = total_steps / elapsed
    logger.info(f"PPO Seed {seed} finished in {elapsed:.1f}s ({throughput:.1f} steps/s)")

    best_path = os.path.join(seed_dir, "best_model.zip")
    model.save(best_path)
    vec_env.close()

    return {
        "model": "PPO",
        "seed": seed,
        "timesteps": total_steps,
        "elapsed_sec": round(elapsed, 1),
        "throughput_fps": round(throughput, 1),
        "final_val_success": stage_logs[-1]["val_success_rate"],
        "final_val_hei": round(stage_logs[-1]["val_mean_hei"], 3),
        "final_val_return": round(stage_logs[-1]["val_mean_return"], 2),
        "model_path": best_path
    }


def train_dqn_seed(seed: int, train_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Trains DQN for one seed on the scaled network."""
    set_seed(seed)
    seed_dir = f"models/large/dqn/seed_{seed}"
    log_dir = f"logs/large/dqn/seed_{seed}"
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    raw_env = EvacuationEnv(config_path=CONFIG_PATH, scenarios=train_scenarios, use_traci=False, curriculum_stage=4)
    env = Monitor(raw_env, filename=os.path.join(log_dir, "monitor.csv"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=2000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        seed=seed,
        device=device,
        verbose=0
    )

    t0 = time.time()
    chunk_size = DQN_TIMESTEPS // 5
    val_metrics = {}

    for chunk in range(1, 6):
        model.learn(total_timesteps=chunk_size, reset_num_timesteps=False)
        val_metrics = evaluate_on_validation(model, is_ppo=False, max_scens=40)
        logger.info(f"DQN Seed {seed} Chunk {chunk}/5 ({chunk*chunk_size} steps) -> Val Success: {val_metrics['val_success_rate']:.1f}%, Val Return: {val_metrics['val_mean_return']:.2f}")

    elapsed = time.time() - t0
    throughput = DQN_TIMESTEPS / elapsed
    logger.info(f"DQN Seed {seed} finished in {elapsed:.1f}s ({throughput:.1f} steps/s)")

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
        "final_val_hei": round(val_metrics["val_mean_hei"], 3),
        "final_val_return": round(val_metrics["val_mean_return"], 2),
        "model_path": best_path
    }


def main():
    logger.info("=" * 80)
    logger.info("PHASE 4: LARGE-SCALE MULTI-SEED TRAINING (5 SEEDS PPO + 5 SEEDS DQN)")
    logger.info("=" * 80)

    train_scenarios = json.load(open(TRAIN_SCENARIOS_PATH))
    summary_records = []

    # 1. Train PPO across 5 seeds
    for seed in SEEDS:
        logger.info(f"\n>>> Starting MaskablePPO Large-Scale Seed {seed} <<<")
        res = train_ppo_seed(seed, train_scenarios)
        summary_records.append(res)

    # 2. Train DQN across 5 seeds
    for seed in SEEDS:
        logger.info(f"\n>>> Starting Masked DQN Large-Scale Seed {seed} <<<")
        res = train_dqn_seed(seed, train_scenarios)
        summary_records.append(res)

    os.makedirs("results/tables", exist_ok=True)
    out_csv = "results/tables/multiseed_large_training_summary.csv"
    df = pd.DataFrame(summary_records)
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("LARGE-SCALE MULTI-SEED TRAINING SUMMARY (10 RUNS, 500,000 TOTAL STEPS)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")
    logger.info(f"Large-scale multi-seed training complete! Summary saved to {out_csv}")


if __name__ == "__main__":
    main()
