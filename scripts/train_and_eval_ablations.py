"""Rigorous Ablation Framework: Retraining PPO from scratch for each State and Reward ablation condition

and evaluating independently on the 100 unseen test disaster scenarios.
"""

import os
import sys
import json
import yaml
import copy
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.monitor import Monitor

from src.utils.logger import setup_logger
from src.environment.evacuation_env import EvacuationEnv

logger = setup_logger("AblationSuite")

ABLATION_MODELS_DIR = "models/ablations"
os.makedirs(ABLATION_MODELS_DIR, exist_ok=True)
os.makedirs("results/tables", exist_ok=True)


def train_ablation_model(
    model_name: str,
    ablation_flags: Dict[str, bool] = None,
    reward_weights: Dict[str, float] = None,
    timesteps: int = 6144,
    seed: int = 42
) -> str:
    """Trains a MaskablePPO model from scratch under a specific ablation condition."""
    ablation_flags = ablation_flags or {}
    reward_weights = reward_weights or {}

    train_scenarios = json.load(open("data/generated/train_scenarios.json"))

    # Prepare custom config if reward weights are provided
    config_path = "configs/default.yaml"
    temp_cfg_path = None
    if reward_weights:
        base_cfg = yaml.safe_load(open(config_path))
        base_cfg["environment"]["reward"].update(reward_weights)
        temp_cfg_path = os.path.join(ABLATION_MODELS_DIR, f"temp_{model_name}.yaml")
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(base_cfg, f)
        config_path = temp_cfg_path

    # Build environment with specific ablation flags and reward
    raw_env = EvacuationEnv(
        config_path=config_path,
        scenarios=train_scenarios,
        use_traci=False,
        curriculum_stage=3,  # Train with active hazards and congestion to expose feature dependence
        ablation_flags=ablation_flags
    )
    masked_env = ActionMasker(raw_env, lambda env: env.unwrapped.action_masks())
    env = Monitor(masked_env)

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        seed=seed,
        verbose=0
    )

    logger.info(f"Training ablation model [{model_name}] for {timesteps} steps...")
    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    t_elapsed = time.time() - t0
    logger.info(f"Trained [{model_name}] in {t_elapsed:.1f}s")

    model_save_path = os.path.join(ABLATION_MODELS_DIR, f"{model_name}.zip")
    model.save(model_save_path)

    env.close()
    if temp_cfg_path and os.path.isfile(temp_cfg_path):
        os.remove(temp_cfg_path)

    return model_save_path


def evaluate_ablation_model(
    model_path: str,
    ablation_flags: Dict[str, bool] = None,
    reward_weights: Dict[str, float] = None,
    test_scenarios_path: str = "data/generated/test_scenarios.json",
    max_scenarios: int = 100
) -> Dict[str, float]:
    """Evaluates a trained ablation model on the unseen test scenarios."""
    ablation_flags = ablation_flags or {}
    reward_weights = reward_weights or {}
    test_scenarios = json.load(open(test_scenarios_path))[:max_scenarios]

    config_path = "configs/default.yaml"
    temp_cfg_path = None
    if reward_weights:
        base_cfg = yaml.safe_load(open(config_path))
        base_cfg["environment"]["reward"].update(reward_weights)
        temp_cfg_path = os.path.join(ABLATION_MODELS_DIR, f"eval_temp.yaml")
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(base_cfg, f)
        config_path = temp_cfg_path

    env = EvacuationEnv(
        config_path=config_path,
        scenarios=test_scenarios,
        use_traci=False,
        curriculum_stage=4,  # Evaluate under realistic disaster spread and peak traffic
        ablation_flags=ablation_flags
    )

    model = MaskablePPO.load(model_path)

    successes = []
    heis = []
    times = []
    closures = []
    rewards = []

    for scen in test_scenarios:
        obs, info = env.reset(options={"scenario": scen})
        done = False
        ep_rew = 0.0

        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, term, trunc, step_info = env.step(int(action))
            ep_rew += r
            done = term or trunc

        successes.append(step_info.get("success", False))
        heis.append(step_info.get("hazard_exposure_index", 0.0))
        times.append(step_info.get("cumulative_travel_time", 0.0))
        closures.append(step_info.get("closure_encounters", 0))
        rewards.append(ep_rew)

    env.close()
    if temp_cfg_path and os.path.isfile(temp_cfg_path):
        os.remove(temp_cfg_path)

    return {
        "success_rate": float(np.mean(successes)) * 100.0,
        "mean_hei": float(np.mean(heis)),
        "mean_travel_time": float(np.mean(times)),
        "mean_closure_encounters": float(np.mean(closures)),
        "mean_reward": float(np.mean(rewards))
    }


def run_state_ablations():
    """Retrains and evaluates PPO models under 4 State Space ablation conditions."""
    logger.info("=" * 80)
    logger.info("PHASE A.1: STATE SPACE FEATURE ABLATIONS (TRAINED FROM SCRATCH)")
    logger.info("=" * 80)

    state_configs = [
        ("Full Proposed State", {}, "state_full"),
        ("Ablation 1 (No Congestion Info)", {"no_congestion": True}, "state_no_congestion"),
        ("Ablation 2 (No Hazard Info)", {"no_hazard": True}, "state_no_hazard"),
        ("Ablation 3 (No Closure Info)", {"no_closure": True}, "state_no_closure"),
    ]

    records = []
    for cond_name, flags, tag in state_configs:
        logger.info(f"--- Training state ablation: {cond_name} ---")
        model_path = train_ablation_model(tag, ablation_flags=flags, timesteps=6144)
        metrics = evaluate_ablation_model(model_path, ablation_flags=flags, max_scenarios=100)

        records.append({
            "Ablation_Condition": cond_name,
            "Success_Rate": round(metrics["success_rate"], 2),
            "Mean_HEI": round(metrics["mean_hei"], 2),
            "Mean_Travel_Time": round(metrics["mean_travel_time"], 2),
            "Mean_Closure_Encounters": round(metrics["mean_closure_encounters"], 2),
            "Mean_Episode_Reward": round(metrics["mean_reward"], 2)
        })

    df = pd.DataFrame(records)
    df.to_csv("results/tables/state_ablation.csv", index=False)
    logger.info("Saved results/tables/state_ablation.csv")
    print("\n" + "=" * 80)
    print("STATE ABLATION RESULTS (EVALUATED ON 100 UNSEEN SCENARIOS)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")
    return df


def run_reward_ablations():
    """Retrains and evaluates PPO models under 4 Reward Decomposition ablation weightings."""
    logger.info("=" * 80)
    logger.info("PHASE A.2: REWARD DECOMPOSITION ABLATIONS (TRAINED FROM SCRATCH)")
    logger.info("=" * 80)

    reward_configs = [
        ("Balanced (Proposed)", {"progress_weight": 4.0, "hazard_penalty_weight": 4.0, "congestion_penalty_weight": 1.5, "arrival_bonus": 100.0}, "rew_balanced"),
        ("Distance-Focused", {"progress_weight": 8.0, "hazard_penalty_weight": 0.5, "congestion_penalty_weight": 0.2, "arrival_bonus": 50.0}, "rew_distance_focused"),
        ("Safety-Focused", {"progress_weight": 1.0, "hazard_penalty_weight": 12.0, "congestion_penalty_weight": 1.0, "arrival_bonus": 100.0}, "rew_safety_focused"),
        ("Congestion-Focused", {"progress_weight": 1.0, "hazard_penalty_weight": 1.0, "congestion_penalty_weight": 8.0, "arrival_bonus": 50.0}, "rew_congestion_focused"),
    ]

    records = []
    for cond_name, weights, tag in reward_configs:
        logger.info(f"--- Training reward ablation: {cond_name} ---")
        model_path = train_ablation_model(tag, reward_weights=weights, timesteps=6144)
        metrics = evaluate_ablation_model(model_path, reward_weights=weights, max_scenarios=100)

        records.append({
            "Reward_Formulation": cond_name,
            "Success_Rate": round(metrics["success_rate"], 2),
            "Mean_HEI": round(metrics["mean_hei"], 2),
            "Mean_Travel_Time": round(metrics["mean_travel_time"], 2),
            "Mean_Episode_Reward": round(metrics["mean_reward"], 2)
        })

    df = pd.DataFrame(records)
    df.to_csv("results/tables/reward_ablation.csv", index=False)
    logger.info("Saved results/tables/reward_ablation.csv")
    print("\n" + "=" * 80)
    print("REWARD ABLATION RESULTS (EVALUATED ON 100 UNSEEN SCENARIOS)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")
    return df


if __name__ == "__main__":
    run_state_ablations()
    run_reward_ablations()
