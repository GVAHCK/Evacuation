"""Ablation Studies Script evaluating State Ablations and Reward Decomposition Ablations."""

import os
import sys
import copy
import yaml
import json
import argparse
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.environment.evacuation_env import EvacuationEnv
from src.baselines.dijkstra import DijkstraBaseline
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

logger = setup_logger("Ablations")


def run_state_ablations(ppo_model_path: str, test_scenarios_path: str, n_scens: int = 50) -> pd.DataFrame:
    """Evaluates the effect of masking state features (congestion, hazard, road closures)."""
    logger.info("=== RUNNING STATE ABLATION STUDIES ===")
    with open(test_scenarios_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)[:n_scens]

    try:
        ppo_model = MaskablePPO.load(ppo_model_path) if os.path.isfile(ppo_model_path) else None
    except Exception:
        ppo_model = PPO.load(ppo_model_path) if os.path.isfile(ppo_model_path) else None

    conditions = [
        ("Full Proposed State", {}),
        ("Ablation 1 (No Congestion Info)", {"no_congestion": True}),
        ("Ablation 2 (No Hazard Info)", {"no_hazard": True}),
        ("Ablation 3 (No Closure Info)", {"no_closure": True}),
    ]

    records = []
    for cond_name, flags in conditions:
        env = EvacuationEnv(scenarios=scenarios, use_traci=False, ablation_flags=flags)
        successes = []
        heis = []
        times = []
        closures = []

        for scen in scenarios:
            obs, info = env.reset(options={"scenario": scen})
            done = False
            while not done:
                if ppo_model is not None:
                    try:
                        mask = env.action_masks()
                        action, _ = ppo_model.predict(obs, action_masks=mask, deterministic=True)
                    except TypeError:
                        action, _ = ppo_model.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    action = 0
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated

            successes.append(step_info.get("success", False))
            heis.append(step_info.get("hazard_exposure_index", 0.0))
            times.append(step_info.get("cumulative_travel_time", 0.0))
            closures.append(step_info.get("closure_encounters", 0))

        env.close()

        records.append({
            "Ablation_Condition": cond_name,
            "Success_Rate": round(float(np.mean(successes)) * 100, 2),
            "Mean_HEI": round(float(np.mean(heis)), 2),
            "Mean_Travel_Time": round(float(np.mean(times)), 2),
            "Mean_Closure_Encounters": round(float(np.mean(closures)), 2)
        })

    df = pd.DataFrame(records)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/state_ablation.csv", index=False)
    logger.info("State ablation results saved to results/tables/state_ablation.csv")
    return df


def run_reward_ablations(test_scenarios_path: str, n_scens: int = 50) -> pd.DataFrame:
    """Evaluates different reward formulations: Distance-focused, Safety-focused,

    Congestion-focused, and Balanced.
    """
    logger.info("=== RUNNING REWARD ABLATION STUDIES ===")
    with open(test_scenarios_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)[:n_scens]

    reward_variants = [
        ("Balanced (Proposed)", {"progress_weight": 2.0, "hazard_penalty_weight": 4.0, "congestion_penalty_weight": 1.5, "arrival_bonus": 100.0}),
        ("Distance-Focused", {"progress_weight": 6.0, "hazard_penalty_weight": 0.5, "congestion_penalty_weight": 0.2, "arrival_bonus": 50.0}),
        ("Safety-Focused", {"progress_weight": 1.0, "hazard_penalty_weight": 10.0, "congestion_penalty_weight": 1.0, "arrival_bonus": 100.0}),
        ("Congestion-Focused", {"progress_weight": 1.0, "hazard_penalty_weight": 1.0, "congestion_penalty_weight": 6.0, "arrival_bonus": 50.0})
    ]

    records = []
    base_cfg = yaml.safe_load(open("configs/default.yaml"))

    for variant_name, weights in reward_variants:
        cfg = copy.deepcopy(base_cfg)
        cfg["environment"]["reward"].update(weights)

        # Write temp config or pass into env
        temp_cfg_path = f"configs/temp_{variant_name.replace(' ', '_').lower()}.yaml"
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        env = EvacuationEnv(config_path=temp_cfg_path, scenarios=scenarios, use_traci=False)
        dijkstra = DijkstraBaseline()

        successes = []
        heis = []
        times = []
        rewards = []

        for scen in scenarios:
            res = dijkstra.run_episode(env, scenario=scen)
            successes.append(res.get("success", False))
            heis.append(res.get("hazard_exposure_index", 0.0))
            times.append(res.get("cumulative_travel_time", 0.0))
            rewards.append(res.get("total_reward", 0.0))

        env.close()
        if os.path.isfile(temp_cfg_path):
            os.remove(temp_cfg_path)

        records.append({
            "Reward_Formulation": variant_name,
            "Success_Rate": round(float(np.mean(successes)) * 100, 2),
            "Mean_HEI": round(float(np.mean(heis)), 2),
            "Mean_Travel_Time": round(float(np.mean(times)), 2),
            "Mean_Episode_Reward": round(float(np.mean(rewards)), 2)
        })

    df = pd.DataFrame(records)
    df.to_csv("results/tables/reward_ablation.csv", index=False)
    logger.info("Reward ablation results saved to results/tables/reward_ablation.csv")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-model", type=str, default="models/ppo/best_model.zip")
    parser.add_argument("--test-scenarios", type=str, default="data/generated/test_scenarios.json")
    parser.add_argument("--n-scenarios", type=int, default=50)
    args = parser.parse_args()

    df_state = run_state_ablations(args.ppo_model, args.test_scenarios, n_scens=args.n_scenarios)
    print("\n--- STATE ABLATION RESULTS ---")
    print(df_state.to_string(index=False))

    df_reward = run_reward_ablations(args.test_scenarios, n_scens=args.n_scenarios)
    print("\n--- REWARD ABLATION RESULTS ---")
    print(df_reward.to_string(index=False))


if __name__ == "__main__":
    main()
