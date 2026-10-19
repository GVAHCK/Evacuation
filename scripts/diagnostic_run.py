"""Diagnostic experiment script running 20 scenarios with full trajectory,

action mapping, and reward breakdown inspection.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment.evacuation_env import EvacuationEnv
from src.utils.logger import setup_logger

logger = setup_logger("DiagnosticRun")


def run_diagnostic(n_episodes: int = 20):
    scen_file = "data/generated/test_scenarios.json"
    with open(scen_file, "r", encoding="utf-8") as f:
        scenarios = json.load(f)[:n_episodes]

    ppo_path = "models/ppo/best_model.zip"
    dqn_path = "models/dqn/best_model.zip"

    ppo_model = PPO.load(ppo_path) if os.path.isfile(ppo_path) else None
    dqn_model = DQN.load(dqn_path) if os.path.isfile(dqn_path) else None

    env = EvacuationEnv(scenarios=scenarios, use_traci=False, curriculum_stage=5)

    print("\n" + "=" * 100)
    print(f"DIAGNOSTIC RUN: INSPECTING {n_episodes} EPISODES (PPO)")
    print("=" * 100)

    results = []

    for ep_idx, scen in enumerate(scenarios):
        obs, info = env.reset(options={"scenario": scen})
        done = False
        step = 0
        ep_reward = 0.0

        print(f"\n--- Episode {ep_idx+1:02d}: Scenario {scen['scenario_id']} ---")
        print(f"Origin: {scen['evac_origin']} -> Destination: {scen['evac_destination']}")
        print(f"Hazard: {scen['hazard_type']} at node {scen['hazard_origin']}, Traffic: {scen['traffic_level']}")

        trajectory_actions = []
        trajectory_edges = []
        trajectory_nodes = [env.current_node]

        while not done:
            curr_node = env.current_node
            successors = list(env.graph.successors(curr_node))
            mask = env.action_masks()

            if ppo_model is not None:
                action, _ = ppo_model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = 0

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step += 1

            next_node = env.current_node
            edge_str = f"{curr_node}->{next_node}" if curr_node != next_node else f"{curr_node}(STUCK)"
            trajectory_actions.append(action)
            trajectory_edges.append(edge_str)
            trajectory_nodes.append(next_node)

            if step <= 5 or done:
                print(
                    f"  Step {step:02d}: Node={curr_node}, Successors={successors}, Action={action}, "
                    f"NextNode={next_node}, ValidMask={mask.tolist()[:len(successors)+1]}, "
                    f"R_prog={step_info.get('progress_reward', 0):.2f}, "
                    f"P_close={step_info.get('closure_penalty', 0):.2f}, "
                    f"P_dead={step_info.get('dead_end_penalty', 0):.2f}, "
                    f"Reward={reward:.2f}"
                )

        status = "SUCCESS" if step_info.get("success", False) else ("TIMEOUT" if truncated else "DEAD_END")
        print(f"Episode Outcome: {status} in {step} steps | Total Return: {ep_reward:.2f} | Unique Nodes: {len(set(trajectory_nodes))}")
        print(f"Action Sequence: {trajectory_actions[:15]}...")

        results.append({
            "scenario_id": scen["scenario_id"],
            "origin": scen["evac_origin"],
            "destination": scen["evac_destination"],
            "steps": step,
            "status": status,
            "total_reward": ep_reward,
            "unique_visited": len(set(trajectory_nodes)),
            "action_counts": {a: trajectory_actions.count(a) for a in set(trajectory_actions)}
        })

    env.close()

    print("\n" + "=" * 100)
    print("DIAGNOSTIC SUMMARY OVER 20 EPISODES")
    print("=" * 100)
    df_res = pd.DataFrame(results)
    print(df_res[["scenario_id", "origin", "destination", "steps", "status", "total_reward", "unique_visited", "action_counts"]].to_string())


if __name__ == "__main__":
    run_diagnostic(20)
