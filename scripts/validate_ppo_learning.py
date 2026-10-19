"""Validation experiment script to verify that MaskablePPO is actively learning

and exploring multiple actions before running long training.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.monitor import Monitor

from src.utils.logger import setup_logger
from src.environment.evacuation_env import EvacuationEnv
from src.network.scenario_generator import ScenarioGenerator
from src.network.osm_loader import OSMLoader

logger = setup_logger("ValidatePPOLearning")


def validate_learning(timesteps: int = 5000):
    logger.info("=== VALIDATION EXPERIMENT: VERIFYING PPO LEARNING & EXPLORATION ===")

    graph = OSMLoader.load_graph_json()
    import json
    scenarios = json.load(open("data/generated/train_scenarios.json"))

    raw_env = EvacuationEnv(scenarios=scenarios, use_traci=False, curriculum_stage=1)
    masked_env = ActionMasker(raw_env, lambda e: e.unwrapped.action_masks())
    env = Monitor(masked_env)

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.02,
        verbose=1
    )

    logger.info(f"Training MaskablePPO for {timesteps} validation steps...")
    model.learn(total_timesteps=timesteps)

    # Test policy on 20 unseen test scenarios
    test_scenarios = json.load(open("data/generated/test_scenarios.json"))[:20]
    eval_env = EvacuationEnv(scenarios=test_scenarios, use_traci=False, curriculum_stage=2)

    logger.info("Evaluating trained validation policy on 20 test scenarios with full trajectory logging...")
    successes = []
    actions_chosen = []
    steps_list = []
    returns = []

    print("\n" + "=" * 100)
    print("DETAILED DECISION TRAJECTORIES ACROSS 20 UNSEEN SCENARIOS")
    print("=" * 100)

    for ep_idx, scen in enumerate(test_scenarios):
        obs, info = eval_env.reset(options={"scenario": scen})
        origin = scen.get("evac_origin", scen.get("origin_node"))
        dest = scen.get("evac_destination", scen.get("destination_node"))
        scen_id = scen.get("scenario_id", scen.get("id", f"test_{ep_idx:04d}"))

        done = False
        ep_ret = 0.0
        ep_acts = []
        ep_edges = []
        tot_prog = 0.0
        tot_arr = 0.0
        tot_haz = 0.0
        tot_delay = 0.0
        max_haz = 0.0

        while not done:
            mask = eval_env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
            ep_acts.append(action)
            actions_chosen.append(action)

            curr = eval_env.current_node
            succs = eval_env._get_ordered_successors()
            next_node = succs[action] if action < len(succs) else "NONE"
            ep_edges.append(f"{curr}->{next_node}")

            obs, r, term, trunc, step_info = eval_env.step(action)
            ep_ret += r
            done = term or trunc

            tot_prog += step_info.get("progress_reward", 0.0)
            tot_arr += step_info.get("arrival_bonus", 0.0)
            tot_haz += step_info.get("hazard_penalty", 0.0)
            tot_delay += step_info.get("delay_penalty", 0.0)

            haz = step_info.get("hazard_exposure_index", 0.0)
            if haz > max_haz:
                max_haz = haz

        succ = step_info.get("success", False)
        timeout = step_info.get("timeout", False)
        outcome = "SUCCESS" if succ else ("TIMEOUT" if timeout else "DEAD_END")

        successes.append(succ)
        steps_list.append(len(ep_acts))
        returns.append(ep_ret)

        print(f"[{ep_idx+1:02d}/20] Scenario: {scen_id} | Origin: {origin} -> Dest: {dest} | Outcome: {outcome}")
        print(f"       Actions: {ep_acts[:12]}{'...' if len(ep_acts) > 12 else ''} (Total: {len(ep_acts)})")
        print(f"       Edges: {ep_edges[:5]}{'...' if len(ep_edges) > 5 else ''}")
        print(f"       Max Hazard: {max_haz:.3f} | Return: {ep_ret:.2f}")
        print(f"       Rewards: Progress={tot_prog:.2f}, Arrival={tot_arr:.1f}, HazPenalty={tot_haz:.2f}, Delay={tot_delay:.2f}\n")

    eval_env.close()

    action_dist = {a: actions_chosen.count(a) for a in sorted(set(actions_chosen))}
    success_rate = float(np.mean(successes)) * 100

    print("=" * 100)
    print("VALIDATION EXPERIMENT SUMMARY")
    print("=" * 100)
    print(f"Success Rate on 20 test scenarios: {success_rate:.1f}%")
    print(f"Mean Episode Return: {np.mean(returns):.2f}")
    print(f"Mean Steps to Evacuation: {np.mean(steps_list):.1f}")
    print(f"Total Action Distribution across all steps: {action_dist}")
    print("=" * 100 + "\n")

    assert success_rate >= 50.0, f"FAILURE: Success rate {success_rate:.1f}% is below 50%!"
    logger.info("SUCCESS: Validation experiment passed! PPO is genuinely learning, exploring, and routing!")


if __name__ == "__main__":
    validate_learning(5000)
