"""Verification script for Multi-Seed Rigor:
1. Model file checksums and parameter weight differences.
2. Raw per-seed metrics dump for PPO (42, 101, 2024) and DQN (42, 101, 2024).
3. Step-by-step action sequence comparison for 5 test scenarios.
4. Investigation of deterministic argmax convergence vs graph topology.
"""

import os
import sys
import json
import hashlib
import torch
import numpy as np
import pandas as pd

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN
from src.environment.evacuation_env import EvacuationEnv


def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("STEP 1: CHECKSUM & MODEL WEIGHT COMPARISON")
    print("=" * 80)

    ppo_paths = {
        "PPO_seed_42": "models/ppo/seed_42/best_model.zip",
        "PPO_seed_101": "models/ppo/seed_101/best_model.zip",
        "PPO_seed_2024": "models/ppo/seed_2024/best_model.zip",
    }
    dqn_paths = {
        "DQN_seed_42": "models/dqn/seed_42/best_model.zip",
        "DQN_seed_101": "models/dqn/seed_101/best_model.zip",
        "DQN_seed_2024": "models/dqn/seed_2024/best_model.zip",
    }

    print("\n--- File Checksums (SHA-256) ---")
    for name, p in {**ppo_paths, **dqn_paths}.items():
        if os.path.exists(p):
            sha = get_file_sha256(p)
            sz = os.path.getsize(p)
            print(f"{name:15s} | Size: {sz:7d} bytes | SHA256: {sha}")
        else:
            print(f"{name:15s} | NOT FOUND: {p}")

    print("\n--- Model Parameter Weight Differences (L2 Norm of Delta) ---")
    ppo_models = {k: MaskablePPO.load(v) for k, v in ppo_paths.items()}
    dqn_models = {k: DQN.load(v) for k, v in dqn_paths.items()}

    # Compare PPO weights
    p42_params = dict(ppo_models["PPO_seed_42"].policy.named_parameters())
    p101_params = dict(ppo_models["PPO_seed_101"].policy.named_parameters())
    p2024_params = dict(ppo_models["PPO_seed_2024"].policy.named_parameters())

    diff_42_101 = sum(torch.norm(p42_params[k] - p101_params[k]).item() for k in p42_params)
    diff_42_2024 = sum(torch.norm(p42_params[k] - p2024_params[k]).item() for k in p42_params)
    diff_101_2024 = sum(torch.norm(p101_params[k] - p2024_params[k]).item() for k in p101_params)

    print(f"PPO Weight Diff (Seed 42 vs 101):   L2 Delta = {diff_42_101:.4f}")
    print(f"PPO Weight Diff (Seed 42 vs 2024):  L2 Delta = {diff_42_2024:.4f}")
    print(f"PPO Weight Diff (Seed 101 vs 2024): L2 Delta = {diff_101_2024:.4f}")

    # Compare DQN weights
    d42_params = dict(dqn_models["DQN_seed_42"].q_net.named_parameters())
    d101_params = dict(dqn_models["DQN_seed_101"].q_net.named_parameters())
    d2024_params = dict(dqn_models["DQN_seed_2024"].q_net.named_parameters())

    d_diff_42_101 = sum(torch.norm(d42_params[k] - d101_params[k]).item() for k in d42_params)
    d_diff_42_2024 = sum(torch.norm(d42_params[k] - d2024_params[k]).item() for k in d42_params)
    d_diff_101_2024 = sum(torch.norm(d101_params[k] - d2024_params[k]).item() for k in d101_params)

    print(f"DQN Weight Diff (Seed 42 vs 101):   L2 Delta = {d_diff_42_101:.4f}")
    print(f"DQN Weight Diff (Seed 42 vs 2024):  L2 Delta = {d_diff_42_2024:.4f}")
    print(f"DQN Weight Diff (Seed 101 vs 2024): L2 Delta = {d_diff_101_2024:.4f}")

    print("\n" + "=" * 80)
    print("STEP 2: RAW PER-SEED METRICS BREAKDOWN (FROM SAVED RAW EVAL RECORDS)")
    print("=" * 80)

    if os.path.exists("results/tables/raw_multiseed_records.csv"):
        df_raw = pd.read_csv("results/tables/raw_multiseed_records.csv")
        grouped = df_raw.groupby(["model", "seed"]).agg(
            success_rate=("success", lambda x: np.mean(x) * 100.0),
            mean_evac_time=("evacuation_time", "mean"),
            std_evac_time=("evacuation_time", "std"),
            mean_hei=("hei", "mean"),
            std_hei=("hei", "std"),
            mean_distance=("distance", "mean"),
            timeout_rate=("timeout", lambda x: np.mean(x) * 100.0),
            mean_return=("return", "mean"),
            count=("scenario_id", "count")
        ).reset_index()
        print(grouped.to_string(index=False))

    print("\n" + "=" * 80)
    print("STEP 3: ACTION SEQUENCE COMPARISON ACROSS 5 TEST SCENARIOS")
    print("=" * 80)

    test_scenarios = json.load(open("data/generated/test_scenarios.json"))[:5]
    env = EvacuationEnv(scenarios=test_scenarios, use_traci=False, curriculum_stage=4)

    models_to_test = {
        "PPO_42": (ppo_models["PPO_seed_42"], "ppo"),
        "PPO_101": (ppo_models["PPO_seed_101"], "ppo"),
        "PPO_2024": (ppo_models["PPO_seed_2024"], "ppo"),
        "DQN_42": (dqn_models["DQN_seed_42"], "dqn"),
        "DQN_101": (dqn_models["DQN_seed_101"], "dqn"),
        "DQN_2024": (dqn_models["DQN_seed_2024"], "dqn"),
    }

    scenario_action_logs = {}

    for i, scen in enumerate(test_scenarios):
        scen_id = scen.get("scenario_id", f"scen_{i}")
        origin = scen.get("origin_edge")
        dest = scen.get("destination_edge")
        htype = scen.get("hazard_type")
        print(f"\n--- Scenario {i+1}: ID={scen_id} | Origin={origin} -> Dest={dest} | Hazard={htype} ---")

        scenario_action_logs[scen_id] = {}

        for m_name, (m_obj, m_type) in models_to_test.items():
            obs, info = env.reset(options={"scenario": scen})
            done = False
            actions = []
            node_seq = [env.current_node]
            logits_or_q_samples = []

            while not done:
                mask = env.action_masks()
                if m_type == "ppo":
                    # Get policy distribution / logits
                    obs_tensor, _ = m_obj.policy.obs_to_tensor(obs)
                    with torch.no_grad():
                        dist = m_obj.policy.get_distribution(obs_tensor, action_masks=mask)
                        probs = dist.distribution.probs.cpu().numpy().squeeze(0)
                    act, _ = m_obj.predict(obs, action_masks=mask, deterministic=True)
                    act = int(act)
                    valid_probs = {a: round(float(probs[a]), 4) for a in range(len(mask)) if mask[a]}
                    logits_or_q_samples.append((env.current_node, act, valid_probs))
                else:
                    obs_t = torch.as_tensor(obs, device=m_obj.device).unsqueeze(0)
                    with torch.no_grad():
                        q = m_obj.q_net(obs_t).squeeze(0).cpu().numpy()
                    q_valid = {a: round(float(q[a]), 2) for a in range(len(mask)) if mask[a]}
                    q[~mask] = -1e9
                    act = int(np.argmax(q))
                    logits_or_q_samples.append((env.current_node, act, q_valid))

                actions.append(act)
                obs, r, term, trunc, s_info = env.step(act)
                node_seq.append(env.current_node)
                done = term or trunc

            scenario_action_logs[scen_id][m_name] = {
                "actions": actions,
                "node_seq": node_seq,
                "time": s_info.get("cumulative_travel_time", 0.0),
                "hei": s_info.get("hazard_exposure_index", 0.0),
                "success": s_info.get("success", False),
                "decision_details": logits_or_q_samples
            }

            print(f"[{m_name:8s}] Success={s_info.get('success')}, Time={s_info.get('cumulative_travel_time'):.2f}s, Steps={len(actions)}, Actions={actions}")
            print(f"            Nodes: {' -> '.join(node_seq)}")

        # Check if actions are identical across models for this scenario
        acts_p42 = scenario_action_logs[scen_id]["PPO_42"]["actions"]
        acts_p101 = scenario_action_logs[scen_id]["PPO_101"]["actions"]
        acts_d42 = scenario_action_logs[scen_id]["DQN_42"]["actions"]
        same_ppo = (acts_p42 == acts_p101)
        same_all = (acts_p42 == acts_p101 == acts_d42)
        print(f"  >>> PPO_42 == PPO_101: {same_ppo} | PPO_42 == DQN_42: {same_all}")

    env.close()

    print("\n" + "=" * 80)
    print("STEP 4: DETAILED LOGIT & Q-VALUE ANALYSIS AT INITIAL STEP (SCENARIO 1)")
    print("=" * 80)
    scen0_id = test_scenarios[0]["scenario_id"]
    for m_name in ["PPO_42", "PPO_101", "DQN_42"]:
        details = scenario_action_logs[scen0_id][m_name]["decision_details"]
        print(f"Model {m_name} decisions:")
        for step_i, (node, act, vals) in enumerate(details):
            print(f"  Step {step_i} (Node {node}): Chosen Action={act} | Values/Probs={vals}")


if __name__ == "__main__":
    main()
