"""Verification script for Multi-Seed Rigor on Large-Scale Network (336 Nodes):
1. Model file checksums (SHA-256) and exact byte counts for 5 PPO seeds.
2. Pairwise parameter weight L2 divergence matrix across all 5 PPO seeds (10 pairs).
3. Continuous action probability distributions across seeds on test scenarios.
4. Investigation of deterministic argmax convergence vs true model diversity vs collapse.
"""

import os
import sys
import json
import hashlib
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from src.environment.evacuation_env import EvacuationEnv

SEEDS = [42, 101, 2024, 777, 999]
CONFIG_PATH = "configs/large.yaml"
TEST_SCENARIOS_PATH = "data/generated/test_large_scenarios.json"


def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 85)
    print("TASK 1: MULTI-SEED RIGOR VERIFICATION ON 5 LARGE-SCALE PPO SEEDS")
    print("=" * 85)

    ppo_paths = {seed: f"models/large/ppo/seed_{seed}/best_model.zip" for seed in SEEDS}

    print("\n--- 1. Checkpoint Integrity & SHA-256 Checksums ---")
    checksums = {}
    for seed, path in ppo_paths.items():
        if os.path.exists(path):
            sha = get_file_sha256(path)
            sz = os.path.getsize(path)
            checksums[seed] = {"path": path, "size_bytes": sz, "sha256": sha}
            print(f"PPO Seed {seed:4d} | Size: {sz:7d} bytes | SHA-256: {sha}")
        else:
            print(f"PPO Seed {seed:4d} | NOT FOUND: {path}")

    # Check for identical checksums
    shas = [c["sha256"] for c in checksums.values()]
    if len(shas) == len(set(shas)):
        print(">> VERIFIED: All 5 PPO checkpoint files have strictly UNIQUE SHA-256 checksums.")
    else:
        print(">> WARNING: Checksum collision detected!")

    print("\n--- 2. Pairwise Parameter Weight L2 Divergence Matrix ---")
    models = {seed: MaskablePPO.load(path) for seed, path in ppo_paths.items()}

    # Extract named parameters for each model's policy
    param_dicts = {
        seed: dict(models[seed].policy.named_parameters())
        for seed in SEEDS
    }

    l2_matrix = np.zeros((len(SEEDS), len(SEEDS)))
    total_params = sum(p.numel() for p in param_dicts[SEEDS[0]].values())
    print(f"Total Policy Parameters per Model: {total_params:,}")

    pairwise_results = []
    for i, s1 in enumerate(SEEDS):
        for j, s2 in enumerate(SEEDS):
            if i == j:
                l2_matrix[i, j] = 0.0
            elif i < j:
                p1 = param_dicts[s1]
                p2 = param_dicts[s2]
                diff = sum(torch.norm(p1[k] - p2[k]).item() for k in p1.keys())
                l2_matrix[i, j] = diff
                l2_matrix[j, i] = diff
                pairwise_results.append({
                    "seed_pair": f"{s1} vs {s2}",
                    "l2_divergence": float(diff)
                })

    df_l2 = pd.DataFrame(l2_matrix, index=[f"Seed_{s}" for s in SEEDS], columns=[f"Seed_{s}" for s in SEEDS])
    print("\nPairwise L2 Weight Divergence Matrix (Sum of L2 norms of delta across layers):")
    print(df_l2.to_string())
    print(f"\nMean Pairwise L2 Divergence: {np.mean([r['l2_divergence'] for r in pairwise_results]):.4f}")
    print(f"Min Pairwise L2 Divergence:  {np.min([r['l2_divergence'] for r in pairwise_results]):.4f}")
    print(f"Max Pairwise L2 Divergence:  {np.max([r['l2_divergence'] for r in pairwise_results]):.4f}")

    print("\n--- 3. Continuous Probability Distribution Comparison ---")
    with open(TEST_SCENARIOS_PATH, "r") as f:
        test_scenarios = json.load(f)

    # Sample 5 distinct scenarios
    sample_scenarios = test_scenarios[:5]
    env = EvacuationEnv(config_path=CONFIG_PATH, scenarios=sample_scenarios, use_traci=False, curriculum_stage=4)

    prob_comparison_records = []

    for scen_idx, scen in enumerate(sample_scenarios):
        scen_id = scen.get("scenario_id", f"scen_{scen_idx}")
        origin = scen.get("origin_edge")
        dest = scen.get("destination_edge")
        hazard = scen.get("hazard_type")

        print(f"\nEvaluating Scenario {scen_idx+1}: {scen_id} (Origin: {origin} -> Dest: {dest}, Hazard: {hazard})")

        obs, info = env.reset(options={"scenario": scen})
        mask = env.action_masks()

        # Compare action distributions at the starting state
        step_probs = {}
        argmax_actions = {}

        for seed in SEEDS:
            m = models[seed]
            obs_tensor, _ = m.policy.obs_to_tensor(obs)
            with torch.no_grad():
                dist = m.policy.get_distribution(obs_tensor, action_masks=mask)
                probs = dist.distribution.probs.cpu().numpy().squeeze(0)
            act, _ = m.predict(obs, action_masks=mask, deterministic=True)

            step_probs[seed] = [round(float(p), 5) for p in probs]
            argmax_actions[seed] = int(act)

        print(f"  Valid Actions Mask: {mask}")
        for seed in SEEDS:
            valid_p_dict = {a: step_probs[seed][a] for a in range(len(mask)) if mask[a]}
            print(f"  Seed {seed:4d} | Probs: {valid_p_dict} | Argmax: {argmax_actions[seed]}")

        # Compute probability variance across seeds for the top actions
        probs_matrix = np.array([step_probs[s] for s in SEEDS])  # shape (5, num_actions)
        prob_stds = np.std(probs_matrix, axis=0)

        prob_comparison_records.append({
            "scenario_id": scen_id,
            "origin": origin,
            "destination": dest,
            "mask": [bool(m) for m in mask],
            "step_probs": step_probs,
            "argmax_actions": argmax_actions,
            "prob_stds": [float(s) for s in prob_stds]
        })

    # Step through multi-step trajectory on scenario 1 to see branching over time
    print("\n--- 4. Multi-Step Trajectory Branching Analysis (Scenario 1) ---")
    obs, info = env.reset(options={"scenario": sample_scenarios[0]})
    done = False
    step_num = 0

    while not done and step_num < 8:
        step_num += 1
        mask = env.action_masks()
        current_node = env.current_node

        step_data = {}
        for seed in SEEDS:
            m = models[seed]
            obs_tensor, _ = m.policy.obs_to_tensor(obs)
            with torch.no_grad():
                dist = m.policy.get_distribution(obs_tensor, action_masks=mask)
                probs = dist.distribution.probs.cpu().numpy().squeeze(0)
            act, _ = m.predict(obs, action_masks=mask, deterministic=True)
            step_data[seed] = {
                "act": int(act),
                "probs": {a: round(float(probs[a]), 4) for a in range(len(mask)) if mask[a]}
            }

        acts = [step_data[s]["act"] for s in SEEDS]
        all_same = len(set(acts)) == 1
        print(f"Step {step_num:2d} (Node: {current_node}): Valid Actions: {[a for a in range(len(mask)) if mask[a]]}")
        for seed in SEEDS:
            print(f"    Seed {seed:4d} -> Action {step_data[seed]['act']} | Probs: {step_data[seed]['probs']}")
        print(f"    All 5 Seeds Choose Same Action: {all_same}")

        # Step environment using seed 42's action
        obs, r, term, trunc, info = env.step(acts[0])
        done = term or trunc

    env.close()

    # Save verification output to json
    out_path = "results/tables/multiseed_large_verification.json"
    summary = {
        "seeds": SEEDS,
        "total_parameters": total_params,
        "checksums": checksums,
        "pairwise_l2_divergence": pairwise_results,
        "mean_l2_divergence": float(np.mean([r["l2_divergence"] for r in pairwise_results])),
        "sample_scenarios_comparison": prob_comparison_records
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n>> Saved verification summary to {out_path}")


if __name__ == "__main__":
    main()
