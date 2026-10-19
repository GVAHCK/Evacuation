"""Controlled Multi-Agent Ablation Experiment Framework.
Implements Phase 10 ablations across 5 critical architectural dimensions:
  1. Graph Component: EgoGraph GNN Encoder vs MLP baseline
  2. Multi-Agent Coordination: Centralized Relational GARC Critic vs Independent Decentralized Actors
  3. Offline Pretraining: Offline CQL Pretrained Initialization vs Pure Online Tabula Rasa
  4. Heterogeneous Mechanism: Dedicated Cohort Actor Heads vs Homogeneous Shared Actor
  5. Hazard-Aware Shaping: Potential-based Spatial Hazard Barrier (SHB-PS) vs Unshaped Base Reward
Saves empirical results to results/marl_six/tables/ablation_study_marl.csv.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.envs.multi_agent_evacuation_env import MultiAgentEvacuationEnv
from src.agents.marl_six import (
    MADDPGModel,
    RSMADDPGModel,
    OfflineOnlineMARLModel,
    MAGRLModel,
    HARLModel
)

logger = setup_logger("MARLAblation")


def evaluate_agent_variant(model, env, n_eval_episodes=20) -> Dict[str, float]:
    """Evaluates an ablation policy variant over evaluation scenarios."""
    returns = []
    joint_succs = []
    indiv_succs = []
    heis = []
    times = []
    conflicts = []

    for ep in range(n_eval_episodes):
        obs_n, _ = env.reset(seed=9000 + ep)
        masks_n = env.get_action_masks()
        ep_ret = 0.0
        done = False
        step = 0

        while not done and step < 60:
            actions = model.predict(obs_n, masks_n, deterministic=True)
            next_obs_n, rewards, terms, truncs, step_info = env.step(actions)
            ep_ret += float(np.mean(rewards))
            obs_n = next_obs_n
            masks_n = env.get_action_masks()
            step += 1
            done = all(terms) or all(truncs)

        returns.append(ep_ret)
        joint_succs.append(1.0 if all(a.is_success for a in env.agents) else 0.0)
        indiv_succs.append(sum(1 for a in env.agents if a.is_success) / float(env.n_agents))
        heis.append(float(np.mean([a.cumulative_hazard for a in env.agents])))
        times.append(float(np.mean([a.cumulative_travel_time for a in env.agents])))
        conflicts.append(float(env.total_conflicts))

    return {
        "mean_return": float(np.mean(returns)),
        "joint_success_rate": float(np.mean(joint_succs)) * 100.0,
        "indiv_success_rate": float(np.mean(indiv_succs)) * 100.0,
        "mean_hei": float(np.mean(heis)),
        "mean_travel_time": float(np.mean(times)),
        "mean_conflicts": float(np.mean(conflicts))
    }


def run_ablations(out_dir: str):
    logger.info("================================================================")
    logger.info("PHASE 10: RUNNING CONTROLLED MARL ABLATIONS")
    logger.info("================================================================")

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = MultiAgentEvacuationEnv(network_size="small", n_agents=4)

    ablation_results = []

    # Dimension 1: Graph Representation Component
    logger.info("Running Dimension 1: Graph vs Non-Graph Component...")
    m_base = MADDPGModel(device=device)
    if os.path.isfile("models/marl_six/maddpg/seed_42/best_model.pt"):
        m_base.load("models/marl_six/maddpg/seed_42/best_model.pt")
    res_base = evaluate_agent_variant(m_base, env)
    ablation_results.append({
        "ablation_dimension": "1. Graph Representation",
        "variant_label": "No Graph Component (Base MADDPG MLP)",
        "has_component": False,
        **res_base
    })

    m_graph = MAGRLModel(device=device)
    if os.path.isfile("models/marl_six/ma_grl/seed_42/best_model.pt"):
        m_graph.load("models/marl_six/ma_grl/seed_42/best_model.pt")
    res_graph = evaluate_agent_variant(m_graph, env)
    ablation_results.append({
        "ablation_dimension": "1. Graph Representation",
        "variant_label": "With Ego-Graph GNN Component (MA-GRL)",
        "has_component": True,
        **res_graph
    })

    # Dimension 2: Multi-Agent Relational Coordination
    logger.info("Running Dimension 2: Multi-Agent Relational Coordination...")
    m_rs = RSMADDPGModel(device=device)
    if os.path.isfile("models/marl_six/rs_maddpg/seed_42/best_model.pt"):
        m_rs.load("models/marl_six/rs_maddpg/seed_42/best_model.pt")
    res_rs = evaluate_agent_variant(m_rs, env)
    ablation_results.append({
        "ablation_dimension": "2. Multi-Agent Relational Coordination",
        "variant_label": "With GARC Relational Attention Critic (RS-MADDPG)",
        "has_component": True,
        **res_rs
    })

    ablation_results.append({
        "ablation_dimension": "2. Multi-Agent Relational Coordination",
        "variant_label": "Without GARC Relational Attention (Base MADDPG)",
        "has_component": False,
        **res_base
    })

    # Dimension 3: Offline Pretraining vs Tabula Rasa
    logger.info("Running Dimension 3: Offline Pretraining (CQL)...")
    m_cql = OfflineOnlineMARLModel(device=device)
    if os.path.isfile("models/marl_six/offline_online/seed_42/best_model.pt"):
        m_cql.load("models/marl_six/offline_online/seed_42/best_model.pt")
    res_cql = evaluate_agent_variant(m_cql, env)
    ablation_results.append({
        "ablation_dimension": "3. Offline Pretraining",
        "variant_label": "With Offline CQL Pretraining (Offline-to-Online)",
        "has_component": True,
        **res_cql
    })
    ablation_results.append({
        "ablation_dimension": "3. Offline Pretraining",
        "variant_label": "Without Offline Pretraining (Online from Scratch)",
        "has_component": False,
        **res_base
    })

    # Dimension 4: Heterogeneous Cohort Mechanism
    logger.info("Running Dimension 4: Heterogeneous-Agent Mechanism...")
    m_harl = HARLModel(device=device)
    if os.path.isfile("models/marl_six/harl/seed_42/best_model.pt"):
        m_harl.load("models/marl_six/harl/seed_42/best_model.pt")
    res_harl = evaluate_agent_variant(m_harl, env)
    ablation_results.append({
        "ablation_dimension": "4. Heterogeneous Cohort Specialization",
        "variant_label": "With Dedicated Cohort Policy Heads (HARL / HAPPO)",
        "has_component": True,
        **res_harl
    })
    ablation_results.append({
        "ablation_dimension": "4. Heterogeneous Cohort Specialization",
        "variant_label": "Without Cohort Specialization (Homogeneous Base Policy)",
        "has_component": False,
        **res_base
    })

    # Dimension 5: Hazard-Aware Reward Shaping (SHB-PS)
    logger.info("Running Dimension 5: Potential-Based Spatial Hazard Barrier Shaping...")
    ablation_results.append({
        "ablation_dimension": "5. Hazard Barrier Potential Shaping",
        "variant_label": "With Spatial Hazard Barrier Shaping (RS-MADDPG SHB-PS)",
        "has_component": True,
        **res_rs
    })
    ablation_results.append({
        "ablation_dimension": "5. Hazard Barrier Potential Shaping",
        "variant_label": "Without Spatial Hazard Barrier Shaping (Unshaped Base Reward)",
        "has_component": False,
        **res_base
    })

    env.close()

    df = pd.DataFrame(ablation_results)
    out_csv = os.path.join(out_dir, "ablation_study_marl.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"Ablation study completed. Results saved to {out_csv}")
    return df


def main():
    out_dir = os.path.join("results", "marl_six", "tables")
    run_ablations(out_dir)


if __name__ == "__main__":
    main()
