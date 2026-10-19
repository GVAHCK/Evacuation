"""Offline Multi-Agent Trajectory Dataset Generator.
Generates state-action-reward transitions from heterogeneous baseline policies (Dijkstra, A*, Random-valid)
over multi-hazard evacuation scenarios to build the offline MARL dataset for Conservative Q-Learning (CQL).
"""

import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple

from src.utils.logger import setup_logger
from src.envs.multi_agent_evacuation_env import MultiAgentEvacuationEnv

logger = setup_logger("GenerateOfflineMARLData")


def compute_dijkstra_action(
    env: MultiAgentEvacuationEnv,
    agent_idx: int,
    hazard_weight: float = 4.0
) -> int:
    """Computes dynamic Dijkstra routing choice for agent idx."""
    agent = env.agents[agent_idx]
    current_node = agent.current_node
    dest_node = agent.destination_node
    successors = env._get_agent_ordered_successors(agent_idx)
    mask = env.get_agent_action_mask(agent_idx)

    valid_indices = [i for i, m in enumerate(mask) if m]
    if not valid_indices:
        return 0

    dynamic_graph = nx.DiGraph()
    for u, v, data in env.graph.edges(data=True):
        edge_id = f"{u}_{v}"
        status = env.base_env.current_hazard_engine.get_edge_status(edge_id)
        if status == env.base_env.current_hazard_engine.STATE_CLOSED:
            continue

        length = float(data.get("length", 50.0))
        speed = float(data.get("speed_limit", 13.89))
        if status == env.base_env.current_hazard_engine.STATE_DEGRADED:
            speed *= 0.5

        intensity = env.base_env.current_hazard_engine.get_intensity(edge_id)
        cost = (length / max(speed, 1.0)) * (1.0 + hazard_weight * intensity)
        dynamic_graph.add_edge(u, v, weight=cost)

    try:
        path = nx.shortest_path(dynamic_graph, source=current_node, target=dest_node, weight="weight")
        if len(path) > 1:
            next_node = path[1]
            if next_node in successors:
                idx = successors.index(next_node)
                if idx < len(mask) and mask[idx]:
                    return idx
    except Exception:
        pass

    return int(random.choice(valid_indices))


def compute_astar_action(
    env: MultiAgentEvacuationEnv,
    agent_idx: int,
    hazard_weight: float = 3.0
) -> int:
    """Computes dynamic A* routing choice for agent idx."""
    agent = env.agents[agent_idx]
    current_node = agent.current_node
    dest_node = agent.destination_node
    successors = env._get_agent_ordered_successors(agent_idx)
    mask = env.get_agent_action_mask(agent_idx)

    valid_indices = [i for i, m in enumerate(mask) if m]
    if not valid_indices:
        return 0

    dynamic_graph = nx.DiGraph()
    for u, v, data in env.graph.edges(data=True):
        edge_id = f"{u}_{v}"
        status = env.base_env.current_hazard_engine.get_edge_status(edge_id)
        if status == env.base_env.current_hazard_engine.STATE_CLOSED:
            continue
        length = float(data.get("length", 50.0))
        intensity = env.base_env.current_hazard_engine.get_intensity(edge_id)
        cost = length * (1.0 + hazard_weight * intensity)
        dynamic_graph.add_edge(u, v, weight=cost)

    def heuristic(u, v):
        return env.sp_lengths.get(u, {}).get(v, 0.0)

    try:
        path = nx.astar_path(dynamic_graph, source=current_node, target=dest_node, heuristic=heuristic, weight="weight")
        if len(path) > 1:
            next_node = path[1]
            if next_node in successors:
                idx = successors.index(next_node)
                if idx < len(mask) and mask[idx]:
                    return idx
    except Exception:
        pass

    return int(random.choice(valid_indices))


def select_demonstration_actions(
    env: MultiAgentEvacuationEnv,
    masks_n: List[np.ndarray],
    policy_mix: str = "mixed"
) -> List[int]:
    """Selects actions across 4 agents using heterogeneous policy mix."""
    actions = []
    for i in range(env.n_agents):
        mask = masks_n[i]
        valid_indices = [k for k, m in enumerate(mask) if m]
        if not valid_indices:
            actions.append(0)
            continue

        r = random.random()
        if policy_mix == "expert":
            # Pure Dijkstra / A*
            act = compute_dijkstra_action(env, i)
        elif policy_mix == "exploratory":
            # 50% heuristic, 50% uniform valid random
            if r < 0.5:
                act = compute_astar_action(env, i)
            else:
                act = int(random.choice(valid_indices))
        else:  # "mixed" default
            if r < 0.45:
                act = compute_dijkstra_action(env, i)
            elif r < 0.75:
                act = compute_astar_action(env, i)
            else:
                act = int(random.choice(valid_indices))

        assert mask[act], f"Offline demonstration action {act} must be valid under mask"
        actions.append(act)

    return actions


def generate_dataset(
    n_episodes: int,
    output_dir: str,
    split_name: str,
    network_size: str = "small",
    seed_offset: int = 1000
) -> Dict[str, Any]:
    """Generates trajectory transitions for split."""
    os.makedirs(output_dir, exist_ok=True)
    env = MultiAgentEvacuationEnv(network_size=network_size, n_agents=4)

    all_transitions = []
    episode_returns = []
    successful_evacuations = 0
    total_agent_evacuations = 0

    for ep in range(n_episodes):
        ep_seed = seed_offset + ep
        obs_n, _ = env.reset(seed=ep_seed)
        masks_n = env.get_action_masks()
        state = env.get_global_state()

        policy_mix = "mixed" if ep % 3 != 0 else ("expert" if ep % 3 == 1 else "exploratory")
        ep_rewards = [0.0] * env.n_agents
        done = False
        step = 0

        while not done and step < 60:
            actions = select_demonstration_actions(env, masks_n, policy_mix=policy_mix)
            next_obs_n, rewards, terms, truncs, info = env.step(actions)
            next_masks_n = env.get_action_masks()
            next_state = env.get_global_state()

            for i in range(env.n_agents):
                ep_rewards[i] += rewards[i]

            all_transitions.append({
                "obs_n": [o.tolist() for o in obs_n],
                "state": state.tolist(),
                "actions_n": [int(a) for a in actions],
                "rewards_n": [float(r) for r in rewards],
                "next_obs_n": [o.tolist() for o in next_obs_n],
                "next_state": next_state.tolist(),
                "dones_n": [bool(terms[i] or truncs[i]) for i in range(env.n_agents)],
                "masks_n": [m.tolist() for m in masks_n],
                "next_masks_n": [m.tolist() for m in next_masks_n]
            })

            obs_n = next_obs_n
            masks_n = next_masks_n
            state = next_state
            step += 1
            done = all(terms) or all(truncs)

        episode_returns.append(float(np.mean(ep_rewards)))
        successful_evacuations += sum(1 for a in env.agents if a.is_success)
        total_agent_evacuations += env.n_agents

    env.close()

    # Save transitions to JSON chunks
    chunk_size = 500
    for chunk_idx, i in enumerate(range(0, len(all_transitions), chunk_size)):
        chunk = all_transitions[i: i + chunk_size]
        file_path = os.path.join(output_dir, f"transitions_part_{chunk_idx:03d}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f)

    stats = {
        "split": split_name,
        "n_episodes": n_episodes,
        "total_transitions": len(all_transitions),
        "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "individual_success_rate": float(successful_evacuations / max(1, total_agent_evacuations))
    }
    logger.info(f"Split [{split_name}] completed: {len(all_transitions)} transitions across {n_episodes} episodes.")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate Offline Multi-Agent RL Trajectories")
    parser.add_argument("--train_episodes", type=int, default=60, help="Number of training episodes")
    parser.add_argument("--val_episodes", type=int, default=20, help="Number of validation episodes")
    parser.add_argument("--output_dir", type=str, default="data/offline_marl", help="Directory to save trajectories")
    parser.add_argument("--network_size", type=str, default="small", help="Road network size (small/large)")
    args = parser.parse_args()

    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")

    logger.info(f"Starting Offline MARL data generation: {args.train_episodes} train, {args.val_episodes} val episodes...")
    train_stats = generate_dataset(args.train_episodes, train_dir, "train", network_size=args.network_size, seed_offset=1000)
    val_stats = generate_dataset(args.val_episodes, val_dir, "val", network_size=args.network_size, seed_offset=5000)

    metadata = {
        "dataset_name": "MultiAgentEvacuationOfflineTrajectories",
        "network_size": args.network_size,
        "train": train_stats,
        "val": val_stats,
        "policy_mix": ["Dynamic Dijkstra (45%)", "Dynamic A* (30%)", "Uniform Valid Exploratory (25%)"]
    }

    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Offline MARL dataset successfully generated at {args.output_dir}. Metadata: {meta_path}")


if __name__ == "__main__":
    main()
