"""Master Training Pipeline for the Six MARL Models.
Supports MADDPG, RS-MADDPG, Offline-to-Online MARL, MA-GRL, GraphSAGE-MAPPO, and HARL.
Enforces hard action masking, CTDE paradigm, heterogeneous cohort evaluation, and multi-seed checkpoints.
"""

import os
import sys
import json
import yaml
import time
import random
import argparse
import numpy as np
import torch
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.envs.multi_agent_evacuation_env import MultiAgentEvacuationEnv
from src.agents.marl_six import (
    MADDPGModel,
    RSMADDPGModel,
    OfflineOnlineMARLModel,
    MAGRLModel,
    GraphSAGEMAPPOModel,
    HARLModel
)

logger = setup_logger("TrainMARLSix")

MODEL_REGISTRY = {
    "maddpg": MADDPGModel,
    "rs_maddpg": RSMADDPGModel,
    "offline_online": OfflineOnlineMARLModel,
    "ma_grl": MAGRLModel,
    "graphsage_mappo": GraphSAGEMAPPOModel,
    "harl": HARLModel
}

DEFAULT_SEEDS = [42, 101, 2024, 777, 999]


def set_seed(seed: int):
    """Sets random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_config(model_key: str) -> Dict[str, Any]:
    """Loads configuration YAML for specified model."""
    cfg_file = os.path.join("configs", "marl_six", f"{model_key}.yaml")
    if os.path.isfile(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def build_model(model_key: str, cfg: Dict[str, Any], device: Optional[str] = None):
    """Factory creating an initialized instance of the selected MARL model."""
    agent_cfg = cfg.get("agent", {})
    cls = MODEL_REGISTRY[model_key]

    if model_key == "maddpg":
        return cls(
            n_agents=agent_cfg.get("n_agents", 4),
            obs_dim=agent_cfg.get("obs_dim", 34),
            act_dim=agent_cfg.get("act_dim", 4),
            actor_hidden_dim=agent_cfg.get("actor_hidden_dim", 128),
            critic_hidden_dim=agent_cfg.get("critic_hidden_dim", 128),
            actor_lr=agent_cfg.get("actor_lr", 0.0003),
            critic_lr=agent_cfg.get("critic_lr", 0.001),
            gamma=agent_cfg.get("gamma", 0.98),
            tau=agent_cfg.get("tau", 0.005),
            gumbel_temp=agent_cfg.get("gumbel_temperature", 1.0),
            gumbel_min_temp=agent_cfg.get("gumbel_min_temperature", 0.2),
            device=device
        )
    elif model_key == "rs_maddpg":
        return cls(
            n_agents=agent_cfg.get("n_agents", 4),
            obs_dim=agent_cfg.get("obs_dim", 34),
            act_dim=agent_cfg.get("act_dim", 4),
            actor_hidden_dim=agent_cfg.get("actor_hidden_dim", 128),
            critic_hidden_dim=agent_cfg.get("critic_hidden_dim", 128),
            actor_lr=agent_cfg.get("actor_lr", 0.0003),
            critic_lr=agent_cfg.get("critic_lr", 0.0008),
            gamma=agent_cfg.get("gamma", 0.98),
            tau=agent_cfg.get("tau", 0.005),
            per_alpha=agent_cfg.get("per_alpha", 0.6),
            per_beta_start=agent_cfg.get("per_beta_start", 0.4),
            hazard_barrier_weight=agent_cfg.get("hazard_barrier_weight", 2.0),
            device=device
        )
    elif model_key == "offline_online":
        return cls(
            n_agents=agent_cfg.get("n_agents", 4),
            obs_dim=agent_cfg.get("obs_dim", 34),
            act_dim=agent_cfg.get("act_dim", 4),
            hidden_dim=agent_cfg.get("hidden_dim", 128),
            actor_lr=agent_cfg.get("actor_lr", 0.0003),
            critic_lr=agent_cfg.get("critic_lr", 0.001),
            gamma=agent_cfg.get("gamma", 0.98),
            cql_alpha=agent_cfg.get("cql_alpha", 1.0),
            policy_reg_weight=agent_cfg.get("policy_reg_weight", 0.2),
            device=device
        )
    elif model_key == "ma_grl":
        return cls(
            n_agents=agent_cfg.get("n_agents", 4),
            obs_dim=agent_cfg.get("obs_dim", 34),
            act_dim=agent_cfg.get("act_dim", 4),
            gnn_hidden_dim=agent_cfg.get("gnn_hidden_dim", 64),
            critic_hidden_dim=agent_cfg.get("critic_hidden_dim", 128),
            actor_lr=agent_cfg.get("actor_lr", 0.0003),
            critic_lr=agent_cfg.get("critic_lr", 0.0008),
            gamma=agent_cfg.get("gamma", 0.98),
            device=device
        )
    elif model_key == "graphsage_mappo":
        return cls(
            n_agents=agent_cfg.get("n_agents", 4),
            obs_dim=agent_cfg.get("obs_dim", 34),
            act_dim=agent_cfg.get("act_dim", 4),
            graphsage_hidden_dim=agent_cfg.get("graphsage_hidden_dim", 64),
            mlp_hidden_dim=agent_cfg.get("mlp_hidden_dim", 128),
            actor_lr=agent_cfg.get("actor_lr", 0.0003),
            critic_lr=agent_cfg.get("critic_lr", 0.001),
            gamma=agent_cfg.get("gamma", 0.99),
            gae_lambda=agent_cfg.get("gae_lambda", 0.95),
            clip_range=agent_cfg.get("clip_range", 0.2),
            entropy_coef=agent_cfg.get("entropy_coef", 0.01),
            ppo_epochs=agent_cfg.get("ppo_epochs", 4),
            device=device
        )
    elif model_key == "harl":
        return cls(
            n_agents=agent_cfg.get("n_agents", 4),
            obs_dim=agent_cfg.get("obs_dim", 34),
            act_dim=agent_cfg.get("act_dim", 4),
            actor_hidden_dim=agent_cfg.get("actor_hidden_dim", 128),
            critic_hidden_dim=agent_cfg.get("critic_hidden_dim", 128),
            actor_lr=agent_cfg.get("actor_lr", 0.0003),
            critic_lr=agent_cfg.get("critic_lr", 0.001),
            gamma=agent_cfg.get("gamma", 0.99),
            gae_lambda=agent_cfg.get("gae_lambda", 0.95),
            clip_range=agent_cfg.get("clip_range", 0.2),
            entropy_coef=agent_cfg.get("entropy_coef", 0.01),
            ppo_epochs=agent_cfg.get("ppo_epochs", 4),
            batch_size=agent_cfg.get("batch_size", 64),
            device=device
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def evaluate_policy(
    model: Any,
    env: MultiAgentEvacuationEnv,
    n_eval_episodes: int = 5,
    seed_offset: int = 8000
) -> Dict[str, float]:
    """Evaluates multi-agent model deterministically across validation episodes."""
    eval_returns = []
    joint_successes = []
    indiv_success_rates = []
    mean_heis = []
    mean_travel_times = []
    conflict_counts = []
    overlaps = []

    for ep in range(n_eval_episodes):
        obs_n, info = env.reset(seed=seed_offset + ep)
        masks_n = env.get_action_masks()
        ep_rewards = [0.0] * env.n_agents
        done = False
        step = 0

        while not done and step < 60:
            actions = model.predict(obs_n, masks_n, deterministic=True)
            for i in range(env.n_agents):
                assert masks_n[i][actions[i]], f"Action {actions[i]} violates mask"

            next_obs_n, rewards, terms, truncs, step_info = env.step(actions)
            for i in range(env.n_agents):
                ep_rewards[i] += rewards[i]

            obs_n = next_obs_n
            masks_n = env.get_action_masks()
            step += 1
            done = all(terms) or all(truncs)

        eval_returns.append(float(np.mean(ep_rewards)))
        joint_successes.append(1.0 if all(a.is_success for a in env.agents) else 0.0)
        indiv_success_rates.append(sum(1 for a in env.agents if a.is_success) / float(env.n_agents))
        mean_heis.append(float(np.mean([a.cumulative_hazard for a in env.agents])))
        mean_travel_times.append(float(np.mean([a.cumulative_travel_time for a in env.agents])))
        conflict_counts.append(float(env.total_conflicts))
        overlaps.append(float(step_info.get("mean_route_overlap", 0.0)))

    return {
        "mean_return": float(np.mean(eval_returns)),
        "joint_success_rate": float(np.mean(joint_successes)),
        "indiv_success_rate": float(np.mean(indiv_success_rates)),
        "mean_hei": float(np.mean(mean_heis)),
        "mean_travel_time": float(np.mean(mean_travel_times)),
        "conflict_count": float(np.mean(conflict_counts)),
        "route_overlap": float(np.mean(overlaps))
    }


def train_single_model(
    model_key: str,
    seed: int,
    mode: str = "quick",
    network_size: str = "small",
    offline_data_dir: str = "data/offline_marl"
) -> Dict[str, Any]:
    """Runs complete training loop for a single model and seed."""
    set_seed(seed)
    cfg = load_model_config(model_key)
    training_cfg = cfg.get("training", {})

    total_episodes = training_cfg.get("quick_episodes", 50) if mode == "quick" else training_cfg.get("full_episodes", 300)
    eval_freq = training_cfg.get("eval_freq", 25)
    batch_size = training_cfg.get("batch_size", 64)

    logger.info(f"[{model_key.upper()}] Starting training: seed={seed}, mode={mode}, episodes={total_episodes}, network={network_size}")

    train_env = MultiAgentEvacuationEnv(network_size=network_size, n_agents=4, seed=seed)
    val_env = MultiAgentEvacuationEnv(network_size=network_size, n_agents=4, seed=seed + 500)
    model = build_model(model_key, cfg)

    checkpoint_dir = os.path.join("models", "marl_six", model_key, f"seed_{seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    curve_dir = os.path.join("results", "marl_six", "training_curves")
    os.makedirs(curve_dir, exist_ok=True)

    is_on_policy = model_key in ["graphsage_mappo", "harl"]
    is_offline_online = model_key == "offline_online"

    # Phase A for Offline-Online RL: Offline CQL Pretraining
    if is_offline_online:
        train_data_dir = os.path.join(offline_data_dir, "train")
        logger.info(f"[{model_key.upper()}] Loading offline dataset from {train_data_dir}...")
        model.load_offline_dataset(train_data_dir)
        pretrain_steps = 100 if mode == "quick" else 400
        logger.info(f"[{model_key.upper()}] Running {pretrain_steps} Conservative Q-Learning (CQL) offline steps...")
        for step_idx in range(pretrain_steps):
            cql_res = model.update_offline(batch_size=batch_size)
            if step_idx % 100 == 0 and cql_res:
                logger.info(f"  [CQL Pretrain {step_idx}/{pretrain_steps}] Critic Loss: {cql_res.get('critic_loss', 0.0):.4f}, CQL Penalty: {cql_res.get('cql_penalty', 0.0):.4f}")
        model.freeze_for_online()
        logger.info(f"[{model_key.upper()}] Offline pretraining complete. Anchored policy for conservative online fine-tuning.")

    episode_history = []
    best_eval_return = -1e9
    start_time = time.time()

    total_steps = 0
    for ep in range(1, total_episodes + 1):
        obs_n, _ = train_env.reset(seed=seed + ep * 13)
        masks_n = train_env.get_action_masks()
        state = train_env.get_global_state()

        ep_reward = [0.0] * train_env.n_agents
        ep_losses = []
        done = False
        step = 0

        while not done and step < 60:
            if is_on_policy:
                actions, log_probs = model.select_actions(obs_n, masks_n, deterministic=False)
                value = model.get_value(state)
            elif model_key in ["maddpg", "rs_maddpg", "ma_grl"]:
                actions = model.select_actions(obs_n, masks_n, explore=True)
            else:  # offline_online online fine-tuning
                actions = model.select_actions(obs_n, masks_n, deterministic=False)

            # Enforce hard action masking verification
            for i in range(train_env.n_agents):
                assert masks_n[i][actions[i]], f"CRITICAL: Agent {i} selected invalid action {actions[i]}"

            next_obs_n, rewards, terms, truncs, info = train_env.step(actions)
            next_masks_n = train_env.get_action_masks()
            next_state = train_env.get_global_state()

            for i in range(train_env.n_agents):
                ep_reward[i] += rewards[i]

            step_dones = [bool(terms[i] or truncs[i]) for i in range(train_env.n_agents)]

            # Buffer Storage & Model Updates
            if is_on_policy:
                model.buffer.add(
                    obs_n=obs_n,
                    state=state,
                    actions_n=actions,
                    log_probs_n=log_probs,
                    rewards_n=rewards,
                    dones_n=step_dones,
                    masks_n=masks_n,
                    value=value
                )
                if len(model.buffer.states) >= training_cfg.get("steps_per_rollout", 64):
                    loss_dict = model.update(next_state=next_state)
                    if loss_dict:
                        ep_losses.append(loss_dict)
            elif is_offline_online:
                model.store_online_transition(
                    obs_n, state, actions, rewards, next_obs_n, next_state, step_dones, masks_n, next_masks_n
                )
                if len(model.replay_buffer) >= batch_size:
                    loss_dict = model.update_online(batch_size=batch_size)
                    if loss_dict:
                        ep_losses.append(loss_dict)
            elif model_key == "rs_maddpg":
                model.store_transition(
                    obs_n, state, actions, rewards, next_obs_n, next_state, step_dones, masks_n, next_masks_n
                )
                if len(model.replay_buffer) >= batch_size:
                    loss_dict = model.update(batch_size=batch_size)
                    if loss_dict:
                        ep_losses.append(loss_dict)
            else:  # maddpg, ma_grl
                model.buffer.add(
                    obs_n, state, actions, rewards, next_obs_n, next_state, step_dones, masks_n, next_masks_n
                )
                if len(model.replay_buffer) >= batch_size:
                    loss_dict = model.update(batch_size=batch_size)
                    if loss_dict:
                        ep_losses.append(loss_dict)

            obs_n = next_obs_n
            masks_n = next_masks_n
            state = next_state
            step += 1
            total_steps += 1
            done = all(terms) or all(truncs)

        # On-policy end of episode update if residual steps remain
        if is_on_policy and len(model.buffer.states) > 0:
            loss_dict = model.update(next_state=next_state)
            if loss_dict:
                ep_losses.append(loss_dict)

        mean_ep_ret = float(np.mean(ep_reward))
        success_rate = sum(1 for a in train_env.agents if a.is_success) / float(train_env.n_agents)
        mean_hei = float(np.mean([a.cumulative_hazard for a in train_env.agents]))

        ep_record = {
            "episode": ep,
            "return": mean_ep_ret,
            "success_rate": success_rate,
            "mean_hei": mean_hei,
            "conflicts": train_env.total_conflicts,
            "steps": step
        }
        episode_history.append(ep_record)

        if ep % eval_freq == 0 or ep == total_episodes:
            eval_metrics = evaluate_policy(model, val_env, n_eval_episodes=5, seed_offset=8000 + ep)
            logger.info(
                f"[{model_key.upper()} | Ep {ep}/{total_episodes}] "
                f"Train Ret: {mean_ep_ret:.1f} | "
                f"Val Ret: {eval_metrics['mean_return']:.1f} | "
                f"Val Joint Succ: {eval_metrics['joint_success_rate']*100:.1f}% | "
                f"Val HEI: {eval_metrics['mean_hei']:.2f} | "
                f"Conflicts: {eval_metrics['conflict_count']:.1f}"
            )

            if eval_metrics["mean_return"] > best_eval_return:
                best_eval_return = eval_metrics["mean_return"]
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                model.save(best_path)

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    model.save(final_path)

    elapsed_time = time.time() - start_time
    train_env.close()
    val_env.close()

    # Save training curves
    curve_file = os.path.join(curve_dir, f"{model_key}_seed_{seed}.json")
    curve_data = {
        "model": model_key,
        "seed": seed,
        "mode": mode,
        "network_size": network_size,
        "elapsed_seconds": elapsed_time,
        "episodes": episode_history,
        "final_best_eval_return": best_eval_return
    }
    with open(curve_file, "w", encoding="utf-8") as f:
        json.dump(curve_data, f, indent=2)

    logger.info(f"[{model_key.upper()}] Training finished in {elapsed_time:.1f}s. Checkpoints in {checkpoint_dir}")
    return curve_data


def main():
    parser = argparse.ArgumentParser(description="Train Six MARL Models on Multi-Hazard Road Networks")
    parser.add_argument("--model", type=str, default="rs_maddpg", choices=list(MODEL_REGISTRY.keys()) + ["all"],
                        help="Specific model to train, or 'all'")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"],
                        help="Training mode: 'quick' (50 eps) or 'full' (300 eps)")
    parser.add_argument("--seed", type=int, default=42, help="Single seed to train")
    parser.add_argument("--all_seeds", action="store_true", help="Train over all 5 benchmark seeds [42, 101, 2024, 777, 999]")
    parser.add_argument("--network_size", type=str, default="small", choices=["small", "large"],
                        help="Network topology (small 84-node or large 336-node)")
    parser.add_argument("--offline_data_dir", type=str, default="data/offline_marl",
                        help="Path to offline multi-agent dataset")
    args = parser.parse_args()

    models_to_train = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    seeds_to_run = DEFAULT_SEEDS if args.all_seeds else [args.seed]

    logger.info(f"MARL Training Launch: Models={models_to_train}, Seeds={seeds_to_run}, Mode={args.mode}, Network={args.network_size}")

    for m in models_to_train:
        for s in seeds_to_run:
            train_single_model(
                model_key=m,
                seed=s,
                mode=args.mode,
                network_size=args.network_size,
                offline_data_dir=args.offline_data_dir
            )

    logger.info("All requested MARL models and seeds trained successfully!")


if __name__ == "__main__":
    main()
