"""Model 3: Offline-to-Online Reinforcement Learning
Phase A: Conservative Multi-Agent Q-Learning (CQL) pretraining on offline trajectory dataset.
Phase B: Online interactive environment fine-tuning with conservative policy regularization.
"""

import os
import copy
import json
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logger import setup_logger
from src.agents.marl_six.maddpg_agent import DiscreteActor, CentralizedCritic, MultiAgentReplayBuffer

logger = setup_logger("OfflineOnlineMARL")


class OfflineOnlineMARLModel:
    """Offline-to-Online Multi-Agent Reinforcement Learning framework."""

    def __init__(
        self,
        n_agents: int = 4,
        obs_dim: int = 34,
        act_dim: int = 4,
        hidden_dim: int = 128,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.0008,
        gamma: float = 0.98,
        cql_alpha: float = 1.0,
        policy_reg_weight: float = 0.1,
        tau: float = 0.005,
        device: Optional[str] = None
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.policy_reg_weight = policy_reg_weight
        self.tau = tau

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        global_state_dim = obs_dim * n_agents + 8
        total_act_dim = act_dim * n_agents

        self.actors = [DiscreteActor(obs_dim, act_dim, hidden_dim).to(self.device) for _ in range(n_agents)]
        self.target_actors = [copy.deepcopy(a).to(self.device) for a in self.actors]
        self.offline_anchor_actors = [copy.deepcopy(a).to(self.device) for a in self.actors]

        self.critics = [CentralizedCritic(global_state_dim, total_act_dim, hidden_dim).to(self.device) for _ in range(n_agents)]
        self.target_critics = [copy.deepcopy(c).to(self.device) for c in self.critics]

        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=actor_lr) for a in self.actors]
        self.critic_optimizers = [torch.optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics]

        self.replay_buffer = MultiAgentReplayBuffer(max_size=50000)
        self.buffer = self.replay_buffer
        self.offline_dataset: List[Dict[str, Any]] = []

    def store_offline_transition(self, obs_n, state, actions_n, rewards_n, next_obs_n, next_state, dones_n, masks_n, next_masks_n):
        self.offline_dataset.append({
            "obs_n": [o.copy() for o in obs_n],
            "state": state.copy(),
            "actions_n": list(actions_n),
            "rewards_n": list(rewards_n),
            "next_obs_n": [o.copy() for o in next_obs_n],
            "next_state": next_state.copy(),
            "dones_n": list(dones_n),
            "masks_n": [m.copy() for m in masks_n],
            "next_masks_n": [m.copy() for m in next_masks_n]
        })

    def store_online_transition(self, obs_n, state, actions_n, rewards_n, next_obs_n, next_state, dones_n, masks_n, next_masks_n):
        self.replay_buffer.add(
            obs_n=obs_n,
            state=state,
            actions_n=actions_n,
            rewards_n=rewards_n,
            next_obs_n=next_obs_n,
            next_state=next_state,
            dones_n=dones_n,
            masks_n=masks_n,
            next_masks_n=next_masks_n
        )

    def update_offline(self, batch_size: int = 128) -> Dict[str, float]:
        res = self.pretrain_offline_step(batch_size=batch_size)
        if res:
            res["critic_loss"] = res.get("offline_critic_td_loss", 0.0)
            res["cql_penalty"] = res.get("offline_cql_penalty", 0.0)
        return res

    def freeze_for_online(self):
        self.anchor_offline_policy()

    def update_online(self, batch_size: int = 128) -> Dict[str, float]:
        return self.online_finetune_step(batch_size=batch_size)

    def fine_tune_online_step(self, batch_size: int = 128) -> Dict[str, float]:
        return self.online_finetune_step(batch_size=batch_size)

    def load_offline_dataset(self, dataset_path: str):
        """Loads offline trajectory transitions from JSON/NPZ."""
        if not os.path.exists(dataset_path):
            logger.warning(f"Offline dataset path {dataset_path} does not exist.")
            return

        json_files = glob.glob(os.path.join(dataset_path, "*.json"))
        loaded_count = 0
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for trans in data:
                        self.offline_dataset.append(trans)
                        loaded_count += 1
            except Exception as e:
                logger.debug(f"Error reading {jf}: {e}")

        logger.info(f"Loaded {loaded_count} offline multi-agent transitions from {dataset_path}")

    def pretrain_offline_step(self, batch_size: int = 128) -> Dict[str, float]:
        """Performs one offline Conservative Q-Learning (CQL) pretraining step."""
        if len(self.offline_dataset) < batch_size:
            return {}

        batch = random.sample(self.offline_dataset, batch_size)
        state = torch.as_tensor(np.array([b["state"] for b in batch]), dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(np.array([b["next_state"] for b in batch]), dtype=torch.float32, device=self.device)

        obs_t = [torch.as_tensor(np.array([b["obs_n"][i] for b in batch]), dtype=torch.float32, device=self.device) for i in range(self.n_agents)]
        next_obs_t = [torch.as_tensor(np.array([b["next_obs_n"][i] for b in batch]), dtype=torch.float32, device=self.device) for i in range(self.n_agents)]
        acts_t = [torch.as_tensor(np.array([b["actions_n"][i] for b in batch]), dtype=torch.int64, device=self.device) for i in range(self.n_agents)]
        rewards_t = [torch.as_tensor(np.array([b["rewards_n"][i] for b in batch]), dtype=torch.float32, device=self.device).unsqueeze(-1) for i in range(self.n_agents)]
        dones_t = [torch.as_tensor(np.array([b["dones_n"][i] for b in batch]), dtype=torch.float32, device=self.device).unsqueeze(-1) for i in range(self.n_agents)]
        masks_t = [torch.as_tensor(np.array([b["masks_n"][i] for b in batch]), dtype=torch.bool, device=self.device) for i in range(self.n_agents)]
        next_masks_t = [torch.as_tensor(np.array([b["next_masks_n"][i] for b in batch]), dtype=torch.bool, device=self.device) for i in range(self.n_agents)]

        one_hot_acts = [F.one_hot(acts_t[i], num_classes=self.act_dim).float() for i in range(self.n_agents)]
        joint_actions = torch.cat(one_hot_acts, dim=-1)

        critic_losses = []
        cql_losses = []

        with torch.no_grad():
            next_one_hots = []
            for i in range(self.n_agents):
                act_i, _ = self.target_actors[i].sample_action(next_obs_t[i], next_masks_t[i], deterministic=True)
                next_one_hots.append(act_i)
            next_joint_acts = torch.cat(next_one_hots, dim=-1)

        for i in range(self.n_agents):
            with torch.no_grad():
                target_q = self.target_critics[i](next_state, next_joint_acts)
                y = rewards_t[i] + self.gamma * (1.0 - dones_t[i]) * target_q

            current_q = self.critics[i](state, joint_actions)
            td_loss = F.mse_loss(current_q, y)

            # CQL Regularizer: push down unseen action values
            # Sample random valid actions for counterfactuals
            rand_acts = []
            for j in range(self.n_agents):
                r_act = F.one_hot(torch.randint(0, self.act_dim, (batch_size,), device=self.device), num_classes=self.act_dim).float()
                rand_acts.append(r_act)
            rand_joint_acts = torch.cat(rand_acts, dim=-1)

            q_rand = self.critics[i](state, rand_joint_acts)
            cql_penalty = torch.mean(q_rand) - torch.mean(current_q)
            cql_loss = self.cql_alpha * cql_penalty

            total_c_loss = td_loss + cql_loss

            self.critic_optimizers[i].zero_grad()
            total_c_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_optimizers[i].step()

            critic_losses.append(td_loss.item())
            cql_losses.append(cql_penalty.item())

        # Actor Behavior Cloning + Value Maximization
        actor_losses = []
        for i in range(self.n_agents):
            logits = self.actors[i](obs_t[i], masks_t[i])
            bc_loss = F.cross_entropy(logits, acts_t[i])

            act_sample, _ = self.actors[i].sample_action(obs_t[i], masks_t[i], deterministic=False)
            current_acts = []
            for j in range(self.n_agents):
                current_acts.append(act_sample if j == i else one_hot_acts[j].detach())
            joint_acts = torch.cat(current_acts, dim=-1)
            q_val = self.critics[i](state, joint_acts)
            q_loss = -torch.mean(q_val)

            total_a_loss = bc_loss + 0.1 * q_loss

            self.actor_optimizers[i].zero_grad()
            total_a_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()
            actor_losses.append(total_a_loss.item())

        # Target updates
        for i in range(self.n_agents):
            for param, target_param in zip(self.actors[i].parameters(), self.target_actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for param, target_param in zip(self.critics[i].parameters(), self.target_critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "offline_critic_td_loss": float(np.mean(critic_losses)),
            "offline_cql_penalty": float(np.mean(cql_losses)),
            "offline_actor_loss": float(np.mean(actor_losses))
        }

    def anchor_offline_policy(self):
        """Copies offline pretrained actors to anchor networks to prevent online catastrophic forgetting."""
        for i in range(self.n_agents):
            self.offline_anchor_actors[i].load_state_dict(self.actors[i].state_dict())
            for param in self.offline_anchor_actors[i].parameters():
                param.requires_grad = False
        logger.info("Anchored offline pretrained policies for conservative online fine-tuning.")

    def select_actions(
        self,
        obs_n: List[np.ndarray],
        action_masks_n: List[np.ndarray],
        deterministic: bool = False
    ) -> List[int]:
        """Decentralized execution with hard action masking."""
        actions = []
        with torch.no_grad():
            for i in range(len(obs_n)):
                actor_idx = i % self.n_agents
                o = torch.as_tensor(obs_n[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                m = torch.as_tensor(action_masks_n[i], dtype=torch.bool, device=self.device).unsqueeze(0)
                _, act_idx = self.actors[actor_idx].sample_action(o, m, deterministic=deterministic)
                act = int(act_idx.item())

                assert action_masks_n[i][act], f"CRITICAL: Offline-to-Online Agent {i} selected masked invalid action {act}"
                actions.append(act)
        return actions

    def predict(
        self,
        obs_n: List[np.ndarray],
        action_masks_n: List[np.ndarray],
        deterministic: bool = True
    ) -> List[int]:
        return self.select_actions(obs_n, action_masks_n, deterministic=deterministic)

    def online_finetune_step(self, batch_size: int = 128) -> Dict[str, float]:
        """Online fine-tuning with conservative policy regularization toward offline anchor."""
        if len(self.replay_buffer) < batch_size:
            return {}

        batch = self.replay_buffer.sample(batch_size)
        state = torch.as_tensor(batch["state"], device=self.device)
        next_state = torch.as_tensor(batch["next_state"], device=self.device)

        obs_t = [torch.as_tensor(batch["obs_n"][i], device=self.device) for i in range(self.n_agents)]
        next_obs_t = [torch.as_tensor(batch["next_obs_n"][i], device=self.device) for i in range(self.n_agents)]
        acts_t = [torch.as_tensor(batch["actions_n"][i], device=self.device) for i in range(self.n_agents)]
        rewards_t = [torch.as_tensor(batch["rewards_n"][i], device=self.device).unsqueeze(-1) for i in range(self.n_agents)]
        dones_t = [torch.as_tensor(batch["dones_n"][i], device=self.device).unsqueeze(-1) for i in range(self.n_agents)]
        masks_t = [torch.as_tensor(batch["masks_n"][i], device=self.device) for i in range(self.n_agents)]
        next_masks_t = [torch.as_tensor(batch["next_masks_n"][i], device=self.device) for i in range(self.n_agents)]

        one_hot_acts = [F.one_hot(acts_t[i], num_classes=self.act_dim).float() for i in range(self.n_agents)]
        joint_actions = torch.cat(one_hot_acts, dim=-1)

        # 1. Critic Update
        critic_losses = []
        with torch.no_grad():
            next_one_hots = []
            for i in range(self.n_agents):
                act_i, _ = self.target_actors[i].sample_action(next_obs_t[i], next_masks_t[i], deterministic=True)
                next_one_hots.append(act_i)
            next_joint_acts = torch.cat(next_one_hots, dim=-1)

        for i in range(self.n_agents):
            with torch.no_grad():
                target_q = self.target_critics[i](next_state, next_joint_acts)
                y = rewards_t[i] + self.gamma * (1.0 - dones_t[i]) * target_q

            current_q = self.critics[i](state, joint_actions)
            c_loss = F.mse_loss(current_q, y)

            self.critic_optimizers[i].zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_optimizers[i].step()
            critic_losses.append(c_loss.item())

        # 2. Actor Update with KL Anchor Regularization
        actor_losses = []
        kl_losses = []
        for i in range(self.n_agents):
            act_i, _ = self.actors[i].sample_action(obs_t[i], masks_t[i], deterministic=False)
            current_acts = []
            for j in range(self.n_agents):
                current_acts.append(act_i if j == i else one_hot_acts[j].detach())
            joint_acts = torch.cat(current_acts, dim=-1)

            actor_q = self.critics[i](state, joint_acts)
            rl_loss = -torch.mean(actor_q)

            # Anchor regularization: keep actor close to offline pretrained anchor
            with torch.no_grad():
                anchor_logits = self.offline_anchor_actors[i](obs_t[i], masks_t[i])
                anchor_dist = F.softmax(anchor_logits, dim=-1)
            current_logits = self.actors[i](obs_t[i], masks_t[i])
            current_log_dist = F.log_softmax(current_logits, dim=-1)
            kl_anchor = F.kl_div(current_log_dist, anchor_dist, reduction="batchmean")

            total_a_loss = rl_loss + self.policy_reg_weight * kl_anchor

            self.actor_optimizers[i].zero_grad()
            total_a_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()
            actor_losses.append(total_a_loss.item())
            kl_losses.append(kl_anchor.item())

        # Target updates
        for i in range(self.n_agents):
            for param, target_param in zip(self.actors[i].parameters(), self.target_actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for param, target_param in zip(self.critics[i].parameters(), self.target_critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "online_critic_loss": float(np.mean(critic_losses)),
            "online_actor_loss": float(np.mean(actor_losses)),
            "kl_loss": float(np.mean(kl_losses))
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics],
            "anchor_actors": [a.state_dict() for a in self.offline_anchor_actors],
            "n_agents": self.n_agents
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved Offline-to-Online checkpoint to {filepath}")

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        for i, a in enumerate(self.actors):
            a.load_state_dict(checkpoint["actors"][i])
            self.target_actors[i].load_state_dict(checkpoint["actors"][i])
        for i, c in enumerate(self.critics):
            c.load_state_dict(checkpoint["critics"][i])
            self.target_critics[i].load_state_dict(checkpoint["critics"][i])
        if "anchor_actors" in checkpoint:
            for i, a in enumerate(self.offline_anchor_actors):
                a.load_state_dict(checkpoint["anchor_actors"][i])
        logger.info(f"Loaded Offline-to-Online checkpoint from {filepath}")
