"""Model 2: RS-MADDPG (Refined Multi-Agent Deep Deterministic Policy Gradient)
Refinements over base MADDPG:
  1. Multi-Agent Prioritized Experience Replay (MA-PER)
  2. Relational Graph-Aware Centralized Critic (GARC) with Multi-Head Inter-Agent Attention
  3. Potential-Based Spatial Hazard Barrier Shaping (SHB-PS)
"""

import os
import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logger import setup_logger
from src.agents.marl_six.maddpg_agent import DiscreteActor

logger = setup_logger("RS-MADDPG")


class RelationalGraphAwareCritic(nn.Module):
    """Centralized Critic with multi-head relational attention over agent topological states."""

    def __init__(self, obs_dim: int = 34, act_dim: int = 4, n_agents: int = 4, hidden_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        # Agent node encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Multi-head attention across agents
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)

        # Global readout
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * n_agents + 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, obs_n: List[torch.Tensor], joint_actions: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        # Split joint actions
        acts_split = torch.split(joint_actions, self.act_dim, dim=-1)

        # Encode each agent node: [batch, n_agents, hidden_dim]
        agent_nodes = []
        for i in range(self.n_agents):
            node_in = torch.cat([obs_n[i], acts_split[i]], dim=-1)
            agent_nodes.append(self.agent_encoder(node_in))

        nodes_tensor = torch.stack(agent_nodes, dim=1)  # [batch, n_agents, hidden_dim]

        # Inter-agent attention
        attn_out, _ = self.mha(nodes_tensor, nodes_tensor, nodes_tensor)  # [batch, n_agents, hidden_dim]
        attn_flat = attn_out.reshape(batch_size, -1)

        # Global context features (last 8 features of global state)
        global_ctx = state[:, -8:]
        x = torch.cat([attn_flat, global_ctx], dim=-1)
        return self.value_head(x)


class PrioritizedReplayBuffer:
    """Multi-Agent Prioritized Experience Replay (MA-PER)."""

    def __init__(self, max_size: int = 50000, alpha: float = 0.6):
        self.max_size = max_size
        self.alpha = alpha
        self.buffer: List[Dict[str, Any]] = []
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            transition = args[0]
            priority = kwargs.get("priority", None)
        elif len(args) >= 9:
            transition = {
                "obs_n": [o.copy() for o in args[0]],
                "state": args[1].copy(),
                "actions_n": list(args[2]),
                "rewards_n": list(args[3]),
                "next_obs_n": [o.copy() for o in args[4]],
                "next_state": args[5].copy(),
                "dones_n": list(args[6]),
                "masks_n": [m.copy() for m in args[7]],
                "next_masks_n": [m.copy() for m in args[8]]
            }
            priority = kwargs.get("priority", None)
        else:
            raise ValueError(f"Invalid arguments to PrioritizedReplayBuffer.add: {len(args)} args provided")

        max_prio = self.priorities.max() if self.size > 0 else 1.0
        prio = priority if priority is not None else max_prio

        if self.size < self.max_size:
            self.buffer.append(transition)
            self.priorities[self.size] = prio
            self.size += 1
        else:
            self.buffer[self.ptr] = transition
            self.priorities[self.ptr] = prio
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("Buffer is empty")

        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Importance sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        collated = {
            "obs_n": [np.array([b["obs_n"][i] for b in batch], dtype=np.float32) for i in range(len(batch[0]["obs_n"]))],
            "state": np.array([b["state"] for b in batch], dtype=np.float32),
            "actions_n": [np.array([b["actions_n"][i] for b in batch], dtype=np.int64) for i in range(len(batch[0]["actions_n"]))],
            "rewards_n": [np.array([b["rewards_n"][i] for b in batch], dtype=np.float32) for i in range(len(batch[0]["rewards_n"]))],
            "next_obs_n": [np.array([b["next_obs_n"][i] for b in batch], dtype=np.float32) for i in range(len(batch[0]["next_obs_n"]))],
            "next_state": np.array([b["next_state"] for b in batch], dtype=np.float32),
            "dones_n": [np.array([b["dones_n"][i] for b in batch], dtype=np.float32) for i in range(len(batch[0]["dones_n"]))],
            "masks_n": [np.array([b["masks_n"][i] for b in batch], dtype=bool) for i in range(len(batch[0]["masks_n"]))],
            "next_masks_n": [np.array([b["next_masks_n"][i] for b in batch], dtype=bool) for i in range(len(batch[0]["next_masks_n"]))]
        }
        return collated, indices, weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = float(np.abs(err)) + 1e-6

    def __len__(self):
        return self.size


class RSMADDPGModel:
    """Refined MADDPG (RS-MADDPG) Model with MA-PER, GARC, and SHB-PS."""

    def __init__(
        self,
        n_agents: int = 4,
        obs_dim: int = 34,
        act_dim: int = 4,
        hidden_dim: int = 128,
        actor_hidden_dim: Optional[int] = None,
        critic_hidden_dim: Optional[int] = None,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.0008,
        gamma: float = 0.98,
        tau: float = 0.005,
        gumbel_temp: float = 1.0,
        gumbel_min_temp: float = 0.2,
        gumbel_anneal_rate: float = 0.9995,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        hazard_barrier_weight: float = 2.0,
        device: Optional[str] = None
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.temperature = gumbel_temp
        self.min_temperature = gumbel_min_temp
        self.anneal_rate = gumbel_anneal_rate

        self.beta = per_beta_start
        self.beta_end = per_beta_end
        self.hazard_barrier_weight = hazard_barrier_weight

        act_hidden = actor_hidden_dim or hidden_dim
        crt_hidden = critic_hidden_dim or hidden_dim

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Actors
        self.actors = [DiscreteActor(obs_dim, act_dim, act_hidden).to(self.device) for _ in range(n_agents)]
        self.target_actors = [copy.deepcopy(a).to(self.device) for a in self.actors]

        # Relational Graph-Aware Critics
        self.critics = [RelationalGraphAwareCritic(obs_dim, act_dim, n_agents, crt_hidden).to(self.device) for _ in range(n_agents)]
        self.target_critics = [copy.deepcopy(c).to(self.device) for c in self.critics]

        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=actor_lr) for a in self.actors]
        self.critic_optimizers = [torch.optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics]

        self.replay_buffer = PrioritizedReplayBuffer(max_size=50000, alpha=per_alpha)
        self.buffer = self.replay_buffer

    def compute_hazard_barrier_shaping(self, curr_obs: np.ndarray, next_obs: np.ndarray) -> float:
        """Admissible Potential-based Hazard Barrier Shaping: gamma * Phi(s') - Phi(s)."""
        # Feature 5 is curr_haz in the 34-dim vector
        h_curr = float(curr_obs[5])
        h_next = float(next_obs[5])

        phi_curr = -self.hazard_barrier_weight * math.log(1.0 + h_curr / max(0.01, 1.01 - h_curr))
        phi_next = -self.hazard_barrier_weight * math.log(1.0 + h_next / max(0.01, 1.01 - h_next))

        return float(self.gamma * phi_next - phi_curr)

    def store_transition(
        self,
        obs_n: List[np.ndarray],
        state: np.ndarray,
        actions_n: List[int],
        rewards_n: List[float],
        next_obs_n: List[np.ndarray],
        next_state: np.ndarray,
        dones_n: List[bool],
        masks_n: List[np.ndarray],
        next_masks_n: List[np.ndarray]
    ):
        """Applies spatial hazard potential shaping to rewards before adding to prioritized buffer."""
        shaped_rewards = []
        for i in range(self.n_agents):
            f_shaping = self.compute_hazard_barrier_shaping(obs_n[i], next_obs_n[i])
            shaped_rewards.append(rewards_n[i] + f_shaping)

        transition = {
            "obs_n": [o.copy() for o in obs_n],
            "state": state.copy(),
            "actions_n": list(actions_n),
            "rewards_n": shaped_rewards,
            "next_obs_n": [o.copy() for o in next_obs_n],
            "next_state": next_state.copy(),
            "dones_n": list(dones_n),
            "masks_n": [m.copy() for m in masks_n],
            "next_masks_n": [m.copy() for m in next_masks_n]
        }
        self.replay_buffer.add(transition)

    def select_actions(
        self,
        obs_n: List[np.ndarray],
        action_masks_n: List[np.ndarray],
        deterministic: bool = False,
        explore: Optional[bool] = None
    ) -> List[int]:
        """Decentralized execution with hard action masking."""
        if explore is not None:
            deterministic = not explore
        actions = []
        with torch.no_grad():
            for i in range(self.n_agents):
                o = torch.as_tensor(obs_n[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                m = torch.as_tensor(action_masks_n[i], dtype=torch.bool, device=self.device).unsqueeze(0)
                _, act_idx = self.actors[i].sample_action(o, m, temperature=self.temperature, deterministic=deterministic)
                act = int(act_idx.item())

                assert action_masks_n[i][act], f"CRITICAL: RS-MADDPG Agent {i} selected masked invalid action {act}"
                actions.append(act)
        return actions

    def predict(
        self,
        obs_n: List[np.ndarray],
        action_masks_n: List[np.ndarray],
        deterministic: bool = True
    ) -> List[int]:
        return self.select_actions(obs_n, action_masks_n, deterministic=deterministic)

    def update(self, batch_size: int = 128) -> Dict[str, float]:
        """Performs RS-MADDPG update with importance sampling and relational attention."""
        if len(self.replay_buffer) < batch_size:
            return {}

        batch, indices, is_weights = self.replay_buffer.sample(batch_size, beta=self.beta)
        weights_t = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(-1)

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

        # 1. Update Critics with Relational Attention & PER
        critic_losses = []
        td_errors = []

        with torch.no_grad():
            next_one_hots = []
            for i in range(self.n_agents):
                target_act, _ = self.target_actors[i].sample_action(next_obs_t[i], next_masks_t[i], temperature=self.temperature, deterministic=False)
                next_one_hots.append(target_act)
            next_joint_actions = torch.cat(next_one_hots, dim=-1)

        for i in range(self.n_agents):
            with torch.no_grad():
                target_q = self.target_critics[i](next_state, next_obs_t, next_joint_actions)
                y = rewards_t[i] + self.gamma * (1.0 - dones_t[i]) * target_q

            current_q = self.critics[i](state, obs_t, joint_actions)
            td_err = (current_q - y).squeeze(-1).detach().cpu().numpy()
            td_errors.append(np.abs(td_err))

            c_loss = torch.mean(weights_t * ((current_q - y) ** 2))

            self.critic_optimizers[i].zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_optimizers[i].step()
            critic_losses.append(c_loss.item())

        # Update PER priorities using mean absolute TD error across agents
        mean_td_err = np.mean(td_errors, axis=0)
        self.replay_buffer.update_priorities(indices, mean_td_err)

        # 2. Update Actors
        actor_losses = []
        for i in range(self.n_agents):
            current_acts_for_grad = []
            for j in range(self.n_agents):
                if j == i:
                    act_i, _ = self.actors[i].sample_action(obs_t[i], masks_t[i], temperature=self.temperature, deterministic=False)
                    current_acts_for_grad.append(act_i)
                else:
                    current_acts_for_grad.append(one_hot_acts[j].detach())

            joint_acts_for_grad = torch.cat(current_acts_for_grad, dim=-1)
            actor_q = self.critics[i](state, obs_t, joint_acts_for_grad)
            a_loss = -torch.mean(actor_q)

            self.actor_optimizers[i].zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()
            actor_losses.append(a_loss.item())

        # 3. Soft Target Updates
        for i in range(self.n_agents):
            for param, target_param in zip(self.actors[i].parameters(), self.target_actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for param, target_param in zip(self.critics[i].parameters(), self.target_critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # Anneal parameters
        self.temperature = max(self.min_temperature, self.temperature * self.anneal_rate)
        self.beta = min(self.beta_end, self.beta + 0.0002)

        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses)),
            "gumbel_temperature": self.temperature,
            "per_beta": self.beta
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics],
            "temperature": self.temperature,
            "beta": self.beta,
            "n_agents": self.n_agents
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved RS-MADDPG checkpoint to {filepath}")

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        for i, a in enumerate(self.actors):
            a.load_state_dict(checkpoint["actors"][i])
            self.target_actors[i].load_state_dict(checkpoint["actors"][i])
        for i, c in enumerate(self.critics):
            c.load_state_dict(checkpoint["critics"][i])
            self.target_critics[i].load_state_dict(checkpoint["critics"][i])
        self.temperature = checkpoint.get("temperature", 0.2)
        self.beta = checkpoint.get("beta", 1.0)
        logger.info(f"Loaded RS-MADDPG checkpoint from {filepath}")
