"""Model 1: MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
Discrete-compatible action representation via Gumbel-Softmax with hard action masking.
Centralized Training with Decentralized Execution (CTDE).
"""

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logger import setup_logger

logger = setup_logger("MADDPG")


class DiscreteActor(nn.Module):
    """Decentralized actor policy network with action masking."""

    def __init__(self, obs_dim: int = 34, act_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.net(obs)
        if mask is not None:
            # Mask out invalid transitions with -1e9
            logits = torch.where(mask, logits, torch.tensor(-1e9, device=logits.device, dtype=logits.dtype))
        return logits

    def sample_action(
        self,
        obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns one-hot action tensor and discrete action indices."""
        logits = self.forward(obs, mask)
        if deterministic:
            act_idx = torch.argmax(logits, dim=-1)
            one_hot = F.one_hot(act_idx, num_classes=logits.shape[-1]).float()
            return one_hot, act_idx
        else:
            # Gumbel-Softmax sampling
            relaxed_action = F.gumbel_softmax(logits, tau=max(0.1, temperature), hard=True)
            act_idx = torch.argmax(relaxed_action, dim=-1)
            return relaxed_action, act_idx


class CentralizedCritic(nn.Module):
    """Centralized Critic conditioned on global state and joint actions."""

    def __init__(self, global_state_dim: int, total_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        in_dim = global_state_dim + total_action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, joint_actions], dim=-1)
        return self.net(x)


class MultiAgentReplayBuffer:
    """Joint transition replay buffer for CTDE algorithms."""

    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.buffer: List[Dict[str, Any]] = []
        self.ptr = 0

    def add(
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
        transition = {
            "obs_n": [o.copy() for o in obs_n],
            "state": state.copy(),
            "actions_n": list(actions_n),
            "rewards_n": list(rewards_n),
            "next_obs_n": [o.copy() for o in next_obs_n],
            "next_state": next_state.copy(),
            "dones_n": list(dones_n),
            "masks_n": [m.copy() for m in masks_n],
            "next_masks_n": [m.copy() for m in next_masks_n]
        }
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.ptr] = transition
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size: int) -> Dict[str, Any]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return {
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

    def __len__(self):
        return len(self.buffer)


class MADDPGModel:
    """MADDPG implementation with hard action masking and Centralized Critic."""

    def __init__(
        self,
        n_agents: int = 4,
        obs_dim: int = 34,
        act_dim: int = 4,
        hidden_dim: int = 128,
        actor_hidden_dim: Optional[int] = None,
        critic_hidden_dim: Optional[int] = None,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.001,
        gamma: float = 0.98,
        tau: float = 0.005,
        gumbel_temp: float = 1.0,
        gumbel_min_temp: float = 0.2,
        gumbel_anneal_rate: float = 0.9995,
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

        act_hidden = actor_hidden_dim or hidden_dim
        crt_hidden = critic_hidden_dim or hidden_dim

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        global_state_dim = obs_dim * n_agents + 8
        total_act_dim = act_dim * n_agents

        # Distinct decentralized actors and centralized critics per agent
        self.actors = [DiscreteActor(obs_dim, act_dim, act_hidden).to(self.device) for _ in range(n_agents)]
        self.target_actors = [copy.deepcopy(a).to(self.device) for a in self.actors]

        self.critics = [CentralizedCritic(global_state_dim, total_act_dim, crt_hidden).to(self.device) for _ in range(n_agents)]
        self.target_critics = [copy.deepcopy(c).to(self.device) for c in self.critics]

        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=actor_lr) for a in self.actors]
        self.critic_optimizers = [torch.optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics]

        self.replay_buffer = MultiAgentReplayBuffer(max_size=50000)
        self.buffer = self.replay_buffer

    def select_actions(
        self,
        obs_n: List[np.ndarray],
        action_masks_n: List[np.ndarray],
        deterministic: bool = False,
        explore: Optional[bool] = None
    ) -> List[int]:
        """Decentralized inference for all agents with hard action masking."""
        if explore is not None:
            deterministic = not explore
        actions = []
        with torch.no_grad():
            for i in range(self.n_agents):
                o = torch.as_tensor(obs_n[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                m = torch.as_tensor(action_masks_n[i], dtype=torch.bool, device=self.device).unsqueeze(0)
                _, act_idx = self.actors[i].sample_action(o, m, temperature=self.temperature, deterministic=deterministic)
                act = int(act_idx.item())

                # Hard action masking assertion
                assert action_masks_n[i][act], f"CRITICAL: MADDPG Agent {i} selected masked invalid action {act}"
                actions.append(act)
        return actions

    def predict(
        self,
        obs_n: List[np.ndarray],
        action_masks_n: List[np.ndarray],
        deterministic: bool = True
    ) -> List[int]:
        """Standard evaluation prediction interface."""
        return self.select_actions(obs_n, action_masks_n, deterministic=deterministic)

    def update(self, batch_size: int = 128) -> Dict[str, float]:
        """Performs one CTDE update step for actors and critics."""
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

        # One-hot encoding of current actions
        one_hot_acts = [F.one_hot(acts_t[i], num_classes=self.act_dim).float() for i in range(self.n_agents)]
        joint_actions = torch.cat(one_hot_acts, dim=-1)

        # 1. Update Critics
        critic_losses = []
        with torch.no_grad():
            next_one_hots = []
            for i in range(self.n_agents):
                target_act, _ = self.target_actors[i].sample_action(next_obs_t[i], next_masks_t[i], temperature=self.temperature, deterministic=False)
                next_one_hots.append(target_act)
            next_joint_actions = torch.cat(next_one_hots, dim=-1)

        for i in range(self.n_agents):
            with torch.no_grad():
                target_q = self.target_critics[i](next_state, next_joint_actions)
                y = rewards_t[i] + self.gamma * (1.0 - dones_t[i]) * target_q

            current_q = self.critics[i](state, joint_actions)
            c_loss = F.mse_loss(current_q, y)

            self.critic_optimizers[i].zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_optimizers[i].step()
            critic_losses.append(c_loss.item())

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
            actor_q = self.critics[i](state, joint_acts_for_grad)
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

        # Anneal Gumbel temperature
        self.temperature = max(self.min_temperature, self.temperature * self.anneal_rate)

        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses)),
            "gumbel_temperature": self.temperature
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics],
            "temperature": self.temperature,
            "n_agents": self.n_agents
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved MADDPG checkpoint to {filepath}")

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        for i, a in enumerate(self.actors):
            a.load_state_dict(checkpoint["actors"][i])
            self.target_actors[i].load_state_dict(checkpoint["actors"][i])
        for i, c in enumerate(self.critics):
            c.load_state_dict(checkpoint["critics"][i])
            self.target_critics[i].load_state_dict(checkpoint["critics"][i])
        self.temperature = checkpoint.get("temperature", 0.2)
        logger.info(f"Loaded MADDPG checkpoint from {filepath}")
