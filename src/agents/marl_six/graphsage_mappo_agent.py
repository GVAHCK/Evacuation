"""Model 5: GraphSAGE-MAPPO
Inductive GraphSAGE neighbor aggregation layers paired with Multi-Agent PPO (MAPPO).
Decentralized Actor + Centralized Value Function Critic with PPO clipping and GAE.
"""

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logger import setup_logger

logger = setup_logger("GraphSAGE-MAPPO")


class GraphSAGELayer(nn.Module):
    """Inductive GraphSAGE layer with mean neighbor aggregation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim, bias=False)
        self.w_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()

    def forward(self, root_feat: torch.Tensor, neigh_feats: torch.Tensor) -> torch.Tensor:
        """root_feat: [batch, in_dim]
        neigh_feats: [batch, max_neighbors, in_dim]
        """
        mean_neigh = torch.mean(neigh_feats, dim=1)  # [batch, in_dim]
        out = self.w_self(root_feat) + self.w_neigh(mean_neigh)
        return self.act(self.norm(out))


class GraphSAGENet(nn.Module):
    """2-layer GraphSAGE module processing candidate road edges."""

    def __init__(self, in_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.sage1 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, hidden_dim)

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        """edge_feats: [batch, 4, 5]"""
        h = F.relu(self.proj(edge_feats))  # [batch, 4, hidden]
        root = torch.mean(h, dim=1)         # [batch, hidden]
        h1 = self.sage1(root, h)            # [batch, hidden]
        # Repeat root for layer 2
        h_all = h + h1.unsqueeze(1)
        h2 = self.sage2(h1, h_all)          # [batch, hidden]
        return h2


class MAPPOActor(nn.Module):
    """Decentralized MAPPO Actor with GraphSAGE encoder and hard action masking."""

    def __init__(self, obs_dim: int = 34, hidden_dim: int = 64, act_dim: int = 4):
        super().__init__()
        self.sage = GraphSAGENet(in_dim=5, hidden_dim=hidden_dim)
        self.actor_mlp = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract edge features from obs
        edge_feats = obs[:, 14:].reshape(obs.shape[0], 4, 5)
        sage_emb = self.sage(edge_feats)
        combined = torch.cat([obs, sage_emb], dim=-1)
        logits = self.actor_mlp(combined)

        if mask is not None:
            logits = torch.where(mask, logits, torch.tensor(-1e9, device=logits.device, dtype=logits.dtype))
        return logits

    def get_action_and_log_prob(
        self,
        obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(obs, mask)
        dist = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


class MAPPOCritic(nn.Module):
    """Centralized MAPPO Critic evaluating joint state values V(s)."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class RolloutBuffer:
    """Stores on-policy multi-agent rollouts for MAPPO PPO updates."""

    def __init__(self, n_agents: int = 4):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.obs_n: List[List[np.ndarray]] = [[] for _ in range(self.n_agents)]
        self.states: List[np.ndarray] = []
        self.actions_n: List[List[int]] = [[] for _ in range(self.n_agents)]
        self.log_probs_n: List[List[float]] = [[] for _ in range(self.n_agents)]
        self.rewards_n: List[List[float]] = [[] for _ in range(self.n_agents)]
        self.dones_n: List[List[bool]] = [[] for _ in range(self.n_agents)]
        self.masks_n: List[List[np.ndarray]] = [[] for _ in range(self.n_agents)]
        self.values: List[float] = []

    def add(
        self,
        obs_n: List[np.ndarray],
        state: np.ndarray,
        actions_n: List[int],
        log_probs_n: List[float],
        rewards_n: List[float],
        dones_n: List[bool],
        masks_n: List[np.ndarray],
        value: float
    ):
        for i in range(self.n_agents):
            self.obs_n[i].append(obs_n[i])
            self.actions_n[i].append(actions_n[i])
            self.log_probs_n[i].append(log_probs_n[i])
            self.rewards_n[i].append(rewards_n[i])
            self.dones_n[i].append(dones_n[i])
            self.masks_n[i].append(masks_n[i])
        self.states.append(state)
        self.values.append(value)


class GraphSAGEMAPPOModel:
    """GraphSAGE-MAPPO: Inductive graph representation with Multi-Agent PPO."""

    def __init__(
        self,
        n_agents: int = 4,
        obs_dim: int = 34,
        act_dim: int = 4,
        graphsage_hidden_dim: int = 64,
        mlp_hidden_dim: int = 128,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.001,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        device: Optional[str] = None
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        global_state_dim = obs_dim * n_agents + 8

        self.actors = [MAPPOActor(obs_dim, graphsage_hidden_dim, act_dim).to(self.device) for _ in range(n_agents)]
        self.critic = MAPPOCritic(global_state_dim, mlp_hidden_dim).to(self.device)

        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=actor_lr) for a in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = RolloutBuffer(n_agents=n_agents)

    def select_actions(
        self,
        obs_n: List[np.ndarray],
        action_masks_n: List[np.ndarray],
        deterministic: bool = False
    ) -> Tuple[List[int], List[float]]:
        """Decentralized inference returning discrete actions and their log-probabilities."""
        actions = []
        log_probs = []
        with torch.no_grad():
            for i in range(self.n_agents):
                o = torch.as_tensor(obs_n[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                m = torch.as_tensor(action_masks_n[i], dtype=torch.bool, device=self.device).unsqueeze(0)
                act, lp, _ = self.actors[i].get_action_and_log_prob(o, m, deterministic=deterministic)
                a_int = int(act.item())

                assert action_masks_n[i][a_int], f"CRITICAL: GraphSAGE-MAPPO Agent {i} picked masked action {a_int}"
                actions.append(a_int)
                log_probs.append(float(lp.item()))

        return actions, log_probs

    def predict(
        self,
        obs_n: List[np.ndarray],
        action_masks_n: List[np.ndarray],
        deterministic: bool = True
    ) -> List[int]:
        acts, _ = self.select_actions(obs_n, action_masks_n, deterministic=deterministic)
        return acts

    def get_value(self, state: np.ndarray) -> float:
        with torch.no_grad():
            s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            val = self.critic(s_t)
            return float(val.item())

    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        """Calculates GAE advantages and executes PPO clipping epochs."""
        steps = len(self.buffer.states)
        if steps == 0:
            return {}

        # Compute GAE and returns
        last_val = self.get_value(next_state)
        values = self.buffer.values + [last_val]

        # Shared team reward
        team_rewards = [np.mean([self.buffer.rewards_n[i][t] for i in range(self.n_agents)]) for t in range(steps)]
        team_dones = [any(self.buffer.dones_n[i][t] for i in range(self.n_agents)) for t in range(steps)]

        advantages = np.zeros(steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(steps)):
            non_terminal = 1.0 - float(team_dones[t])
            delta = team_rewards[t] + self.gamma * values[t + 1] * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.buffer.values)
        adv_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        ret_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(-1)
        states_tensor = torch.as_tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)

        actor_losses = []
        critic_losses = []

        for _ in range(self.ppo_epochs):
            # 1. Update Critic
            curr_values = self.critic(states_tensor)
            val_loss = F.mse_loss(curr_values, ret_tensor)

            self.critic_optimizer.zero_grad()
            val_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            critic_losses.append(val_loss.item())

            # 2. Update Actors
            for i in range(self.n_agents):
                obs_i = torch.as_tensor(np.array(self.buffer.obs_n[i]), dtype=torch.float32, device=self.device)
                acts_i = torch.as_tensor(np.array(self.buffer.actions_n[i]), dtype=torch.int64, device=self.device)
                old_lps_i = torch.as_tensor(np.array(self.buffer.log_probs_n[i]), dtype=torch.float32, device=self.device)
                masks_i = torch.as_tensor(np.array(self.buffer.masks_n[i]), dtype=torch.bool, device=self.device)

                _, new_lps, entropy = self.actors[i].get_action_and_log_prob(obs_i, masks_i, action=acts_i)
                ratio = torch.exp(new_lps - old_lps_i)

                surr1 = ratio * adv_tensor
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_tensor
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.actor_optimizers[i].zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                self.actor_optimizers[i].step()
                actor_losses.append(policy_loss.item())

        self.buffer.reset()
        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses))
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "actors": [a.state_dict() for a in self.actors],
            "critic": self.critic.state_dict(),
            "n_agents": self.n_agents
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved GraphSAGE-MAPPO checkpoint to {filepath}")

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        for i, a in enumerate(self.actors):
            a.load_state_dict(checkpoint["actors"][i])
        self.critic.load_state_dict(checkpoint["critic"])
        logger.info(f"Loaded GraphSAGE-MAPPO checkpoint from {filepath}")
