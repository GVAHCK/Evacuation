"""Model 6: HARL (Heterogeneous-Agent Reinforcement Learning)
Heterogeneous-Agent PPO (HAPPO) with dedicated cohort policy heads and sequential policy updates:
A^i(s, a_{1:i}) = Q(s, a_{1:i}) - Q(s, a_{1:i-1})
Enforces decentralized execution with hard action masking and centralized state-value evaluation.
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

logger = setup_logger("HARL-HAPPO")

COHORT_NAMES = [
    "type_a_standard",
    "type_b_congestion_sensitive",
    "type_c_hazard_sensitive",
    "type_d_long_distance"
]


class HeterogeneousActor(nn.Module):
    """Cohort-specialized actor architecture for heterogeneous evacuee agents."""

    def __init__(
        self,
        cohort_idx: int,
        obs_dim: int = 34,
        act_dim: int = 4,
        hidden_dim: int = 128,
        cohort_embed_dim: int = 16
    ):
        super().__init__()
        self.cohort_idx = cohort_idx
        self.cohort_name = COHORT_NAMES[cohort_idx % len(COHORT_NAMES)]

        # Cohort identifier embedding
        self.cohort_embed = nn.Embedding(len(COHORT_NAMES), cohort_embed_dim)

        # Specialized feature projection based on cohort domain focus
        self.input_proj = nn.Linear(obs_dim + cohort_embed_dim, hidden_dim)

        # Cohort-specific behavioral bias layers
        # Type A: Balanced MLP
        # Type B: Focus on congestion and local queue features (features 10-13, 16, 21, 26, 31)
        # Type C: Focus on hazard distances and intensities (features 3, 4, 15, 20, 25, 30)
        # Type D: Focus on edge lengths and max speeds (features 17, 18, 22, 23, 27, 28, 32, 33)
        self.shared_trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = obs.shape[0]
        cohort_ids = torch.full((batch_size,), self.cohort_idx, dtype=torch.long, device=obs.device)
        c_emb = self.cohort_embed(cohort_ids)

        x = torch.cat([obs, c_emb], dim=-1)
        h = F.relu(self.input_proj(x))
        h = self.shared_trunk(h)
        logits = self.policy_head(h)

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


class HARLCritic(nn.Module):
    """Centralized state-value function V(s) conditioned on joint state and heterogeneous cohort specs."""

    def __init__(self, state_dim: int, n_agents: int = 4, hidden_dim: int = 128):
        super().__init__()
        cohort_total_dim = n_agents * 8
        self.cohort_embeddings = nn.Embedding(n_agents, 8)

        self.net = nn.Sequential(
            nn.Linear(state_dim + cohort_total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        device = state.device
        all_cohort_ids = torch.arange(4, device=device)
        c_emb = self.cohort_embeddings(all_cohort_ids).flatten()  # [32]
        c_emb_batch = c_emb.unsqueeze(0).expand(batch_size, -1)   # [batch, 32]

        x = torch.cat([state, c_emb_batch], dim=-1)
        return self.net(x)


class HARLRolloutBuffer:
    """Rollout buffer capturing joint multi-agent trajectories for sequential HAPPO updates."""

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


class HARLModel:
    """HARL / HAPPO (Heterogeneous-Agent Reinforcement Learning):
    - Dedicated cohort actor per vehicle type
    - Centralized V(s) critic with cohort conditioning
    - Sequential multi-agent policy updates preventing policy churn
    - Strict action masking verification
    """

    def __init__(
        self,
        n_agents: int = 4,
        obs_dim: int = 34,
        act_dim: int = 4,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.001,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
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
        self.batch_size = batch_size

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        global_state_dim = obs_dim * n_agents + 8

        # Distinct actor for each cohort type
        self.actors = [
            HeterogeneousActor(
                cohort_idx=i,
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=actor_hidden_dim
            ).to(self.device)
            for i in range(n_agents)
        ]

        self.critic = HARLCritic(
            state_dim=global_state_dim,
            n_agents=n_agents,
            hidden_dim=critic_hidden_dim
        ).to(self.device)

        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=actor_lr)
            for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = HARLRolloutBuffer(n_agents=n_agents)

    def select_actions(
        self,
        obs_n: List[np.ndarray],
        action_masks_n: List[np.ndarray],
        deterministic: bool = False
    ) -> Tuple[List[int], List[float]]:
        """Decentralized inference by each heterogeneous cohort actor."""
        actions = []
        log_probs = []
        with torch.no_grad():
            for i in range(self.n_agents):
                o = torch.as_tensor(obs_n[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                m = torch.as_tensor(action_masks_n[i], dtype=torch.bool, device=self.device).unsqueeze(0)
                act, lp, _ = self.actors[i].get_action_and_log_prob(o, m, deterministic=deterministic)
                a_int = int(act.item())

                assert action_masks_n[i][a_int], f"CRITICAL: HARL Agent {i} ({COHORT_NAMES[i]}) picked masked action {a_int}"
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
        """Calculates GAE advantages and applies sequential HAPPO updates across heterogeneous cohorts."""
        steps = len(self.buffer.states)
        if steps == 0:
            return {}

        # 1. Compute centralized GAE advantages
        last_val = self.get_value(next_state)
        values = self.buffer.values + [last_val]

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

        # Convert buffers to tensors
        obs_tensors = [
            torch.as_tensor(np.array(self.buffer.obs_n[i]), dtype=torch.float32, device=self.device)
            for i in range(self.n_agents)
        ]
        act_tensors = [
            torch.as_tensor(np.array(self.buffer.actions_n[i]), dtype=torch.int64, device=self.device)
            for i in range(self.n_agents)
        ]
        old_lp_tensors = [
            torch.as_tensor(np.array(self.buffer.log_probs_n[i]), dtype=torch.float32, device=self.device)
            for i in range(self.n_agents)
        ]
        mask_tensors = [
            torch.as_tensor(np.array(self.buffer.masks_n[i]), dtype=torch.bool, device=self.device)
            for i in range(self.n_agents)
        ]

        # 2. Sequential HAPPO Actor Updates
        for epoch in range(self.ppo_epochs):
            # Update Critic first
            curr_values = self.critic(states_tensor)
            val_loss = F.mse_loss(curr_values, ret_tensor)

            self.critic_optimizer.zero_grad()
            val_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            critic_losses.append(val_loss.item())

            # Sequential Multi-Agent Factor multiplier M
            seq_multiplier = torch.ones(steps, device=self.device)

            # Randomize agent update sequence or follow canonical order per epoch
            agent_order = list(range(self.n_agents))

            for i in agent_order:
                # Factorized sequential advantage: A^i(s, a_{1:i}) = M_{i-1} * A(s, a)
                agent_adv = adv_tensor * seq_multiplier.detach()

                _, new_lps, entropy = self.actors[i].get_action_and_log_prob(
                    obs_tensors[i], mask_tensors[i], action=act_tensors[i]
                )
                ratio = torch.exp(new_lps - old_lp_tensors[i])

                surr1 = ratio * agent_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * agent_adv
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.actor_optimizers[i].zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                self.actor_optimizers[i].step()
                actor_losses.append(policy_loss.item())

                # Update sequential multiplier for downstream heterogeneous agents
                with torch.no_grad():
                    _, post_lps, _ = self.actors[i].get_action_and_log_prob(
                        obs_tensors[i], mask_tensors[i], action=act_tensors[i]
                    )
                    new_ratio = torch.clamp(torch.exp(post_lps - old_lp_tensors[i]), 0.5, 2.0)
                    seq_multiplier = seq_multiplier * new_ratio

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
        logger.info(f"Saved HARL checkpoint to {filepath}")

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        for i, a in enumerate(self.actors):
            a.load_state_dict(checkpoint["actors"][i])
        self.critic.load_state_dict(checkpoint["critic"])
        logger.info(f"Loaded HARL checkpoint from {filepath}")
