"""Model 4: MA-GRL (Multi-Agent Graph Reinforcement Learning)
Graph Neural Message Passing over variable-scale road network topologies G = (V, E).
Decentralized Ego-Graph Actors + Centralized Graph Readout Critic.
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
from src.agents.marl_six.maddpg_agent import MultiAgentReplayBuffer

logger = setup_logger("MA-GRL")


class EgoGraphEncoder(nn.Module):
    """Native Ego-Graph message passing encoder over 14 root + 4x5 candidate edge features."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.root_proj = nn.Linear(14, hidden_dim)
        self.edge_proj = nn.Linear(5, hidden_dim)
        self.msg1 = nn.Linear(hidden_dim, hidden_dim)
        self.msg2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, root_feat: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        h_root = F.relu(self.root_proj(root_feat))
        h_edge = F.relu(self.edge_proj(edge_feat))

        # Edge to root aggregation
        agg_edge = torch.mean(self.msg1(h_edge), dim=1)
        h_root = self.norm(h_root + agg_edge)

        # Root to edge broadcast
        h_edge = self.norm(h_edge + self.msg2(h_root).unsqueeze(1))

        edge_mean = torch.mean(h_edge, dim=1)
        edge_max, _ = torch.max(h_edge, dim=1)
        return torch.cat([h_root, edge_mean, edge_max], dim=-1)


class MAGRLActor(nn.Module):
    """Decentralized Ego-Graph Actor Policy with hard action masking."""

    def __init__(self, node_in_dim: int = 14, edge_in_dim: int = 5, hidden_dim: int = 64, act_dim: int = 4):
        super().__init__()
        self.gnn = EgoGraphEncoder(hidden_dim=hidden_dim)
        # Readout: hidden_dim * 3 + 34 (with residual observation skip) -> 4 logits
        in_readout = hidden_dim * 3 + 34
        self.policy_head = nn.Sequential(
            nn.Linear(in_readout, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Deconstruct 34-dim observation into 14 root features and 4x5 candidate edge features
        root_feat = obs[:, :14]
        edge_feat = obs[:, 14:].reshape(obs.shape[0], 4, 5)

        g_embed = self.gnn(root_feat, edge_feat)
        combined = torch.cat([g_embed, obs], dim=-1)
        logits = self.policy_head(combined)

        if mask is not None:
            logits = torch.where(mask, logits, torch.tensor(-1e9, device=logits.device, dtype=logits.dtype))
        return logits

    def sample_action(
        self,
        obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs, mask)
        if deterministic:
            act_idx = torch.argmax(logits, dim=-1)
            one_hot = F.one_hot(act_idx, num_classes=logits.shape[-1]).float()
            return one_hot, act_idx
        else:
            relaxed_action = F.gumbel_softmax(logits, tau=max(0.1, temperature), hard=True)
            act_idx = torch.argmax(relaxed_action, dim=-1)
            return relaxed_action, act_idx


class MAGRLCritic(nn.Module):
    """Centralized Graph Critic pooling all-agent graph features."""

    def __init__(self, node_in_dim: int = 14, edge_in_dim: int = 5, hidden_dim: int = 64, n_agents: int = 4, act_dim: int = 4):
        super().__init__()
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.gnn = EgoGraphEncoder(hidden_dim=hidden_dim)
        # Global critic input: (hidden_dim * 3 + 34) * n_agents + act_dim * n_agents + 8
        in_dim = (hidden_dim * 3 + 34) * n_agents + (act_dim * n_agents) + 8
        self.critic_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor, obs_n: List[torch.Tensor], joint_actions: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        agent_g_embeds = []
        for i in range(self.n_agents):
            obs_i = obs_n[i]
            root_f = obs_i[:, :14]
            edge_f = obs_i[:, 14:].reshape(batch_size, 4, 5)
            g_emb = self.gnn(root_f, edge_f)
            comb = torch.cat([g_emb, obs_i], dim=-1)
            agent_g_embeds.append(comb)

        all_g_embeds = torch.cat(agent_g_embeds, dim=-1)
        global_ctx = state[:, -8:]
        full_in = torch.cat([all_g_embeds, joint_actions, global_ctx], dim=-1)
        return self.critic_head(full_in)


class MAGRLModel:
    """Multi-Agent Graph Reinforcement Learning Model."""

    def __init__(
        self,
        n_agents: int = 4,
        obs_dim: int = 34,
        act_dim: int = 4,
        node_in_dim: int = 14,
        edge_in_dim: int = 5,
        gnn_hidden_dim: int = 64,
        critic_hidden_dim: int = 128,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.0008,
        gamma: float = 0.98,
        tau: float = 0.005,
        device: Optional[str] = None
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.temperature = 1.0

        crt_hidden = critic_hidden_dim or gnn_hidden_dim

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.actors = [MAGRLActor(node_in_dim, edge_in_dim, gnn_hidden_dim, act_dim).to(self.device) for _ in range(n_agents)]
        self.target_actors = [copy.deepcopy(a).to(self.device) for a in self.actors]

        self.critics = [MAGRLCritic(node_in_dim, edge_in_dim, crt_hidden, n_agents, act_dim).to(self.device) for _ in range(n_agents)]
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
        """Decentralized ego-graph inference with hard action masking."""
        if explore is not None:
            deterministic = not explore
        actions = []
        with torch.no_grad():
            for i in range(self.n_agents):
                o = torch.as_tensor(obs_n[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                m = torch.as_tensor(action_masks_n[i], dtype=torch.bool, device=self.device).unsqueeze(0)
                _, act_idx = self.actors[i].sample_action(o, m, temperature=self.temperature, deterministic=deterministic)
                act = int(act_idx.item())

                assert action_masks_n[i][act], f"CRITICAL: MA-GRL Agent {i} selected masked invalid action {act}"
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
                target_act, _ = self.target_actors[i].sample_action(next_obs_t[i], next_masks_t[i], temperature=self.temperature, deterministic=False)
                next_one_hots.append(target_act)
            next_joint_actions = torch.cat(next_one_hots, dim=-1)

        for i in range(self.n_agents):
            with torch.no_grad():
                target_q = self.target_critics[i](next_state, next_obs_t, next_joint_actions)
                y = rewards_t[i] + self.gamma * (1.0 - dones_t[i]) * target_q

            current_q = self.critics[i](state, obs_t, joint_actions)
            c_loss = F.mse_loss(current_q, y)

            self.critic_optimizers[i].zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_optimizers[i].step()
            critic_losses.append(c_loss.item())

        # 2. Actor Update
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

        self.temperature = max(0.2, self.temperature * 0.9995)

        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses))
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
        logger.info(f"Saved MA-GRL checkpoint to {filepath}")

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        for i, a in enumerate(self.actors):
            a.load_state_dict(checkpoint["actors"][i])
            self.target_actors[i].load_state_dict(checkpoint["actors"][i])
        for i, c in enumerate(self.critics):
            c.load_state_dict(checkpoint["critics"][i])
            self.target_critics[i].load_state_dict(checkpoint["critics"][i])
        self.temperature = checkpoint.get("temperature", 0.2)
        logger.info(f"Loaded MA-GRL checkpoint from {filepath}")
