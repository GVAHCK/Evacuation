"""Model 2: RecurrentMaskablePPO Agent
Temporal LSTM Actor-Critic with Action Masking and Episodic State Resets.
"""

import os
import yaml
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from torch.distributions.categorical import Categorical

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.environment.evacuation_env import EvacuationEnv

logger = setup_logger("RecurrentMaskablePPO")


class RecurrentActorCriticNet(nn.Module):
    """Recurrent Actor-Critic network with LSTM temporal memory and masked discrete action selection."""

    def __init__(
        self,
        obs_dim: int = 34,
        action_dim: int = 4,
        hidden_dim: int = 64,
        lstm_hidden_dim: int = 64
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # Feedforward MLP feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # Recurrent temporal cell
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)

        # Policy & Value heads
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_initial_state(self, batch_size: int = 1, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns zeros for initial (h_0, c_0) LSTM states."""
        return (
            torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device),
            torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        )

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        lstm_state: Tuple[torch.Tensor, torch.Tensor],
        done_seq: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Processes a sequence through the encoder and LSTM with episodic boundary resets."""
        # obs_seq: (B, T, obs_dim)
        B, T, _ = obs_seq.shape
        flat_obs = obs_seq.view(B * T, self.obs_dim)
        feats = self.encoder(flat_obs).view(B, T, -1)

        if done_seq is None or not torch.any(done_seq):
            lstm_out, next_state = self.lstm(feats, lstm_state)
        else:
            # Step by step through sequence to reset LSTM state on episode boundaries
            outputs = []
            h, c = lstm_state
            for t in range(T):
                step_feat = feats[:, t:t+1, :]
                is_done = done_seq[:, t:t+1, None]  # (B, 1, 1)
                # Reset state if previous step ended an episode
                h = h * (1.0 - is_done.permute(1, 0, 2))
                c = c * (1.0 - is_done.permute(1, 0, 2))
                step_out, (h, c) = self.lstm(step_feat, (h, c))
                outputs.append(step_out)
            lstm_out = torch.cat(outputs, dim=1)
            next_state = (h, c)

        flat_out = lstm_out.view(B * T, self.lstm_hidden_dim)
        logits = self.actor(flat_out).view(B, T, self.action_dim)
        values = self.critic(flat_out).view(B, T)
        return logits, values, next_state

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        lstm_state: Tuple[torch.Tensor, torch.Tensor],
        action_masks: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Computes action, log prob, entropy, and value for a single timestep with action masking."""
        # obs: (B, obs_dim)
        B = obs.shape[0]
        feats = self.encoder(obs).unsqueeze(1)  # (B, 1, H)
        lstm_out, next_state = self.lstm(feats, lstm_state)
        lstm_out = lstm_out.squeeze(1)          # (B, H)

        logits = self.actor(lstm_out)           # (B, 4)
        values = self.critic(lstm_out).squeeze(-1)  # (B,)

        if action_masks is not None:
            # Hard masking: set invalid actions to -1e9
            logits = torch.where(action_masks.bool(), logits, torch.full_like(logits, -1e9))

        dist = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, values, next_state


class RecurrentMaskablePPOAgent:
    """Inference-compatible Recurrent Maskable PPO Agent supporting predict() with action_masks."""

    def __init__(self, model_path: Optional[str] = None, net: Optional[RecurrentActorCriticNet] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.net = net
        self.current_lstm_state = None
        if model_path and os.path.isfile(model_path):
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """Loads model weights from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        hp = checkpoint.get("hyperparameters", {})
        self.net = RecurrentActorCriticNet(
            obs_dim=hp.get("obs_dim", 34),
            action_dim=hp.get("action_dim", 4),
            hidden_dim=hp.get("hidden_dim", 64),
            lstm_hidden_dim=hp.get("lstm_hidden_dim", 64)
        ).to(self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()

    def save(self, save_path: str, hp: Optional[Dict[str, Any]] = None) -> None:
        """Saves model weights and hyperparameters."""
        torch.save({
            "state_dict": self.net.state_dict(),
            "hyperparameters": hp or {}
        }, save_path)

    def reset_state(self, batch_size: int = 1) -> None:
        """Resets the episodic LSTM hidden state."""
        self.current_lstm_state = self.net.get_initial_state(batch_size=batch_size, device=self.device)

    def predict(
        self,
        obs: np.ndarray,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        episode_start: Optional[bool] = None,
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Predicts next action matching Stable-Baselines3 interface."""
        if state is not None:
            self.current_lstm_state = state
        elif self.current_lstm_state is None or episode_start:
            self.reset_state(batch_size=1)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        mask_t = None
        if action_masks is not None:
            mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
            if mask_t.dim() == 1:
                mask_t = mask_t.unsqueeze(0)

        with torch.no_grad():
            act, _, _, _, next_state = self.net.get_action_and_value(
                obs_t, self.current_lstm_state, action_masks=mask_t, deterministic=deterministic
            )

        self.current_lstm_state = next_state
        action = int(act.squeeze().cpu().numpy())
        return action, self.current_lstm_state


class RecurrentPPOTrainer:
    """Trainer for RecurrentMaskablePPO with episodic rollouts and clipped PPO objective."""

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        lstm_config_path: str = "configs/hybrid/lstm_ppo.yaml",
        seed: int = 42,
        use_traci: bool = False
    ):
        self.seed = seed
        set_seed(seed)
        self.config_path = config_path
        self.lstm_config_path = lstm_config_path
        self.use_traci = use_traci

        if os.path.isfile(lstm_config_path):
            with open(lstm_config_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_dir = f"models/hybrid/lstm_ppo/seed_{seed}"
        self.logs_dir = f"logs/hybrid/lstm_ppo/seed_{seed}"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        lstm_hp = self.cfg.get("lstm", {})
        self.hidden_dim = int(lstm_hp.get("hidden_dim", 64))
        self.lstm_hidden_dim = int(lstm_hp.get("lstm_hidden_dim", 64))

        self.net = RecurrentActorCriticNet(
            obs_dim=34,
            action_dim=4,
            hidden_dim=self.hidden_dim,
            lstm_hidden_dim=self.lstm_hidden_dim
        ).to(self.device)

        hp = self.cfg.get("hyperparameters", {})
        self.lr = float(hp.get("learning_rate", 3e-4))
        self.gamma = float(hp.get("gamma", 0.99))
        self.gae_lambda = float(hp.get("gae_lambda", 0.95))
        self.clip_range = float(hp.get("clip_range", 0.2))
        self.ent_coef = float(hp.get("ent_coef", 0.01))
        self.vf_coef = float(hp.get("vf_coef", 0.5))
        self.max_grad_norm = float(hp.get("max_grad_norm", 0.5))

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def train(
        self,
        train_scenarios: List[Dict[str, Any]],
        val_scenarios: Optional[List[Dict[str, Any]]] = None,
        total_timesteps: int = 50000,
        rollout_steps: int = 512
    ) -> str:
        """Trains Recurrent Maskable PPO agent using episodic experience rollouts."""
        env = EvacuationEnv(
            config_path=self.config_path,
            scenarios=train_scenarios,
            use_traci=self.use_traci,
            curriculum_stage=4
        )

        logger.info(f"Training RecurrentMaskablePPO (LSTM) for {total_timesteps} steps on {self.device}...")
        timesteps = 0
        best_reward = -float("inf")

        obs, _ = env.reset()
        lstm_state = self.net.get_initial_state(batch_size=1, device=self.device)

        while timesteps < total_timesteps:
            # Collect rollouts
            b_obs = []
            b_actions = []
            b_masks = []
            b_logprobs = []
            b_rewards = []
            b_dones = []
            b_values = []

            for _ in range(rollout_steps):
                mask = env.action_masks()
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    action_t, logprob_t, _, val_t, next_lstm = self.net.get_action_and_value(
                        obs_t, lstm_state, action_masks=mask_t
                    )

                action = int(action_t.item())
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc

                b_obs.append(obs)
                b_actions.append(action)
                b_masks.append(mask)
                b_logprobs.append(logprob_t.item())
                b_rewards.append(reward)
                b_dones.append(done)
                b_values.append(val_t.item())

                obs = next_obs
                timesteps += 1

                if done:
                    obs, _ = env.reset()
                    lstm_state = self.net.get_initial_state(batch_size=1, device=self.device)
                else:
                    lstm_state = next_lstm

                if timesteps >= total_timesteps:
                    break

            # Compute GAE
            b_returns = np.zeros(len(b_rewards), dtype=np.float32)
            b_advantages = np.zeros(len(b_rewards), dtype=np.float32)
            last_gae = 0.0

            with torch.no_grad():
                next_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                _, next_val = self.net.encoder(next_obs_t), self.net.critic(self.net.lstm(self.net.encoder(next_obs_t).unsqueeze(1), lstm_state)[0].squeeze(1)).squeeze(-1)
                next_val = next_val.item()

            for t in reversed(range(len(b_rewards))):
                next_v = next_val if t == len(b_rewards) - 1 else b_values[t + 1]
                non_terminal = 1.0 - float(b_dones[t])
                delta = b_rewards[t] + self.gamma * next_v * non_terminal - b_values[t]
                last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
                b_advantages[t] = last_gae
                b_returns[t] = b_advantages[t] + b_values[t]

            # PPO Update
            t_obs = torch.as_tensor(np.array(b_obs), dtype=torch.float32, device=self.device)
            t_act = torch.as_tensor(np.array(b_actions), dtype=torch.long, device=self.device)
            t_masks = torch.as_tensor(np.array(b_masks), dtype=torch.bool, device=self.device)
            t_logprobs = torch.as_tensor(np.array(b_logprobs), dtype=torch.float32, device=self.device)
            t_returns = torch.as_tensor(b_returns, dtype=torch.float32, device=self.device)
            t_advantages = torch.as_tensor(b_advantages, dtype=torch.float32, device=self.device)
            t_advantages = (t_advantages - t_advantages.mean()) / (t_advantages.std() + 1e-8)

            B_batch = t_obs.shape[0]
            init_lstm = self.net.get_initial_state(batch_size=B_batch, device=self.device)
            for _ in range(4):
                _, new_logprobs, entropy, new_values, _ = self.net.get_action_and_value(
                    t_obs, init_lstm, action_masks=t_masks, action=t_act
                )
                ratio = torch.exp(new_logprobs - t_logprobs)
                surr1 = ratio * t_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * t_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((new_values - t_returns) ** 2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        env.close()

        # Save model
        agent = RecurrentMaskablePPOAgent(net=self.net, device=self.device)
        best_path = os.path.join(self.models_dir, "best_model.pt")
        agent.save(best_path, hp=dict(
            obs_dim=34,
            action_dim=4,
            hidden_dim=self.hidden_dim,
            lstm_hidden_dim=self.lstm_hidden_dim
        ))

        with open(os.path.join(self.models_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(self.cfg, f)

        logger.info(f"Saved RecurrentMaskablePPO checkpoint to {best_path}")
        return best_path


def load_recurrent_ppo_agent(model_path: str, device: str = "cpu") -> RecurrentMaskablePPOAgent:
    """Helper to load a trained RecurrentMaskablePPOAgent."""
    agent = RecurrentMaskablePPOAgent(device=device)
    agent.load(model_path)
    return agent
