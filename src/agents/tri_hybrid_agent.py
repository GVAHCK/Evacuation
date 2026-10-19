"""Model 5: TriHybridEvacuationAgent
Multi-Objective Ensemble combining GNN-MaskablePPO, GNN-QRDQN, and Dynamic Dijkstra
with Step-by-Step Disagreement Logging.
"""

import os
import math
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List

from src.utils.logger import setup_logger
from src.environment.evacuation_env import EvacuationEnv
from src.baselines.dijkstra import DijkstraBaseline

logger = setup_logger("TriHybridAgent")


class TriHybridEvacuationAgent:
    """Ensemble Agent combining:

      1. GNN-MaskablePPO (Policy gradient topology confidence)
      2. GNN-QRDQN (Distributional value estimation)
      3. Dynamic Dijkstra (Exact dynamic shortest path heuristic)

    Normalized Decision Score:
      score(a) = w_ppo * PPO_prob(a) + w_dqn * DQN_norm_Q(a) + w_dijkstra * Dijkstra_vote(a) + w_safety * Safety(a)
    """

    def __init__(
        self,
        ppo_agent: Any,
        qrdqn_agent: Any,
        dijkstra_baseline: Optional[DijkstraBaseline] = None,
        w_ppo: float = 0.40,
        w_dqn: float = 0.30,
        w_dijkstra: float = 0.20,
        w_safety: float = 0.10
    ):
        self.ppo_agent = ppo_agent
        self.qrdqn_agent = qrdqn_agent
        self.dijkstra = dijkstra_baseline or DijkstraBaseline()
        self.w_ppo = w_ppo
        self.w_dqn = w_dqn
        self.w_dijkstra = w_dijkstra
        self.w_safety = w_safety

        # Decision trace log for research analysis
        self.decision_records: List[Dict[str, Any]] = []

    def clear_decision_records(self) -> None:
        """Clears the internal decision trace log."""
        self.decision_records.clear()

    def select_action(self, env: EvacuationEnv) -> int:
        """Selects action and records multi-agent decision audit record."""
        obs = env._get_observation()
        mask = env.action_masks()

        # 1. PPO Action Probabilities
        ppo_probs = np.zeros(4, dtype=np.float32)
        if hasattr(self.ppo_agent, "policy"):
            obs_t = torch.as_tensor(obs, device=self.ppo_agent.policy.device).float().unsqueeze(0)
            mask_t = torch.as_tensor(mask, device=self.ppo_agent.policy.device).bool().unsqueeze(0)
            with torch.no_grad():
                dist = self.ppo_agent.policy.get_distribution(obs_t, action_masks=mask_t)
                ppo_probs = dist.distribution.probs.squeeze(0).cpu().numpy()
        else:
            p_act, _ = self.ppo_agent.predict(obs, action_masks=mask, deterministic=True)
            ppo_probs[p_act] = 1.0

        # 2. DQN Normalized Q-Values / Quantiles
        dqn_q = np.full(4, -1e9, dtype=np.float32)
        if hasattr(self.qrdqn_agent, "get_quantiles"):
            quantiles = self.qrdqn_agent.get_quantiles(obs)  # (4, N)
            dqn_q = np.mean(quantiles, axis=1)
        elif hasattr(self.qrdqn_agent, "q_net"):
            obs_t = torch.as_tensor(obs, device=self.qrdqn_agent.device).float().unsqueeze(0)
            with torch.no_grad():
                dqn_q = self.qrdqn_agent.q_net(obs_t).squeeze(0).cpu().numpy()
        else:
            q_act, _ = self.qrdqn_agent.predict(obs, action_masks=mask, deterministic=True)
            dqn_q[q_act] = 100.0

        # Normalize valid DQN scores into [0, 1]
        dqn_norm = np.zeros(4, dtype=np.float32)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) > 0:
            valid_q = dqn_q[valid_indices]
            q_min, q_max = np.min(valid_q), np.max(valid_q)
            if q_max > q_min:
                dqn_norm[valid_indices] = (valid_q - q_min) / (q_max - q_min)
            else:
                dqn_norm[valid_indices] = 1.0

        # 3. Dijkstra Recommendation
        dijkstra_action = self.dijkstra.select_action(env)
        dijkstra_votes = np.zeros(4, dtype=np.float32)
        if 0 <= dijkstra_action < 4 and mask[dijkstra_action]:
            dijkstra_votes[dijkstra_action] = 1.0

        # 4. Environmental Candidate Safety Score from observation
        safety_scores = np.zeros(4, dtype=np.float32)
        for k in range(4):
            if mask[k]:
                base_idx = 14 + k * 5
                cand_haz = float(obs[base_idx + 3])
                cand_cong = float(obs[base_idx + 4])
                safety_scores[k] = max(0.0, 1.0 - (cand_haz * 0.7 + cand_cong * 0.3))

        # 5. Composite Normalized Score
        final_scores = np.full(4, -float("inf"))
        for k in valid_indices:
            final_scores[k] = (
                self.w_ppo * float(ppo_probs[k]) +
                self.w_dqn * float(dqn_norm[k]) +
                self.w_dijkstra * float(dijkstra_votes[k]) +
                self.w_safety * float(safety_scores[k])
            )

        chosen_action = int(np.argmax(final_scores))

        # Individual recommendations for disagreement analysis
        ppo_choice = int(np.argmax(ppo_probs)) if np.any(ppo_probs > 0) else -1
        dqn_choice = int(np.argmax(dqn_norm)) if len(valid_indices) > 0 else -1
        models_disagreed = bool(
            (ppo_choice != dqn_choice) or
            (ppo_choice != dijkstra_action) or
            (dqn_choice != dijkstra_action)
        )

        # Log decision step
        step_record = {
            "step": env.step_count,
            "current_node": env.current_node,
            "destination_node": env.destination_node,
            "ppo_probs": [round(float(p), 4) for p in ppo_probs],
            "dqn_scores": [round(float(q), 4) for q in dqn_norm],
            "dijkstra_action": int(dijkstra_action),
            "safety_scores": [round(float(s), 4) for s in safety_scores],
            "final_scores": [round(float(f), 4) if f != -float("inf") else -999.0 for f in final_scores],
            "selected_action": chosen_action,
            "models_disagreed": models_disagreed
        }
        self.decision_records.append(step_record)

        assert mask[chosen_action], f"TriHybrid selected invalid masked action {chosen_action}"
        return chosen_action

    def predict(
        self,
        obs: np.ndarray,
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[int, None]:
        """Predict method for benchmarking with observation vectors."""
        if action_masks is None:
            action_masks = np.ones(4, dtype=bool)

        mask = np.array(action_masks, dtype=bool)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return 0, None

        # PPO
        ppo_probs = np.zeros(4, dtype=np.float32)
        if hasattr(self.ppo_agent, "policy"):
            obs_t = torch.as_tensor(obs, device=self.ppo_agent.policy.device).float().unsqueeze(0)
            mask_t = torch.as_tensor(mask, device=self.ppo_agent.policy.device).bool().unsqueeze(0)
            with torch.no_grad():
                dist = self.ppo_agent.policy.get_distribution(obs_t, action_masks=mask_t)
                ppo_probs = dist.distribution.probs.squeeze(0).cpu().numpy()
        else:
            p_act, _ = self.ppo_agent.predict(obs, action_masks=mask, deterministic=True)
            ppo_probs[p_act] = 1.0

        # DQN
        dqn_norm = np.zeros(4, dtype=np.float32)
        if hasattr(self.qrdqn_agent, "get_quantiles"):
            quantiles = self.qrdqn_agent.get_quantiles(obs)
            dqn_q = np.mean(quantiles, axis=1)
        elif hasattr(self.qrdqn_agent, "q_net"):
            obs_t = torch.as_tensor(obs, device=self.qrdqn_agent.device).float().unsqueeze(0)
            with torch.no_grad():
                dqn_q = self.qrdqn_agent.q_net(obs_t).squeeze(0).cpu().numpy()
        else:
            q_act, _ = self.qrdqn_agent.predict(obs, action_masks=mask, deterministic=True)
            dqn_q = np.zeros(4)
            dqn_q[q_act] = 1.0

        valid_q = dqn_q[valid_indices]
        q_min, q_max = np.min(valid_q), np.max(valid_q)
        if q_max > q_min:
            dqn_norm[valid_indices] = (valid_q - q_min) / (q_max - q_min)
        else:
            dqn_norm[valid_indices] = 1.0

        # Slot 0 in canonical ordering is shortest open graph distance to destination
        dijkstra_votes = np.zeros(4, dtype=np.float32)
        dijkstra_votes[0] = 1.0 if mask[0] else 0.0

        scores = np.full(4, -float("inf"))
        for k in valid_indices:
            base_idx = 14 + k * 5
            cand_haz = float(obs[base_idx + 3])
            cand_cong = float(obs[base_idx + 4])
            safety = max(0.0, 1.0 - (cand_haz * 0.7 + cand_cong * 0.3))
            scores[k] = (
                self.w_ppo * float(ppo_probs[k]) +
                self.w_dqn * float(dqn_norm[k]) +
                self.w_dijkstra * float(dijkstra_votes[k]) +
                self.w_safety * safety
            )

        chosen = int(np.argmax(scores))
        return chosen, None
