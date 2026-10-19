"""Model 4: SafetyShieldPPO Agent
Hybrid Constrained Action-Selection combining PPO Policy Distribution
with a Deterministic Multi-Factor Safety Shield.
"""

import os
import math
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List

from src.utils.logger import setup_logger
from src.environment.evacuation_env import EvacuationEnv

logger = setup_logger("SafetyShieldPPO")


class SafetyShieldPPOAgent:
    """Hybrid agent combining a learned MaskablePPO policy with a deterministic

    predictive safety layer enforcing dynamic risk constraints.

    Transparent Action Scoring:
      final_score(a) = alpha * policy_prob(a) + beta * safety_score(a) + gamma * goal_progress(a)
    """

    def __init__(
        self,
        base_ppo_agent: Any,
        alpha: float = 0.50,  # Policy weight
        beta: float = 0.35,   # Safety weight (hazard avoidance & congestion)
        gamma: float = 0.15,  # Goal progress weight
        hazard_weight: float = 2.5,
        congestion_weight: float = 1.0,
        closure_threshold: float = 0.65
    ):
        self.base_ppo_agent = base_ppo_agent
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.hazard_weight = hazard_weight
        self.congestion_weight = congestion_weight
        self.closure_threshold = closure_threshold

    def compute_action_probabilities(self, obs: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
        """Extracts normalized policy action probabilities from the underlying MaskablePPO policy."""
        # Handle both sb3_contrib.MaskablePPO and custom agents
        if hasattr(self.base_ppo_agent, "policy"):
            policy = self.base_ppo_agent.policy
            obs_t = torch.as_tensor(obs, device=policy.device).float()
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            mask_t = torch.as_tensor(action_masks, device=policy.device).bool().unsqueeze(0)

            with torch.no_grad():
                dist = policy.get_distribution(obs_t, action_masks=mask_t)
                probs = dist.distribution.probs.squeeze(0).cpu().numpy()
            return probs
        else:
            # Fallback uniform over valid mask
            probs = np.zeros(4, dtype=np.float32)
            valid_indices = np.where(action_masks)[0]
            if len(valid_indices) > 0:
                probs[valid_indices] = 1.0 / len(valid_indices)
            return probs

    def evaluate_candidate_safety(self, env: EvacuationEnv, action_idx: int) -> Tuple[float, float, bool]:
        """Evaluates safety score, goal progress, and hard constraint violation for candidate action.

        Returns: (safety_score, goal_progress, is_safe)
        """
        successors = env._get_ordered_successors()
        if action_idx >= len(successors):
            return 0.0, 0.0, False

        succ = successors[action_idx]
        edge_id = f"{env.current_node}_{succ}"
        edge_status = env.current_hazard_engine.get_edge_status(edge_id)

        # Hard constraint: Closed road is impassable
        if edge_status == env.current_hazard_engine.STATE_CLOSED:
            return 0.0, 0.0, False

        # Live hazard on transition edge and lookahead at target node
        edge_haz = float(env.current_hazard_engine.get_intensity(edge_id))
        succ_haz = float(env.current_hazard_engine.get_intensity(succ))
        combined_haz = max(edge_haz, succ_haz)

        # Traffic congestion
        cong, _, _ = env._get_traffic_metrics()

        # Dead-end check: does target node have zero open outgoing successors?
        succ_outgoing = list(env.graph.successors(succ))
        is_dead_end = (len(succ_outgoing) == 0 and succ != env.destination_node)

        # Dead-end or extreme hazard violates hard safety constraint if alternatives exist
        if is_dead_end:
            is_safe = False
        else:
            is_safe = True

        # Safety score: inverted penalty in [0, 1]
        penalty = (self.hazard_weight * combined_haz + self.congestion_weight * cong) / (self.hazard_weight + self.congestion_weight + 1e-6)
        safety_score = max(0.0, 1.0 - penalty)

        # Topological goal progress: reduction in remaining road network distance
        curr_dist = env.sp_lengths.get(env.current_node, {}).get(env.destination_node, 1e5)
        succ_dist = env.sp_lengths.get(succ, {}).get(env.destination_node, 1e5)
        scale = max(env.max_graph_dist * 0.1, 20.0)
        dist_gain = (curr_dist - succ_dist) / scale
        goal_progress = float(np.clip((dist_gain + 1.0) / 2.0, 0.0, 1.0))

        return safety_score, goal_progress, is_safe

    def select_action(self, env: EvacuationEnv) -> int:
        """Selects action combining policy probability, deterministic safety, and goal progress."""
        obs = env._get_observation()
        mask = env.action_masks()

        # 1. PPO policy distribution over valid actions
        probs = self.compute_action_probabilities(obs, mask)

        # 2. Safety layer evaluation for all candidate slots
        candidate_scores = []
        safe_candidates = []

        for a in range(len(mask)):
            if not mask[a]:
                candidate_scores.append(-float("inf"))
                continue

            safety_score, goal_progress, is_safe = self.evaluate_candidate_safety(env, a)

            score = (
                self.alpha * float(probs[a]) +
                self.beta * safety_score +
                self.gamma * goal_progress
            )
            candidate_scores.append(score)
            if is_safe:
                safe_candidates.append(a)

        # Constrained action selection: choose highest score among safe candidates if any exist
        if safe_candidates:
            best_action = max(safe_candidates, key=lambda a: candidate_scores[a])
        else:
            # Fallback among all unclosed actions in mask
            valid_actions = [a for a in range(len(mask)) if mask[a]]
            best_action = max(valid_actions, key=lambda a: candidate_scores[a]) if valid_actions else 0

        # Safety invariant: guarantee mask validity
        assert mask[best_action], f"SafetyShield selected invalid masked action {best_action}"
        return best_action

    def predict(
        self,
        obs: np.ndarray,
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[int, None]:
        """Predict method for evaluation benchmarks where env context is available via fallback."""
        if action_masks is not None:
            probs = self.compute_action_probabilities(obs, action_masks)
            # From obs: candidate slot features are at [14 + 5*k]
            # [is_valid, dest_alignment_cos, dist_gain_norm, cand_haz, cand_cong]
            scores = np.full(4, -float("inf"))
            for k in range(4):
                if action_masks[k]:
                    base_idx = 14 + k * 5
                    is_valid = obs[base_idx]
                    alignment = obs[base_idx + 1]
                    dist_gain = obs[base_idx + 2]
                    cand_haz = obs[base_idx + 3]
                    cand_cong = obs[base_idx + 4]

                    safety = 1.0 - (cand_haz * 0.7 + cand_cong * 0.3)
                    progress = (dist_gain + alignment) / 2.0
                    scores[k] = (
                        self.alpha * probs[k] +
                        self.beta * safety +
                        self.gamma * progress
                    )
            action = int(np.argmax(scores))
            return action, None
        else:
            return int(self.base_ppo_agent.predict(obs, deterministic=deterministic)[0]), None
