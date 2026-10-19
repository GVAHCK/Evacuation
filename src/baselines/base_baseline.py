"""Base class for classical evacuation routing baseline algorithms."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import networkx as nx
from src.environment.evacuation_env import EvacuationEnv


class BaseEvacuationBaseline(ABC):
    """Abstract interface for classical routing baselines operating on EvacuationEnv."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_action(self, env: EvacuationEnv) -> int:
        """Selects discrete action index for current environment state."""
        pass

    def run_episode(self, env: EvacuationEnv, scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Runs a complete evaluation episode using this baseline."""
        obs, info = env.reset(options={"scenario": scenario} if scenario else None)
        done = False
        step = 0
        total_reward = 0.0

        while not done:
            action = self.select_action(env)
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

        result = dict(step_info)
        result["model"] = self.name
        result["total_reward"] = total_reward
        result["steps"] = step
        return result
