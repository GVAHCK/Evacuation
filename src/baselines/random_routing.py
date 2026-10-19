"""Random valid routing baseline selecting uniformly among open transitions."""

import random
import numpy as np
from src.baselines.base_baseline import BaseEvacuationBaseline
from src.environment.evacuation_env import EvacuationEnv


class RandomValidRoutingBaseline(BaseEvacuationBaseline):
    """Weak baseline selecting uniformly at random from available unclosed transitions."""

    def __init__(self, seed: int = 42):
        super().__init__(name="Random")
        self.rng = random.Random(seed)

    def select_action(self, env: EvacuationEnv) -> int:
        """Selects random valid action from action mask."""
        mask = env.action_masks()
        valid_indices = np.where(mask)[0]
        if len(valid_indices) > 0:
            return int(self.rng.choice(valid_indices))
        return 0
