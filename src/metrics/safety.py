"""Safety evaluation metrics including Hazard Exposure Index (HEI)."""

import numpy as np
from typing import List, Dict, Any


def compute_hazard_exposure_index(hazard_intensities: List[float], dt: float = 5.0) -> float:
    """Computes Hazard Exposure Index (HEI = sum(hazard_intensity * dt)).

    Lower values indicate safer routing trajectories with less hazard contact.
    """
    if not hazard_intensities:
        return 0.0
    return float(np.sum(np.array(hazard_intensities) * dt))


def compute_safety_metrics(episode_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregates safety and survivability metrics across episodes."""
    if not episode_records:
        return {
            "success_rate": 0.0,
            "mean_hei": 0.0,
            "mean_closure_encounters": 0.0,
            "casualty_rate": 1.0
        }

    successes = [r.get("success", False) for r in episode_records]
    heis = [r.get("hazard_exposure_index", 0.0) for r in episode_records]
    closures = [r.get("closure_encounters", 0) for r in episode_records]

    success_rate = float(np.mean(successes))
    return {
        "success_rate": success_rate,
        "mean_hei": float(np.mean(heis)),
        "median_hei": float(np.median(heis)),
        "mean_closure_encounters": float(np.mean(closures)),
        "casualty_rate": float(1.0 - success_rate)
    }
