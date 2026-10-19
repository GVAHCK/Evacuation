"""Traffic performance metrics for evacuation routes."""

import numpy as np
from typing import List, Dict, Any


def compute_traffic_metrics(episode_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes traffic interaction metrics including evacuation travel times,

    network distances, and congestion exposure.
    """
    if not episode_records:
        return {
            "mean_travel_time": 0.0,
            "median_travel_time": 0.0,
            "mean_distance": 0.0,
            "mean_congestion_exposure": 0.0,
            "timeout_rate": 0.0
        }

    travel_times = [r.get("cumulative_travel_time", 0.0) for r in episode_records]
    distances = [r.get("cumulative_distance", 0.0) for r in episode_records]
    timeouts = [r.get("timeout", False) for r in episode_records]

    return {
        "mean_travel_time": float(np.mean(travel_times)),
        "median_travel_time": float(np.median(travel_times)),
        "std_travel_time": float(np.std(travel_times)),
        "mean_distance": float(np.mean(distances)),
        "timeout_rate": float(np.mean(timeouts))
    }
