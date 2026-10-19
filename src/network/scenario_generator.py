"""Scenario dataset generator creating train, val, and test disaster evacuation scenarios."""

import os
import json
import random
import yaml
import networkx as nx
from typing import List, Dict, Any, Tuple
from src.utils.logger import setup_logger

logger = setup_logger("ScenarioGenerator")


class ScenarioGenerator:
    """Generates strictly disjoint, reproducible disaster evacuation scenarios

    for training, validation, and benchmarking.
    """

    def __init__(self, graph: nx.DiGraph, config_path: str = "configs/default.yaml"):
        self.graph = graph
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.scen_cfg = self.config.get("scenarios", {})
        self.output_dir = self.scen_cfg.get("output_dir", "data/generated")
        self.traffic_levels = ["low", "normal", "morning_peak", "evening_peak", "heavy"]
        self.hazard_types = ["flood", "fire"]

        # Cache nodes and compute candidate O-D pairs with reasonable shortest-path distances
        self.nodes = list(self.graph.nodes())
        self._precompute_valid_od_pairs()

    def _precompute_valid_od_pairs(self) -> None:
        """Finds node pairs (origin, destination) with topological distance between 2 and 15 hops."""
        self.valid_od_pairs = []
        for u in self.nodes:
            lengths = nx.single_source_shortest_path_length(self.graph, u)
            for v, dist in lengths.items():
                if 2 <= dist <= 12 and u != v:
                    self.valid_od_pairs.append((u, v))

        if not self.valid_od_pairs:
            # Fallback if graph is small or sparsely connected
            for u in self.nodes:
                for v in self.nodes:
                    if u != v:
                        self.valid_od_pairs.append((u, v))

        logger.info(f"Precomputed {len(self.valid_od_pairs)} valid Evacuation Origin-Destination pairs")

    def generate_split(
        self,
        split_name: str,
        n_scenarios: int,
        base_seed: int
    ) -> List[Dict[str, Any]]:
        """Generates a list of distinct disaster evacuation scenarios for a split."""
        rng = random.Random(base_seed)
        scenarios = []

        for i in range(n_scenarios):
            scen_seed = base_seed + i
            scen_rng = random.Random(scen_seed)

            od = scen_rng.choice(self.valid_od_pairs)
            evac_origin = od[0]
            evac_destination = od[1]

            hazard_type = scen_rng.choice(self.hazard_types)
            hazard_origin = scen_rng.choice(self.nodes)
            traffic_level = scen_rng.choice(self.traffic_levels)
            spread_mult = scen_rng.uniform(0.6, 1.4)
            wind_dir = scen_rng.uniform(0.0, 360.0)
            wind_spd = scen_rng.uniform(2.0, 8.0)

            scen = {
                "scenario_id": f"{split_name}_{i+1:04d}",
                "split": split_name,
                "seed": scen_seed,
                "hazard_type": hazard_type,
                "hazard_origin": hazard_origin,
                "evac_origin": evac_origin,
                "evac_destination": evac_destination,
                "traffic_level": traffic_level,
                "spread_rate_mult": round(spread_mult, 3),
                "wind_direction": round(wind_dir, 1),
                "wind_speed": round(wind_spd, 1)
            }
            scenarios.append(scen)

        os.makedirs(self.output_dir, exist_ok=True)
        out_file = os.path.join(self.output_dir, f"{split_name}_scenarios.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2)

        logger.info(f"Saved {len(scenarios)} {split_name} scenarios to {out_file}")
        return scenarios

    def generate_all_datasets(self) -> Dict[str, str]:
        """Generates train, validation, and test datasets with guaranteed disjoint seeds."""
        n_train = self.scen_cfg.get("n_train", 1000)
        n_val = self.scen_cfg.get("n_val", 200)
        n_test = self.scen_cfg.get("n_test", 100)

        # Disjoint seed domains
        train_file = os.path.join(self.output_dir, "train_scenarios.json")
        val_file = os.path.join(self.output_dir, "val_scenarios.json")
        test_file = os.path.join(self.output_dir, "test_scenarios.json")

        self.generate_split("train", n_train, base_seed=10000)
        self.generate_split("val", n_val, base_seed=20000)
        self.generate_split("test", n_test, base_seed=30000)

        return {
            "train": train_file,
            "val": val_file,
            "test": test_file
        }
