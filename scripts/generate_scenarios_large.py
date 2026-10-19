"""Generate large-scale disaster evacuation scenario datasets (train: 5000, val: 1000, test: 500)
for the scaled multi-district urban road network with disjoint seed spaces.
"""

import os
import sys
import json
import random
import yaml
import networkx as nx
import pandas as pd

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.network.osm_loader import OSMLoader

logger = setup_logger("GenerateScenariosLarge")


def generate_large_scenarios():
    graph_path = "data/processed/road_graph_large.json"
    logger.info(f"Loading large road graph from {graph_path}...")
    graph = OSMLoader.load_graph_json(graph_path)
    nodes = list(graph.nodes())
    logger.info(f"Graph loaded with {len(nodes)} nodes and {graph.number_of_edges()} edges.")

    logger.info("Precomputing valid Origin-Destination pairs across multi-district network (hop distance 3 to 25)...")
    valid_od_pairs = []
    # Sample nodes to find rich OD pairs with diverse routing paths
    for u in nodes:
        lengths = nx.single_source_shortest_path_length(graph, u)
        for v, dist in lengths.items():
            if 3 <= dist <= 25 and u != v:
                valid_od_pairs.append((u, v))

    logger.info(f"Precomputed {len(valid_od_pairs)} valid Evacuation Origin-Destination pairs.")

    traffic_levels = ["low", "normal", "morning_peak", "evening_peak", "heavy"]
    hazard_types = ["flood", "fire"]

    splits = [
        ("train_large", 5000, 50000),
        ("val_large", 1000, 60000),
        ("test_large", 500, 70000)
    ]

    output_dir = "data/generated"
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for split_name, n_scens, base_seed in splits:
        logger.info(f"Generating {n_scens} scenarios for {split_name} (base seed {base_seed})...")
        rng = random.Random(base_seed)
        scenarios = []

        for i in range(n_scens):
            scen_seed = base_seed + i
            scen_rng = random.Random(scen_seed)

            od = scen_rng.choice(valid_od_pairs)
            evac_origin = od[0]
            evac_destination = od[1]

            hazard_type = scen_rng.choice(hazard_types)
            hazard_origin = scen_rng.choice(nodes)
            traffic_level = scen_rng.choice(traffic_levels)
            spread_mult = scen_rng.uniform(0.6, 1.4)
            wind_dir = scen_rng.uniform(0.0, 360.0)
            wind_spd = scen_rng.uniform(2.0, 8.0)

            scen = {
                "scenario_id": f"{split_name}_{i+1:05d}",
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

        out_path = os.path.join(output_dir, f"{split_name}_scenarios.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2)

        sz_kb = os.path.getsize(out_path) / 1024.0
        df = pd.DataFrame(scenarios)
        haz_counts = df["hazard_type"].value_counts().to_dict()
        logger.info(f"Saved {len(scenarios)} scenarios to {out_path} ({sz_kb:.1f} KB / {sz_kb/1024.0:.2f} MB)")
        logger.info(f"   Hazard Distribution: {haz_counts}")
        results[split_name] = {
            "count": len(scenarios),
            "size_kb": sz_kb,
            "size_mb": round(sz_kb / 1024.0, 3),
            "seed_range": [base_seed, base_seed + n_scens - 1],
            "hazard_dist": haz_counts,
            "path": out_path
        }

    return results


if __name__ == "__main__":
    generate_large_scenarios()
