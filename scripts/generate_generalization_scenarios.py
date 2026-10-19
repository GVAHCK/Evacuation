"""Generate Held-Out Test Scenarios for Cross-Region Generalization Study:
Region 1: Dense Urban Grid (George Town) - 250 test scenarios (seeds 80000..80249)
Region 2: Sparse Suburban Corridor (OMR) - 250 test scenarios (seeds 85000..85249)
Strictly test-only: No training data is generated for these regions.
"""

import os
import sys
import json
import random
import networkx as nx
import pandas as pd

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.network.osm_loader import OSMLoader

logger = setup_logger("GenerateGeneralizationScenarios")

REGIONS = {
    "dense_grid": {
        "name": "George Town (Dense Urban Grid)",
        "graph_path": "data/processed/road_graph_dense_grid.json",
        "out_file": "data/generated/test_dense_grid_scenarios.json",
        "base_seed": 80000,
        "n_scenarios": 250
    },
    "sparse_suburban": {
        "name": "OMR Suburban (Sparse Linear Corridor)",
        "graph_path": "data/processed/road_graph_sparse_suburban.json",
        "out_file": "data/generated/test_sparse_suburban_scenarios.json",
        "base_seed": 85000,
        "n_scenarios": 250
    }
}

TRAFFIC_LEVELS = ["low", "normal", "morning_peak", "evening_peak", "heavy"]
HAZARD_TYPES = ["flood", "fire"]


def generate_scenarios_for_region(region_key: str, cfg: dict):
    graph_path = cfg["graph_path"]
    logger.info(f"Loading road graph for {region_key} from {graph_path}...")
    graph = OSMLoader.load_graph_json(graph_path)
    nodes = list(graph.nodes())
    n_nodes = len(nodes)
    n_edges = graph.number_of_edges()
    logger.info(f"{region_key}: Loaded {n_nodes} nodes, {n_edges} edges.")

    logger.info(f"Precomputing valid OD pairs (hop distance 3 to 25) for {region_key}...")
    valid_od_pairs = []
    for u in nodes:
        lengths = nx.single_source_shortest_path_length(graph, u)
        for v, dist in lengths.items():
            if 3 <= dist <= 25 and u != v:
                valid_od_pairs.append((u, v))

    logger.info(f"{region_key}: Found {len(valid_od_pairs)} valid OD pairs.")
    assert len(valid_od_pairs) > 500, f"Insufficient OD pairs ({len(valid_od_pairs)}) for {region_key}"

    base_seed = cfg["base_seed"]
    n_scens = cfg["n_scenarios"]
    scenarios = []

    for i in range(n_scens):
        scen_seed = base_seed + i
        scen_rng = random.Random(scen_seed)

        od = scen_rng.choice(valid_od_pairs)
        evac_origin = od[0]
        evac_destination = od[1]

        hazard_type = scen_rng.choice(HAZARD_TYPES)
        hazard_origin = scen_rng.choice(nodes)
        traffic_level = scen_rng.choice(TRAFFIC_LEVELS)
        spread_mult = scen_rng.uniform(0.6, 1.4)
        wind_dir = scen_rng.uniform(0.0, 360.0)
        wind_spd = scen_rng.uniform(2.0, 8.0)

        scen = {
            "scenario_id": f"zero_shot_{region_key}_{i+1:04d}",
            "region": region_key,
            "split": "zero_shot_test",
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

    out_path = cfg["out_file"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)

    sz_kb = os.path.getsize(out_path) / 1024.0
    df = pd.DataFrame(scenarios)
    haz_counts = df["hazard_type"].value_counts().to_dict()
    traffic_counts = df["traffic_level"].value_counts().to_dict()

    logger.info(f"Saved {len(scenarios)} test scenarios to {out_path} ({sz_kb:.1f} KB)")
    logger.info(f"   Hazard Breakdown: {haz_counts}")
    logger.info(f"   Traffic Breakdown: {traffic_counts}")

    return {
        "region": region_key,
        "n_scenarios": len(scenarios),
        "valid_od_pairs": len(valid_od_pairs),
        "file_size_kb": round(sz_kb, 1),
        "out_path": out_path,
        "hazard_breakdown": haz_counts,
        "seed_range": [base_seed, base_seed + n_scens - 1]
    }


def main():
    logger.info("=== STARTING PHASE B: GENERATE TEST-ONLY SCENARIOS PER REGION ===")
    summary = {}
    for r_key, cfg in REGIONS.items():
        res = generate_scenarios_for_region(r_key, cfg)
        summary[r_key] = res

    summary_file = "results/tables/generalization_scenarios_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved scenario summary to {summary_file}")

    print("\n" + "=" * 90)
    print("PHASE B: ZERO-SHOT TEST SCENARIOS GENERATED")
    print("=" * 90)
    for k, v in summary.items():
        print(f"Region: {k}")
        print(f"  Scenarios: {v['n_scenarios']} (Test-Only, Zero-Shot)")
        print(f"  Seed Range: {v['seed_range'][0]} to {v['seed_range'][1]} (Disjoint)")
        print(f"  Valid OD Space: {v['valid_od_pairs']} pairs")
        print(f"  Hazard Distribution: {v['hazard_breakdown']}")
        print(f"  Saved File: {v['out_path']} ({v['file_size_kb']} KB)\n")


if __name__ == "__main__":
    main()
