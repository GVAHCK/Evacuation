"""Generate disaster evacuation scenario datasets (train, val, test) with disjoint seeds."""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.network.osm_loader import OSMLoader
from src.network.scenario_generator import ScenarioGenerator

logger = setup_logger("GenerateScenarios")


def main(config_path: str = "configs/default.yaml") -> None:
    logger.info("Loading road graph...")
    graph = OSMLoader.load_graph_json("data/processed/road_graph.json")
    generator = ScenarioGenerator(graph, config_path=config_path)

    logger.info("Generating disjoint scenario splits...")
    files = generator.generate_all_datasets()
    for split, path in files.items():
        logger.info(f"Generated {split} scenarios at {path}")

    logger.info("Scenario generation complete!")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(cfg)
