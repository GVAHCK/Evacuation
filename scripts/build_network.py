"""Build Network Pipeline: Downloads OSM, processes graph, converts to SUMO network,

and builds initial traffic demand scenarios.
"""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from src.utils.logger import setup_logger
from src.network.osm_loader import OSMLoader
from src.network.sumo_converter import SUMOConverter
from src.network.traffic_generator import TrafficGenerator

logger = setup_logger("BuildNetwork")


def build_pipeline(config_path: str = "configs/default.yaml") -> None:
    logger.info("Starting End-to-End Digital Twin Network Pipeline...")

    # Step 1: OSM Download & Spatial Graph
    logger.info("=== STEP 1: OSM Network Download & Graph Extraction ===")
    osm_loader = OSMLoader(config_path)
    graph = osm_loader.download_and_process()
    logger.info(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    # Step 2: SUMO Network Conversion via netconvert
    logger.info("=== STEP 2: SUMO Network Conversion (netconvert) ===")
    converter = SUMOConverter(config_path)
    net_file = converter.convert_osm_to_sumo()
    logger.info(f"SUMO network file created at {net_file}")

    # Step 3: Traffic Generation for Baseline Regimes
    logger.info("=== STEP 3: Multi-Density Traffic Generation ===")
    traffic_gen = TrafficGenerator(config_path)
    routes = traffic_gen.generate_all_traffic_regimes(seed=42)
    for regime, route_path in routes.items():
        logger.info(f"Traffic regime '{regime}' ready at {route_path}")

    logger.info("Digital Twin Network Pipeline completed successfully!")


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    build_pipeline(cfg_file)
