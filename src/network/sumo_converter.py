"""SUMO network converter using netconvert and sumolib."""

import os
import json
import subprocess
import yaml
from typing import Dict, List, Any, Optional
import sumolib
from src.utils.logger import setup_logger

logger = setup_logger("SUMOConverter")


class SUMOConverter:
    """Converts OSM XML or node/edge XML files to a SUMO .net.xml file

    and maps SUMO network topology to RL state-action structures.
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.net_cfg = self.config.get("network", {})
        self.osm_file = self.net_cfg.get("osm_file", "data/raw/network.osm")
        self.net_file = self.net_cfg.get("net_file", "data/processed/network.net.xml")
        self.mapping_file = "data/processed/sumo_mappings.json"

        try:
            self.netconvert_bin = sumolib.checkBinary("netconvert")
        except Exception:
            # Fallback path in virtualenv
            self.netconvert_bin = os.path.abspath(
                os.path.join(".venv", "Lib", "site-packages", "sumo", "bin", "netconvert.exe")
            )

    def convert_osm_to_sumo(self) -> str:
        """Runs netconvert on the OSM XML file with geometry cleanup and junction guessing."""
        os.makedirs(os.path.dirname(self.net_file), exist_ok=True)

        if not os.path.isfile(self.osm_file):
            raise FileNotFoundError(f"OSM file not found: {self.osm_file}")

        logger.info(f"Running netconvert on {self.osm_file} -> {self.net_file}")

        cmd = [
            self.netconvert_bin,
            "--osm-files", self.osm_file,
            "-o", self.net_file,
            "--geometry.remove",
            "--roundabouts.guess",
            "--ramps.guess",
            "--junctions.join",
            "--tls.guess-signals",
            "--default.lanenumber", "2",
            "--default.speed", "13.89",
            "--no-warnings"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"netconvert failed (code {result.returncode}):\n{result.stderr}")
            raise RuntimeError(f"netconvert execution failed: {result.stderr}")

        logger.info(f"SUMO network generated successfully at {self.net_file}")
        self.build_sumo_mappings()
        return self.net_file

    def build_sumo_mappings(self) -> Dict[str, Any]:
        """Extracts junctions, edges, and outgoing transitions from the SUMO net file."""
        if not os.path.isfile(self.net_file):
            raise FileNotFoundError(f"SUMO net file not found: {self.net_file}")

        net = sumolib.net.readNet(self.net_file)
        edges = net.getEdges()
        nodes = net.getNodes()

        logger.info(f"Loaded SUMO network: {len(nodes)} junctions, {len(edges)} edges")

        # Map junctions and edges
        junction_map = {}
        edge_map = {}
        outgoing_transitions = {}

        for j in nodes:
            j_id = j.getID()
            coord = j.getCoord()
            out_edges = [e.getID() for e in j.getOutgoing()]
            in_edges = [e.getID() for e in j.getIncoming()]
            junction_map[j_id] = {
                "coord": coord,
                "incoming": in_edges,
                "outgoing": out_edges
            }

        for e in edges:
            e_id = e.getID()
            from_node = e.getFromNode().getID()
            to_node = e.getToNode().getID()
            length = e.getLength()
            speed = e.getSpeed()
            num_lanes = e.getLaneNumber()

            # Connected outgoing edges after this edge
            out_edges = [out_e.getID() for out_e in e.getOutgoing()]

            edge_map[e_id] = {
                "from_node": from_node,
                "to_node": to_node,
                "length": length,
                "speed": speed,
                "lanes": num_lanes,
                "outgoing_edges": out_edges
            }
            outgoing_transitions[e_id] = out_edges

        mappings = {
            "num_junctions": len(nodes),
            "num_edges": len(edges),
            "junctions": junction_map,
            "edges": edge_map,
            "outgoing_transitions": outgoing_transitions
        }

        with open(self.mapping_file, "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=2)

        logger.info(f"SUMO mappings saved to {self.mapping_file}")
        return mappings

    @staticmethod
    def load_mappings(mapping_file: str = "data/processed/sumo_mappings.json") -> Dict[str, Any]:
        """Loads precomputed SUMO topology mappings."""
        with open(mapping_file, "r", encoding="utf-8") as f:
            return json.load(f)
