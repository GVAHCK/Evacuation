"""OpenStreetMap network loader and spatial graph processor using OSMnx."""

import os
import math
import json
import yaml
import networkx as nx
import osmnx as ox
from typing import Dict, Any, Tuple, Optional
from src.utils.logger import setup_logger

logger = setup_logger("OSMLoader")


class OSMLoader:
    """Handles downloading OpenStreetMap networks, calculating spatial features,

    and exporting OSM XML for SUMO conversion.
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.net_cfg = self.config.get("network", {})
        self.north = float(self.net_cfg.get("north", 13.0050))
        self.south = float(self.net_cfg.get("south", 12.9850))
        self.east = float(self.net_cfg.get("east", 80.2450))
        self.west = float(self.net_cfg.get("west", 80.2250))
        self.network_type = self.net_cfg.get("network_type", "drive")
        self.osm_file = self.net_cfg.get("osm_file", "data/raw/network.osm")
        self.graph_file = self.net_cfg.get("graph_file", "data/processed/road_graph.json")

    def download_and_process(self) -> nx.DiGraph:
        """Downloads the OSM network, saves unsimplified OSM XML for SUMO,

        and builds a clean directed graph.
        """
        os.makedirs(os.path.dirname(self.osm_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.graph_file), exist_ok=True)

        logger.info(
            f"Fetching OSM road network bbox: W={self.west}, S={self.south}, E={self.east}, N={self.north}"
        )

        try:
            # OSMnx 2.x requires all_oneway=True and simplify=False to export valid OSM XML for netconvert
            ox.settings.all_oneway = True
            ox.settings.use_cache = True
            ox.settings.log_console = False

            # bbox format in OSMnx 2.x: (left, bottom, right, top) = (west, south, east, north)
            raw_multi_graph = ox.graph_from_bbox(
                bbox=(self.west, self.south, self.east, self.north),
                network_type=self.network_type,
                simplify=False,
                retain_all=True
            )

            # Export OSM XML for SUMO netconvert
            ox.save_graph_xml(raw_multi_graph, filepath=self.osm_file)
            logger.info(f"Exported raw OSM XML to {self.osm_file}")

            # Now simplify for clean RL topological graph
            simplified_graph = ox.simplify_graph(raw_multi_graph)
            di_graph = self._convert_to_clean_digraph(simplified_graph)

        except Exception as e:
            logger.warning(f"Overpass download failed or timed out: {e}. Generating synthetic urban grid fallback.")
            di_graph = self._generate_synthetic_urban_grid()

        # Save processed graph JSON
        self._save_graph_json(di_graph, self.graph_file)
        logger.info(f"Processed graph saved: {di_graph.number_of_nodes()} nodes, {di_graph.number_of_edges()} edges")
        return di_graph

    def _convert_to_clean_digraph(self, multi_graph: nx.MultiDiGraph) -> nx.DiGraph:
        """Converts OSMnx MultiDiGraph to a clean nx.DiGraph with normalized edge attributes."""
        di_graph = nx.DiGraph()

        for node, data in multi_graph.nodes(data=True):
            di_graph.add_node(
                str(node),
                x=float(data.get("x", 0.0)),
                y=float(data.get("y", 0.0)),
                lon=float(data.get("x", 0.0)),
                lat=float(data.get("y", 0.0)),
                elevation=float(data.get("elevation", 5.0))
            )

        for u, v, k, data in multi_graph.edges(keys=True, data=True):
            u_str, v_str = str(u), str(v)
            if u_str == v_str:
                continue

            length = float(data.get("length", 50.0))
            lanes = data.get("lanes", 1)
            if isinstance(lanes, list):
                lanes = int(lanes[0])
            else:
                try:
                    lanes = int(lanes)
                except (ValueError, TypeError):
                    lanes = 1

            maxspeed = data.get("maxspeed", 13.89)
            if isinstance(maxspeed, list):
                maxspeed = maxspeed[0]
            try:
                if isinstance(maxspeed, str) and "mph" in maxspeed.lower():
                    speed_val = float(maxspeed.lower().replace("mph", "").strip()) * 0.44704
                else:
                    speed_val = float(str(maxspeed).replace("km/h", "").strip()) / 3.6
            except (ValueError, TypeError):
                speed_val = 13.89  # default ~50 km/h in m/s

            edge_id = data.get("name", f"e_{u_str}_{v_str}")
            if isinstance(edge_id, list):
                edge_id = str(edge_id[0])

            oneway = data.get("oneway", False)
            if isinstance(oneway, list):
                oneway = oneway[0]
            is_oneway = oneway in (True, "yes", "1")

            edge_attrs = dict(
                id=f"{u_str}_{v_str}",
                length=max(length, 10.0),
                lanes=max(lanes, 1),
                speed_limit=max(speed_val, 5.0),
                free_flow_time=max(length, 10.0) / max(speed_val, 5.0),
                hazard_intensity=0.0,
                hazard_state=0  # 0: safe, 1: degraded, 2: closed
            )
            di_graph.add_edge(u_str, v_str, **edge_attrs)

            # Unless explicitly marked as oneway, urban roads accommodate two-way traffic
            if not is_oneway and not di_graph.has_edge(v_str, u_str):
                rev_attrs = dict(edge_attrs)
                rev_attrs["id"] = f"{v_str}_{u_str}"
                di_graph.add_edge(v_str, u_str, **rev_attrs)

        # Retain the largest strongly connected component so all nodes have valid paths
        if di_graph.number_of_nodes() > 0:
            sccs = list(nx.strongly_connected_components(di_graph))
            if sccs:
                largest_scc = max(sccs, key=len)
                di_graph = di_graph.subgraph(largest_scc).copy()
                logger.info(f"Retained largest strongly connected component: {di_graph.number_of_nodes()} nodes, {di_graph.number_of_edges()} edges")

        return di_graph

    def _generate_synthetic_urban_grid(self) -> nx.DiGraph:
        """Generates a synthetic urban road grid with multi-lane avenues and cross streets."""
        logger.info("Building synthetic urban grid (8x8 topology)...")
        grid = nx.grid_2d_graph(8, 8)
        di_graph = nx.DiGraph()

        node_map = {}
        idx = 0
        lat_step = (self.north - self.south) / 8.0
        lon_step = (self.east - self.west) / 8.0

        for r, c in grid.nodes():
            node_id = f"node_{r}_{c}"
            node_map[(r, c)] = node_id
            lon = self.west + c * lon_step
            lat = self.south + r * lat_step
            di_graph.add_node(
                node_id,
                x=lon,
                y=lat,
                lon=lon,
                lat=lat,
                elevation=5.0 + (r + c) * 0.5
            )

        for (r1, c1), (r2, c2) in grid.edges():
            u = node_map[(r1, c1)]
            v = node_map[(r2, c2)]
            length = 150.0
            speed = 13.89

            # Bidirectional street
            for src, dst in [(u, v), (v, u)]:
                di_graph.add_edge(
                    src,
                    dst,
                    id=f"{src}_{dst}",
                    length=length,
                    lanes=2 if (r1 == 3 or c1 == 3) else 1,
                    speed_limit=speed,
                    free_flow_time=length / speed,
                    hazard_intensity=0.0,
                    hazard_state=0
                )

        return di_graph

    def _save_graph_json(self, G: nx.DiGraph, filepath: str) -> None:
        """Serializes the graph structure and attributes to JSON."""
        data = {
            "nodes": {n: d for n, d in G.nodes(data=True)},
            "edges": [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_graph_json(filepath: str = "data/processed/road_graph.json") -> nx.DiGraph:
        """Loads a processed graph from JSON or gpickle format."""
        if not os.path.isfile(filepath):
            # Check alternative extensions
            alt = "data/processed/road_graph.gpickle"
            if os.path.isfile(alt):
                import pickle
                with open(alt, "rb") as f:
                    return pickle.load(f)
            raise FileNotFoundError(f"Graph file not found at {filepath} or {alt}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        G = nx.DiGraph()
        for node_id, attrs in data["nodes"].items():
            G.add_node(node_id, **attrs)

        for edge in data["edges"]:
            src = edge["source"]
            tgt = edge["target"]
            attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
            G.add_edge(src, tgt, **attrs)

        return G
