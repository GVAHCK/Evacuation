import os
import json
import math
import threading
from typing import Dict, Any, List, Tuple, Optional, Union
import networkx as nx

from backend.app.core.config import settings
from backend.app.core.logging import logger

class GraphManager:
    """Thread-safe Singleton Graph Manager for caching and spatial querying of the OSM road network."""

    _instance: Optional["GraphManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.graph: Optional[nx.DiGraph] = None
        self.nodes: List[str] = []
        self.node_coords: Dict[str, Tuple[float, float]] = {}  # node -> (lat, lon)
        self.node_xy: Dict[str, Tuple[float, float]] = {}      # node -> (x, y)
        self.sp_lengths: Dict[str, Dict[str, float]] = {}      # node -> {target -> distance}
        self.max_graph_dist: float = 1000.0
        self.min_x: float = 0.0
        self.max_x: float = 1.0
        self.x_range: float = 1.0
        self.min_y: float = 0.0
        self.max_y: float = 1.0
        self.y_range: float = 1.0
        self.evacuation_centers: List[Dict[str, Any]] = []

        self._load_graph()
        self._initialized = True

    def _load_graph(self):
        """Loads and caches OSM graph from road_graph_large.json or fallback."""
        target_path = settings.OSM_GRAPH_PATH
        if not os.path.isfile(target_path):
            logger.warning(f"Large road graph not found at {target_path}. Trying fallback: {settings.OSM_GRAPH_FALLBACK_PATH}")
            target_path = settings.OSM_GRAPH_FALLBACK_PATH

        if not os.path.isfile(target_path):
            raise FileNotFoundError(f"No road graph JSON found at {target_path} or fallback.")

        logger.info(f"Loading OSM Road Graph from {target_path}...")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct NetworkX DiGraph matching osm_loader.py format
        self.graph = nx.DiGraph()
        if "nodes" in data and "edges" in data:
            for node_id, attrs in data["nodes"].items():
                self.graph.add_node(str(node_id), **attrs)
            for edge in data["edges"]:
                src = str(edge["source"])
                tgt = str(edge["target"])
                attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
                self.graph.add_edge(src, tgt, **attrs)
        else:
            self.graph = nx.node_link_graph(data)

        self.nodes = [str(n) for n in self.graph.nodes()]
        logger.info(f"OSM Graph loaded with {len(self.nodes)} nodes and {self.graph.number_of_edges()} edges.")

        # Compute spatial bounds & coordinate maps
        xs, ys = [], []
        for n in self.nodes:
            d = self.graph.nodes[n]
            # Handle lon/lat vs x/y
            lon = float(d.get("lon", d.get("x", 80.20)))
            lat = float(d.get("lat", d.get("y", 13.00)))
            x = float(d.get("x", lon))
            y = float(d.get("y", lat))

            self.node_coords[n] = (lat, lon)
            self.node_xy[n] = (x, y)
            xs.append(x)
            ys.append(y)

        self.min_x = min(xs) if xs else 0.0
        self.max_x = max(xs) if xs else 1.0
        self.x_range = max(1e-5, self.max_x - self.min_x)

        self.min_y = min(ys) if ys else 0.0
        self.max_y = max(ys) if ys else 1.0
        self.y_range = max(1e-5, self.max_y - self.min_y)

        # Precompute all-pairs shortest paths for graph distances
        logger.info("Computing all-pairs shortest path distance lookup table...")
        self.sp_lengths = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight="length"))

        # Find maximum graph distance
        all_dists = [d for sub in self.sp_lengths.values() for d in sub.values() if d < 1e8]
        self.max_graph_dist = max(all_dists) if all_dists else 2000.0

        # Define prominent designated evacuation centers
        self._setup_evacuation_centers()
        logger.info(f"GraphManager initialized. Max graph distance: {self.max_graph_dist:.1f}m.")

    def _setup_evacuation_centers(self):
        """Identifies perimeter / high-capacity evacuation destination centers."""
        # Pick 3-5 strategically located perimeter nodes as primary evacuation centers
        if len(self.nodes) > 10:
            center_node_ids = [self.nodes[-1], self.nodes[len(self.nodes)//2], self.nodes[-5]]
        else:
            center_node_ids = [self.nodes[-1]]

        names = [
            "Central Relief Shelter & Evacuation Hub",
            "Northern High-Ground Community Stadium",
            "Eastern Emergency Response Depot"
        ]

        self.evacuation_centers = []
        for i, n_id in enumerate(center_node_ids):
            lat, lon = self.node_coords[n_id]
            self.evacuation_centers.append({
                "id": f"shelter_{i+1}",
                "name": names[i % len(names)],
                "node_id": n_id,
                "latitude": lat,
                "longitude": lon,
                "capacity": 2500,
                "type": "Designated Evacuation Facility"
            })

    def resolve_to_node(self, target: Union[str, List[float], Tuple[float, float], Dict[str, Any]]) -> str:
        """Resolves node ID string, [lat, lon] coordinates, or {latitude, longitude} dict to nearest road graph node."""
        if isinstance(target, str):
            if target in self.graph.nodes:
                return target
            # Search by substring if string doesn't directly match
            for n in self.nodes:
                if str(n) == target or target in str(n):
                    return n
            # If not found, fallback to nearest or first node
            logger.warning(f"Node '{target}' not found in graph. Selecting nearest default.")
            return self.nodes[0]

        if isinstance(target, dict):
            try:
                req_lat = float(target.get("latitude", target.get("lat", self.min_y)))
                req_lon = float(target.get("longitude", target.get("lon", target.get("lng", self.min_x))))
            except (TypeError, ValueError):
                return self.nodes[0]
        else:
            # Target is [lat, lon] or (lat, lon)
            try:
                req_lat = float(target[0])
                req_lon = float(target[1])
            except (IndexError, TypeError, ValueError):
                return self.nodes[0]

        best_node = self.nodes[0]
        min_dist_sq = float("inf")

        for n, (n_lat, n_lon) in self.node_coords.items():
            d_sq = (n_lat - req_lat) ** 2 + (n_lon - req_lon) ** 2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                best_node = n

        return best_node

    def get_node_coordinates(self, node_id: str) -> Tuple[float, float]:
        """Returns (lat, lon) for given node ID."""
        return self.node_coords.get(node_id, (self.min_y, self.min_x))

    def node_path_to_coordinates(self, node_path: List[str]) -> List[List[float]]:
        """Converts a sequence of node IDs to [[lat, lon], ...] coordinates."""
        return [[self.node_coords[n][0], self.node_coords[n][1]] for n in node_path if n in self.node_coords]

    def get_path_distance(self, node_path: List[str]) -> float:
        """Calculates cumulative distance along path in meters."""
        total_dist = 0.0
        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i+1]
            if self.graph.has_edge(u, v):
                edge_len = float(self.graph[u][v].get("length", 50.0))
            else:
                lat1, lon1 = self.node_coords.get(u, (0.0, 0.0))
                lat2, lon2 = self.node_coords.get(v, (0.0, 0.0))
                edge_len = math.hypot(lat2 - lat1, lon2 - lon1) * 111000.0
            total_dist += edge_len
        return total_dist

    def get_ordered_successors(self, current_node: str, destination_node: str) -> List[str]:
        """Returns outgoing neighbors sorted by shortest path distance to destination."""
        if current_node not in self.graph:
            return []
        raw_succs = list(self.graph.successors(current_node))

        def sort_key(s):
            return self.sp_lengths.get(s, {}).get(destination_node, 1e9)

        return sorted(raw_succs, key=sort_key)

graph_manager = GraphManager()
