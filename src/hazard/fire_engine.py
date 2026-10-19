"""Diffusion and wind-advection based dynamic fire propagation engine."""

import random
import math
import networkx as nx
from typing import Dict, Any, Optional, Set
from src.hazard.hazard_engine import HazardEngine


class FireEngine(HazardEngine):
    """Simulates dynamic wildfire/urban fire propagation using anisotropic diffusion

    driven by wind direction, wind speed, and fuel/vegetation factors.
    """

    def __init__(self, graph: nx.DiGraph, config: Dict[str, Any]):
        super().__init__(graph, config)
        fire_cfg = config.get("hazard", {}).get("fire", {})
        self.base_spread_rate = float(fire_cfg.get("spread_rate", 0.10))
        self.wind_direction = float(fire_cfg.get("wind_direction", 45.0))  # degrees
        self.wind_strength = float(fire_cfg.get("wind_strength", 5.0))     # m/s
        self.vegetation_factor = float(fire_cfg.get("vegetation_factor", 0.6))
        self.degradation_threshold = float(fire_cfg.get("degradation_threshold", 0.30))
        self.closure_threshold = float(fire_cfg.get("closure_threshold", 0.65))

        self.spread_rate = self.base_spread_rate
        self.ignition_node: Optional[str] = None
        self.ign_x = 0.0
        self.ign_y = 0.0
        self.rng = random.Random(42)

    def reset(
        self,
        seed: int = 42,
        origin_node: Optional[str] = None,
        spread_rate_mult: float = 1.0,
        wind_direction: Optional[float] = None,
        wind_strength: Optional[float] = None,
        **kwargs
    ) -> None:
        """Resets fire propagation with new ignition point, wind, and seed."""
        self.rng = random.Random(seed)
        self.current_time = 0.0
        self.spread_rate = self.base_spread_rate * spread_rate_mult
        if wind_direction is not None:
            self.wind_direction = wind_direction
        if wind_strength is not None:
            self.wind_strength = wind_strength

        nodes = list(self.graph.nodes())
        if origin_node and origin_node in self.graph:
            self.ignition_node = origin_node
        else:
            self.ignition_node = self.rng.choice(nodes)

        ign_data = self.graph.nodes[self.ignition_node]
        self.ign_x = float(ign_data.get("x", ign_data.get("lon", 0.0)))
        self.ign_y = float(ign_data.get("y", ign_data.get("lat", 0.0)))

        self.node_intensities = {n: 0.0 for n in nodes}
        self.node_intensities[self.ignition_node] = 0.8  # Ignition core

        self.edge_intensities = {}
        self.edge_states = {}
        self.closed_edges.clear()
        self.degraded_edges.clear()

        for u, v in self.graph.edges():
            edge_id = f"{u}_{v}"
            self.edge_intensities[edge_id] = 0.0
            self.edge_states[edge_id] = self.STATE_SAFE

        self._update_edge_states()

    def update(self, dt: float) -> None:
        """Advances fire diffusion by dt seconds."""
        self.current_time += dt
        new_node_intensities = dict(self.node_intensities)

        # Convert wind angle to radians (0 = East, 90 = North)
        wind_rad = math.radians(self.wind_direction)
        wind_vec_x = math.cos(wind_rad)
        wind_vec_y = math.sin(wind_rad)

        for node, intensity in self.node_intensities.items():
            if intensity > 0.05:
                # Fire intensifies up to max
                new_node_intensities[node] = min(1.0, intensity + self.spread_rate * (dt / 8.0))

                node_data = self.graph.nodes[node]
                nx_coord = float(node_data.get("x", node_data.get("lon", 0.0)))
                ny_coord = float(node_data.get("y", node_data.get("lat", 0.0)))

                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                for nbr in neighbors:
                    nbr_data = self.graph.nodes[nbr]
                    bx_coord = float(nbr_data.get("x", nbr_data.get("lon", 0.0)))
                    by_coord = float(nbr_data.get("y", nbr_data.get("lat", 0.0)))

                    dx = bx_coord - nx_coord
                    dy = by_coord - ny_coord
                    dist = math.hypot(dx, dy)
                    if dist < 1e-6:
                        continue

                    # Direction vector from burning node to neighbor
                    dir_x = dx / dist
                    dir_y = dy / dist

                    # Dot product with wind vector (1 if aligned with wind, -1 if against)
                    alignment = dir_x * wind_vec_x + dir_y * wind_vec_y
                    wind_boost = 1.0 + max(0.0, alignment) * (self.wind_strength / 3.0)

                    # Thermal diffusion probability
                    spread_prob = self.spread_rate * wind_boost * self.vegetation_factor * (dt / 5.0)
                    spread_prob = min(0.95, max(0.02, spread_prob))

                    if self.rng.random() < spread_prob:
                        transfer = 0.20 * intensity * (dt / 5.0)
                        new_node_intensities[nbr] = min(1.0, new_node_intensities[nbr] + transfer)

        self.node_intensities = new_node_intensities
        self._update_edge_states()

    def _update_edge_states(self) -> None:
        """Updates edge states (safe/degraded/closed) based on node fire intensities."""
        self.closed_edges.clear()
        self.degraded_edges.clear()

        for u, v in self.graph.edges():
            edge_id = f"{u}_{v}"
            intensity = (self.node_intensities[u] + self.node_intensities[v]) / 2.0
            self.edge_intensities[edge_id] = intensity

            if intensity >= self.closure_threshold:
                self.edge_states[edge_id] = self.STATE_CLOSED
                self.closed_edges.add(edge_id)
            elif intensity >= self.degradation_threshold:
                self.edge_states[edge_id] = self.STATE_DEGRADED
                self.degraded_edges.add(edge_id)
            else:
                self.edge_states[edge_id] = self.STATE_SAFE
