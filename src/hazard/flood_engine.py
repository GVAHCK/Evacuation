"""Cellular Automata based dynamic flood propagation engine."""

import random
import math
import networkx as nx
from typing import Dict, Any, Optional, Set
from src.hazard.hazard_engine import HazardEngine


class FloodEngine(HazardEngine):
    """Simulates dynamic flood spread using stochastic cellular automata

    guided by elevation proxies and road connectivity.
    """

    def __init__(self, graph: nx.DiGraph, config: Dict[str, Any]):
        super().__init__(graph, config)
        flood_cfg = config.get("hazard", {}).get("flood", {})
        self.base_spread_rate = float(flood_cfg.get("spread_rate", 0.08))
        self.propagation_prob = float(flood_cfg.get("propagation_prob", 0.65))
        self.elevation_sensitivity = float(flood_cfg.get("elevation_sensitivity", 0.5))
        self.degradation_threshold = float(flood_cfg.get("degradation_threshold", 0.35))
        self.closure_threshold = float(flood_cfg.get("closure_threshold", 0.70))

        self.spread_rate = self.base_spread_rate
        self.origin_node: Optional[str] = None
        self.rng = random.Random(42)

    def reset(
        self,
        seed: int = 42,
        origin_node: Optional[str] = None,
        spread_rate_mult: float = 1.0,
        **kwargs
    ) -> None:
        """Resets flood propagation with new origin and seed."""
        self.rng = random.Random(seed)
        self.current_time = 0.0
        self.spread_rate = self.base_spread_rate * spread_rate_mult

        nodes = list(self.graph.nodes())
        if origin_node and origin_node in self.graph:
            self.origin_node = origin_node
        else:
            # Pick a node with relatively low elevation or random
            sorted_nodes = sorted(
                nodes,
                key=lambda n: self.graph.nodes[n].get("elevation", 5.0)
            )
            # Pick from lowest 20%
            lowest_pool = sorted_nodes[:max(1, len(sorted_nodes) // 5)]
            self.origin_node = self.rng.choice(lowest_pool)

        # Initialize node intensities
        self.node_intensities = {n: 0.0 for n in nodes}
        self.node_intensities[self.origin_node] = 0.5  # Seed initial water

        # Initialize edge intensities & states
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
        """Advances cellular automata flood spread by dt seconds."""
        self.current_time += dt
        new_node_intensities = dict(self.node_intensities)

        # Inundation growth at currently flooded nodes
        for node, intensity in self.node_intensities.items():
            if intensity > 0.0:
                # Accumulate water
                new_node_intensities[node] = min(1.0, intensity + self.spread_rate * (dt / 10.0))

                # Spread to adjacent nodes (out-edges and in-edges representing physical flow)
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                current_elev = self.graph.nodes[node].get("elevation", 5.0)

                for nbr in neighbors:
                    if self.node_intensities[nbr] < 1.0:
                        nbr_elev = self.graph.nodes[nbr].get("elevation", 5.0)
                        elev_diff = current_elev - nbr_elev  # positive if neighbor is lower

                        # Water flows downhill faster
                        eff_prob = self.propagation_prob * (
                            1.0 + self.elevation_sensitivity * math.tanh(elev_diff / 5.0)
                        )
                        eff_prob = max(0.05, min(0.95, eff_prob))

                        if self.rng.random() < eff_prob * (self.spread_rate * dt):
                            inflow = 0.15 * intensity * (dt / 5.0)
                            new_node_intensities[nbr] = min(1.0, new_node_intensities[nbr] + inflow)

        self.node_intensities = new_node_intensities
        self._update_edge_states()

    def _update_edge_states(self) -> None:
        """Updates edge hazard intensity and closures based on endpoint nodes."""
        self.closed_edges.clear()
        self.degraded_edges.clear()

        for u, v in self.graph.edges():
            edge_id = f"{u}_{v}"
            # Edge intensity is average of connected endpoints
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
