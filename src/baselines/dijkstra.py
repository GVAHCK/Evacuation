"""Dynamic Dijkstra routing baseline for evacuation route optimization."""

import math
import networkx as nx
from typing import Optional, List
from src.baselines.base_baseline import BaseEvacuationBaseline
from src.environment.evacuation_env import EvacuationEnv


class DijkstraBaseline(BaseEvacuationBaseline):
    """Dynamic Dijkstra algorithm re-evaluating shortest path based on live road closures

    and hazard exposure at every simulation timestep.
    """

    def __init__(self, hazard_weight: float = 5.0):
        super().__init__(name="Dijkstra")
        self.hazard_weight = hazard_weight

    def select_action(self, env: EvacuationEnv) -> int:
        """Finds shortest weighted path to destination and returns action index."""
        current_node = env.current_node
        dest_node = env.destination_node
        successors = env._get_ordered_successors()

        if not successors:
            return 0

        # Build dynamic subgraph excluding closed edges
        dynamic_graph = nx.DiGraph()
        for u, v, data in env.graph.edges(data=True):
            edge_id = f"{u}_{v}"
            status = env.current_hazard_engine.get_edge_status(edge_id)
            if status == env.current_hazard_engine.STATE_CLOSED:
                continue  # impassable

            length = float(data.get("length", 50.0))
            speed = float(data.get("speed_limit", 13.89))
            if status == env.current_hazard_engine.STATE_DEGRADED:
                speed *= 0.5  # degraded speed

            intensity = env.current_hazard_engine.get_intensity(edge_id)
            travel_time = length / max(speed, 1.0)
            cost = travel_time * (1.0 + self.hazard_weight * intensity)

            dynamic_graph.add_edge(u, v, weight=cost)

        # Compute shortest path
        try:
            path = nx.shortest_path(dynamic_graph, source=current_node, target=dest_node, weight="weight")
            if len(path) > 1:
                next_node = path[1]
                if next_node in successors:
                    return successors.index(next_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        # Fallback: choose open successor minimizing Euclidean distance to destination
        dest_data = env.graph.nodes[dest_node]
        dest_x = float(dest_data.get("x", dest_data.get("lon", 0.0)))
        dest_y = float(dest_data.get("y", dest_data.get("lat", 0.0)))

        best_idx = 0
        min_dist = float("inf")
        for idx, succ in enumerate(successors[:env.max_neighbors]):
            edge_id = f"{current_node}_{succ}"
            if env.current_hazard_engine.get_edge_status(edge_id) != env.current_hazard_engine.STATE_CLOSED:
                succ_data = env.graph.nodes[succ]
                sx = float(succ_data.get("x", succ_data.get("lon", 0.0)))
                sy = float(succ_data.get("y", succ_data.get("lat", 0.0)))
                d = math.hypot(dest_x - sx, dest_y - sy)
                if d < min_dist:
                    min_dist = d
                    best_idx = idx

        return best_idx
