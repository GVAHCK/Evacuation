"""Dynamic A* heuristic routing baseline for evacuation optimization."""

import math
import networkx as nx
from typing import Optional
from src.baselines.base_baseline import BaseEvacuationBaseline
from src.environment.evacuation_env import EvacuationEnv


class AStarBaseline(BaseEvacuationBaseline):
    """Dynamic A* heuristic search algorithm re-planning shortest paths toward

    evacuation destination using spatial Euclidean heuristics and live hazard costs.
    """

    def __init__(self, hazard_weight: float = 5.0):
        super().__init__(name="A*")
        self.hazard_weight = hazard_weight

    def select_action(self, env: EvacuationEnv) -> int:
        """Computes A* path to destination and returns corresponding discrete action index."""
        current_node = env.current_node
        dest_node = env.destination_node
        successors = env._get_ordered_successors()

        if not successors:
            return 0

        # Build dynamic subgraph without closed roads
        dynamic_graph = nx.DiGraph()
        for u, v, data in env.graph.edges(data=True):
            edge_id = f"{u}_{v}"
            status = env.current_hazard_engine.get_edge_status(edge_id)
            if status == env.current_hazard_engine.STATE_CLOSED:
                continue

            length = float(data.get("length", 50.0))
            speed = float(data.get("speed_limit", 13.89))
            if status == env.current_hazard_engine.STATE_DEGRADED:
                speed *= 0.5

            intensity = env.current_hazard_engine.get_intensity(edge_id)
            cost = (length / max(speed, 1.0)) * (1.0 + self.hazard_weight * intensity)
            dynamic_graph.add_edge(u, v, weight=cost)

        # Euclidean heuristic function
        dest_data = env.graph.nodes[dest_node]
        dx = float(dest_data.get("x", dest_data.get("lon", 0.0)))
        dy = float(dest_data.get("y", dest_data.get("lat", 0.0)))

        def heuristic(u: str, v: str) -> float:
            ud = env.graph.nodes[u]
            ux = float(ud.get("x", ud.get("lon", 0.0)))
            uy = float(ud.get("y", ud.get("lat", 0.0)))
            return math.hypot(dx - ux, dy - uy) / 13.89

        try:
            path = nx.astar_path(dynamic_graph, current_node, dest_node, heuristic=heuristic, weight="weight")
            if len(path) > 1:
                next_node = path[1]
                if next_node in successors:
                    return successors.index(next_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        # Fallback: greedy step toward destination among open neighbors
        best_idx = 0
        min_h = float("inf")
        for idx, succ in enumerate(successors[:env.max_neighbors]):
            edge_id = f"{current_node}_{succ}"
            if env.current_hazard_engine.get_edge_status(edge_id) != env.current_hazard_engine.STATE_CLOSED:
                h_val = heuristic(succ, dest_node)
                if h_val < min_h:
                    min_h = h_val
                    best_idx = idx

        return best_idx
