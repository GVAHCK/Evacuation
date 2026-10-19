import heapq
import math
from typing import List, Dict, Tuple, Optional
import networkx as nx

from backend.app.routing.graph_manager import graph_manager
from backend.app.hazards.hazard_manager import HazardManager
from src.hazard.hazard_engine import HazardEngine

class DynamicAStarRouter:
    """Dynamic A* shortest path router with hazard degradation and traffic congestion penalties.
    Serves as the transparent fallback planner when Offline-to-Online RL inference encounters
    disconnected transitions or severe blockage.
    """

    def __init__(self, hazard_manager: HazardManager):
        self.hazard_manager = hazard_manager
        self.graph = graph_manager.graph

    def plan_route(
        self,
        origin: str,
        destination: str,
        congestion_bias: float = 0.20
    ) -> Tuple[List[str], float, float, float, str]:
        """Plans a dynamic risk-aware route from origin to destination.
        Returns:
            - node_path: List[str]
            - distance_m: float
            - estimated_time_s: float
            - risk_score: float
            - reason: str
        """
        if origin == destination:
            return [origin], 0.0, 0.0, 0.0, "Origin equals destination"

        closed_edges = self.hazard_manager.get_closed_edges()

        # Custom weight function penalizing hazards and congestion
        def edge_weight(u, v, data):
            edge_id = f"{u}_{v}"
            if edge_id in closed_edges:
                return float("inf")

            base_len = float(data.get("length", 50.0))
            haz = self.hazard_manager.get_intensity(edge_id)
            status = self.hazard_manager.get_edge_status(edge_id)

            status_mult = 1.0
            if status == HazardEngine.STATE_DEGRADED:
                status_mult = 2.5

            hazard_cost = 1.0 + 4.0 * haz
            traffic_cost = 1.0 + 2.0 * congestion_bias

            return base_len * status_mult * hazard_cost * traffic_cost

        # Heuristic: Euclidean distance to destination
        def heuristic(u, v):
            u_coords = graph_manager.get_node_coordinates(u)
            v_coords = graph_manager.get_node_coordinates(v)
            dx = (u_coords[0] - v_coords[0]) * 111000.0
            dy = (u_coords[1] - v_coords[1]) * 111000.0
            return math.hypot(dx, dy)

        try:
            path = nx.astar_path(self.graph, origin, destination, heuristic=heuristic, weight=edge_weight)
            distance = graph_manager.get_path_distance(path)
            # Estimate time: 40 km/h baseline (~11.1 m/s) with hazard/traffic delay
            avg_speed = max(3.0, 11.1 * (1.0 - congestion_bias * 0.5))
            time_est = distance / avg_speed

            # Calculate path cumulative risk
            risk = sum(self.hazard_manager.get_intensity(n) for n in path) / max(len(path), 1)

            return path, distance, time_est, risk, "Dynamic A* path computed successfully"

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Fallback to standard shortest path ignoring closures if all paths are compromised
            try:
                path = nx.shortest_path(self.graph, origin, destination, weight="length")
                distance = graph_manager.get_path_distance(path)
                return path, distance, distance / 8.0, 0.85, "Severe blockage: compromised emergency corridor selected"
            except Exception as e:
                # If truly disconnected, return single node
                return [origin], 0.0, 0.0, 1.0, f"No route possible: {e}"
