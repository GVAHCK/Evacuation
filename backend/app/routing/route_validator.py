from typing import List, Tuple, Dict, Any
from backend.app.routing.graph_manager import graph_manager
from backend.app.hazards.hazard_manager import HazardManager

class RouteValidator:
    """Validates structural continuity, graph connectivity, and hazard constraints of generated routes."""

    @staticmethod
    def validate_route(
        path: List[str],
        origin: str,
        destination: str,
        hazard_manager: HazardManager
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """Validates:
        1. Non-empty path
        2. Origin and destination match
        3. Node existence
        4. Edge connectivity
        5. Road closure flags
        """
        issues = []
        if not path:
            return False, "Route path is empty", [{"type": "EMPTY_PATH"}]

        if path[0] != origin:
            issues.append({"type": "ORIGIN_MISMATCH", "expected": origin, "actual": path[0]})

        if path[-1] != destination:
            issues.append({"type": "DESTINATION_MISMATCH", "expected": destination, "actual": path[-1]})

        closed_edges = hazard_manager.get_closed_edges()

        # Check edge-by-edge continuity
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if not graph_manager.graph.has_node(u):
                issues.append({"type": "INVALID_NODE", "node": u})
            if not graph_manager.graph.has_node(v):
                issues.append({"type": "INVALID_NODE", "node": v})

            if not graph_manager.graph.has_edge(u, v):
                issues.append({"type": "DISCONTINUOUS_EDGE", "u": u, "v": v})

            edge_id = f"{u}_{v}"
            if edge_id in closed_edges:
                issues.append({"type": "CLOSED_ROAD_TRAVERSED", "edge": edge_id})

        is_valid = len(issues) == 0
        summary = "Route passed validation" if is_valid else f"Route validation issues detected ({len(issues)} issues)"
        return is_valid, summary, issues
