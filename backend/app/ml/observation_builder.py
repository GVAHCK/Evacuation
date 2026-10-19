import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from backend.app.routing.graph_manager import graph_manager
from backend.app.hazards.hazard_manager import HazardManager
from src.hazard.hazard_engine import HazardEngine

class ObservationBuilder:
    """Constructs the exact 34-dimensional observation vector expected by the trained Offline-to-Online MARL model."""

    def __init__(self, hazard_manager: HazardManager):
        self.hazard_manager = hazard_manager
        self.graph = graph_manager.graph
        self.max_neighbors = 4
        self.max_steps = 150

    def build_agent_observation(
        self,
        current_node: str,
        destination_node: str,
        step_count: int = 0,
        peer_nodes: Optional[List[str]] = None,
        base_congestion: float = 0.20,
        avg_speed: float = 0.70,
        queue_len: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Constructs:
        1. 34-dimensional observation vector (np.float32, clipped [0, 1])
        2. 4-dimensional boolean action mask
        3. Ordered successor node IDs (up to 4)
        """
        peer_nodes = peer_nodes or []
        curr_xy = graph_manager.node_xy.get(current_node, (0.0, 0.0))
        dest_xy = graph_manager.node_xy.get(destination_node, (1.0, 1.0))

        curr_x, curr_y = curr_xy
        dest_x, dest_y = dest_xy

        # 1-5: Normalized spatial coordinates and remaining distance
        curr_x_norm = (curr_x - graph_manager.min_x) / graph_manager.x_range
        curr_y_norm = (curr_y - graph_manager.min_y) / graph_manager.y_range
        dest_x_norm = (dest_x - graph_manager.min_x) / graph_manager.x_range
        dest_y_norm = (dest_y - graph_manager.min_y) / graph_manager.y_range

        rem_dist = graph_manager.sp_lengths.get(current_node, {}).get(destination_node, 0.0)
        rem_dist_norm = min(1.0, rem_dist / max(graph_manager.max_graph_dist, 1e-5))

        # 6-9: Hazard and road closure features
        curr_haz = self.hazard_manager.get_intensity(current_node)
        ordered_succs = graph_manager.get_ordered_successors(current_node, destination_node)

        out_hazs = [
            self.hazard_manager.get_intensity(f"{current_node}_{succ}")
            for succ in ordered_succs
        ]
        mean_out_haz = float(np.mean(out_hazs)) if out_hazs else curr_haz

        if ordered_succs:
            statuses = [
                self.hazard_manager.get_edge_status(f"{current_node}_{succ}")
                for succ in ordered_succs
            ]
            frac_closed = sum(1 for s in statuses if s == HazardEngine.STATE_CLOSED) / len(statuses)
            frac_degraded = sum(1 for s in statuses if s == HazardEngine.STATE_DEGRADED) / len(statuses)
        else:
            frac_closed = 1.0
            frac_degraded = 0.0

        # 10-14: Coupled congestion, speed, queue, time, and node degree
        peer_occupants = sum(1 for p in peer_nodes if p == current_node)
        coupled_congestion = min(1.0, base_congestion + 0.15 * peer_occupants)

        time_norm = min(1.0, step_count / float(self.max_steps))
        degree_norm = min(1.0, len(ordered_succs) / float(self.max_neighbors))

        base_features = [
            curr_x_norm, curr_y_norm, dest_x_norm, dest_y_norm, rem_dist_norm,
            curr_haz, mean_out_haz, frac_closed, frac_degraded, coupled_congestion,
            avg_speed, queue_len, time_norm, degree_norm
        ]

        # 15-34: Candidate successor features (4 slots x 5 attributes = 20 features)
        candidate_features = []
        action_mask = np.zeros(self.max_neighbors, dtype=bool)

        goal_dx = dest_x - curr_x
        goal_dy = dest_y - curr_y
        goal_len = math.hypot(goal_dx, goal_dy)

        for k in range(self.max_neighbors):
            if k < len(ordered_succs):
                succ = ordered_succs[k]
                succ_xy = graph_manager.node_xy.get(succ, (curr_x, curr_y))
                sx, sy = succ_xy
                edge_id = f"{current_node}_{succ}"

                status = self.hazard_manager.get_edge_status(edge_id)
                is_valid = 1.0 if status != HazardEngine.STATE_CLOSED else 0.0
                if is_valid > 0.5:
                    action_mask[k] = True

                road_dx = sx - curr_x
                road_dy = sy - curr_y
                road_len = math.hypot(road_dx, road_dy)

                if road_len > 1e-6 and goal_len > 1e-6:
                    alignment = ((road_dx * goal_dx + road_dy * goal_dy) / (road_len * goal_len) + 1.0) / 2.0
                else:
                    alignment = 0.5

                succ_dest_dist = graph_manager.sp_lengths.get(succ, {}).get(destination_node, graph_manager.max_graph_dist)
                dist_gain = min(1.0, max(0.0, ((rem_dist - succ_dest_dist) / max(graph_manager.max_graph_dist * 0.05, 10.0) + 1.0) / 2.0))
                cand_haz = self.hazard_manager.get_intensity(edge_id)

                succ_occupants = sum(1 for p in peer_nodes if p == succ)
                cand_cong = min(1.0, coupled_congestion + 0.15 * succ_occupants)

                candidate_features.extend([is_valid, alignment, dist_gain, cand_haz, cand_cong])
            else:
                candidate_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Fallback mask: if all actions masked out, allow index 0 to avoid crashes
        if not np.any(action_mask):
            action_mask[0] = True

        obs = np.array(base_features + candidate_features, dtype=np.float32)
        obs_clipped = np.clip(obs, 0.0, 1.0)

        return obs_clipped, action_mask, ordered_succs[:self.max_neighbors]
