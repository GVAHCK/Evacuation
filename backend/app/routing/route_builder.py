import math
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from backend.app.routing.graph_manager import graph_manager
from backend.app.hazards.hazard_manager import HazardManager
from backend.app.ml.model_manager import model_manager
from backend.app.ml.observation_builder import ObservationBuilder
from backend.app.routing.fallback_router import DynamicAStarRouter
from backend.app.routing.route_validator import RouteValidator
from backend.app.schemas.route import AgentRoute, RouteDecisionFactors, RouteCalculationResponse
from backend.app.schemas.traffic import TrafficConfig
from backend.app.core.logging import logger

AGENT_COHORTS = [
    {"type": "Type A (Standard)", "desc": "Balanced evacuation cohort"},
    {"type": "Type B (Congestion-Sensitive)", "desc": "Heavy bus/convoy evacuation cohort"},
    {"type": "Type C (Hazard-Sensitive)", "desc": "Medical/vulnerable evacuation cohort"},
    {"type": "Type D (Perimeter Corridor)", "desc": "High-efficiency perimeter evacuation cohort"}
]

class RouteBuilder:
    """Executes multi-agent Offline-to-Online MARL route generation with hard action masking,
    transparent decision factors, and Dynamic A* fallback.
    """

    def __init__(self, hazard_manager: HazardManager):
        self.hazard_manager = hazard_manager
        self.obs_builder = ObservationBuilder(hazard_manager)
        self.fallback_router = DynamicAStarRouter(hazard_manager)

    def calculate_routes(
        self,
        origin_node: str,
        destination_node: str,
        agent_count: int = 4,
        traffic_config: Optional[TrafficConfig] = None,
        max_steps: int = 150
    ) -> RouteCalculationResponse:
        """Generates synchronized multi-agent evacuation routes using Offline-to-Online MARL."""
        t_start = time.perf_counter()
        session_id = f"calc_{int(time.time() * 1000)}"

        # Validate origin & destination in graph
        origin_node = graph_manager.resolve_to_node(origin_node)
        destination_node = graph_manager.resolve_to_node(destination_node)

        traffic_bias = traffic_config.congestion_bias if traffic_config else 0.20

        # Check if Offline-to-Online model is ready
        if model_manager.status != "ready" or model_manager.model is None:
            logger.warning("Offline-to-Online MARL model not ready. Executing transparent Dynamic A* fallback.")
            return self._calculate_fallback(
                session_id=session_id,
                origin_node=origin_node,
                destination_node=destination_node,
                agent_count=agent_count,
                traffic_bias=traffic_bias,
                reason="Offline-to-Online MARL model unavailable or degraded",
                t_start=t_start
            )

        # Initialize multi-agent positions
        agent_positions = [origin_node]
        agent_paths: List[List[str]] = [[origin_node]]
        agent_dones = [False] * agent_count

        # For heterogeneous multi-agent simulation, pick slight neighboring starting offsets if needed
        # while keeping primary Agent 0 exactly at requested origin
        if agent_count > 1:
            origin_neighbors = list(graph_manager.graph.neighbors(origin_node))
            for i in range(1, agent_count):
                if i - 1 < len(origin_neighbors):
                    start_node = origin_neighbors[i - 1]
                else:
                    start_node = origin_node
                agent_positions.append(start_node)
                agent_paths.append([start_node])

        total_conflicts = 0
        step_count = 0
        fallback_used = False
        fallback_reason = None
        total_inference_ms = 0.0

        # Step simulation loop using Offline-to-Online MARL
        while step_count < max_steps and not all(agent_dones):
            step_count += 1

            # 1. Build observations and action masks for each active agent
            obs_n = []
            masks_n = []
            active_indices = []
            ordered_succs_n = []

            for i in range(agent_count):
                curr = agent_positions[i]
                if curr == destination_node:
                    agent_dones[i] = True

                if agent_dones[i]:
                    # Done agent observation: dummy obs and mask allowing only 0
                    dummy_obs = np.zeros(34, dtype=np.float32)
                    dummy_mask = np.zeros(4, dtype=bool)
                    dummy_mask[0] = True
                    obs_n.append(dummy_obs)
                    masks_n.append(dummy_mask)
                    ordered_succs_n.append([])
                else:
                    peer_nodes = [agent_positions[j] for j in range(agent_count) if j != i and not agent_dones[j]]
                    obs, mask, succs = self.obs_builder.build_agent_observation(
                        current_node=curr,
                        destination_node=destination_node,
                        step_count=step_count,
                        peer_nodes=peer_nodes,
                        base_congestion=traffic_bias
                    )
                    obs_n.append(obs)
                    masks_n.append(mask)
                    ordered_succs_n.append(succs)
                    active_indices.append(i)

            if not active_indices:
                break

            # 2. Run Offline-to-Online MARL multi-agent inference
            try:
                actions, lat_ms = model_manager.predict(obs_n, masks_n, deterministic=True)
                total_inference_ms += lat_ms
            except Exception as e:
                logger.error(f"Inference error during route generation: {e}")
                fallback_used = True
                fallback_reason = f"Inference exception: {str(e)}"
                break

            # 3. Apply actions to advance agents
            intended_next = []
            for i in range(agent_count):
                if agent_dones[i]:
                    intended_next.append(agent_positions[i])
                    continue

                act = actions[i]
                succs = ordered_succs_n[i]

                # Validate hard action mask
                if not masks_n[i][act]:
                    logger.warning(f"Agent {i} chose masked action {act}. Correcting to valid action.")
                    valid_actions = [idx for idx, valid in enumerate(masks_n[i]) if valid]
                    act = valid_actions[0] if valid_actions else 0

                if act < len(succs):
                    next_node = succs[act]
                else:
                    # No-op or destination reached
                    next_node = agent_positions[i]

                # Prevent local 2-cycles / infinite loops
                if len(agent_paths[i]) >= 2 and next_node == agent_paths[i][-2]:
                    # Find alternative valid successor if possible
                    alt_succs = [s for idx, s in enumerate(succs) if s != next_node and masks_n[i][idx]]
                    if alt_succs:
                        next_node = alt_succs[0]

                intended_next.append(next_node)

            # 4. Check bottleneck conflicts
            node_counts: Dict[str, int] = {}
            for n_idx, n_node in enumerate(intended_next):
                if not agent_dones[n_idx]:
                    node_counts[n_node] = node_counts.get(n_node, 0) + 1

            for count in node_counts.values():
                if count > 1:
                    total_conflicts += (count - 1)

            # 5. Commit transitions
            for i in range(agent_count):
                if not agent_dones[i]:
                    agent_positions[i] = intended_next[i]
                    agent_paths[i].append(intended_next[i])
                    if intended_next[i] == destination_node:
                        agent_dones[i] = True

            # Advance hazard propagation slightly
            self.hazard_manager.step(dt=0.5)

        # Check if any agent failed to reach destination within max_steps
        if not all(agent_dones):
            logger.info("Certain agents did not reach destination via pure RL within step budget. Connecting with Dynamic A*.")
            for i in range(agent_count):
                if not agent_dones[i]:
                    sub_path, _, _, _, _ = self.fallback_router.plan_route(
                        agent_positions[i],
                        destination_node,
                        congestion_bias=traffic_bias
                    )
                    if len(sub_path) > 1:
                        agent_paths[i].extend(sub_path[1:])
            fallback_used = True
            fallback_reason = "Hybrid completion: partial RL trajectory spliced with Dynamic A* connecting corridor"

        # Build output AgentRoute objects
        agent_routes: List[AgentRoute] = []
        for i in range(agent_count):
            path = agent_paths[i]
            # Ensure path starts at intended origin and ends at destination
            if path[-1] != destination_node:
                path.append(destination_node)

            # De-duplicate consecutive duplicate nodes
            clean_path = [path[0]]
            for n in path[1:]:
                if n != clean_path[-1]:
                    clean_path.append(n)

            coords = graph_manager.node_path_to_coordinates(clean_path)
            dist_m = graph_manager.get_path_distance(clean_path)

            # Estimated time at ~35-40 km/h (10.0 m/s) with hazard/traffic impedance
            haz_scores = [self.hazard_manager.get_intensity(n) for n in clean_path]
            mean_haz = float(np.mean(haz_scores)) if haz_scores else 0.0
            avg_speed = max(3.0, 10.0 * (1.0 - traffic_bias * 0.4) * (1.0 - mean_haz * 0.5))
            est_time_s = dist_m / avg_speed

            traffic_exp = min(1.0, traffic_bias + (0.05 * total_conflicts))
            risk_index = min(1.0, 0.5 * mean_haz + 0.3 * traffic_exp + 0.2 * (total_conflicts / max(step_count, 1)))

            factors = RouteDecisionFactors(
                travel_time_seconds=round(est_time_s, 1),
                distance_meters=round(dist_m, 1),
                hazard_exposure_score=round(mean_haz, 4),
                traffic_exposure_score=round(traffic_exp, 4),
                agent_conflicts=total_conflicts,
                route_risk_index=round(risk_index, 4)
            )

            cohort_info = AGENT_COHORTS[i % len(AGENT_COHORTS)]
            agent_routes.append(AgentRoute(
                agent_id=i,
                agent_type=cohort_info["type"],
                origin_node=clean_path[0],
                destination_node=destination_node,
                nodes=clean_path,
                coordinates=coords,
                distance_meters=round(dist_m, 1),
                estimated_time_seconds=round(est_time_s, 1),
                hazard_exposure=round(mean_haz, 4),
                traffic_exposure=round(traffic_exp, 4),
                risk_score=round(risk_index, 4),
                decision_factors=factors
            ))

        primary_route = agent_routes[0]
        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        return RouteCalculationResponse(
            session_id=session_id,
            model="offline_to_online",
            requested_model="offline_to_online",
            actual_model="dynamic_astar" if (fallback_used and not total_inference_ms) else "offline_to_online",
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            success=True,
            inference_latency_ms=round(total_inference_ms or total_latency_ms, 2),
            primary_route=primary_route,
            agent_routes=agent_routes,
            hazard_summary={
                "type": self.hazard_manager.current_hazard_type,
                "severity": self.hazard_manager.severity,
                "active_zones_count": len(self.hazard_manager.get_active_hazard_areas()),
                "closed_roads_count": len(self.hazard_manager.get_closed_edges())
            },
            traffic_summary={
                "congestion_bias": traffic_bias,
                "total_conflicts": total_conflicts
            },
            route_decision_factors=primary_route.decision_factors
        )

    def _calculate_fallback(
        self,
        session_id: str,
        origin_node: str,
        destination_node: str,
        agent_count: int,
        traffic_bias: float,
        reason: str,
        t_start: float
    ) -> RouteCalculationResponse:
        """Direct Dynamic A* multi-agent fallback calculation."""
        agent_routes: List[AgentRoute] = []

        path, dist, t_est, risk, fb_reason = self.fallback_router.plan_route(
            origin_node, destination_node, congestion_bias=traffic_bias
        )
        coords = graph_manager.node_path_to_coordinates(path)

        factors = RouteDecisionFactors(
            travel_time_seconds=round(t_est, 1),
            distance_meters=round(dist, 1),
            hazard_exposure_score=round(risk, 4),
            traffic_exposure_score=round(traffic_bias, 4),
            agent_conflicts=0,
            route_risk_index=round(risk, 4)
        )

        for i in range(agent_count):
            cohort_info = AGENT_COHORTS[i % len(AGENT_COHORTS)]
            agent_routes.append(AgentRoute(
                agent_id=i,
                agent_type=cohort_info["type"],
                origin_node=origin_node,
                destination_node=destination_node,
                nodes=path,
                coordinates=coords,
                distance_meters=round(dist, 1),
                estimated_time_seconds=round(t_est, 1),
                hazard_exposure=round(risk, 4),
                traffic_exposure=round(traffic_bias, 4),
                risk_score=round(risk, 4),
                decision_factors=factors
            ))

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return RouteCalculationResponse(
            session_id=session_id,
            model="offline_to_online",
            requested_model="offline_to_online",
            actual_model="dynamic_astar",
            fallback_used=True,
            fallback_reason=reason or fb_reason,
            success=True,
            inference_latency_ms=round(latency_ms, 2),
            primary_route=agent_routes[0],
            agent_routes=agent_routes,
            hazard_summary={
                "type": self.hazard_manager.current_hazard_type,
                "severity": self.hazard_manager.severity
            },
            traffic_summary={"congestion_bias": traffic_bias},
            route_decision_factors=factors
        )
