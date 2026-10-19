import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np
from sqlalchemy.orm import Session

from backend.app.core.logging import logger
from backend.app.database.session import SessionLocal
from backend.app.models import EvacuationSession, Agent, Route, RouteStep, Hazard, TrafficSnapshot
from backend.app.schemas.evacuation import (
    EvacuationStartRequest,
    AgentLiveState,
    TrafficLiveState,
    HazardLiveState,
    RouteLiveState,
    MetricsLiveState,
    WebSocketEvacuationMessage
)
from backend.app.schemas.hazard import HazardConfig
from backend.app.routing.graph_manager import graph_manager
from backend.app.hazards.hazard_manager import HazardManager
from backend.app.routing.route_builder import RouteBuilder
from backend.app.simulation.sumo_manager import sumo_manager
from backend.app.websocket.manager import ws_manager

AGENT_TYPES = [
    "Type A (Standard)",
    "Type B (Congestion-Sensitive)",
    "Type C (Hazard-Sensitive)",
    "Type D (Perimeter Corridor)"
]

class EvacuationService:
    """Coordinates end-to-end multi-agent evacuation lifecycle, background simulation loop,
    real-time WebSocket broadcasting, dynamic rerouting, and database persistence.
    """

    def __init__(self):
        # session_id -> running asyncio.Task
        self.active_tasks: Dict[str, asyncio.Task] = {}
        # session_id -> runtime control dict {stop_requested: bool, reroute_requested: bool, paused: bool, speed_factor: float}
        self.session_controls: Dict[str, Dict[str, Any]] = {}
        # session_id -> list of event records
        self.session_events: Dict[str, List[Dict[str, Any]]] = {}

    def add_event(self, session_id: str, event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        if session_id not in self.session_events:
            self.session_events[session_id] = []
        self.session_events[session_id].append({
            "id": f"evt_{int(time.time()*1000)}_{len(self.session_events[session_id])}",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "message": message,
            "metadata": metadata or {}
        })

    def get_events(self, session_id: str) -> List[Dict[str, Any]]:
        return self.session_events.get(session_id, [])

    def pause_session(self, session_id: str) -> bool:
        if session_id in self.session_controls:
            self.session_controls[session_id]["paused"] = True
            self.add_event(session_id, "pause", "Evacuation simulation paused by operator.")
            return True
        return False

    def resume_session(self, session_id: str) -> bool:
        if session_id in self.session_controls:
            self.session_controls[session_id]["paused"] = False
            self.add_event(session_id, "resume", "Evacuation simulation resumed.")
            return True
        return False

    def set_speed_factor(self, session_id: str, speed_factor: float) -> bool:
        if session_id in self.session_controls:
            clamped = max(0.1, min(10.0, float(speed_factor)))
            self.session_controls[session_id]["speed_factor"] = clamped
            self.add_event(session_id, "speed_change", f"Simulation speed factor set to {clamped:.1f}x.")
            return True
        return False

    def start_session(
        self,
        request: EvacuationStartRequest,
        user_id: Optional[str],
        db: Session
    ) -> EvacuationSession:
        """Initializes session, records to database, and launches live background simulation."""
        origin_node = graph_manager.resolve_to_node(request.origin)
        dest_node = graph_manager.resolve_to_node(request.destination)

        session = EvacuationSession(
            user_id=user_id,
            scenario_id=request.scenario_id,
            model="offline_to_online",
            status="active",
            started_at=datetime.now(timezone.utc),
            origin=origin_node,
            destination=dest_node,
            agent_count=request.agent_count,
            distance=0.0,
            travel_time=0.0,
            hazard_exposure=0.0,
            traffic_exposure=0.0,
            risk_score=0.0,
            inference_latency=3.5,
            success=False,
            fallback_used=False
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        speed_factor = getattr(request, "speed_factor", 1.0) or 1.0

        # Control flags
        self.session_controls[session.id] = {
            "stop_requested": False,
            "reroute_requested": False,
            "paused": False,
            "speed_factor": speed_factor
        }
        self.add_event(session.id, "session_started", f"Evacuation session initialized with {request.agent_count} agents.")

        # Spawn background simulation task
        task = asyncio.create_task(
            self._live_simulation_loop(
                session_id=session.id,
                origin_node=origin_node,
                destination_node=dest_node,
                agent_count=request.agent_count,
                hazard_config=request.hazard_config,
                hazard_type=request.hazard_type,
                speed_factor=speed_factor,
                seed=request.seed or 42
            )
        )
        self.active_tasks[session.id] = task
        def _on_task_done(t):
            if not t.cancelled() and t.exception():
                logger.error(f"Task exception for session '{session.id}': {t.exception()}", exc_info=t.exception())
        task.add_done_callback(_on_task_done)
        logger.info(f"Evacuation session '{session.id}' started in background (speed_factor={speed_factor}x).")

        return session

    def stop_session(self, session_id: str, db: Session) -> Optional[EvacuationSession]:
        """Requests graceful termination of active simulation session."""
        if session_id in self.session_controls:
            self.session_controls[session_id]["stop_requested"] = True
            self.add_event(session_id, "session_stopped", "Evacuation simulation stopped by user command.")

        session = db.query(EvacuationSession).filter(EvacuationSession.id == session_id).first()
        if session and session.status == "active":
            session.status = "stopped"
            session.completed_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(session)

        return session

    def trigger_reroute(self, session_id: str):
        """Signals active evacuation session to trigger immediate dynamic route recalculation."""
        if session_id in self.session_controls:
            self.session_controls[session_id]["reroute_requested"] = True
            self.add_event(session_id, "reroute_requested", "Manual reroute signal dispatched to MARL policy.")
            logger.info(f"Dynamic reroute triggered for session '{session_id}'.")

    async def _live_simulation_loop(
        self,
        session_id: str,
        origin_node: str,
        destination_node: str,
        agent_count: int,
        hazard_config: Optional[Any],
        hazard_type: Optional[str] = None,
        speed_factor: float = 1.0,
        seed: int = 42
    ):
        """Asynchronous live evacuation loop. Advances agents, hazard, and traffic step-by-step,
        broadcasting real-time telemetry over WebSockets and saving final results.
        """
        logger.info(f"Beginning live evacuation loop for session '{session_id}'...")

        # Setup isolated hazard manager and route builder for this session
        hazard_mgr = HazardManager()
        hz_cfg = hazard_config
        if isinstance(hazard_config, dict):
            try:
                hz_cfg = HazardConfig(**hazard_config)
            except Exception:
                hz_cfg = None
        elif hz_cfg is None and hazard_type:
            hz_cfg = HazardConfig(hazard_type=hazard_type, severity=1.0, radius_meters=450)

        hazard_mgr.configure(hz_cfg, seed=seed, origin_node=origin_node)
        route_builder = RouteBuilder(hazard_mgr)

        logger.info(f"Calculating initial routes for session '{session_id}'...")
        # 1. Compute initial Offline-to-Online MARL route
        initial_res = route_builder.calculate_routes(
            origin_node=origin_node,
            destination_node=destination_node,
            agent_count=agent_count
        )
        logger.info(f"Initial route computed for session '{session_id}'. Primary waypoints: {len(initial_res.primary_route.nodes)}")

        agent_paths = [r.nodes.copy() for r in initial_res.agent_routes]
        # Current progress index along path for each agent
        agent_path_indices = [0] * agent_count
        agent_cumulative_hazard = [0.0] * agent_count
        agent_statuses = ["evacuating"] * agent_count

        # Update initial session metrics and seed initial routes and agents in database
        try:
            with SessionLocal() as db:
                s_rec = db.query(EvacuationSession).filter(EvacuationSession.id == session_id).first()
                if s_rec:
                    s_rec.distance = initial_res.primary_route.distance_meters
                    s_rec.travel_time = initial_res.primary_route.estimated_time_seconds
                    s_rec.inference_latency = initial_res.inference_latency_ms
                    s_rec.fallback_used = initial_res.fallback_used
                    s_rec.fallback_reason = initial_res.fallback_reason

                    # Persist initial primary Route
                    coords = graph_manager.node_path_to_coordinates(agent_paths[0])
                    route_rec = Route(
                        session_id=session_id,
                        agent_index=0,
                        total_distance=round(initial_res.primary_route.distance_meters, 1),
                        estimated_time=round(initial_res.primary_route.estimated_time_seconds, 1),
                        risk_score=round(initial_res.primary_route.risk_score, 4),
                        nodes=agent_paths[0],
                        coordinates=coords,
                        decision_factors=initial_res.primary_route.decision_factors.model_dump()
                    )
                    db.add(route_rec)

                    # Persist initial Agent records
                    for i, p in enumerate(agent_paths):
                        ag = Agent(
                            session_id=session_id,
                            agent_index=i,
                            current_node=p[0],
                            destination_node=destination_node,
                            status="evacuating",
                            route=p,
                            distance_travelled=0.0,
                            time_elapsed=0.0,
                            hazard_exposure=0.0
                        )
                        db.add(ag)

                    db.commit()
        except Exception as e:
            logger.warning(f"Could not write initial session metrics/routes: {e}")

        step = 0
        max_steps = 100
        start_time = time.time()
        rerouted_count = 0

        # Broadcast session_started
        primary_coords = graph_manager.node_path_to_coordinates(agent_paths[0])
        await ws_manager.broadcast_to_session(session_id, {
            "event": "session_started",
            "session_id": session_id,
            "timestamp": time.time(),
            "status": "active",
            "message": "Evacuation session commenced. Offline-to-Online MARL route generated.",
            "route": {
                "coordinates": primary_coords,
                "distance": initial_res.primary_route.distance_meters,
                "eta": initial_res.primary_route.estimated_time_seconds,
                "risk": initial_res.primary_route.risk_score
            }
        })

        while step < max_steps:
            # Check for user stop request
            control = self.session_controls.get(session_id, {})
            if control.get("stop_requested", False):
                logger.info(f"Session '{session_id}' stopping due to user request.")
                break

            # Check for pause state
            while control.get("paused", False):
                if control.get("stop_requested", False):
                    break
                await asyncio.sleep(0.5)
                control = self.session_controls.get(session_id, {})

            if control.get("stop_requested", False):
                break

            step += 1
            logger.info(f"[SIM] Step {step}/{max_steps} executing for session '{session_id}'...")
            current_speed = control.get("speed_factor", speed_factor)
            step_dt = max(0.01, 1.0 / max(0.1, current_speed))
            await asyncio.sleep(step_dt)

            # 1. Advance hazard propagation
            hazard_mgr.step(dt=1.5)

            # 2. Advance traffic digital twin
            current_nodes = [
                agent_paths[i][agent_path_indices[i]] if agent_path_indices[i] < len(agent_paths[i]) else destination_node
                for i in range(agent_count)
            ]
            traffic_state = sumo_manager.step(agent_nodes=current_nodes, dt=1.0)
            logger.info(f"[SIM] Step {step} - advanced hazard and traffic.")

            # 3. Dynamic Rerouting check:
            # Check if road ahead of any active agent became closed or heavily flooded
            should_reroute = control.get("reroute_requested", False)
            closed_edges = hazard_mgr.get_closed_edges()

            for i in range(agent_count):
                curr_idx = agent_path_indices[i]
                if curr_idx < len(agent_paths[i]) - 1:
                    next_node = agent_paths[i][curr_idx + 1]
                    curr_node = agent_paths[i][curr_idx]
                    edge_id = f"{curr_node}_{next_node}"
                    if edge_id in closed_edges or hazard_mgr.get_intensity(edge_id) > 0.65:
                        should_reroute = True
                        break

            logger.info(f"[SIM] Step {step} - should_reroute: {should_reroute}")
            if should_reroute:
                control["reroute_requested"] = False
                rerouted_count += 1
                logger.info(f"Dynamic conditions changed: recalculating route for session '{session_id}'...")

                # Recalculate route from current primary agent position to destination
                curr_primary = agent_paths[0][agent_path_indices[0]]
                new_calc = route_builder.calculate_routes(
                    origin_node=curr_primary,
                    destination_node=destination_node,
                    agent_count=agent_count
                )

                # Splice new route into remaining path
                for i in range(agent_count):
                    new_nodes = new_calc.agent_routes[i].nodes
                    curr_idx = agent_path_indices[i]
                    traversed = agent_paths[i][:curr_idx + 1]
                    # Connect traversed with new nodes (avoiding duplicate curr_node)
                    if new_nodes and new_nodes[0] == traversed[-1]:
                        spliced = traversed + new_nodes[1:]
                    else:
                        spliced = traversed + new_nodes
                    agent_paths[i] = spliced

                new_coords = graph_manager.node_path_to_coordinates(agent_paths[0][agent_path_indices[0]:])
                self.add_event(session_id, "reroute", "Route updated due to changing multi-hazard conditions.", {
                    "distance_meters": new_calc.primary_route.distance_meters,
                    "eta_seconds": new_calc.primary_route.estimated_time_seconds,
                    "risk_score": new_calc.primary_route.risk_score
                })
                await ws_manager.broadcast_to_session(session_id, {
                    "event": "reroute",
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "message": "Route updated due to changing multi-hazard conditions.",
                    "route": {
                        "coordinates": new_coords,
                        "distance": new_calc.primary_route.distance_meters,
                        "eta": new_calc.primary_route.estimated_time_seconds,
                        "risk": new_calc.primary_route.risk_score
                    }
                })

            # 4. Advance agents along their routes
            for i in range(agent_count):
                if agent_path_indices[i] < len(agent_paths[i]) - 1:
                    agent_path_indices[i] += 1
                    curr_n = agent_paths[i][agent_path_indices[i]]
                    agent_cumulative_hazard[i] += hazard_mgr.get_intensity(curr_n)
                    if curr_n == destination_node:
                        agent_statuses[i] = "evacuated"
                        self.add_event(session_id, "agent_arrival", f"Agent {i} reached safe evacuation destination.")
                else:
                    if agent_statuses[i] != "evacuated":
                        self.add_event(session_id, "agent_arrival", f"Agent {i} reached safe evacuation destination.")
                    agent_statuses[i] = "evacuated"

            # 5. Build live agent states
            agent_live_states: List[AgentLiveState] = []
            for i in range(agent_count):
                idx = min(agent_path_indices[i], len(agent_paths[i]) - 1)
                node = agent_paths[i][idx]
                lat, lon = graph_manager.get_node_coordinates(node)
                progress = float(idx) / max(len(agent_paths[i]) - 1, 1)

                rem_path = agent_paths[i][idx:]
                rem_dist = graph_manager.get_path_distance(rem_path)

                agent_live_states.append(AgentLiveState(
                    id=i,
                    agent_type=AGENT_TYPES[i % len(AGENT_TYPES)],
                    latitude=lat,
                    longitude=lon,
                    current_node=node,
                    destination_node=destination_node,
                    status=agent_statuses[i],
                    progress=round(progress, 3),
                    distance_remaining_m=round(rem_dist, 1),
                    route=agent_paths[i]
                ))

            # 6. Build hazard states
            hazard_areas = hazard_mgr.get_active_hazard_areas()
            hazard_live = [
                HazardLiveState(
                    type=ha.type,
                    severity=ha.severity,
                    center=ha.center,
                    radius=ha.radius
                )
                for ha in hazard_areas
            ]

            # 7. Remaining primary route
            prim_idx = min(agent_path_indices[0], len(agent_paths[0]) - 1)
            prim_rem_nodes = agent_paths[0][prim_idx:]
            prim_rem_coords = graph_manager.node_path_to_coordinates(prim_rem_nodes)
            prim_rem_dist = graph_manager.get_path_distance(prim_rem_nodes)
            prim_rem_eta = prim_rem_dist / max(traffic_state.average_speed_mps, 3.0)

            # 8. Check if all agents completed evacuation
            all_evacuated = all(s == "evacuated" for s in agent_statuses)
            elapsed_s = time.time() - start_time
            evacuated_count = sum(1 for s in agent_statuses if s == "evacuated")

            event_type = "session_completed" if all_evacuated else "agent_update"

            # 9. Broadcast real-time WebSocket state
            msg = {
                "event": event_type,
                "session_id": session_id,
                "timestamp": time.time(),
                "status": "completed" if all_evacuated else "active",
                "agents": [a.model_dump() for a in agent_live_states],
                "traffic": {
                    "density": traffic_state.density,
                    "congestion": traffic_state.congestion,
                    "average_speed": traffic_state.average_speed_mps
                },
                "hazards": [h.model_dump() for h in hazard_live],
                "route": {
                    "coordinates": prim_rem_coords,
                    "distance": round(prim_rem_dist, 1),
                    "eta": round(prim_rem_eta, 1),
                    "risk": round(float(np.mean([hazard_mgr.get_intensity(n) for n in prim_rem_nodes] or [0.0])), 3)
                },
                "metrics": {
                    "hazard_exposure": round(float(np.mean(agent_cumulative_hazard)), 3),
                    "traffic_exposure": traffic_state.congestion,
                    "elapsed_time": round(elapsed_s, 1),
                    "total_conflicts": rerouted_count,
                    "agents_evacuated": evacuated_count,
                    "agents_remaining": agent_count - evacuated_count
                },
                "message": "All agents successfully evacuated!" if all_evacuated else "Evacuation in progress..."
            }
            await ws_manager.broadcast_to_session(session_id, msg)

            if all_evacuated:
                logger.info(f"All {agent_count} agents successfully evacuated session '{session_id}' in {elapsed_s:.1f}s.")
                break

        # 10. Persist final results to database
        self._persist_session_results(
            session_id=session_id,
            agent_paths=agent_paths,
            elapsed_time_s=time.time() - start_time,
            agent_cumulative_hazard=agent_cumulative_hazard,
            success=all(s == "evacuated" for s in agent_statuses),
            fallback_used=initial_res.fallback_used
        )

        # Cleanup control state
        if session_id in self.session_controls:
            del self.session_controls[session_id]
        if session_id in self.active_tasks:
            del self.active_tasks[session_id]

    def _persist_session_results(
        self,
        session_id: str,
        agent_paths: List[List[str]],
        elapsed_time_s: float,
        agent_cumulative_hazard: List[float],
        success: bool,
        fallback_used: bool
    ):
        """Saves final session record, routes, and agents into database."""
        db = SessionLocal()
        try:
            session = db.query(EvacuationSession).filter(EvacuationSession.id == session_id).first()
            if not session:
                return

            primary_dist = graph_manager.get_path_distance(agent_paths[0])
            if primary_dist <= 0 and hasattr(session, "distance") and session.distance > 0:
                primary_dist = session.distance

            mean_haz = float(np.mean(agent_cumulative_hazard)) if agent_cumulative_hazard else 0.0

            if success:
                session.status = "completed"
            elif session.status == "stopped":
                pass
            else:
                session.status = "incomplete"

            session.completed_at = datetime.now(timezone.utc)
            session.distance = round(primary_dist, 1)
            session.travel_time = round(max(elapsed_time_s, 0.5), 1)
            session.hazard_exposure = round(mean_haz, 4)
            session.traffic_exposure = 0.25
            session.risk_score = round(min(1.0, mean_haz * 0.5 + 0.2), 4)
            session.success = success
            session.fallback_used = fallback_used

            # Persist or update Route record for primary agent
            coords = graph_manager.node_path_to_coordinates(agent_paths[0])
            route_rec = db.query(Route).filter(Route.session_id == session_id, Route.agent_index == 0).first()
            if not route_rec:
                route_rec = Route(session_id=session_id, agent_index=0)
                db.add(route_rec)

            route_rec.total_distance = round(primary_dist, 1)
            route_rec.estimated_time = round(elapsed_time_s, 1)
            route_rec.risk_score = round(session.risk_score, 4)
            route_rec.nodes = agent_paths[0]
            route_rec.coordinates = coords
            route_rec.decision_factors = {
                "travel_time_seconds": round(elapsed_time_s, 1),
                "distance_meters": round(primary_dist, 1),
                "hazard_exposure_score": round(mean_haz, 4),
                "traffic_exposure_score": 0.25,
                "agent_conflicts": 0,
                "route_risk_index": round(session.risk_score, 4)
            }

            # Persist or update individual Agent records
            for i, p in enumerate(agent_paths):
                dist = graph_manager.get_path_distance(p)
                ag = db.query(Agent).filter(Agent.session_id == session_id, Agent.agent_index == i).first()
                if not ag:
                    ag = Agent(session_id=session_id, agent_index=i)
                    db.add(ag)

                ag.current_node = p[-1]
                ag.destination_node = session.destination
                ag.status = "evacuated" if success else "evacuating"
                ag.route = p
                ag.distance_travelled = round(dist, 1)
                ag.time_elapsed = round(elapsed_time_s, 1)
                ag.hazard_exposure = round(agent_cumulative_hazard[i] if i < len(agent_cumulative_hazard) else 0.0, 4)

            db.commit()
            logger.info(f"Persisted final evacuation session results for '{session_id}'.")
        except Exception as e:
            logger.error(f"Error persisting session results: {e}", exc_info=True)
            db.rollback()
        finally:
            db.close()

evacuation_service = EvacuationService()
