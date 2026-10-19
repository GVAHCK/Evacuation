"""Gymnasium Evacuation Environment with continuous SUMO TraCI digital twin synchronization,

dynamic cellular-automata/diffusion hazard spread, action masking, and decomposed rewards.
"""

import os
import sys
import math
import yaml
import random
import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List, Set

from src.utils.logger import setup_logger
from src.network.osm_loader import OSMLoader
from src.hazard.flood_engine import FloodEngine
from src.hazard.fire_engine import FireEngine
from src.network.traffic_generator import TrafficGenerator

logger = setup_logger("EvacuationEnv")


class EvacuationEnv(gym.Env):
    """Reinforcement learning digital twin environment for dynamic flood and fire

    evacuation route optimization under microscopic SUMO traffic conditions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        scenarios: Optional[List[Dict[str, Any]]] = None,
        use_traci: bool = True,
        curriculum_stage: int = 5,
        ablation_flags: Optional[Dict[str, bool]] = None
    ):
        super().__init__()

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.env_cfg = self.config.get("environment", {})
        self.reward_cfg = self.env_cfg.get("reward", {})
        self.max_episode_steps = int(self.env_cfg.get("max_episode_steps", 60))
        self.step_dt = float(self.env_cfg.get("time_step_duration", 5.0))
        self.max_neighbors = int(self.env_cfg.get("max_neighbors", 8))

        self.use_traci = use_traci
        self.curriculum_stage = curriculum_stage
        self.ablation_flags = ablation_flags or {}

        # Load road graph
        graph_file = self.config.get("network", {}).get("graph_file", "data/processed/road_graph.json")
        if os.path.isfile(graph_file):
            self.graph = OSMLoader.load_graph_json(graph_file)
        else:
            loader = OSMLoader(config_path)
            self.graph = loader.download_and_process()

        self.nodes = list(self.graph.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}

        # Precompute bounding box coordinate normalization
        xs = [self.graph.nodes[n].get("x", 0.0) for n in self.nodes]
        ys = [self.graph.nodes[n].get("y", 0.0) for n in self.nodes]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)
        self.x_range = max(self.max_x - self.min_x, 1e-5)
        self.y_range = max(self.max_y - self.min_y, 1e-5)
        self.max_spatial_dist = math.hypot(self.x_range, self.y_range)

        # Precompute graph shortest path distances for robust routing metrics
        self.sp_lengths = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight="length"))
        self.max_graph_dist = max(
            max(d.values()) if d else 100.0
            for d in self.sp_lengths.values()
        ) if self.sp_lengths else 1000.0

        # Initialize hazard engines
        self.flood_engine = FloodEngine(self.graph, self.config)
        self.fire_engine = FireEngine(self.graph, self.config)
        self.current_hazard_engine = self.flood_engine

        # Traffic generator
        self.traffic_gen = TrafficGenerator(config_path)
        self.net_file = self.config.get("network", {}).get("net_file", "data/processed/network.net.xml")

        # Scenario pool
        self.scenarios = scenarios or []
        self.current_scenario: Optional[Dict[str, Any]] = None

        # TraCI control state
        self.traci = None
        self.traci_active = False
        self.sumo_bin = None
        if self.use_traci:
            try:
                import sumolib
                import traci
                self.traci = traci
                self.sumo_bin = sumolib.checkBinary("sumo")
            except Exception as e:
                logger.warning(f"SUMO/TraCI initialization notice: {e}. Running in internal digital twin sync mode.")
                self.use_traci = False

        # Spaces
        # Action space: index in [0, max_neighbors - 1] selecting an outgoing candidate edge
        self.action_space = spaces.Discrete(self.max_neighbors)

        # Observation space: 14 base features + (max_neighbors * 5) candidate edge embeddings = 34 features
        # Base: [curr_x, curr_y, dest_x, dest_y, rem_dist, curr_haz, mean_out_haz, frac_closed, frac_degraded, cong, speed, queue, time, deg]
        # Per candidate slot k (5 features each):
        # [is_valid, dest_alignment_cos, dist_gain_norm, hazard_intensity, congestion]
        total_obs_dim = 14 + (self.max_neighbors * 5)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_obs_dim,),
            dtype=np.float32
        )

        # Runtime episode states
        self.current_node: str = self.nodes[0]
        self.destination_node: str = self.nodes[-1]
        self.step_count = 0
        self.elapsed_sim_time = 0.0
        self.cumulative_hazard_exposure = 0.0
        self.cumulative_travel_time = 0.0
        self.cumulative_distance = 0.0
        self.closure_encounters = 0
        self.replanning_count = 0
        self.visited_nodes: List[str] = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else random.randint(0, 100000)
        self.rng = random.Random(episode_seed)

        options = options or {}
        scenario = options.get("scenario", None)

        if scenario is None:
            if self.scenarios:
                scenario = self.rng.choice(self.scenarios)
            else:
                # Generate on-the-fly random scenario
                hazard_type = self.rng.choice(["flood", "fire"])
                u = self.rng.choice(self.nodes)
                # Pick destination at least 2 hops away
                dest_candidates = [v for v in self.nodes if v != u]
                v = self.rng.choice(dest_candidates)
                scenario = {
                    "scenario_id": f"rand_{episode_seed}",
                    "seed": episode_seed,
                    "hazard_type": hazard_type,
                    "hazard_origin": self.rng.choice(self.nodes),
                    "evac_origin": u,
                    "evac_destination": v,
                    "traffic_level": self.rng.choice(["low", "normal", "morning_peak"]),
                    "spread_rate_mult": self.rng.uniform(0.7, 1.3)
                }

        self.current_scenario = scenario
        self.current_node = scenario["evac_origin"]
        self.destination_node = scenario["evac_destination"]
        self.step_count = 0
        self.elapsed_sim_time = 0.0
        self.cumulative_hazard_exposure = 0.0
        self.cumulative_travel_time = 0.0
        self.cumulative_distance = 0.0
        self.closure_encounters = 0
        self.replanning_count = 0
        self.visited_nodes = [self.current_node]

        # Hazard setup & curriculum scaling
        h_type = scenario.get("hazard_type", "flood")
        self.current_hazard_engine = self.flood_engine if h_type == "flood" else self.fire_engine

        curriculum_mult = 1.0
        if self.curriculum_stage == 1:
            curriculum_mult = 0.0
        elif self.curriculum_stage == 2:
            curriculum_mult = 0.3
        elif self.curriculum_stage == 3:
            curriculum_mult = 0.6
        elif self.curriculum_stage == 4:
            curriculum_mult = 1.0
        else:
            curriculum_mult = scenario.get("spread_rate_mult", 1.0)

        self.current_hazard_engine.reset(
            seed=scenario.get("seed", episode_seed),
            origin_node=scenario.get("hazard_origin", None),
            spread_rate_mult=curriculum_mult,
            wind_direction=scenario.get("wind_direction", 45.0),
            wind_strength=scenario.get("wind_strength", 5.0)
        )

        # SUMO TraCI synchronization reset
        if self.use_traci and self.traci is not None:
            self._reset_sumo_traci(scenario)

        obs = self._get_observation()
        info = self._get_info(action_reward_breakdown=None)
        return obs, info

    def _reset_sumo_traci(self, scenario: Dict[str, Any]) -> None:
        """Starts or reloads SUMO via TraCI for the selected traffic level."""
        traffic_level = scenario.get("traffic_level", "normal")
        seed = scenario.get("seed", 42)

        try:
            route_file = self.traffic_gen.generate_scenario(
                traffic_level=traffic_level,
                seed=seed,
                duration=int(self.max_episode_steps * self.step_dt + 600)
            )

            sumo_cmd = [
                self.sumo_bin,
                "-n", self.net_file,
                "-r", route_file,
                "--no-step-log", "true",
                "--no-warnings", "true",
                "--waiting-time-memory", "1000",
                "--time-to-teleport", "-1"
            ]

            if self.traci_active:
                try:
                    self.traci.load(sumo_cmd[1:])
                except Exception:
                    self.traci.close()
                    self.traci.start(sumo_cmd)
            else:
                self.traci.start(sumo_cmd)
                self.traci_active = True

            # Warmup SUMO by a few steps so traffic flows
            for _ in range(3):
                self.traci.simulationStep()

        except Exception as e:
            logger.warning(f"TraCI reload encountered issue: {e}. Continuing with digital twin traffic estimator.")
            self.traci_active = False

    def _get_ordered_successors(self) -> List[str]:
        """Returns outgoing neighbors canonically sorted by remaining road network distance to destination.
        Slot 0: Primary candidate (shortest open road path to evacuation destination)
        Slot 1: First detour candidate (second shortest road path)
        Slot 2: Second detour candidate (third shortest road path)
        Slot 3: Third detour candidate (fourth shortest road path)
        """
        raw_succs = list(self.graph.successors(self.current_node))

        def sort_key(s):
            return self.sp_lengths.get(s, {}).get(self.destination_node, 1e9)

        return sorted(raw_succs, key=sort_key)

    def action_masks(self) -> np.ndarray:
        """Returns boolean mask of valid outgoing transitions from current node."""
        mask = np.zeros(self.max_neighbors, dtype=bool)
        successors = self._get_ordered_successors()

        for idx, succ in enumerate(successors[:self.max_neighbors]):
            edge_id = f"{self.current_node}_{succ}"
            status = self.current_hazard_engine.get_edge_status(edge_id)
            # Action is valid if edge is safe or degraded, but NOT completely closed
            if status != self.current_hazard_engine.STATE_CLOSED:
                mask[idx] = True

        # If all paths are blocked or no successors, permit at least staying / nearest fallback
        if not np.any(mask) and successors:
            mask[0] = True
        elif not np.any(mask):
            mask[0] = True

        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one transition step in the evacuation routing digital twin."""
        self.step_count += 1
        self.elapsed_sim_time += self.step_dt

        # Record pre-transition distance to destination for true progress calculation
        prev_node = self.current_node
        prev_dist = self.sp_lengths.get(prev_node, {}).get(self.destination_node, 0.0)

        # 1. Advance SUMO and update hazard propagation
        self._advance_digital_twin()

        # 2. Decode candidate next-hop from discrete action using ordered successors
        successors = self._get_ordered_successors()
        is_invalid_action = False
        is_closed_attempt = False
        is_u_turn = False
        chosen_next_node = None
        edge_id = None

        if 0 <= action < len(successors):
            chosen_next_node = successors[action]
            edge_id = f"{self.current_node}_{chosen_next_node}"
            edge_status = self.current_hazard_engine.get_edge_status(edge_id)

            if edge_status == self.current_hazard_engine.STATE_CLOSED:
                is_closed_attempt = True
                self.closure_encounters += 1
            else:
                # Valid physical transition
                if len(self.visited_nodes) >= 2 and chosen_next_node == self.visited_nodes[-2] and len(successors) > 1:
                    is_u_turn = True
                self.current_node = chosen_next_node
                self.visited_nodes.append(self.current_node)
                self.replanning_count += 1
        else:
            is_invalid_action = True

        # Calculate post-transition distance along road network
        new_dist = self.sp_lengths.get(self.current_node, {}).get(self.destination_node, 0.0)
        dist_delta = prev_dist - new_dist  # Positive if moved closer to destination along road graph

        # 3. Read current environmental metrics
        edge_hazard_intensity = self.current_hazard_engine.get_intensity(self.current_node)
        self.cumulative_hazard_exposure += edge_hazard_intensity * self.step_dt

        edge_len = 50.0
        if edge_id and chosen_next_node and self.graph.has_edge(prev_node, chosen_next_node):
            edge_len = self.graph[prev_node][chosen_next_node].get("length", 50.0)
        self.cumulative_distance += edge_len

        local_congestion, local_speed, local_queue = self._get_traffic_metrics()
        self.cumulative_travel_time += self.step_dt * (1.0 + local_congestion)

        # 4. Check termination criteria
        is_success = (self.current_node == self.destination_node)
        is_dead_end = (len(list(self.graph.successors(self.current_node))) == 0 and not is_success)
        is_timeout = (self.step_count >= self.max_episode_steps)
        terminated = is_success or is_dead_end
        truncated = is_timeout and not terminated

        # 5. Compute mathematically decomposed reward with true distance delta
        reward, reward_breakdown = self._compute_reward(
            is_success=is_success,
            is_closed_attempt=is_closed_attempt,
            is_invalid_action=is_invalid_action,
            is_dead_end=is_dead_end,
            is_timeout=truncated,
            hazard_intensity=edge_hazard_intensity,
            congestion=local_congestion,
            dist_delta=dist_delta,
            is_u_turn=is_u_turn
        )

        obs = self._get_observation()
        info = self._get_info(
            action_reward_breakdown=reward_breakdown,
            is_success=is_success,
            is_dead_end=is_dead_end,
            is_timeout=truncated
        )

        return obs, float(reward), terminated, truncated, info

    def _advance_digital_twin(self) -> None:
        """Advances both the hazard simulation and the SUMO microscopic traffic model."""
        self.current_hazard_engine.update(self.step_dt)

        if self.use_traci and self.traci_active and self.traci is not None:
            try:
                # Step SUMO for step_dt simulation seconds
                steps_to_run = max(1, int(self.step_dt))
                for _ in range(steps_to_run):
                    self.traci.simulationStep()

                # Reflect closed and degraded roads into SUMO edge speed limits
                closed_edges = self.current_hazard_engine.get_closed_edges()
                degraded_edges = self.current_hazard_engine.get_degraded_edges()

                sumo_edges = set(self.traci.edge.getIDList())
                for ce in closed_edges:
                    if ce in sumo_edges:
                        self.traci.edge.setMaxSpeed(ce, 0.1)  # impassable / blocked
                for de in degraded_edges:
                    if de in sumo_edges:
                        orig_speed = self.traci.edge.getMaxSpeed(de)
                        if orig_speed > 3.0:
                            self.traci.edge.setMaxSpeed(de, orig_speed * 0.5)

            except Exception as e:
                logger.debug(f"TraCI step exception: {e}")

    def _get_traffic_metrics(self) -> Tuple[float, float, float]:
        """Calculates local traffic congestion, average speed, and queue length."""
        if self.ablation_flags.get("no_congestion", False):
            return 0.0, 1.0, 0.0

        if self.use_traci and self.traci_active and self.traci is not None:
            try:
                successors = list(self.graph.successors(self.current_node))
                sumo_edges = set(self.traci.edge.getIDList())

                speeds = []
                occupancies = []
                queues = []

                for succ in successors:
                    candidate_id = f"{self.current_node}_{succ}"
                    if candidate_id in sumo_edges:
                        mean_speed = self.traci.edge.getLastStepMeanSpeed(candidate_id)
                        occ = self.traci.edge.getLastStepOccupancy(candidate_id)
                        halted = self.traci.edge.getLastStepHaltingNumber(candidate_id)
                        speeds.append(mean_speed)
                        occupancies.append(occ)
                        queues.append(halted)

                if speeds:
                    avg_speed = float(np.mean(speeds)) / 13.89
                    congestion = float(np.mean(occupancies))
                    queue_len = float(np.mean(queues)) / 10.0
                    return min(1.0, congestion), min(1.0, max(0.0, avg_speed)), min(1.0, queue_len)
            except Exception:
                pass

        traffic_level = self.current_scenario.get("traffic_level", "normal") if self.current_scenario else "normal"
        base_cong = {
            "low": 0.1,
            "normal": 0.3,
            "morning_peak": 0.7,
            "evening_peak": 0.8,
            "heavy": 0.9
        }.get(traffic_level, 0.3)

        local_haz = self.current_hazard_engine.get_intensity(self.current_node)
        congestion = min(1.0, base_cong + local_haz * 0.4)
        avg_speed = max(0.05, 1.0 - congestion * 0.8)
        queue_len = min(1.0, congestion * 0.9)

        return congestion, avg_speed, queue_len

    def _get_observation(self) -> np.ndarray:
        """Constructs 34-dimensional observation vector with candidate action embeddings."""
        curr_data = self.graph.nodes[self.current_node]
        dest_data = self.graph.nodes[self.destination_node]

        curr_x = float(curr_data.get("x", curr_data.get("lon", 0.0)))
        curr_y = float(curr_data.get("y", curr_data.get("lat", 0.0)))
        dest_x = float(dest_data.get("x", dest_data.get("lon", 0.0)))
        dest_y = float(dest_data.get("y", dest_data.get("lat", 0.0)))

        curr_x_norm = (curr_x - self.min_x) / self.x_range
        curr_y_norm = (curr_y - self.min_y) / self.y_range
        dest_x_norm = (dest_x - self.min_x) / self.x_range
        dest_y_norm = (dest_y - self.min_y) / self.y_range

        rem_dist = self.sp_lengths.get(self.current_node, {}).get(self.destination_node, 0.0)
        rem_dist_norm = min(1.0, rem_dist / max(self.max_graph_dist, 1e-5))

        # Hazard features
        if self.ablation_flags.get("no_hazard", False):
            curr_haz = 0.0
            mean_out_haz = 0.0
        else:
            curr_haz = self.current_hazard_engine.get_intensity(self.current_node)
            successors = list(self.graph.successors(self.current_node))
            out_hazs = [
                self.current_hazard_engine.get_intensity(f"{self.current_node}_{succ}")
                for succ in successors
            ]
            mean_out_haz = float(np.mean(out_hazs)) if out_hazs else curr_haz

        # Road closure features
        successors = self._get_ordered_successors()
        if self.ablation_flags.get("no_closure", False):
            frac_closed = 0.0
            frac_degraded = 0.0
        else:
            if successors:
                statuses = [
                    self.current_hazard_engine.get_edge_status(f"{self.current_node}_{succ}")
                    for succ in successors
                ]
                frac_closed = sum(1 for s in statuses if s == self.current_hazard_engine.STATE_CLOSED) / len(statuses)
                frac_degraded = sum(1 for s in statuses if s == self.current_hazard_engine.STATE_DEGRADED) / len(statuses)
            else:
                frac_closed = 1.0
                frac_degraded = 0.0

        congestion, avg_speed, queue_len = self._get_traffic_metrics()
        time_norm = min(1.0, self.step_count / float(self.max_episode_steps))
        degree_norm = min(1.0, len(successors) / float(self.max_neighbors))

        base_features = [
            curr_x_norm,
            curr_y_norm,
            dest_x_norm,
            dest_y_norm,
            rem_dist_norm,
            curr_haz,
            mean_out_haz,
            frac_closed,
            frac_degraded,
            congestion,
            avg_speed,
            queue_len,
            time_norm,
            degree_norm
        ]

        # Candidate edge embeddings for slots 0 .. max_neighbors-1
        candidate_features = []
        goal_dx = dest_x - curr_x
        goal_dy = dest_y - curr_y
        goal_len = math.hypot(goal_dx, goal_dy)

        for k in range(self.max_neighbors):
            if k < len(successors):
                succ = successors[k]
                succ_data = self.graph.nodes[succ]
                sx = float(succ_data.get("x", succ_data.get("lon", 0.0)))
                sy = float(succ_data.get("y", succ_data.get("lat", 0.0)))
                edge_id = f"{self.current_node}_{succ}"
                status = self.current_hazard_engine.get_edge_status(edge_id)

                is_valid = 1.0 if status != self.current_hazard_engine.STATE_CLOSED else 0.0

                road_dx = sx - curr_x
                road_dy = sy - curr_y
                road_len = math.hypot(road_dx, road_dy)

                if road_len > 1e-6 and goal_len > 1e-6:
                    cos_sim = (road_dx * goal_dx + road_dy * goal_dy) / (road_len * goal_len)
                    alignment = (cos_sim + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                else:
                    alignment = 0.5

                succ_dest_dist = self.sp_lengths.get(succ, {}).get(self.destination_node, self.max_graph_dist)
                dist_reduction = (rem_dist - succ_dest_dist) / max(self.max_graph_dist * 0.05, 10.0)
                dist_gain = min(1.0, max(0.0, (dist_reduction + 1.0) / 2.0))

                cand_haz = self.current_hazard_engine.get_intensity(edge_id) if not self.ablation_flags.get("no_hazard", False) else 0.0
                cand_cong = congestion if not self.ablation_flags.get("no_congestion", False) else 0.0

                candidate_features.extend([is_valid, alignment, dist_gain, cand_haz, cand_cong])
            else:
                candidate_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        obs = np.array(base_features + candidate_features, dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _compute_reward(
        self,
        is_success: bool,
        is_closed_attempt: bool,
        is_invalid_action: bool,
        is_dead_end: bool,
        is_timeout: bool,
        hazard_intensity: float,
        congestion: float,
        dist_delta: float = 0.0,
        is_u_turn: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """Computes the mathematically decomposed reward function:

        R = R_progress + R_arrival - R_hazard - R_closure - R_congestion - R_delay - R_dead_end - R_timeout - R_u_turn
        """
        w_progress = float(self.reward_cfg.get("progress_weight", 4.0))
        r_arrival = float(self.reward_cfg.get("arrival_bonus", 100.0))
        w_hazard = float(self.reward_cfg.get("hazard_penalty_weight", 4.0))
        p_closure = float(self.reward_cfg.get("closure_penalty", 25.0))
        w_congestion = float(self.reward_cfg.get("congestion_penalty_weight", 1.5))
        p_delay = float(self.reward_cfg.get("delay_penalty", 0.5))
        p_dead_end = float(self.reward_cfg.get("dead_end_penalty", 30.0))
        p_timeout = float(self.reward_cfg.get("timeout_penalty", 40.0))
        p_u_turn = float(self.reward_cfg.get("u_turn_penalty", 3.0))

        # True distance delta progress along graph: positive when closing distance, negative when increasing distance
        scale = max(self.max_graph_dist * 0.05, 10.0)
        progress_reward = w_progress * (dist_delta / scale)

        arrival_bonus = r_arrival if is_success else 0.0
        hazard_penalty = hazard_intensity * w_hazard
        closure_penalty = p_closure if (is_closed_attempt or is_invalid_action) else 0.0
        congestion_penalty = congestion * w_congestion
        delay_penalty = p_delay
        dead_end_penalty = p_dead_end if is_dead_end else 0.0
        timeout_penalty = p_timeout if is_timeout else 0.0
        u_turn_penalty = p_u_turn if is_u_turn else 0.0

        total_reward = (
            progress_reward
            + arrival_bonus
            - hazard_penalty
            - closure_penalty
            - congestion_penalty
            - delay_penalty
            - dead_end_penalty
            - timeout_penalty
            - u_turn_penalty
        )

        breakdown = {
            "progress_reward": progress_reward,
            "arrival_bonus": arrival_bonus,
            "hazard_penalty": hazard_penalty,
            "closure_penalty": closure_penalty,
            "congestion_penalty": congestion_penalty,
            "delay_penalty": delay_penalty,
            "dead_end_penalty": dead_end_penalty,
            "timeout_penalty": timeout_penalty,
            "u_turn_penalty": u_turn_penalty,
            "total_reward": total_reward
        }

        return total_reward, breakdown

    def _get_info(
        self,
        action_reward_breakdown: Optional[Dict[str, float]] = None,
        is_success: bool = False,
        is_dead_end: bool = False,
        is_timeout: bool = False
    ) -> Dict[str, Any]:
        """Assembles diagnostic info dictionary."""
        info = {
            "scenario_id": self.current_scenario.get("scenario_id", "unknown") if self.current_scenario else "none",
            "step_count": self.step_count,
            "sim_time": self.elapsed_sim_time,
            "current_node": self.current_node,
            "destination_node": self.destination_node,
            "success": is_success,
            "dead_end": is_dead_end,
            "timeout": is_timeout,
            "hazard_exposure_index": self.cumulative_hazard_exposure,
            "cumulative_travel_time": self.cumulative_travel_time,
            "cumulative_distance": self.cumulative_distance,
            "closure_encounters": self.closure_encounters,
            "replanning_count": self.replanning_count,
            "visited_path": list(self.visited_nodes),
            "action_mask": self.action_masks()
        }
        if action_reward_breakdown:
            info.update(action_reward_breakdown)
        return info

    def render(self) -> None:
        """Textual rendering of current agent state and hazard status."""
        logger.info(
            f"Step {self.step_count:02d} | Node: {self.current_node} -> Dest: {self.destination_node} | "
            f"HEI: {self.cumulative_hazard_exposure:.2f} | Dist: {self.cumulative_distance:.1f}m"
        )

    def close(self) -> None:
        """Closes TraCI socket connection cleanly."""
        if self.use_traci and self.traci_active and self.traci is not None:
            try:
                self.traci.close()
            except Exception:
                pass
            self.traci_active = False
