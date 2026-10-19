"""Multi-Agent Gymnasium Evacuation Environment:
Coordinates N heterogeneous evacuation units traversing a shared road network
under microscopic traffic congestion, dynamic flood/fire hazards, road closures,
and spatial bottleneck contention.
"""

import os
import sys
import math
import copy
import yaml
import random
import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List, Set

from src.utils.logger import setup_logger
from src.environment.evacuation_env import EvacuationEnv

logger = setup_logger("MultiAgentEvacuationEnv")

# Heterogeneous Cohort Definitions
COHORT_TYPES = {
    0: {
        "name": "Type A (Standard)",
        "desc": "Balanced evacuation cohort with standard progress, safety, and delay weights",
        "w_progress": 4.0,
        "w_hazard": 4.0,
        "w_congestion": 1.5,
        "w_closure": 25.0
    },
    1: {
        "name": "Type B (Congestion-Sensitive)",
        "desc": "Heavy bus/convoy evacuation cohort highly sensitive to road bottlenecks and queue delays",
        "w_progress": 3.5,
        "w_hazard": 4.0,
        "w_congestion": 3.5,
        "w_closure": 25.0
    },
    2: {
        "name": "Type C (Hazard-Sensitive / Vulnerable)",
        "desc": "Medical/elderly evacuation cohort with amplified hazard and flood avoidance priorities",
        "w_progress": 3.0,
        "w_hazard": 8.0,
        "w_congestion": 1.5,
        "w_closure": 40.0
    },
    3: {
        "name": "Type D (Long-Distance Perimeter)",
        "desc": "Peripheral evacuation cohort prioritizing high route efficiency and highway corridor progress",
        "w_progress": 6.0,
        "w_hazard": 4.0,
        "w_congestion": 1.0,
        "w_closure": 25.0
    }
}


class AgentState:
    """Encapsulates runtime state and metrics for an individual evacuation agent."""

    def __init__(self, agent_id: int, agent_type: int, origin: str, destination: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.cohort_info = COHORT_TYPES.get(agent_type % len(COHORT_TYPES), COHORT_TYPES[0])
        self.current_node = origin
        self.destination_node = destination
        self.origin_node = origin

        self.step_count = 0
        self.cumulative_hazard = 0.0
        self.cumulative_travel_time = 0.0
        self.cumulative_distance = 0.0
        self.conflict_count = 0
        self.visited_nodes = [origin]
        self.visited_edges: List[str] = []

        self.is_done = False
        self.is_success = False
        self.is_timeout = False
        self.is_dead_end = False


class MultiAgentEvacuationEnv(gym.Env):
    """Multi-Agent Reinforcement Learning Environment for Multi-Hazard Evacuation Routing.

    N heterogeneous agents simultaneously navigate a shared road network.
    Agents interact through:
      1. Dynamic edge congestion coupling (co-occupancy raises edge travel time & delay).
      2. Bottleneck intersection contention (conflicts when multiple agents select the same edge/node).
      3. Shared dynamic hazard propagation (flood/fire cellular automata digital twin).
      4. Joint mission success (all agents evacuating safely).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        n_agents: int = 4,
        network_size: str = "small",
        scenarios: Optional[Any] = None,
        use_traci: bool = False,
        curriculum_stage: int = 5,
        seed: Optional[int] = None
    ):
        super().__init__()

        self.n_agents = n_agents
        self.network_size = network_size

        network_config_map = {
            "small": "configs/default.yaml",
            "large": "configs/large.yaml",
            "dense_grid": "configs/dense_grid.yaml",
            "george_town": "configs/dense_grid.yaml",
            "sparse_suburban": "configs/sparse_suburban.yaml",
            "omr": "configs/sparse_suburban.yaml"
        }
        if network_size in network_config_map and config_path == "configs/default.yaml":
            config_path = network_config_map[network_size]
        self.config_path = config_path

        if isinstance(scenarios, str) and os.path.isfile(scenarios):
            import json
            with open(scenarios, "r", encoding="utf-8") as f:
                scenarios = json.load(f)

        # Core single-agent digital twin environment acting as underlying physical simulator
        self.base_env = EvacuationEnv(
            config_path=config_path,
            scenarios=scenarios,
            use_traci=use_traci,
            curriculum_stage=curriculum_stage
        )

        self.graph = self.base_env.graph
        self.nodes = self.base_env.nodes
        self.sp_lengths = self.base_env.sp_lengths
        self.max_graph_dist = self.base_env.max_graph_dist
        self.max_episode_steps = self.base_env.max_episode_steps
        self.step_dt = self.base_env.step_dt
        self.max_neighbors = self.base_env.max_neighbors  # 4

        # Action and observation spaces
        self.action_spaces = [spaces.Discrete(self.max_neighbors) for _ in range(self.n_agents)]
        self.observation_spaces = [spaces.Box(low=0.0, high=1.0, shape=(34,), dtype=np.float32) for _ in range(self.n_agents)]

        # Global observation space for centralized critics (34 * N + global context)
        global_dim = 34 * self.n_agents + 8
        self.state_space = spaces.Box(low=0.0, high=1.0, shape=(global_dim,), dtype=np.float32)

        self.agents: List[AgentState] = []
        self.current_scenario: Optional[Dict[str, Any]] = None
        self.step_count = 0
        self.total_conflicts = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Resets environment and spawns N heterogeneous evacuation agents."""
        options = options or {}
        scenario = options.get("scenario", None)

        # Reset base environmental simulation (hazard, weather, ambient traffic)
        _, base_info = self.base_env.reset(seed=seed, options=options)
        self.current_scenario = self.base_env.current_scenario

        ep_seed = seed if seed is not None else random.randint(0, 100000)
        rng = random.Random(ep_seed)

        # Spawn N agents with disjoint origins and valid destinations
        self.agents = []
        used_origins: Set[str] = set()

        base_origin = self.current_scenario.get("evac_origin", self.nodes[0])
        base_dest = self.current_scenario.get("evac_destination", self.nodes[-1])

        for i in range(self.n_agents):
            agent_type = i % len(COHORT_TYPES)

            if i == 0:
                u = base_origin
                v = base_dest
            else:
                # Deterministic diverse origins and destinations from scenario seed
                cand_origins = [n for n in self.nodes if n not in used_origins and n != base_dest]
                u = rng.choice(cand_origins) if cand_origins else rng.choice(self.nodes)

                cand_dests = [n for n in self.nodes if n != u and self.sp_lengths.get(u, {}).get(n, 0) > 100.0]
                v = rng.choice(cand_dests) if cand_dests else base_dest

            used_origins.add(u)
            self.agents.append(AgentState(agent_id=i, agent_type=agent_type, origin=u, destination=v))

        self.step_count = 0
        self.total_conflicts = 0

        obs_n = [self._get_agent_observation(i) for i in range(self.n_agents)]
        masks_n = [self.get_agent_action_mask(i) for i in range(self.n_agents)]

        info = {
            "scenario_id": self.current_scenario.get("scenario_id", "unknown"),
            "action_masks": masks_n,
            "agent_types": [a.agent_type for a in self.agents],
            "n_agents": self.n_agents
        }
        return obs_n, info

    def get_agent_action_mask(self, agent_idx: int) -> np.ndarray:
        """Returns boolean action mask for agent: True for open outgoing edges."""
        agent = self.agents[agent_idx]
        if agent.is_done:
            # Done agent can only pick no-op / stay in place (slot 0)
            mask = np.zeros(self.max_neighbors, dtype=bool)
            mask[0] = True
            return mask

        mask = np.zeros(self.max_neighbors, dtype=bool)
        successors = self._get_agent_ordered_successors(agent_idx)

        for idx, succ in enumerate(successors[:self.max_neighbors]):
            edge_id = f"{agent.current_node}_{succ}"
            status = self.base_env.current_hazard_engine.get_edge_status(edge_id)
            if status != self.base_env.current_hazard_engine.STATE_CLOSED:
                mask[idx] = True

        if not np.any(mask) and successors:
            mask[0] = True
        elif not np.any(mask):
            mask[0] = True

        return mask

    def action_masks(self) -> List[np.ndarray]:
        """Returns action masks for all agents."""
        return [self.get_agent_action_mask(i) for i in range(self.n_agents)]

    def get_action_masks(self) -> List[np.ndarray]:
        """Returns action masks for all agents."""
        return self.action_masks()

    def _get_agent_ordered_successors(self, agent_idx: int) -> List[str]:
        """Returns outgoing neighbors sorted by shortest road distance to this agent's destination."""
        agent = self.agents[agent_idx]
        raw_succs = list(self.graph.successors(agent.current_node))

        def sort_key(s):
            return self.sp_lengths.get(s, {}).get(agent.destination_node, 1e9)

        return sorted(raw_succs, key=sort_key)

    def _get_agent_observation(self, agent_idx: int) -> np.ndarray:
        """Constructs 34-dimensional observation vector for specified agent."""
        agent = self.agents[agent_idx]
        curr_data = self.graph.nodes[agent.current_node]
        dest_data = self.graph.nodes[agent.destination_node]

        curr_x = float(curr_data.get("x", curr_data.get("lon", 0.0)))
        curr_y = float(curr_data.get("y", curr_data.get("lat", 0.0)))
        dest_x = float(dest_data.get("x", dest_data.get("lon", 0.0)))
        dest_y = float(dest_data.get("y", dest_data.get("lat", 0.0)))

        curr_x_norm = (curr_x - self.base_env.min_x) / self.base_env.x_range
        curr_y_norm = (curr_y - self.base_env.min_y) / self.base_env.y_range
        dest_x_norm = (dest_x - self.base_env.min_x) / self.base_env.x_range
        dest_y_norm = (dest_y - self.base_env.min_y) / self.base_env.y_range

        rem_dist = self.sp_lengths.get(agent.current_node, {}).get(agent.destination_node, 0.0)
        rem_dist_norm = min(1.0, rem_dist / max(self.max_graph_dist, 1e-5))

        # Hazard features
        curr_haz = self.base_env.current_hazard_engine.get_intensity(agent.current_node)
        successors = list(self.graph.successors(agent.current_node))
        out_hazs = [
            self.base_env.current_hazard_engine.get_intensity(f"{agent.current_node}_{succ}")
            for succ in successors
        ]
        mean_out_haz = float(np.mean(out_hazs)) if out_hazs else curr_haz

        # Road closure fractions
        ordered_succs = self._get_agent_ordered_successors(agent_idx)
        if ordered_succs:
            statuses = [
                self.base_env.current_hazard_engine.get_edge_status(f"{agent.current_node}_{succ}")
                for succ in ordered_succs
            ]
            frac_closed = sum(1 for s in statuses if s == self.base_env.current_hazard_engine.STATE_CLOSED) / len(statuses)
            frac_degraded = sum(1 for s in statuses if s == self.base_env.current_hazard_engine.STATE_DEGRADED) / len(statuses)
        else:
            frac_closed = 1.0
            frac_degraded = 0.0

        # Ambient traffic plus inter-agent local density
        peer_occupants = sum(1 for other in self.agents if other.agent_id != agent.agent_id and other.current_node == agent.current_node)
        base_cong, avg_speed, queue_len = self.base_env._get_traffic_metrics()
        coupled_congestion = min(1.0, base_cong + 0.15 * peer_occupants)

        time_norm = min(1.0, self.step_count / float(self.max_episode_steps))
        degree_norm = min(1.0, len(ordered_succs) / float(self.max_neighbors))

        base_features = [
            curr_x_norm, curr_y_norm, dest_x_norm, dest_y_norm, rem_dist_norm,
            curr_haz, mean_out_haz, frac_closed, frac_degraded, coupled_congestion,
            avg_speed, queue_len, time_norm, degree_norm
        ]

        # Candidate edge embeddings (4 slots * 5 features = 20 features)
        candidate_features = []
        goal_dx = dest_x - curr_x
        goal_dy = dest_y - curr_y
        goal_len = math.hypot(goal_dx, goal_dy)

        for k in range(self.max_neighbors):
            if k < len(ordered_succs):
                succ = ordered_succs[k]
                succ_data = self.graph.nodes[succ]
                sx = float(succ_data.get("x", succ_data.get("lon", 0.0)))
                sy = float(succ_data.get("y", succ_data.get("lat", 0.0)))
                edge_id = f"{agent.current_node}_{succ}"
                status = self.base_env.current_hazard_engine.get_edge_status(edge_id)
                is_valid = 1.0 if status != self.base_env.current_hazard_engine.STATE_CLOSED else 0.0

                road_dx = sx - curr_x
                road_dy = sy - curr_y
                road_len = math.hypot(road_dx, road_dy)
                if road_len > 1e-6 and goal_len > 1e-6:
                    alignment = ((road_dx * goal_dx + road_dy * goal_dy) / (road_len * goal_len) + 1.0) / 2.0
                else:
                    alignment = 0.5

                succ_dest_dist = self.sp_lengths.get(succ, {}).get(agent.destination_node, self.max_graph_dist)
                dist_gain = min(1.0, max(0.0, ((rem_dist - succ_dest_dist) / max(self.max_graph_dist * 0.05, 10.0) + 1.0) / 2.0))
                cand_haz = self.base_env.current_hazard_engine.get_intensity(edge_id)

                # Peer intent congestion: check how many peers are at this successor
                succ_occupants = sum(1 for o in self.agents if o.agent_id != agent.agent_id and o.current_node == succ)
                cand_cong = min(1.0, coupled_congestion + 0.15 * succ_occupants)

                candidate_features.extend([is_valid, alignment, dist_gain, cand_haz, cand_cong])
            else:
                candidate_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        obs = np.array(base_features + candidate_features, dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def get_global_state(self) -> np.ndarray:
        """Constructs global centralized state vector for critics."""
        agent_obs_flat = np.concatenate([self._get_agent_observation(i) for i in range(self.n_agents)])
        # Global context: [active_agents, mean_hazard, closed_edge_frac, total_conflicts, time_norm, mean_dist_norm, joint_success_prob, congestion]
        mean_haz = float(np.mean([self.base_env.current_hazard_engine.get_intensity(a.current_node) for a in self.agents]))
        active_frac = sum(1 for a in self.agents if not a.is_done) / float(self.n_agents)
        conflicts_norm = min(1.0, self.total_conflicts / 20.0)
        time_norm = min(1.0, self.step_count / float(self.max_episode_steps))
        mean_dist_norm = float(np.mean([
            min(1.0, self.sp_lengths.get(a.current_node, {}).get(a.destination_node, 0.0) / max(self.max_graph_dist, 1.0))
            for a in self.agents
        ]))
        global_ctx = np.array([
            active_frac, mean_haz, conflicts_norm, time_norm, mean_dist_norm,
            float(self.total_conflicts), float(self.n_agents), 0.5
        ], dtype=np.float32)

        return np.concatenate([agent_obs_flat, global_ctx])

    def step(
        self,
        actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], Dict[str, Any]]:
        """Executes simultaneous joint action across all N agents."""
        self.step_count += 1
        masks_before = self.action_masks()

        # Enforce Hard Action Masking Assertion for all agents
        for i, (act, mask) in enumerate(zip(actions, masks_before)):
            assert mask[act], f"CRITICAL MARL INVARIANT VIOLATION: Agent {i} selected invalid masked action {act}"

        # 1. Advance physical digital twin (hazard diffusion & traffic step)
        self.base_env._advance_digital_twin()

        # 2. Track intended transitions and detect bottleneck contention
        intended_edges: List[Optional[str]] = []
        intended_next_nodes: List[str] = []

        for i, agent in enumerate(self.agents):
            if agent.is_done:
                intended_edges.append(None)
                intended_next_nodes.append(agent.current_node)
                continue

            successors = self._get_agent_ordered_successors(i)
            action = actions[i]

            if action < len(successors):
                next_node = successors[action]
                edge_id = f"{agent.current_node}_{next_node}"
            else:
                next_node = agent.current_node
                edge_id = None

            intended_edges.append(edge_id)
            intended_next_nodes.append(next_node)

        # 3. Detect Inter-Agent Conflicts (concurrent edge or node occupancy contention)
        edge_counts: Dict[str, int] = {}
        node_counts: Dict[str, int] = {}

        for edge_id in intended_edges:
            if edge_id:
                edge_counts[edge_id] = edge_counts.get(edge_id, 0) + 1

        for next_node in intended_next_nodes:
            node_counts[next_node] = node_counts.get(next_node, 0) + 1

        step_conflicts = 0
        for edge_id, count in edge_counts.items():
            if count > 1:
                step_conflicts += (count - 1)

        self.total_conflicts += step_conflicts

        # 4. Update Agent Positions and Compute Individual Rewards
        rewards: List[float] = []
        terminations: List[bool] = []
        truncations: List[bool] = []

        for i, agent in enumerate(self.agents):
            if agent.is_done:
                rewards.append(0.0)
                terminations.append(agent.is_success or agent.is_dead_end)
                truncations.append(agent.is_timeout)
                continue

            prev_node = agent.current_node
            next_node = intended_next_nodes[i]
            edge_id = intended_edges[i]

            prev_dist = self.sp_lengths.get(prev_node, {}).get(agent.destination_node, 0.0)
            new_dist = self.sp_lengths.get(next_node, {}).get(agent.destination_node, 0.0)
            dist_delta = prev_dist - new_dist

            # Move agent
            agent.current_node = next_node
            agent.visited_nodes.append(next_node)
            if edge_id:
                agent.visited_edges.append(edge_id)

            # Cumulative metrics
            edge_hazard = self.base_env.current_hazard_engine.get_intensity(next_node)
            agent.cumulative_hazard += edge_hazard * self.step_dt

            edge_len = 50.0
            if edge_id and self.graph.has_edge(prev_node, next_node):
                edge_len = self.graph[prev_node][next_node].get("length", 50.0)
            agent.cumulative_distance += edge_len

            # Congestion coupling: increase travel time based on peer road occupancy
            shared_edge_occupancy = edge_counts.get(edge_id, 1) if edge_id else 1
            local_congestion = min(1.0, 0.2 + 0.25 * (shared_edge_occupancy - 1))
            agent.cumulative_travel_time += self.step_dt * (1.0 + local_congestion)

            if shared_edge_occupancy > 1:
                agent.conflict_count += (shared_edge_occupancy - 1)

            # Termination checks
            agent.is_success = (agent.current_node == agent.destination_node)
            agent.is_dead_end = (len(list(self.graph.successors(agent.current_node))) == 0 and not agent.is_success)
            agent.is_timeout = (self.step_count >= self.max_episode_steps and not agent.is_success)

            term = agent.is_success or agent.is_dead_end
            trunc = agent.is_timeout
            agent.is_done = term or trunc

            # Heterogeneous Decomposed Reward Function
            c_info = agent.cohort_info
            w_prog = c_info["w_progress"]
            w_haz = c_info["w_hazard"]
            w_cong = c_info["w_congestion"]
            w_close = c_info["w_closure"]

            scale = max(self.max_graph_dist * 0.05, 10.0)
            r_prog = w_prog * (dist_delta / scale)
            r_arrival = 100.0 if agent.is_success else 0.0
            p_haz = edge_hazard * w_haz
            p_cong = local_congestion * w_cong
            p_conflict = 3.0 * (shared_edge_occupancy - 1)
            p_dead_end = 30.0 if agent.is_dead_end else 0.0
            p_timeout = 40.0 if trunc else 0.0
            p_delay = 0.5

            agent_reward = (
                r_prog + r_arrival
                - p_haz - p_cong - p_conflict
                - p_dead_end - p_timeout - p_delay
            )

            # Joint success coordination bonus
            if all(a.is_success for a in self.agents):
                agent_reward += 25.0

            rewards.append(float(agent_reward))
            terminations.append(term)
            truncations.append(trunc)

        # 5. Multi-Agent Coordination Metrics
        joint_done = all(a.is_done for a in self.agents)
        joint_success = all(a.is_success for a in self.agents)
        indiv_success_rate = sum(1 for a in self.agents if a.is_success) / float(self.n_agents)

        # Route overlap (Jaccard similarity across visited edges)
        all_pair_overlaps = []
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                edges_i = set(self.agents[i].visited_edges)
                edges_j = set(self.agents[j].visited_edges)
                union = len(edges_i.union(edges_j))
                if union > 0:
                    all_pair_overlaps.append(len(edges_i.intersection(edges_j)) / float(union))
                else:
                    all_pair_overlaps.append(0.0)
        mean_route_overlap = float(np.mean(all_pair_overlaps)) if all_pair_overlaps else 0.0

        # Mean inter-agent graph distance
        pair_distances = []
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                u = self.agents[i].current_node
                v = self.agents[j].current_node
                d = self.sp_lengths.get(u, {}).get(v, self.max_graph_dist)
                pair_distances.append(d)
        mean_inter_agent_dist = float(np.mean(pair_distances)) if pair_distances else 0.0

        obs_next_n = [self._get_agent_observation(i) for i in range(self.n_agents)]
        masks_next_n = [self.get_agent_action_mask(i) for i in range(self.n_agents)]

        mean_hei = float(np.mean([a.cumulative_hazard for a in self.agents]))
        mean_cong = float(np.mean([float(self.base_env.current_scenario.get("ambient_density", 0.3) if self.base_env.current_scenario else 0.3)]))

        info = {
            "scenario_id": self.current_scenario.get("scenario_id", "unknown"),
            "step_count": self.step_count,
            "action_masks": masks_next_n,
            "joint_success": joint_success,
            "individual_success_rate": indiv_success_rate,
            "mean_hei": mean_hei,
            "inter_agent_congestion": mean_cong,
            "bottleneck_conflicts": self.total_conflicts,
            "route_overlap_ratio": mean_route_overlap,
            "step_conflicts": step_conflicts,
            "total_conflicts": self.total_conflicts,
            "conflict_rate": self.total_conflicts / max(1, self.step_count),
            "mean_route_overlap": mean_route_overlap,
            "mean_inter_agent_distance": mean_inter_agent_dist,
            "coordination_efficiency": float(indiv_success_rate / (1.0 + 0.1 * self.total_conflicts)),
            "agent_summaries": [
                {
                    "agent_id": a.agent_id,
                    "agent_type": a.agent_type,
                    "success": a.is_success,
                    "travel_time": a.cumulative_travel_time,
                    "hei": a.cumulative_hazard,
                    "distance": a.cumulative_distance,
                    "timeout": a.is_timeout,
                    "conflicts": a.conflict_count
                }
                for a in self.agents
            ]
        }

        return obs_next_n, rewards, terminations, truncations, info

    def close(self):
        """Closes simulator."""
        self.base_env.close()
