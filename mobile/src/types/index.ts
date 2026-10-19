export interface User {
  id: string;
  name: string;
  email: string;
  role: "USER" | "RESEARCHER" | "ADMIN";
}

export interface RouteDecisionFactors {
  travel_time_seconds: number;
  distance_meters: number;
  hazard_exposure_score: number;
  traffic_exposure_score: number;
  agent_conflicts: number;
  route_risk_index: number;
}

export interface AgentRoute {
  agent_id: number;
  agent_type: string;
  origin_node: string;
  destination_node: string;
  nodes: string[];
  coordinates: [number, number][]; // [lat, lon]
  distance_meters: number;
  estimated_time_seconds: number;
  hazard_exposure: number;
  traffic_exposure: number;
  risk_score: number;
  decision_factors: RouteDecisionFactors;
}

export interface RouteCalculationResponse {
  session_id: string;
  model: string;
  requested_model: string;
  actual_model: string;
  fallback_used: boolean;
  fallback_reason?: string;
  success: boolean;
  inference_latency_ms: number;
  primary_route: AgentRoute;
  agent_routes: AgentRoute[];
  hazard_summary?: Record<string, any>;
  traffic_summary?: Record<string, any>;
  route_decision_factors: RouteDecisionFactors;
}

export interface HazardConfig {
  hazard_type: "normal" | "flood" | "fire" | "combined";
  severity: number;
  center_lat?: number;
  center_lon?: number;
  radius_meters: number;
  propagation_rate?: number;
}

export interface HazardArea {
  type: string;
  severity: number;
  center: [number, number];
  radius: number;
  affected_nodes?: string[];
  affected_edges?: string[];
}

export interface TrafficConfig {
  density_regime: "low" | "medium" | "high" | "extreme";
  congestion_bias: number;
}

export interface AgentLiveState {
  id: number;
  agent_type: string;
  latitude: number;
  longitude: number;
  current_node: string;
  destination_node: string;
  status: "evacuating" | "evacuated" | "trapped" | "rerouted";
  progress: number;
  distance_remaining_m: number;
  route: string[];
}

export interface TrafficLiveState {
  density: number;
  congestion: number;
  average_speed: number;
}

export interface HazardLiveState {
  type: string;
  severity: number;
  center?: [number, number];
  radius?: number;
}

export interface RouteLiveState {
  coordinates: [number, number][];
  distance: number;
  eta: number;
  risk: number;
}

export interface MetricsLiveState {
  hazard_exposure: number;
  traffic_exposure: number;
  elapsed_time: number;
  total_conflicts: number;
  agents_evacuated: number;
  agents_remaining: number;
}

export interface WebSocketEvacuationMessage {
  event: string;
  session_id: string;
  timestamp: number;
  status: string;
  agents: AgentLiveState[];
  traffic: TrafficLiveState;
  hazards: HazardLiveState[];
  route: RouteLiveState;
  metrics: MetricsLiveState;
  message?: string;
}

export interface EvacuationSession {
  id: string;
  user_id?: string;
  scenario_id?: string;
  model: string;
  status: "active" | "completed" | "stopped" | "failed";
  started_at: string;
  completed_at?: string;
  origin: string;
  destination: string;
  agent_count: number;
  distance: number;
  travel_time: number;
  hazard_exposure: number;
  traffic_exposure: number;
  risk_score: number;
  inference_latency: number;
  success: boolean;
  fallback_used: boolean;
  fallback_reason?: string;
}

export interface EvacuationResultResponse {
  session: EvacuationSession;
  routes: AgentRoute[];
  route_decision_factors: RouteDecisionFactors;
}

export interface Scenario {
  id: string;
  name: string;
  description?: string;
  network: string;
  origin: string;
  destination: string;
  agent_count: number;
  hazard_type: string;
  hazard_config?: HazardConfig;
  traffic_config?: TrafficConfig;
  model: string;
  seed: number;
  created_at: string;
}

export interface ModelMetadata {
  model_name: string;
  model_family: string;
  status: string;
  checkpoint_path: string;
  checkpoint_exists: boolean;
  checkpoint_size_kb?: number;
  device: string;
  n_agents: number;
  obs_dim: number;
  act_dim: number;
  parameter_count: number;
  action_masking_supported: boolean;
  warmup_completed?: boolean;
}

export interface ResearchBenchmark {
  model_name: string;
  evaluation_summary: {
    joint_success_rate: string;
    individual_success_rate: string;
    mean_evacuation_time_s: string;
    hazard_exposure_index_hei: string;
    routing_f1: string;
    mean_inference_latency_ms: string;
    parameter_count: number;
    network_evaluated: string;
    scenarios_evaluated: number;
    seeds_evaluated: number;
  };
  all_model_records: any[];
  disclaimer: string;
}
