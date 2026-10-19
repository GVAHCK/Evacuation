import axios from "axios";
import { Platform } from "react-native";
import {
  User,
  RouteCalculationResponse,
  EvacuationSession,
  EvacuationResultResponse,
  Scenario,
  HazardArea,
  TrafficLiveState,
  ModelMetadata,
  ResearchBenchmark
} from "../types";

// Determine base API URL according to environment and platform
const getBaseUrl = () => {
  if (process.env.EXPO_PUBLIC_API_URL) {
    return process.env.EXPO_PUBLIC_API_URL.replace(/\/$/, "");
  }
  if (Platform.OS === "android") {
    // Android emulator alias for host loopback
    return "http://10.0.2.2:8000";
  }
  // iOS simulator or web
  return "http://localhost:8000";
};

export const API_BASE_URL = getBaseUrl();
export const WS_BASE_URL = API_BASE_URL.replace("http://", "ws://").replace("https://", "wss://");

export const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: 15000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Response interceptor for structured errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      const { status, data } = error.response;
      const formattedMessage = data?.message || (typeof data?.details === "string" ? data.details : `Server error (${status})`);
      error.message = formattedMessage;
    } else if (error.request) {
      error.message = `Network connectivity error: Unable to reach backend at ${API_BASE_URL}`;
    }
    return Promise.reject(error);
  }
);

let authToken: string | null = null;

export const setAuthToken = (token: string | null) => {
  authToken = token;
  if (token) {
    apiClient.defaults.headers.common["Authorization"] = `Bearer ${token}`;
  } else {
    delete apiClient.defaults.headers.common["Authorization"];
  }
};

// API Services
export const Api = {
  // Auth
  login: async (email: string, password: string) => {
    const res = await apiClient.post("/auth/login", { email, password });
    return res.data;
  },
  register: async (name: string, email: string, password: string, role: string = "USER") => {
    const res = await apiClient.post("/auth/register", { name, email, password, role });
    return res.data;
  },
  getMe: async (): Promise<User> => {
    const res = await apiClient.get("/users/me");
    return res.data;
  },

  // Routes (Offline-to-Online MARL)
  calculateRoute: async (params: {
    origin: string | [number, number] | { latitude: number; longitude: number };
    destination: string | [number, number] | { latitude: number; longitude: number };
    agent_count?: number;
    hazard_config?: any;
    traffic_config?: any;
  }): Promise<RouteCalculationResponse> => {
    const res = await apiClient.post("/routes/calculate", {
      origin: params.origin,
      destination: params.destination,
      agent_count: params.agent_count ?? 4,
      hazard_config: params.hazard_config,
      traffic_config: params.traffic_config,
    });
    return res.data;
  },

  // Evacuation
  startEvacuation: async (params: {
    origin: string | [number, number] | { latitude: number; longitude: number };
    destination: string | [number, number] | { latitude: number; longitude: number };
    agent_count?: number;
    hazard_type?: string;
    hazard_config?: any;
    traffic_config?: any;
    speed_factor?: number;
    scenario_id?: string;
  }): Promise<EvacuationSession> => {
    const res = await apiClient.post("/evacuation/start", {
      origin: params.origin,
      destination: params.destination,
      agent_count: params.agent_count ?? 4,
      hazard_type: params.hazard_type,
      hazard_config: params.hazard_config,
      traffic_config: params.traffic_config,
      scenario_id: params.scenario_id,
    });
    return res.data;
  },
  getEvacuationSession: async (sessionId: string): Promise<EvacuationSession> => {
    const res = await apiClient.get(`/evacuation/${sessionId}`);
    return res.data;
  },
  stopEvacuation: async (sessionId: string): Promise<EvacuationSession> => {
    const res = await apiClient.post(`/evacuation/${sessionId}/stop`);
    return res.data;
  },
  rerouteEvacuation: async (sessionId: string) => {
    const res = await apiClient.post(`/evacuation/${sessionId}/reroute`);
    return res.data;
  },
  getEvacuationResult: async (sessionId: string): Promise<EvacuationResultResponse> => {
    const res = await apiClient.get(`/evacuation/${sessionId}/result`);
    return res.data;
  },

  // Scenarios
  listScenarios: async (): Promise<Scenario[]> => {
    const res = await apiClient.get("/scenarios");
    return res.data;
  },
  createScenario: async (scenario: Partial<Scenario>): Promise<Scenario> => {
    const res = await apiClient.post("/scenarios", scenario);
    return res.data;
  },
  runScenario: async (scenarioId: string): Promise<EvacuationSession> => {
    const res = await apiClient.post(`/scenarios/${scenarioId}/run`);
    return res.data;
  },

  // Hazards & Traffic
  getActiveHazards: async (): Promise<HazardArea[]> => {
    const res = await apiClient.get("/hazards/active");
    return res.data;
  },
  simulateHazard: async (params: { steps?: number; hazard_config: any }): Promise<any> => {
    const res = await apiClient.post("/hazards/simulate", params);
    return res.data;
  },
  getCurrentTraffic: async (): Promise<TrafficLiveState> => {
    const res = await apiClient.get("/traffic/current");
    return res.data;
  },

  // Models & Research
  getModels: async (): Promise<ModelMetadata[]> => {
    const res = await apiClient.get("/models");
    return res.data;
  },
  getBenchmark: async (): Promise<ResearchBenchmark> => {
    const res = await apiClient.get("/research/benchmark");
    return res.data;
  },
  getZeroShot: async () => {
    const res = await apiClient.get("/research/zero-shot");
    return res.data;
  },
  getAblations: async () => {
    const res = await apiClient.get("/research/ablation");
    return res.data;
  },
  getStatistics: async () => {
    const res = await apiClient.get("/research/statistics");
    return res.data;
  },
  getPareto: async () => {
    const res = await apiClient.get("/research/pareto");
    return res.data;
  },

  // History
  getHistory: async (): Promise<EvacuationSession[]> => {
    const res = await apiClient.get("/history");
    return res.data;
  },
  getHistorySession: async (sessionId: string): Promise<any> => {
    const res = await apiClient.get(`/history/${sessionId}`);
    return res.data;
  },

  // Health & Diagnostics
  getHealth: async (): Promise<any> => {
    const res = await apiClient.get("/health", { baseURL: API_BASE_URL });
    return res.data;
  },
  getHealthReady: async (): Promise<any> => {
    const res = await apiClient.get("/health/ready", { baseURL: API_BASE_URL });
    return res.data;
  },

  // Graph topology
  getGraphSummary: async (): Promise<any> => {
    const res = await apiClient.get("/routes/graph-summary");
    return res.data;
  },

  // Account recovery
  forgotPassword: async (email: string): Promise<any> => {
    const res = await apiClient.post("/auth/forgot-password", { email });
    return res.data;
  },

  // Notifications
  getNotifications: async (): Promise<any[]> => {
    const res = await apiClient.get("/notifications");
    return res.data;
  },
  markNotificationRead: async (id: string): Promise<any> => {
    const res = await apiClient.post(`/notifications/${id}/read`);
    return res.data;
  },

  // Live session controls & events
  pauseEvacuation: async (sessionId: string) => {
    const res = await apiClient.post(`/evacuation/${sessionId}/pause`);
    return res.data;
  },
  resumeEvacuation: async (sessionId: string) => {
    const res = await apiClient.post(`/evacuation/${sessionId}/resume`);
    return res.data;
  },
  setEvacuationSpeed: async (sessionId: string, speedFactor: number) => {
    const res = await apiClient.post(`/evacuation/${sessionId}/speed?speed_factor=${speedFactor}`);
    return res.data;
  },
  getSessionEvents: async (sessionId: string): Promise<any[]> => {
    const res = await apiClient.get(`/evacuation/${sessionId}/events`);
    return res.data;
  },
  getAgentDetail: async (sessionId: string, agentId: number): Promise<any> => {
    const res = await apiClient.get(`/evacuation/${sessionId}/agents/${agentId}`);
    return res.data;
  },

  // Diagnostics & Analytics
  getModelDiagnostics: async (): Promise<any> => {
    const res = await apiClient.get("/models/diagnostics");
    return res.data;
  },
  getHistoryAnalytics: async (): Promise<any> => {
    const res = await apiClient.get("/history/analytics/summary");
    return res.data;
  },
  exportResearchBenchmark: async (): Promise<any> => {
    const res = await apiClient.get("/research/export");
    return res.data;
  },
  getHealthDiagnostics: async (): Promise<any> => {
    const res = await apiClient.get("/health/diagnostics", { baseURL: API_BASE_URL });
    return res.data;
  },
};

// Aliases for multi-screen convenience
export const historyApi = {
  getHistory: Api.getHistory,
  getSession: (id: string) => apiClient.get(`/history/${id}`),
  getAnalytics: Api.getHistoryAnalytics,
};

export const evacuationApi = {
  startEvacuation: (params: any) => apiClient.post("/evacuation/start", params),
  getEvacuationSession: (id: string) => apiClient.get(`/evacuation/${id}`),
  stopEvacuation: (id: string) => apiClient.post(`/evacuation/${id}/stop`),
  rerouteEvacuation: (id: string) => apiClient.post(`/evacuation/${id}/reroute`),
  getEvacuationResult: (id: string) => apiClient.get(`/evacuation/${id}/result`),
  pauseEvacuation: (id: string) => apiClient.post(`/evacuation/${id}/pause`),
  resumeEvacuation: (id: string) => apiClient.post(`/evacuation/${id}/resume`),
  getEvents: (id: string) => apiClient.get(`/evacuation/${id}/events`),
  getAgentDetail: (id: string, agentId: number) => apiClient.get(`/evacuation/${id}/agents/${agentId}`),
};

