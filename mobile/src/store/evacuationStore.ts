import { create } from "zustand";
import {
  EvacuationSession,
  RouteCalculationResponse,
  WebSocketEvacuationMessage,
  AgentLiveState,
  TrafficLiveState,
  HazardLiveState,
  RouteLiveState,
  MetricsLiveState,
  AgentRoute
} from "../types";
import { Api, WS_BASE_URL } from "../api/client";
import { useAuthStore } from "./authStore";

interface EvacuationState {
  activeSession: EvacuationSession | null;
  calculatedRoute: RouteCalculationResponse | null;
  liveStatus: "idle" | "planning" | "evacuating" | "completed" | "stopped" | "error";

  // Live Telemetry from WebSocket
  liveAgents: AgentLiveState[];
  liveTraffic: TrafficLiveState;
  liveHazards: HazardLiveState[];
  liveRoute: RouteLiveState | null;
  liveMetrics: MetricsLiveState;
  recentEvents: { time: string; text: string; type: "info" | "warning" | "reroute" }[];
  rerouteNotification: string | null;

  isWsConnected: boolean;
  wsSocket: WebSocket | null;

  // Actions
  planRoute: (params: { origin: any; destination: any; agent_count?: number; hazard_config?: any }) => Promise<RouteCalculationResponse | null>;
  startEvacuation: (params: { origin: any; destination: any; agent_count?: number; hazard_config?: any; scenario_id?: string }) => Promise<EvacuationSession | null>;
  stopEvacuation: () => Promise<void>;
  requestReroute: () => Promise<void>;
  connectWebSocket: (sessionId: string) => void;
  disconnectWebSocket: () => void;
  dismissRerouteNotification: () => void;
  reset: () => void;
}

const defaultTraffic: TrafficLiveState = { density: 0.25, congestion: 0.2, average_speed: 11.1 };
const defaultMetrics: MetricsLiveState = {
  hazard_exposure: 0.0,
  traffic_exposure: 0.2,
  elapsed_time: 0.0,
  total_conflicts: 0,
  agents_evacuated: 0,
  agents_remaining: 4
};

let heartbeatInterval: any = null;
let reconnectTimeout: any = null;

export const useEvacuationStore = create<EvacuationState>((set, get) => ({
  activeSession: null,
  calculatedRoute: null,
  liveStatus: "idle",

  liveAgents: [],
  liveTraffic: defaultTraffic,
  liveHazards: [],
  liveRoute: null,
  liveMetrics: defaultMetrics,
  recentEvents: [],
  rerouteNotification: null,

  isWsConnected: false,
  wsSocket: null,

  planRoute: async (params) => {
    set({ liveStatus: "planning" });
    try {
      const res = await Api.calculateRoute(params);
      set({
        calculatedRoute: res,
        liveStatus: "idle",
        liveRoute: {
          coordinates: res.primary_route.coordinates,
          distance: res.primary_route.distance_meters,
          eta: res.primary_route.estimated_time_seconds,
          risk: res.primary_route.risk_score
        }
      });
      return res;
    } catch (err: any) {
      set({ liveStatus: "error" });
      return null;
    }
  },

  startEvacuation: async (params) => {
    set({ liveStatus: "planning" });
    try {
      const session = await Api.startEvacuation(params);
      set({
        activeSession: session,
        liveStatus: "evacuating",
        recentEvents: [{
          time: new Date().toLocaleTimeString(),
          text: "Evacuation session initiated. Offline-to-Online MARL route active.",
          type: "info"
        }]
      });

      // Connect WebSocket for live updates
      get().connectWebSocket(session.id);
      return session;
    } catch (err: any) {
      set({ liveStatus: "error" });
      return null;
    }
  },

  stopEvacuation: async () => {
    const session = get().activeSession;
    if (session) {
      await Api.stopEvacuation(session.id);
      set({ liveStatus: "stopped" });
    }
    get().disconnectWebSocket();
  },

  requestReroute: async () => {
    const session = get().activeSession;
    if (session) {
      await Api.rerouteEvacuation(session.id);
    }
  },

  connectWebSocket: (sessionId: string) => {
    get().disconnectWebSocket();

    const token = useAuthStore.getState().token;
    const tokenQuery = token ? `?token=${encodeURIComponent(token)}` : "";
    const url = `${WS_BASE_URL}/ws/evacuation/${sessionId}${tokenQuery}`;
    const ws = new WebSocket(url);

    ws.onopen = () => {
      set({ isWsConnected: true });
      // Start periodic heartbeat ping
      if (heartbeatInterval) clearInterval(heartbeatInterval);
      heartbeatInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send("ping");
        }
      }, 10000);
    };

    ws.onmessage = (event) => {
      try {
        if (event.data === "pong") return;
        const msg = JSON.parse(event.data);
        const timeStr = new Date().toLocaleTimeString();

        if (msg.event === "reroute") {
          set((state) => ({
            liveRoute: msg.route,
            rerouteNotification: msg.message || "Route updated due to changing conditions.",
            recentEvents: [
              { time: timeStr, text: "Route updated due to changing multi-hazard conditions", type: "reroute" },
              ...state.recentEvents.slice(0, 15)
            ]
          }));
        } else if (msg.event === "session_completed") {
          set({
            liveStatus: "completed",
            liveAgents: msg.agents || [],
            liveMetrics: msg.metrics || get().liveMetrics,
            recentEvents: [
              { time: timeStr, text: "All agents evacuated successfully!", type: "info" },
              ...get().recentEvents.slice(0, 15)
            ]
          });
          get().disconnectWebSocket();
        } else if (msg.event === "agent_update") {
          set({
            liveAgents: msg.agents || [],
            liveTraffic: msg.traffic || get().liveTraffic,
            liveHazards: msg.hazards || get().liveHazards,
            liveRoute: msg.route || get().liveRoute,
            liveMetrics: msg.metrics || get().liveMetrics
          });
        }
      } catch (err) {
        console.error("Error parsing WebSocket message:", err);
      }
    };

    ws.onerror = (err) => {
      console.warn("WebSocket error:", err);
      set({ isWsConnected: false });
    };

    ws.onclose = () => {
      set({ isWsConnected: false, wsSocket: null });
      if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
        heartbeatInterval = null;
      }
      // Reconnect automatically if session is still evacuating
      if (get().liveStatus === "evacuating" && get().activeSession?.id === sessionId) {
        if (reconnectTimeout) clearTimeout(reconnectTimeout);
        reconnectTimeout = setTimeout(() => {
          if (get().liveStatus === "evacuating") {
            get().connectWebSocket(sessionId);
          }
        }, 2000);
      }
    };

    set({ wsSocket: ws });
  },

  disconnectWebSocket: () => {
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval);
      heartbeatInterval = null;
    }
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
    const ws = get().wsSocket;
    if (ws) {
      ws.close();
      set({ wsSocket: null, isWsConnected: false });
    }
  },

  dismissRerouteNotification: () => set({ rerouteNotification: null }),

  reset: () => {
    get().disconnectWebSocket();
    set({
      activeSession: null,
      calculatedRoute: null,
      liveStatus: "idle",
      liveAgents: [],
      liveTraffic: defaultTraffic,
      liveHazards: [],
      liveRoute: null,
      liveMetrics: defaultMetrics,
      recentEvents: [],
      rerouteNotification: null
    });
  }
}));
