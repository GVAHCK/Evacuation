import React, { useState, useEffect } from "react";
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, RefreshControl, ActivityIndicator } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { LiquidGlassTokens } from "../theme/tokens";
import { GlassCard } from "../components/glass/GlassCard";
import { GlassPill } from "../components/glass/GlassPill";
import { GlassBadge } from "../components/glass/GlassBadge";
import { GlassButton } from "../components/glass/GlassButton";
import { Api } from "../api/client";
import { useEvacuationStore } from "../store/evacuationStore";
import { useAuthStore } from "../store/authStore";
import { useTheme } from "../theme/ThemeContext";

interface ScreenDirectoryItem {
  id: number;
  title: string;
  route: string;
  desc: string;
  icon: keyof typeof Ionicons.glyphMap;
  badge?: string;
  badgeVariant?: "healthy" | "warning" | "danger" | "neutral" | "ai" | "flood" | "fire";
}

interface ScreenGroup {
  key: string;
  label: string;
  tag: string;
  screens: ScreenDirectoryItem[];
}

const SCREEN_GROUPS: ScreenGroup[] = [
  {
    key: "A",
    label: "Auth & Onboarding",
    tag: "Screens 1–4",
    screens: [
      { id: 1, title: "Welcome & Onboarding", route: "Onboarding", desc: "Capabilities overview & academic non-emergency disclosure", icon: "shield-outline", badge: "Screen 1" },
      { id: 2, title: "Sign In", route: "Login", desc: "JWT authenticated command login", icon: "log-in-outline", badge: "Screen 2" },
      { id: 3, title: "Account Registration", route: "Register", desc: "Operator credential onboarding", icon: "person-add-outline", badge: "Screen 3" },
      { id: 4, title: "Account Recovery", route: "AccountRecovery", desc: "Security token password reset flow", icon: "key-outline", badge: "Screen 4" },
    ],
  },
  {
    key: "B",
    label: "Navigation & Status",
    tag: "Screens 5–8",
    screens: [
      { id: 5, title: "Operations Dashboard", route: "Home", desc: "Real-time command center & connectivity status", icon: "speedometer-outline", badge: "Screen 5" },
      { id: 6, title: "System Health & Probes", route: "SystemHealth", desc: "Subsystem readiness & DB/model/graph probes", icon: "pulse-outline", badge: "Screen 6", badgeVariant: "healthy" },
      { id: 7, title: "Notifications & Inbox", route: "NotificationInbox", desc: "Chronological operational alerts & audit events", icon: "notifications-outline", badge: "Screen 7" },
      { id: 8, title: "Operator Profile", route: "Profile", desc: "Authenticated identity & credential permissions", icon: "person-circle-outline", badge: "Screen 8" },
    ],
  },
  {
    key: "C",
    label: "Geospatial & Hazards",
    tag: "Screens 9–14",
    screens: [
      { id: 9, title: "Interactive Evacuation Map", route: "LiveMap", desc: "336-node / 736-edge OSM topology & shelter nodes", icon: "map-outline", badge: "Screen 9" },
      { id: 10, title: "Hazard Layer Controls", route: "HazardLayerControls", desc: "Toggle flood depths, fire plumes & blocked roads", icon: "layers-outline", badge: "Screen 10" },
      { id: 11, title: "Flood Intelligence", route: "FloodIntelligence", desc: "Hydrodynamic cellular 2D depth propagation", icon: "water-outline", badge: "Screen 11", badgeVariant: "flood" },
      { id: 12, title: "Fire Intelligence", route: "FireIntelligence", desc: "Thermal diffusion & radiant heat thresholds", icon: "flame-outline", badge: "Screen 12", badgeVariant: "fire" },
      { id: 13, title: "Road Network Explorer", route: "RoadNetworkExplorer", desc: "Graph connectivity, edge limits & KDTree snapping", icon: "git-network-outline", badge: "Screen 13" },
      { id: 14, title: "Location & Route Inspector", route: "LocationRouteInspector", desc: "Waypoint inspection & hazard exposure audit", icon: "locate-outline", badge: "Screen 14" },
    ],
  },
  {
    key: "D",
    label: "Scenario Config",
    tag: "Screens 15–20",
    screens: [
      { id: 15, title: "Scenario Overview", route: "ScenarioOverview", desc: "Persisted scenario catalog & batch configurations", icon: "folder-open-outline", badge: "Screen 15" },
      { id: 16, title: "Create Scenario", route: "ScenarioBuilder", desc: "Custom hazard epicenters, origins & shelter selection", icon: "construct-outline", badge: "Screen 16" },
      { id: 17, title: "Agent Configuration", route: "AgentConfiguration", desc: "Cohort archetypes (Civilian, Congestion, Hazard, Perimeter)", icon: "people-outline", badge: "Screen 17" },
      { id: 18, title: "Hazard Configuration", route: "HazardConfiguration", desc: "Severity, propagation velocity & initial radius", icon: "warning-outline", badge: "Screen 18" },
      { id: 19, title: "Traffic Constraints", route: "TrafficConstraints", desc: "BPR digital twin vs SUMO/TraCI microscopic mode", icon: "car-sport-outline", badge: "Screen 19" },
      { id: 20, title: "Scenario Review & Launch", route: "EvacuationPreview", desc: "Pre-computation review & confirmed dispatch", icon: "play-circle-outline", badge: "Screen 20" },
    ],
  },
  {
    key: "E",
    label: "Live Operations",
    tag: "Screens 21–27",
    screens: [
      { id: 21, title: "Active Evacuation Overview", route: "LiveEvacuation", desc: "Real-time 10Hz WebSocket multi-agent navigation", icon: "navigate-circle-outline", badge: "Screen 21" },
      { id: 22, title: "Live Agent Tracking", route: "LiveAgentTracking", desc: "Individual agent coordinates & distance remaining", icon: "compass-outline", badge: "Screen 22" },
      { id: 23, title: "Agent Detail", route: "AgentDetail", desc: "Target agent node trajectory & decision logs", icon: "person-outline", badge: "Screen 23" },
      { id: 24, title: "Route Detail", route: "RouteDetail", desc: "Turn-by-turn waypoints, distances & risk indices", icon: "trail-sign-outline", badge: "Screen 24" },
      { id: 25, title: "Rerouting Events", route: "ReroutingEvents", desc: "Closed-loop hazard breach replanning timeline", icon: "git-branch-outline", badge: "Screen 25" },
      { id: 26, title: "Evacuation Controls", route: "EvacuationControls", desc: "Pause, resume, velocity speed & emergency abort", icon: "pause-circle-outline", badge: "Screen 26" },
      { id: 27, title: "Evacuation Summary", route: "EvacuationResult", desc: "Mission debrief, arrival verification & statistics", icon: "checkmark-done-circle-outline", badge: "Screen 27" },
    ],
  },
  {
    key: "F",
    label: "Routing & MARL AI",
    tag: "Screens 28–33",
    screens: [
      { id: 28, title: "Model Registry", route: "ModelRegistry", desc: "Active PyTorch model checkpoints & parameters", icon: "hardware-chip-outline", badge: "Screen 28", badgeVariant: "ai" },
      { id: 29, title: "MARL Model Details", route: "MarlModelDetails", desc: "327,204-param CQL policy & 34-dim observation vector", icon: "git-commit-outline", badge: "Screen 29", badgeVariant: "ai" },
      { id: 30, title: "Algorithm Comparison", route: "AlgorithmComparison", desc: "Offline-to-Online MARL vs GraphSAGE, HARL, Dijkstra, A*", icon: "git-compare-outline", badge: "Screen 30" },
      { id: 31, title: "Routing Benchmark", route: "Research", desc: "500-scenario peer-reviewed empirical evaluation", icon: "flask-outline", badge: "Screen 31" },
      { id: 32, title: "Algorithm Performance", route: "AlgorithmPerformance", desc: "Wilcoxon signed-rank tests & component ablations", icon: "trending-up-outline", badge: "Screen 32" },
      { id: 33, title: "Model Diagnostics", route: "ModelDiagnostics", desc: "Live inference latency profiler & action mask checks", icon: "terminal-outline", badge: "Screen 33" },
    ],
  },
  {
    key: "G",
    label: "Historical Analytics",
    tag: "Screens 34–38",
    screens: [
      { id: 34, title: "Session History", route: "History", desc: "Persisted evacuation sessions in PostgreSQL", icon: "time-outline", badge: "Screen 34" },
      { id: 35, title: "Session Detail", route: "SessionDetails", desc: "Archived telemetry & route geometry breakdown", icon: "document-text-outline", badge: "Screen 35" },
      { id: 36, title: "Performance Analytics", route: "PerformanceAnalytics", desc: "Joint success rates & travel times across fleet", icon: "bar-chart-outline", badge: "Screen 36" },
      { id: 37, title: "Hazard Safety Analytics", route: "HazardSafetyAnalytics", desc: "Hazard Exposure Index (HEI) & clearance margins", icon: "shield-checkmark-outline", badge: "Screen 37" },
      { id: 38, title: "Research Reports & Export", route: "ResearchReports", desc: "Academic CSV exports & dissertation citations", icon: "download-outline", badge: "Screen 38" },
    ],
  },
  {
    key: "H",
    label: "Diagnostics & Settings",
    tag: "Screens 39–42",
    screens: [
      { id: 39, title: "Connection Diagnostics", route: "ConnectionDiagnostics", desc: "FastAPI reachability, ping latency & WebSocket link", icon: "radio-outline", badge: "Screen 39" },
      { id: 40, title: "Simulation Diagnostics", route: "SimulationDiagnostics", desc: "SUMO/TraCI vs BPR engine provenance verification", icon: "hardware-chip-outline", badge: "Screen 40" },
      { id: 41, title: "Accessibility & Display", route: "AccessibilitySettings", desc: "WCAG 2.1 AA tokens, Dynamic Type & haptic controls", icon: "accessibility-outline", badge: "Screen 41" },
      { id: 42, title: "About Research & Limits", route: "AboutResearch", desc: "Dec-POMDP formulation & simulation boundary limits", icon: "information-circle-outline", badge: "Screen 42" },
    ],
  },
];

export const HomeScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const { user } = useAuthStore();
  const { activeSession, liveStatus } = useEvacuationStore();
  const { haptic } = useTheme();

  const [refreshing, setRefreshing] = useState(false);
  const [hazardsCount, setHazardsCount] = useState(1);
  const [trafficCongestion, setTrafficCongestion] = useState(0.24);
  const [currentRisk, setCurrentRisk] = useState<"LOW" | "MODERATE" | "HIGH">("LOW");
  const [healthStatus, setHealthStatus] = useState<"healthy" | "degraded" | "checking">("checking");
  const [activeGroupKey, setActiveGroupKey] = useState<string>("B");

  const loadData = async () => {
    try {
      const [hazards, traffic, health] = await Promise.all([
        Api.getActiveHazards().catch(() => []),
        Api.getCurrentTraffic().catch(() => ({ congestion: 0.24 })),
        Api.getHealth().catch(() => ({ status: "unavailable" })),
      ]);

      setHazardsCount(hazards.length);
      setTrafficCongestion(traffic.congestion || 0.24);
      setHealthStatus(health.status === "healthy" ? "healthy" : "degraded");

      if (hazards.length > 1 || traffic.congestion > 0.6) {
        setCurrentRisk("HIGH");
      } else if (hazards.length > 0 || traffic.congestion > 0.35) {
        setCurrentRisk("MODERATE");
      } else {
        setCurrentRisk("LOW");
      }
    } catch {
      setHealthStatus("degraded");
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const onRefresh = async () => {
    haptic("light");
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  const currentGroup = SCREEN_GROUPS.find((g) => g.key === activeGroupKey) || SCREEN_GROUPS[1];

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor={LiquidGlassTokens.accent.primary}
        />
      }
    >
      {/* Top Profile / Command Bar */}
      <View style={styles.topBar}>
        <View style={styles.userSection}>
          <Text style={styles.platformBadge}>MULTI-HAZARD INTELLIGENCE PLATFORM</Text>
          <Text style={styles.userName}>{user?.name || "Commander Officer"}</Text>
          <Text style={styles.userRole}>
            {user?.role === "RESEARCHER" ? "Senior MARL Research Lead" : "Incident Command Personnel"}
          </Text>
        </View>

        {/* Top Header Quick Icons */}
        <View style={styles.topIconRow}>
          <TouchableOpacity
            style={styles.topIconBtn}
            onPress={() => {
              haptic("selection");
              navigation.navigate("SystemHealth");
            }}
            accessibilityRole="button"
            accessibilityLabel="Open System Health"
          >
            <Ionicons
              name={healthStatus === "healthy" ? "pulse" : "alert-circle"}
              size={20}
              color={healthStatus === "healthy" ? LiquidGlassTokens.accent.success : LiquidGlassTokens.accent.warning}
            />
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.topIconBtn}
            onPress={() => {
              haptic("selection");
              navigation.navigate("NotificationInbox");
            }}
            accessibilityRole="button"
            accessibilityLabel="Open Event Notifications"
          >
            <Ionicons name="notifications-outline" size={20} color={LiquidGlassTokens.accent.primary} />
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.topIconBtn}
            onPress={() => {
              haptic("selection");
              navigation.navigate("Profile");
            }}
            accessibilityRole="button"
            accessibilityLabel="Open Profile"
          >
            <Ionicons name="person-circle-outline" size={22} color={LiquidGlassTokens.text.primary} />
          </TouchableOpacity>
        </View>
      </View>

      {/* Primary Emergency Risk Status Glass Panel */}
      <GlassCard
        variant={currentRisk === "HIGH" ? "emergency" : currentRisk === "MODERATE" ? "standard" : "safety"}
        style={styles.statusCard}
      >
        <View style={styles.cardHeader}>
          <View style={styles.statusTitleRow}>
            <Ionicons
              name={currentRisk === "LOW" ? "shield-checkmark" : "warning"}
              size={18}
              color={
                currentRisk === "LOW"
                  ? LiquidGlassTokens.accent.success
                  : currentRisk === "MODERATE"
                  ? LiquidGlassTokens.accent.warning
                  : LiquidGlassTokens.accent.danger
              }
              style={{ marginRight: 6 }}
            />
            <Text style={styles.statusTitle}>OPERATIONAL READINESS & HAZARD STATUS</Text>
          </View>
          <GlassPill
            label={`${currentRisk} RISK`}
            variant={currentRisk === "LOW" ? "safety" : currentRisk === "MODERATE" ? "warning" : "emergency"}
          />
        </View>

        <View style={styles.sectorRow}>
          <Ionicons name="location" size={16} color={LiquidGlassTokens.accent.primary} style={{ marginRight: 6 }} />
          <Text style={styles.locationTitle}>Sector 4: Central Highway Corridor</Text>
        </View>
        <Text style={styles.locationSubtitle}>
          Designated Shelter: Central Relief Facility (1.4 km safe perimeter buffer)
        </Text>

        <View style={styles.metricRow}>
          <View style={styles.metricItem}>
            <Text style={styles.metricLabel}>ACTIVE HAZARDS</Text>
            <Text
              style={[
                styles.metricVal,
                { color: hazardsCount > 0 ? LiquidGlassTokens.accent.danger : LiquidGlassTokens.text.primary },
              ]}
            >
              {hazardsCount} Tracked
            </Text>
          </View>
          <View style={styles.metricItem}>
            <Text style={styles.metricLabel}>ROAD CONGESTION</Text>
            <Text style={styles.metricVal}>{(trafficCongestion * 100).toFixed(0)}% Normal</Text>
          </View>
          <View style={styles.metricItem}>
            <Text style={styles.metricLabel}>COHORT AGENTS</Text>
            <Text style={[styles.metricVal, { color: LiquidGlassTokens.accent.primary }]}>4 Ready</Text>
          </View>
        </View>
      </GlassCard>

      {/* Ongoing Active Evacuation Glass Banner */}
      {activeSession && liveStatus === "evacuating" && (
        <TouchableOpacity
          activeOpacity={0.8}
          onPress={() => {
            haptic("medium");
            navigation.navigate("LiveEvacuation");
          }}
          accessibilityRole="button"
          accessibilityLabel="Active evacuation in progress. Tap to open live telemetry."
        >
          <GlassCard variant="emergency" style={styles.activeSessionBanner}>
            <View style={styles.activeBannerContent}>
              <View style={styles.pulseDot} />
              <View style={{ flex: 1, marginLeft: 10 }}>
                <Text style={styles.activeBannerTitle}>ACTIVE EVACUATION IN PROGRESS</Text>
                <Text style={styles.activeBannerSub}>
                  Live multi-agent navigation & cellular hazard monitoring active
                </Text>
              </View>
              <Ionicons name="arrow-forward-circle" size={28} color={LiquidGlassTokens.accent.danger} />
            </View>
          </GlassCard>
        </TouchableOpacity>
      )}

      {/* Primary Action Matrix */}
      <Text style={styles.sectionTitle}>COMMAND & CONTROL WORKFLOWS</Text>
      <View style={styles.actionGrid}>
        <TouchableOpacity
          style={styles.actionTouchable}
          activeOpacity={0.75}
          onPress={() => {
            haptic("light");
            navigation.navigate("StartEvacuation");
          }}
          accessibilityRole="button"
          accessibilityLabel="Start evacuation routing"
        >
          <GlassCard variant="standard" style={styles.actionCard}>
            <View style={[styles.iconCircle, { backgroundColor: "rgba(0, 210, 255, 0.15)" }]}>
              <Ionicons name="navigate-circle" size={26} color={LiquidGlassTokens.accent.primary} />
            </View>
            <Text style={styles.actionTitle}>Start Evacuation</Text>
            <Text style={styles.actionDesc}>Launch 4-agent RL route</Text>
          </GlassCard>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionTouchable}
          activeOpacity={0.75}
          onPress={() => {
            haptic("light");
            navigation.navigate("LiveMap");
          }}
          accessibilityRole="button"
          accessibilityLabel="Open live network map"
        >
          <GlassCard variant="standard" style={styles.actionCard}>
            <View style={[styles.iconCircle, { backgroundColor: "rgba(56, 189, 248, 0.15)" }]}>
              <Ionicons name="map" size={24} color={LiquidGlassTokens.accent.secondary} />
            </View>
            <Text style={styles.actionTitle}>Live Network Map</Text>
            <Text style={styles.actionDesc}>336 nodes / 736 edges</Text>
          </GlassCard>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionTouchable}
          activeOpacity={0.75}
          onPress={() => {
            haptic("light");
            navigation.navigate("ScenarioBuilder");
          }}
          accessibilityRole="button"
          accessibilityLabel="Open scenario builder"
        >
          <GlassCard variant="standard" style={styles.actionCard}>
            <View style={[styles.iconCircle, { backgroundColor: "rgba(245, 158, 11, 0.15)" }]}>
              <Ionicons name="options" size={24} color={LiquidGlassTokens.accent.warning} />
            </View>
            <Text style={styles.actionTitle}>Scenario Builder</Text>
            <Text style={styles.actionDesc}>Configure flood & fire</Text>
          </GlassCard>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionTouchable}
          activeOpacity={0.75}
          onPress={() => {
            haptic("light");
            navigation.navigate("History");
          }}
          accessibilityRole="button"
          accessibilityLabel="View past evacuation sessions"
        >
          <GlassCard variant="standard" style={styles.actionCard}>
            <View style={[styles.iconCircle, { backgroundColor: "rgba(139, 92, 246, 0.15)" }]}>
              <Ionicons name="stats-chart" size={24} color={LiquidGlassTokens.accent.ai} />
            </View>
            <Text style={styles.actionTitle}>Telemetry Logs</Text>
            <Text style={styles.actionDesc}>Persisted decision history</Text>
          </GlassCard>
        </TouchableOpacity>
      </View>

      {/* 42-Screen Operations Command Directory */}
      <View style={styles.directoryHeaderRow}>
        <View style={{ flex: 1 }}>
          <Text style={styles.sectionTitle}>42-SCREEN COMMAND DIRECTORY</Text>
          <Text style={styles.directorySub}>1-Tap Access to Every Architecture Screen</Text>
        </View>
        <GlassBadge label="42 SCREENS" variant="ai" icon="grid-outline" size="sm" />
      </View>

      {/* Group Selector Pill Bar */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.groupTabsScroll}>
        {SCREEN_GROUPS.map((g) => {
          const isActive = g.key === activeGroupKey;
          return (
            <TouchableOpacity
              key={g.key}
              style={[styles.groupTab, isActive && styles.groupTabActive]}
              onPress={() => {
                haptic("selection");
                setActiveGroupKey(g.key);
              }}
              accessibilityRole="tab"
              accessibilityLabel={`Select group ${g.label}`}
            >
              <Text style={[styles.groupTabKey, isActive && styles.groupTabKeyActive]}>{g.key}</Text>
              <Text style={[styles.groupTabLabel, isActive && styles.groupTabLabelActive]}>{g.label}</Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>

      {/* Active Group Screen Cards */}
      <View style={styles.groupContentBox}>
        <View style={styles.groupTitleRow}>
          <Text style={styles.activeGroupTitle}>
            GROUP {currentGroup.key}: {currentGroup.label.toUpperCase()}
          </Text>
          <Text style={styles.activeGroupTag}>{currentGroup.tag}</Text>
        </View>

        <View style={styles.screenList}>
          {currentGroup.screens.map((item) => (
            <TouchableOpacity
              key={item.id}
              activeOpacity={0.78}
              onPress={() => {
                haptic("selection");
                navigation.navigate(item.route);
              }}
              style={styles.screenItemTouchable}
              accessibilityRole="button"
              accessibilityLabel={`Open Screen ${item.id}: ${item.title}`}
            >
              <GlassCard variant="standard" style={styles.screenCard}>
                <View style={styles.screenCardLeft}>
                  <View style={styles.screenIconCircle}>
                    <Ionicons name={item.icon} size={20} color={LiquidGlassTokens.accent.primary} />
                  </View>
                  <View style={{ flex: 1, marginLeft: 12 }}>
                    <View style={{ flexDirection: "row", alignItems: "center", gap: 6, marginBottom: 2 }}>
                      <Text style={styles.screenNum}>#{item.id}</Text>
                      <Text style={styles.screenTitle}>{item.title}</Text>
                    </View>
                    <Text style={styles.screenDesc} numberOfLines={2}>
                      {item.desc}
                    </Text>
                  </View>
                </View>
                <Ionicons name="chevron-forward" size={18} color={LiquidGlassTokens.text.muted} />
              </GlassCard>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Production AI Architecture Info */}
      <GlassCard variant="ultraThin" style={styles.modelCard}>
        <View style={styles.modelHeader}>
          <Ionicons name="hardware-chip-outline" size={20} color={LiquidGlassTokens.accent.ai} style={{ marginRight: 8 }} />
          <Text style={styles.modelCardTitle}>PRODUCTION AI ROUTING ENGINE</Text>
        </View>
        <Text style={styles.modelCardName}>Conservative Q-Learning (CQL) Offline-to-Online MARL</Text>
        <Text style={styles.modelCardDesc}>
          Sole production routing policy loaded from seed 42 checkpoint (327,204 parameters). Operates with 34-dimensional state observations, hard topological action masking, and continuous microscopic hazard integration.
        </Text>
        <TouchableOpacity
          style={styles.benchmarkLinkRow}
          onPress={() => {
            haptic("selection");
            navigation.navigate("Research");
          }}
          accessibilityRole="link"
          accessibilityLabel="View verified benchmark research metrics"
        >
          <Text style={styles.researchLink}>View Verified 500-Scenario Empirical Benchmark</Text>
          <Ionicons name="arrow-forward" size={14} color={LiquidGlassTokens.accent.primary} style={{ marginLeft: 6 }} />
        </TouchableOpacity>
      </GlassCard>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: LiquidGlassTokens.background.primary,
  },
  content: {
    padding: 16,
    paddingBottom: 48,
  },
  topBar: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 18,
    marginTop: 8,
  },
  userSection: {
    flex: 1,
  },
  platformBadge: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 1.2,
    marginBottom: 2,
  },
  userName: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 20,
    fontWeight: "800",
    letterSpacing: 0.2,
  },
  userRole: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 12,
    marginTop: 2,
  },
  topIconRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  topIconBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: "rgba(255, 255, 255, 0.07)",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.14)",
    justifyContent: "center",
    alignItems: "center",
  },
  statusCard: {
    padding: 18,
    marginBottom: 16,
  },
  cardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  statusTitleRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  statusTitle: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 11,
    fontWeight: "700",
    letterSpacing: 0.6,
  },
  sectorRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 2,
  },
  locationTitle: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 15,
    fontWeight: "700",
  },
  locationSubtitle: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 12,
    marginTop: 4,
    marginBottom: 16,
  },
  metricRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    borderTopWidth: 1,
    borderTopColor: "rgba(255, 255, 255, 0.08)",
    paddingTop: 14,
  },
  metricItem: {
    alignItems: "center",
    width: "32%",
  },
  metricLabel: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 0.5,
  },
  metricVal: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 15,
    fontWeight: "800",
    marginTop: 3,
  },
  activeSessionBanner: {
    padding: 16,
    marginBottom: 16,
  },
  activeBannerContent: {
    flexDirection: "row",
    alignItems: "center",
  },
  pulseDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: LiquidGlassTokens.accent.danger,
  },
  activeBannerTitle: {
    color: LiquidGlassTokens.accent.danger,
    fontSize: 13,
    fontWeight: "800",
    letterSpacing: 0.5,
  },
  activeBannerSub: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 11,
    marginTop: 2,
  },
  sectionTitle: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 1.1,
    marginTop: 4,
    marginBottom: 12,
  },
  actionGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    rowGap: 12,
    marginBottom: 20,
  },
  actionTouchable: {
    width: "48%",
  },
  actionCard: {
    padding: 16,
    minHeight: 128,
    justifyContent: "space-between",
  },
  iconCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 8,
  },
  actionTitle: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 14,
    fontWeight: "700",
  },
  actionDesc: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 11,
    marginTop: 2,
  },
  directoryHeaderRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
  },
  directorySub: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 11,
    marginTop: -8,
    marginBottom: 8,
  },
  groupTabsScroll: {
    gap: 8,
    paddingBottom: 8,
    marginBottom: 10,
  },
  groupTab: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.12)",
    borderRadius: LiquidGlassTokens.radius.md,
    paddingHorizontal: 12,
    paddingVertical: 8,
    gap: 6,
  },
  groupTabActive: {
    backgroundColor: "rgba(0, 210, 255, 0.14)",
    borderColor: LiquidGlassTokens.accent.primary,
  },
  groupTabKey: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 11,
    fontWeight: "800",
  },
  groupTabKeyActive: {
    color: LiquidGlassTokens.accent.primary,
  },
  groupTabLabel: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 12,
    fontWeight: "700",
  },
  groupTabLabelActive: {
    color: LiquidGlassTokens.text.primary,
  },
  groupContentBox: {
    marginBottom: 20,
  },
  groupTitleRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
    paddingHorizontal: 2,
  },
  activeGroupTitle: {
    color: LiquidGlassTokens.accent.primary,
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0.8,
  },
  activeGroupTag: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 11,
    fontWeight: "600",
  },
  screenList: {
    gap: 8,
  },
  screenItemTouchable: {
    width: "100%",
  },
  screenCard: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 12,
  },
  screenCardLeft: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
    marginRight: 8,
  },
  screenIconCircle: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: "rgba(0, 210, 255, 0.10)",
    borderWidth: 1,
    borderColor: "rgba(0, 210, 255, 0.22)",
    justifyContent: "center",
    alignItems: "center",
  },
  screenNum: {
    color: LiquidGlassTokens.accent.primary,
    fontSize: 11,
    fontWeight: "800",
  },
  screenTitle: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 13,
    fontWeight: "700",
  },
  screenDesc: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 11,
    lineHeight: 15,
  },
  modelCard: {
    padding: 18,
    marginTop: 4,
  },
  modelHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 4,
  },
  modelCardTitle: {
    color: LiquidGlassTokens.accent.ai,
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 1,
  },
  modelCardName: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 14,
    fontWeight: "800",
    marginTop: 4,
  },
  modelCardDesc: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 12,
    marginTop: 6,
    lineHeight: 18,
  },
  benchmarkLinkRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 14,
  },
  researchLink: {
    color: LiquidGlassTokens.accent.primary,
    fontSize: 12,
    fontWeight: "700",
  },
});
