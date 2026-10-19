import React, { useEffect } from "react";
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { LiquidGlassTokens } from "../theme/tokens";
import { GlassCard } from "../components/glass/GlassCard";
import { GlassPill } from "../components/glass/GlassPill";
import { GlassButton } from "../components/glass/GlassButton";
import { MapRenderer } from "../components/MapRenderer";
import { useEvacuationStore } from "../store/evacuationStore";

export const LiveEvacuationScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const {
    activeSession,
    liveStatus,
    liveAgents,
    liveTraffic,
    liveHazards,
    liveRoute,
    liveMetrics,
    recentEvents,
    rerouteNotification,
    requestReroute,
    stopEvacuation,
    dismissRerouteNotification,
  } = useEvacuationStore();

  useEffect(() => {
    if (liveStatus === "completed") {
      const timer = setTimeout(() => {
        if (activeSession) {
          navigation.navigate("EvacuationResult", { sessionId: activeSession.id });
        }
      }, 1800);
      return () => clearTimeout(timer);
    }
  }, [liveStatus, activeSession, navigation]);

  const progressPct = Math.min(100, Math.round(((4 - liveMetrics.agents_remaining) / 4) * 100));

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Dynamic Reroute Alert Notification Banner */}
      {rerouteNotification && (
        <TouchableOpacity
          activeOpacity={0.8}
          onPress={dismissRerouteNotification}
          accessibilityRole="alert"
          accessibilityLabel={`Dynamic reroute executed: ${rerouteNotification}`}
        >
          <GlassCard variant="emergency" style={styles.rerouteBanner}>
            <View style={styles.rerouteContent}>
              <Ionicons
                name="alert-circle"
                size={24}
                color={LiquidGlassTokens.accent.warning}
                style={{ marginRight: 10 }}
              />
              <View style={{ flex: 1 }}>
                <Text style={styles.rerouteTitle}>DYNAMIC RE-ROUTING EXECUTED</Text>
                <Text style={styles.rerouteDesc}>{rerouteNotification} (Tap to dismiss)</Text>
              </View>
              <Ionicons name="close" size={18} color={LiquidGlassTokens.text.muted} />
            </View>
          </GlassCard>
        </TouchableOpacity>
      )}

      {/* Top Telemetry Header */}
      <GlassCard variant="standard" style={styles.topCard}>
        <View style={styles.statusRow}>
          <GlassPill label="LIVE SIMULATION" variant="emergency" pulsing />
          <GlassPill label="Offline-to-Online MARL" variant="ai" />
        </View>

        {/* Evacuation Progress Bar */}
        <View style={styles.progressContainer}>
          <View style={styles.progressLabelRow}>
            <Text style={styles.progressLabel}>EVACUATION COMPLETION RATIO</Text>
            <Text style={styles.progressVal}>{progressPct}%</Text>
          </View>
          <View style={styles.progressBarBg}>
            <View style={[styles.progressBarFill, { width: `${Math.max(8, progressPct)}%` }]} />
          </View>
        </View>

        {/* Key Metrics Grid */}
        <View style={styles.metricsGrid}>
          <View style={styles.metricBox}>
            <Text style={styles.metricTitle}>SAFE ARRIVALS</Text>
            <Text style={[styles.metricNumber, { color: LiquidGlassTokens.accent.success }]}>
              {liveMetrics.agents_evacuated} / 4
            </Text>
          </View>

          <View style={styles.metricBox}>
            <Text style={styles.metricTitle}>EST. TIME (ETA)</Text>
            <Text style={styles.metricNumber}>
              {Math.round(liveRoute?.eta || 135)}s
            </Text>
          </View>

          <View style={styles.metricBox}>
            <Text style={styles.metricTitle}>DIST. REMAINING</Text>
            <Text style={styles.metricNumber}>
              {((liveRoute?.distance || 1800) / 1000).toFixed(2)} km
            </Text>
          </View>

          <View style={styles.metricBox}>
            <Text style={styles.metricTitle}>HAZARD EXPOSURE</Text>
            <Text
              style={[
                styles.metricNumber,
                {
                  color:
                    liveMetrics.hazard_exposure > 0.2
                      ? LiquidGlassTokens.accent.danger
                      : LiquidGlassTokens.text.secondary,
                },
              ]}
            >
              {(liveMetrics.hazard_exposure * 100).toFixed(1)}%
            </Text>
          </View>
        </View>
      </GlassCard>

      {/* Real-time Map Canvas */}
      <View style={styles.mapContainer}>
        <View style={styles.mapHeaderRow}>
          <Ionicons name="map-outline" size={16} color={LiquidGlassTokens.accent.primary} style={{ marginRight: 6 }} />
          <Text style={styles.sectionHeading}>LIVE MULTI-AGENT SYNCHRONIZATION (OSM ROAD NETWORK)</Text>
        </View>
        <MapRenderer
          route={liveRoute?.coordinates}
          agents={liveAgents}
          hazards={liveHazards}
          height={320}
        />
      </View>

      {/* Control Actions */}
      <View style={styles.controlRow}>
        <View style={{ flex: 1, marginRight: 6 }}>
          <GlassButton
            title="FORCE REROUTE"
            onPress={requestReroute}
            variant="secondary"
            icon={<Ionicons name="shuffle" size={16} color={LiquidGlassTokens.accent.warning} style={{ marginRight: 6 }} />}
          />
        </View>
        <View style={{ flex: 1, marginLeft: 6 }}>
          <GlassButton
            title="ABORT MISSION"
            onPress={stopEvacuation}
            variant="danger"
            icon={<Ionicons name="stop" size={16} color="#FFF" style={{ marginRight: 6 }} />}
          />
        </View>
      </View>

      {/* Telemetry Event Stream Log */}
      <GlassCard variant="ultraThin" style={styles.eventLogCard}>
        <View style={styles.streamHeaderRow}>
          <Ionicons name="pulse" size={16} color={LiquidGlassTokens.accent.secondary} style={{ marginRight: 6 }} />
          <Text style={styles.eventLogTitle}>TELEMETRY EVENT STREAM</Text>
        </View>
        {recentEvents.length === 0 ? (
          <Text style={styles.emptyEvents}>Connecting to live agent telemetry stream...</Text>
        ) : (
          recentEvents.slice(0, 6).map((evt, i) => (
            <View key={i} style={styles.eventItem}>
              <Text style={styles.eventTime}>{evt.time}</Text>
              <Ionicons
                name={
                  evt.type === "reroute"
                    ? "shuffle"
                    : evt.type === "warning"
                    ? "warning-outline"
                    : "navigate-circle-outline"
                }
                size={14}
                color={
                  evt.type === "reroute"
                    ? LiquidGlassTokens.accent.warning
                    : evt.type === "warning"
                    ? LiquidGlassTokens.accent.danger
                    : LiquidGlassTokens.accent.primary
                }
                style={{ marginRight: 6 }}
              />
              <Text
                style={[
                  styles.eventText,
                  evt.type === "reroute" && { color: LiquidGlassTokens.accent.warning },
                  evt.type === "warning" && { color: LiquidGlassTokens.accent.danger },
                ]}
              >
                {evt.text}
              </Text>
            </View>
          ))
        )}
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
  rerouteBanner: {
    padding: 14,
    marginBottom: 14,
  },
  rerouteContent: {
    flexDirection: "row",
    alignItems: "center",
  },
  rerouteTitle: {
    color: LiquidGlassTokens.accent.warning,
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0.5,
  },
  rerouteDesc: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 11,
    marginTop: 2,
  },
  topCard: {
    padding: 16,
    marginBottom: 14,
  },
  statusRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 14,
  },
  progressContainer: {
    marginBottom: 14,
  },
  progressLabelRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 6,
  },
  progressLabel: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 0.6,
  },
  progressVal: {
    color: LiquidGlassTokens.accent.primary,
    fontSize: 13,
    fontWeight: "800",
  },
  progressBarBg: {
    height: 8,
    backgroundColor: "rgba(255, 255, 255, 0.08)",
    borderRadius: 4,
    overflow: "hidden",
  },
  progressBarFill: {
    height: "100%",
    backgroundColor: LiquidGlassTokens.accent.primary,
    borderRadius: 4,
  },
  metricsGrid: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 8,
  },
  metricBox: {
    flex: 1,
    backgroundColor: "rgba(255, 255, 255, 0.04)",
    paddingVertical: 10,
    paddingHorizontal: 6,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.08)",
    alignItems: "center",
  },
  metricTitle: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 8,
    fontWeight: "800",
    letterSpacing: 0.4,
  },
  metricNumber: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 14,
    fontWeight: "800",
    marginTop: 3,
  },
  mapContainer: {
    marginBottom: 14,
  },
  mapHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },
  sectionHeading: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 0.6,
  },
  controlRow: {
    flexDirection: "row",
    marginBottom: 14,
  },
  eventLogCard: {
    padding: 16,
  },
  streamHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 10,
  },
  eventLogTitle: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 0.8,
  },
  emptyEvents: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 11,
    fontStyle: "italic",
    paddingVertical: 8,
  },
  eventItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255, 255, 255, 0.05)",
  },
  eventTime: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 10,
    fontWeight: "700",
    width: 55,
  },
  eventText: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 11,
    flex: 1,
  },
});
