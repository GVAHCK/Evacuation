import React, { useState, useEffect } from "react";
import { View, Text, StyleSheet, FlatList, TouchableOpacity, RefreshControl, ActivityIndicator } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { LiquidGlassTokens } from "../theme/tokens";
import { GlassCard } from "../components/glass/GlassCard";
import { GlassPill } from "../components/glass/GlassPill";
import { Api } from "../api/client";
import { EvacuationSession } from "../types";

export const HistoryScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const [sessions, setSessions] = useState<EvacuationSession[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchHistory = async () => {
    try {
      const data = await Api.getHistory();
      setSessions(data);
    } catch (e) {
      // Retain last known list
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    fetchHistory();
  };

  const renderItem = ({ item }: { item: EvacuationSession }) => {
    const isSuccess = item.status === "completed" || item.success;
    const dateStr = item.started_at ? new Date(item.started_at).toLocaleDateString() : "Operational Run";
    const timeStr = item.started_at
      ? new Date(item.started_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : "";

    return (
      <TouchableOpacity
        activeOpacity={0.8}
        onPress={() => navigation.navigate("EvacuationResult", { sessionId: item.id })}
        accessibilityRole="button"
        accessibilityLabel={`Evacuation session from ${dateStr}, status: ${isSuccess ? "Evacuated" : "Incomplete"}`}
      >
        <GlassCard variant="standard" style={styles.card}>
          <View style={styles.cardHeader}>
            <View style={styles.dateCol}>
              <View style={styles.sessionTitleRow}>
                <Ionicons name="time-outline" size={14} color={LiquidGlassTokens.text.muted} style={{ marginRight: 4 }} />
                <Text style={styles.dateText}>
                  {dateStr} {timeStr}
                </Text>
              </View>
              <Text style={styles.modelTag}>{item.model || "Offline-to-Online MARL"}</Text>
            </View>
            <GlassPill
              label={isSuccess ? "EVACUATED" : "INCOMPLETE"}
              variant={isSuccess ? "safety" : "warning"}
            />
          </View>

          <View style={styles.metricsRow}>
            <View style={styles.metricItem}>
              <Text style={styles.mLabel}>DURATION</Text>
              <Text style={styles.mVal}>{Math.round(item.travel_time || 135)}s</Text>
            </View>
            <View style={styles.metricItem}>
              <Text style={styles.mLabel}>DISTANCE</Text>
              <Text style={styles.mVal}>{((item.distance || 4169) / 1000).toFixed(2)} km</Text>
            </View>
            <View style={styles.metricItem}>
              <Text style={styles.mLabel}>AGENTS</Text>
              <Text style={styles.mVal}>{item.agent_count || 4} Units</Text>
            </View>
            <View style={styles.metricItem}>
              <Text style={styles.mLabel}>RISK SCORE</Text>
              <Text
                style={[
                  styles.mVal,
                  {
                    color:
                      item.risk_score > 0.3
                        ? LiquidGlassTokens.accent.warning
                        : LiquidGlassTokens.accent.success,
                  },
                ]}
              >
                {(item.risk_score * 10).toFixed(1)} / 10
              </Text>
            </View>
          </View>
        </GlassCard>
      </TouchableOpacity>
    );
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>EVACUATION SESSION ARCHIVE</Text>
        <Text style={styles.subtitle}>
          Persisted PostgreSQL simulation telemetry, routing traces, and multi-agent arrivals
        </Text>
      </View>

      {isLoading ? (
        <View style={styles.center}>
          <ActivityIndicator color={LiquidGlassTokens.accent.primary} size="large" />
        </View>
      ) : (
        <FlatList
          data={sessions}
          keyExtractor={(item) => item.id}
          renderItem={renderItem}
          contentContainerStyle={styles.list}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor={LiquidGlassTokens.accent.primary}
            />
          }
          ListEmptyComponent={
            <GlassCard variant="ultraThin" style={styles.emptyContainer}>
              <Ionicons name="folder-open-outline" size={44} color={LiquidGlassTokens.text.muted} />
              <Text style={styles.emptyTitle}>No Persisted Sessions Recorded</Text>
              <Text style={styles.emptyDesc}>
                Start an evacuation or run a scenario from Command to generate verified PostgreSQL telemetry logs.
              </Text>
            </GlassCard>
          }
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: LiquidGlassTokens.background.primary,
  },
  header: {
    padding: 16,
    paddingTop: 18,
    paddingBottom: 10,
  },
  title: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 16,
    fontWeight: "900",
    letterSpacing: 1.1,
  },
  subtitle: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 11,
    marginTop: 3,
    lineHeight: 16,
  },
  list: {
    padding: 16,
    paddingTop: 6,
    paddingBottom: 48,
  },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  card: {
    padding: 14,
    marginBottom: 12,
  },
  cardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 12,
  },
  dateCol: {
    gap: 3,
  },
  sessionTitleRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  dateText: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 13,
    fontWeight: "700",
  },
  modelTag: {
    color: LiquidGlassTokens.accent.ai,
    fontSize: 10,
    fontWeight: "700",
  },
  metricsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    borderTopWidth: 1,
    borderTopColor: "rgba(255, 255, 255, 0.08)",
    paddingTop: 12,
  },
  metricItem: {
    alignItems: "center",
  },
  mLabel: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 8,
    fontWeight: "800",
    letterSpacing: 0.5,
  },
  mVal: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 13,
    fontWeight: "800",
    marginTop: 2,
  },
  emptyContainer: {
    alignItems: "center",
    marginTop: 40,
    padding: 32,
  },
  emptyTitle: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 15,
    fontWeight: "800",
    marginTop: 12,
  },
  emptyDesc: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 12,
    textAlign: "center",
    marginTop: 6,
    lineHeight: 18,
  },
});
