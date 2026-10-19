import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { Colors } from "../theme/colors";
import { RouteDecisionFactors } from "../types";

interface Props {
  factors: RouteDecisionFactors;
  modelName?: string;
  fallbackUsed?: boolean;
}

export const RouteDecisionCard: React.FC<Props> = ({
  factors,
  modelName = "Offline-to-Online MARL",
  fallbackUsed = false
}) => {
  return (
    <View style={styles.card}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>Route Decision Factors</Text>
        <View style={[styles.modelBadge, fallbackUsed ? styles.fallbackBadge : styles.marlBadge]}>
          <Text style={styles.modelBadgeText}>{fallbackUsed ? "Dynamic A* (Fallback)" : modelName}</Text>
        </View>
      </View>

      <Text style={styles.subtitle}>
        Transparent quantitative metrics driving multi-agent routing decisions:
      </Text>

      <View style={styles.grid}>
        <View style={styles.metricItem}>
          <Text style={styles.label}>ESTIMATED TIME</Text>
          <Text style={styles.value}>{Math.round(factors.travel_time_seconds)}s</Text>
          <Text style={styles.subtext}>Speed Impedance Calibrated</Text>
        </View>

        <View style={styles.metricItem}>
          <Text style={styles.label}>PATH DISTANCE</Text>
          <Text style={styles.value}>{(factors.distance_meters / 1000).toFixed(2)} km</Text>
          <Text style={styles.subtext}>{Math.round(factors.distance_meters)} meters total</Text>
        </View>

        <View style={styles.metricItem}>
          <Text style={styles.label}>HAZARD EXPOSURE</Text>
          <Text style={[styles.value, { color: factors.hazard_exposure_score > 0.3 ? Colors.fire : Colors.success }]}>
            {(factors.hazard_exposure_score * 100).toFixed(1)}%
          </Text>
          <Text style={styles.subtext}>Mean Cellular Intensity</Text>
        </View>

        <View style={styles.metricItem}>
          <Text style={styles.label}>TRAFFIC DELAY</Text>
          <Text style={[styles.value, { color: factors.traffic_exposure_score > 0.4 ? Colors.warning : Colors.success }]}>
            {(factors.traffic_exposure_score * 100).toFixed(1)}%
          </Text>
          <Text style={styles.subtext}>Coupled Density Factor</Text>
        </View>

        <View style={styles.metricItem}>
          <Text style={styles.label}>AGENT CONFLICTS</Text>
          <Text style={styles.value}>{factors.agent_conflicts}</Text>
          <Text style={styles.subtext}>Contention Avoided</Text>
        </View>

        <View style={styles.metricItem}>
          <Text style={styles.label}>COMPOSITE RISK</Text>
          <Text style={[styles.value, { color: factors.route_risk_index > 0.4 ? Colors.warning : Colors.accent }]}>
            {(factors.route_risk_index * 10).toFixed(2)} / 10
          </Text>
          <Text style={styles.subtext}>Safety Objective Function</Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: Colors.card,
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: Colors.border,
    marginVertical: 8,
  },
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 4,
  },
  title: {
    color: Colors.text,
    fontSize: 15,
    fontWeight: "700",
  },
  subtitle: {
    color: Colors.textMuted,
    fontSize: 11,
    marginBottom: 12,
  },
  modelBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  marlBadge: {
    backgroundColor: Colors.marlPillSurface,
    borderWidth: 1,
    borderColor: Colors.marlPill,
  },
  fallbackBadge: {
    backgroundColor: "rgba(245, 158, 11, 0.15)",
    borderWidth: 1,
    borderColor: Colors.warning,
  },
  modelBadgeText: {
    color: Colors.text,
    fontSize: 10,
    fontWeight: "600",
  },
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    rowGap: 12,
  },
  metricItem: {
    width: "48%",
    backgroundColor: Colors.surface,
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  label: {
    color: Colors.textSecondary,
    fontSize: 9,
    fontWeight: "700",
    letterSpacing: 0.5,
  },
  value: {
    color: Colors.text,
    fontSize: 18,
    fontWeight: "800",
    marginVertical: 2,
  },
  subtext: {
    color: Colors.textMuted,
    fontSize: 9,
  },
});
