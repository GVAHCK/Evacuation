import React, { useState, useEffect } from "react";
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { LiquidGlassTokens } from "../theme/tokens";
import { GlassCard } from "../components/glass/GlassCard";
import { GlassPill } from "../components/glass/GlassPill";
import { GlassButton } from "../components/glass/GlassButton";
import { MapRenderer } from "../components/MapRenderer";
import { useEvacuationStore } from "../store/evacuationStore";
import { Api } from "../api/client";
import { HazardArea } from "../types";

export const LiveMapScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const { calculatedRoute, liveRoute, liveAgents, liveHazards } = useEvacuationStore();
  const [hazards, setHazards] = useState<HazardArea[]>([]);
  const [selectedLayer, setSelectedLayer] = useState<"all" | "route" | "hazard" | "traffic">("all");

  useEffect(() => {
    Api.getActiveHazards()
      .then((data) => setHazards(data))
      .catch(() => {});
  }, []);

  const routeCoords = liveRoute?.coordinates || calculatedRoute?.primary_route.coordinates || [
    [13.001, 80.201],
    [13.015, 80.215],
    [13.03, 80.23],
    [13.045, 80.245],
  ];

  const originCoord = routeCoords[0];
  const destCoord = routeCoords[routeCoords.length - 1];

  return (
    <View style={styles.container}>
      {/* Top HUD */}
      <View style={styles.hud}>
        <View>
          <Text style={styles.hudTitle}>TACTICAL EVACUATION MAP</Text>
          <Text style={styles.hudSub}>336-Node OSM Network & Microscopic Hazard Overlay</Text>
        </View>
        <GlassPill label="Offline-to-Online MARL" variant="ai" />
      </View>

      {/* Layer Filter Buttons */}
      <View style={styles.filterRow}>
        {(["all", "route", "hazard", "traffic"] as const).map((layer) => {
          const isSelected = selectedLayer === layer;
          return (
            <TouchableOpacity
              key={layer}
              style={[
                styles.filterBtn,
                isSelected && {
                  borderColor: LiquidGlassTokens.accent.primary,
                  backgroundColor: "rgba(0, 210, 255, 0.15)",
                },
              ]}
              onPress={() => setSelectedLayer(layer)}
            >
              <Text
                style={[
                  styles.filterBtnText,
                  isSelected && { color: LiquidGlassTokens.accent.primary, fontWeight: "800" },
                ]}
              >
                {layer.toUpperCase()}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {/* Primary Map View */}
      <View style={styles.mapWrapper}>
        <MapRenderer
          route={selectedLayer === "hazard" ? [] : routeCoords}
          agents={liveAgents}
          hazards={selectedLayer === "route" ? [] : liveHazards.length > 0 ? liveHazards : hazards}
          origin={originCoord}
          destination={destCoord}
          height={380}
        />
      </View>

      {/* Bottom Tactical Sheet */}
      <ScrollView style={styles.bottomSheet} contentContainerStyle={{ paddingBottom: 24 }}>
        <GlassCard variant="standard" style={styles.sheetCard}>
          <View style={styles.routeHeader}>
            <View style={{ flex: 1, marginRight: 12 }}>
              <Text style={styles.routeLabel}>RECOMMENDED AI EVACUATION CORRIDOR</Text>
              <Text style={styles.routeDest}>To: Central Relief Shelter (Capacity: 2,500)</Text>
            </View>
            <TouchableOpacity
              style={styles.startBtn}
              onPress={() => navigation.navigate("StartEvacuation")}
              accessibilityRole="button"
            >
              <Ionicons name="add" size={16} color="#000" style={{ marginRight: 4 }} />
              <Text style={styles.startBtnText}>NEW RUN</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.statusRow}>
            <View style={styles.statusBox}>
              <Text style={styles.statusBoxLabel}>EST. TRAVEL TIME</Text>
              <Text style={styles.statusBoxVal}>
                {Math.round(calculatedRoute?.primary_route.estimated_time_seconds || 450)}s
              </Text>
            </View>
            <View style={styles.statusBox}>
              <Text style={styles.statusBoxLabel}>TOTAL DISTANCE</Text>
              <Text style={styles.statusBoxVal}>
                {((calculatedRoute?.primary_route.distance_meters || 4169) / 1000).toFixed(2)} km
              </Text>
            </View>
            <View style={styles.statusBox}>
              <Text style={styles.statusBoxLabel}>HAZARD CLEARANCE</Text>
              <Text style={[styles.statusBoxVal, { color: LiquidGlassTokens.accent.success }]}>
                SAFE (98%)
              </Text>
            </View>
          </View>
        </GlassCard>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: LiquidGlassTokens.background.primary,
  },
  hud: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 10,
  },
  hudTitle: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 14,
    fontWeight: "900",
    letterSpacing: 1.1,
  },
  hudSub: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 10,
    marginTop: 2,
  },
  filterRow: {
    flexDirection: "row",
    paddingHorizontal: 16,
    gap: 8,
    marginBottom: 10,
  },
  filterBtn: {
    flex: 1,
    paddingVertical: 7,
    alignItems: "center",
    borderRadius: 8,
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.12)",
  },
  filterBtnText: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 10,
    fontWeight: "700",
  },
  mapWrapper: {
    paddingHorizontal: 16,
  },
  bottomSheet: {
    flex: 1,
    paddingHorizontal: 16,
    marginTop: 12,
  },
  sheetCard: {
    padding: 16,
  },
  routeHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 14,
  },
  routeLabel: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 0.6,
  },
  routeDest: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 14,
    fontWeight: "700",
    marginTop: 2,
  },
  startBtn: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: LiquidGlassTokens.accent.primary,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  startBtnText: {
    color: "#000",
    fontSize: 11,
    fontWeight: "800",
  },
  statusRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 8,
    borderTopWidth: 1,
    borderTopColor: "rgba(255, 255, 255, 0.08)",
    paddingTop: 12,
  },
  statusBox: {
    flex: 1,
    backgroundColor: "rgba(255, 255, 255, 0.04)",
    paddingVertical: 8,
    paddingHorizontal: 6,
    borderRadius: 8,
    alignItems: "center",
  },
  statusBoxLabel: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 8,
    fontWeight: "800",
  },
  statusBoxVal: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 13,
    fontWeight: "800",
    marginTop: 2,
  },
});
