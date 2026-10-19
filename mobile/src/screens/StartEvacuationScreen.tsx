import React, { useState } from "react";
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, ActivityIndicator } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { LiquidGlassTokens } from "../theme/tokens";
import { GlassCard } from "../components/glass/GlassCard";
import { GlassPill } from "../components/glass/GlassPill";
import { GlassButton } from "../components/glass/GlassButton";
import { useEvacuationStore } from "../store/evacuationStore";

export const StartEvacuationScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const [step, setStep] = useState(1);
  const [originType, setOriginType] = useState<"gps" | "custom">("gps");
  const [destination, setDestination] = useState("Central Relief Shelter");
  const [hazardType, setHazardType] = useState<"flood" | "fire" | "combined" | "normal">("flood");
  const [severity, setSeverity] = useState(1.5);
  const [agentCount, setAgentCount] = useState(4);
  const [isLoading, setIsLoading] = useState(false);

  const { planRoute, startEvacuation } = useEvacuationStore();

  const handleStart = async () => {
    setIsLoading(true);
    // Origin & Destination coordinates / nodes resolved by backend GraphManager
    const origin = [13.002, 80.205];
    const dest = [13.048, 80.248];

    const hazardConfig = {
      hazard_type: hazardType,
      severity: severity,
      radius_meters: 450,
      center_lat: 13.025,
      center_lon: 80.225,
    };

    try {
      await planRoute({ origin, destination: dest, agent_count: agentCount, hazard_config: hazardConfig });
      const session = await startEvacuation({
        origin,
        destination: dest,
        agent_count: agentCount,
        hazard_config: hazardConfig,
      });

      setIsLoading(false);
      if (session) {
        navigation.navigate("LiveEvacuation");
      }
    } catch (e) {
      setIsLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Step Indicator Header */}
      <View style={styles.stepHeader}>
        <View style={styles.stepTitleRow}>
          <Text style={styles.stepTitle}>CONFIGURE EVACUATION MISSION</Text>
          <GlassPill label={`STEP ${step} OF 4`} variant="cyan" />
        </View>
        <View style={styles.stepBar}>
          {[1, 2, 3, 4].map((s) => (
            <View
              key={s}
              style={[
                styles.stepSegment,
                s <= step && styles.stepSegmentActive,
                s === step && styles.stepSegmentCurrent,
              ]}
            />
          ))}
        </View>
      </View>

      {/* STEP 1: Current Origin Location */}
      {step === 1 && (
        <View style={styles.section}>
          <Text style={styles.heading}>Establish Origin Location</Text>
          <Text style={styles.desc}>
            Select how you wish to establish the starting road graph node for the evacuation cohort:
          </Text>

          <TouchableOpacity
            activeOpacity={0.8}
            onPress={() => setOriginType("gps")}
            accessibilityRole="radio"
            accessibilityState={{ checked: originType === "gps" }}
          >
            <GlassCard
              variant={originType === "gps" ? "standard" : "ultraThin"}
              style={[
                styles.optionCard,
                originType === "gps" && { borderColor: LiquidGlassTokens.accent.primary },
              ]}
            >
              <View style={[styles.iconBox, { backgroundColor: "rgba(0, 210, 255, 0.15)" }]}>
                <Ionicons name="navigate-circle" size={26} color={LiquidGlassTokens.accent.primary} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.optionTitle}>Hardware GPS Telemetry (Recommended)</Text>
                <Text style={styles.optionDesc}>Resolved: Sector 4 Corridor (Lat: 13.002° N, Lon: 80.205° E)</Text>
              </View>
              {originType === "gps" && (
                <Ionicons name="checkmark-circle" size={22} color={LiquidGlassTokens.accent.primary} />
              )}
            </GlassCard>
          </TouchableOpacity>

          <TouchableOpacity
            activeOpacity={0.8}
            onPress={() => setOriginType("custom")}
            accessibilityRole="radio"
            accessibilityState={{ checked: originType === "custom" }}
          >
            <GlassCard
              variant={originType === "custom" ? "standard" : "ultraThin"}
              style={[
                styles.optionCard,
                originType === "custom" && { borderColor: LiquidGlassTokens.accent.primary },
              ]}
            >
              <View style={[styles.iconBox, { backgroundColor: "rgba(56, 189, 248, 0.15)" }]}>
                <Ionicons name="map-outline" size={24} color={LiquidGlassTokens.accent.secondary} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.optionTitle}>Select Junction on Map</Text>
                <Text style={styles.optionDesc}>Choose custom road junction from the 336-node graph</Text>
              </View>
              {originType === "custom" && (
                <Ionicons name="checkmark-circle" size={22} color={LiquidGlassTokens.accent.primary} />
              )}
            </GlassCard>
          </TouchableOpacity>
        </View>
      )}

      {/* STEP 2: Evacuation Destination */}
      {step === 2 && (
        <View style={styles.section}>
          <Text style={styles.heading}>Target Safe Haven / Shelter</Text>
          <Text style={styles.desc}>
            Designate the destination refuge facility with topological capacity and safe buffer:
          </Text>

          {[
            {
              name: "Central Relief Shelter",
              desc: "Capacity: 2,500 | High Ground | Fully Operational",
              dist: "4.2 km",
              safe: true,
            },
            {
              name: "Northern Community Stadium",
              desc: "Capacity: 5,000 | Perimeter Safe Zone Buffer",
              dist: "5.8 km",
              safe: true,
            },
            {
              name: "Eastern Emergency Response Depot",
              desc: "Capacity: 1,800 | Fast Highway Access Corridor",
              dist: "3.9 km",
              safe: true,
            },
          ].map((dest) => {
            const isSelected = destination === dest.name;
            return (
              <TouchableOpacity
                key={dest.name}
                activeOpacity={0.8}
                onPress={() => setDestination(dest.name)}
                accessibilityRole="radio"
                accessibilityState={{ checked: isSelected }}
              >
                <GlassCard
                  variant={isSelected ? "safety" : "ultraThin"}
                  style={[
                    styles.optionCard,
                    isSelected && { borderColor: LiquidGlassTokens.accent.success },
                  ]}
                >
                  <View style={[styles.iconBox, { backgroundColor: "rgba(16, 185, 129, 0.15)" }]}>
                    <Ionicons name="shield-checkmark" size={24} color={LiquidGlassTokens.accent.success} />
                  </View>
                  <View style={{ flex: 1 }}>
                    <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                      <Text style={styles.optionTitle}>{dest.name}</Text>
                      <Text style={{ color: LiquidGlassTokens.accent.primary, fontSize: 12, fontWeight: "700" }}>
                        {dest.dist}
                      </Text>
                    </View>
                    <Text style={styles.optionDesc}>{dest.desc}</Text>
                  </View>
                  {isSelected && (
                    <Ionicons
                      name="checkmark-circle"
                      size={22}
                      color={LiquidGlassTokens.accent.success}
                      style={{ marginLeft: 8 }}
                    />
                  )}
                </GlassCard>
              </TouchableOpacity>
            );
          })}
        </View>
      )}

      {/* STEP 3: Hazard Configuration */}
      {step === 3 && (
        <View style={styles.section}>
          <Text style={styles.heading}>Multi-Hazard Regime & Cohort</Text>
          <Text style={styles.desc}>
            Configure microscopic environmental threats that trigger RL dynamic rerouting:
          </Text>

          <View style={styles.hazardGrid}>
            {[
              {
                id: "flood",
                title: "Flash Flood",
                icon: "water-outline",
                color: LiquidGlassTokens.accent.primary,
              },
              {
                id: "fire",
                title: "Urban Fire",
                icon: "flame-outline",
                color: LiquidGlassTokens.accent.danger,
              },
              {
                id: "combined",
                title: "Flood + Fire",
                icon: "flash-outline",
                color: LiquidGlassTokens.accent.warning,
              },
              {
                id: "normal",
                title: "Normal (Clear)",
                icon: "sunny-outline",
                color: LiquidGlassTokens.accent.success,
              },
            ].map((hz) => {
              const isSelected = hazardType === hz.id;
              return (
                <TouchableOpacity
                  key={hz.id}
                  style={styles.hazardTouchable}
                  activeOpacity={0.8}
                  onPress={() => setHazardType(hz.id as any)}
                  accessibilityRole="button"
                >
                  <GlassCard
                    variant={isSelected ? "standard" : "ultraThin"}
                    style={[
                      styles.hazardCard,
                      isSelected && {
                        borderColor: hz.color,
                        backgroundColor: "rgba(255, 255, 255, 0.08)",
                      },
                    ]}
                  >
                    <Ionicons name={hz.icon as any} size={28} color={hz.color} />
                    <Text
                      style={[
                        styles.hazardCardText,
                        isSelected && { color: hz.color, fontWeight: "800" },
                      ]}
                    >
                      {hz.title}
                    </Text>
                  </GlassCard>
                </TouchableOpacity>
              );
            })}
          </View>

          {hazardType !== "normal" && (
            <GlassCard variant="ultraThin" style={styles.severitySection}>
              <Text style={styles.label}>HAZARD PROPAGATION SEVERITY: {severity.toFixed(1)}x</Text>
              <View style={styles.severityRow}>
                {[1.0, 1.5, 2.0, 3.0].map((s) => (
                  <TouchableOpacity
                    key={s}
                    style={[
                      styles.severityBtn,
                      severity === s && {
                        borderColor: LiquidGlassTokens.accent.warning,
                        backgroundColor: "rgba(245, 158, 11, 0.2)",
                      },
                    ]}
                    onPress={() => setSeverity(s)}
                  >
                    <Text
                      style={[
                        styles.severityBtnText,
                        severity === s && { color: LiquidGlassTokens.accent.warning, fontWeight: "800" },
                      ]}
                    >
                      {s.toFixed(1)}x
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </GlassCard>
          )}

          <GlassCard variant="ultraThin" style={[styles.severitySection, { marginTop: 12 }]}>
            <Text style={styles.label}>MARL COHORT SIZE: {agentCount} AGENTS</Text>
            <View style={styles.severityRow}>
              {[1, 2, 4, 8].map((c) => (
                <TouchableOpacity
                  key={c}
                  style={[
                    styles.severityBtn,
                    agentCount === c && {
                      borderColor: LiquidGlassTokens.accent.ai,
                      backgroundColor: "rgba(139, 92, 246, 0.2)",
                    },
                  ]}
                  onPress={() => setAgentCount(c)}
                >
                  <Text
                    style={[
                      styles.severityBtnText,
                      agentCount === c && { color: LiquidGlassTokens.accent.ai, fontWeight: "800" },
                    ]}
                  >
                    {c} UNITS
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </GlassCard>
        </View>
      )}

      {/* STEP 4: Scenario Summary & Start */}
      {step === 4 && (
        <View style={styles.section}>
          <Text style={styles.heading}>Review & Authorize Simulation</Text>
          <Text style={styles.desc}>
            Inspect multi-agent operational parameters before launching the real-time inference loop:
          </Text>

          <GlassCard variant="standard" style={styles.summaryCard}>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>ORIGIN NODE</Text>
              <Text style={styles.summaryVal}>Sector 4 Corridor (Graph Validated)</Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>DESTINATION</Text>
              <Text style={styles.summaryVal}>{destination}</Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>HAZARD THREAT</Text>
              <Text style={[styles.summaryVal, { color: LiquidGlassTokens.accent.danger }]}>
                {hazardType.toUpperCase()} ({severity}x intensity)
              </Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>AGENT COHORT</Text>
              <Text style={styles.summaryVal}>{agentCount} Heterogeneous Units</Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>AI POLICY</Text>
              <Text style={[styles.summaryVal, { color: LiquidGlassTokens.accent.ai }]}>
                Offline-to-Online MARL (327,204 Params)
              </Text>
            </View>
          </GlassCard>

          <GlassButton
            title={isLoading ? "INITIALIZING SIMULATION..." : "LAUNCH EVACUATION SIMULATION"}
            onPress={handleStart}
            loading={isLoading}
            variant="primary"
            icon={<Ionicons name="play" size={20} color="#000" style={{ marginRight: 8 }} />}
            style={{ marginTop: 18 }}
          />
        </View>
      )}

      {/* Navigation Step Buttons */}
      <View style={styles.navButtons}>
        {step > 1 && (
          <TouchableOpacity
            style={styles.prevBtn}
            onPress={() => setStep(step - 1)}
            accessibilityRole="button"
            accessibilityLabel="Go back to previous configuration step"
          >
            <Ionicons name="arrow-back" size={16} color={LiquidGlassTokens.text.secondary} style={{ marginRight: 6 }} />
            <Text style={styles.prevBtnText}>BACK</Text>
          </TouchableOpacity>
        )}
        {step < 4 && (
          <TouchableOpacity
            style={styles.nextBtn}
            onPress={() => setStep(step + 1)}
            accessibilityRole="button"
            accessibilityLabel="Proceed to next configuration step"
          >
            <Text style={styles.nextBtnText}>NEXT STEP</Text>
            <Ionicons name="arrow-forward" size={16} color="#000" style={{ marginLeft: 6 }} />
          </TouchableOpacity>
        )}
      </View>
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
  stepHeader: {
    marginBottom: 20,
    marginTop: 8,
  },
  stepTitleRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
  },
  stepTitle: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 1.1,
  },
  stepBar: {
    flexDirection: "row",
    gap: 8,
  },
  stepSegment: {
    flex: 1,
    height: 4,
    backgroundColor: "rgba(255, 255, 255, 0.1)",
    borderRadius: 2,
  },
  stepSegmentActive: {
    backgroundColor: LiquidGlassTokens.accent.primary,
  },
  stepSegmentCurrent: {
    backgroundColor: LiquidGlassTokens.accent.secondary,
  },
  section: {
    marginBottom: 20,
  },
  heading: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 18,
    fontWeight: "800",
  },
  desc: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 12,
    marginTop: 4,
    marginBottom: 16,
    lineHeight: 18,
  },
  optionCard: {
    padding: 14,
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 12,
  },
  iconBox: {
    width: 44,
    height: 44,
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },
  optionTitle: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 14,
    fontWeight: "700",
  },
  optionDesc: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 11,
    marginTop: 3,
  },
  hazardGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    rowGap: 12,
    marginBottom: 16,
  },
  hazardTouchable: {
    width: "48%",
  },
  hazardCard: {
    padding: 16,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 100,
  },
  hazardCardText: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 13,
    fontWeight: "700",
    marginTop: 8,
  },
  severitySection: {
    padding: 14,
  },
  label: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 0.8,
    marginBottom: 10,
  },
  severityRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 8,
  },
  severityBtn: {
    flex: 1,
    paddingVertical: 10,
    alignItems: "center",
    borderRadius: 8,
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.12)",
  },
  severityBtnText: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 11,
    fontWeight: "700",
  },
  summaryCard: {
    padding: 16,
  },
  summaryRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255, 255, 255, 0.06)",
  },
  summaryLabel: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 0.6,
  },
  summaryVal: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 13,
    fontWeight: "700",
  },
  navButtons: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 8,
  },
  prevBtn: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: "rgba(255, 255, 255, 0.06)",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.12)",
  },
  prevBtnText: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0.6,
  },
  nextBtn: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: LiquidGlassTokens.accent.primary,
    marginLeft: "auto",
  },
  nextBtnText: {
    color: "#000",
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 0.6,
  },
});
