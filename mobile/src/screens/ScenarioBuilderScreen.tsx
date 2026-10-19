import React, { useState } from "react";
import { View, Text, StyleSheet, ScrollView, TextInput, TouchableOpacity } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { LiquidGlassTokens } from "../theme/tokens";
import { GlassCard } from "../components/glass/GlassCard";
import { GlassButton } from "../components/glass/GlassButton";
import { GlassPill } from "../components/glass/GlassPill";
import { Api } from "../api/client";
import { useEvacuationStore } from "../store/evacuationStore";

export const ScenarioBuilderScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const [name, setName] = useState("Severe Coastal Inundation Scenario");
  const [description, setDescription] = useState("Category 3 coastal surge causing extensive low-lying road blockages");
  const [network, setNetwork] = useState("336_nodes");
  const [hazardType, setHazardType] = useState<"flood" | "fire" | "combined">("flood");
  const [severity, setSeverity] = useState("1.8");
  const [trafficRegime, setTrafficRegime] = useState<"low" | "medium" | "high">("medium");
  const [agentCount, setAgentCount] = useState("4");
  const [seed, setSeed] = useState("42");
  const [isSaving, setIsSaving] = useState(false);

  const { startEvacuation } = useEvacuationStore();

  const handleSaveAndRun = async () => {
    setIsSaving(true);
    try {
      const scenario = await Api.createScenario({
        name,
        description,
        network,
        origin: "node_0",
        destination: "node_335",
        agent_count: parseInt(agentCount) || 4,
        hazard_type: hazardType,
        hazard_config: {
          hazard_type: hazardType,
          severity: parseFloat(severity) || 1.5,
          radius_meters: 500,
        },
        traffic_config: {
          density_regime: trafficRegime,
          congestion_bias: trafficRegime === "high" ? 0.45 : 0.25,
        },
        seed: parseInt(seed) || 42,
      });

      // Launch simulation with created scenario
      const session = await startEvacuation({
        origin: [13.002, 80.205],
        destination: [13.048, 80.248],
        agent_count: parseInt(agentCount) || 4,
        hazard_config: {
          hazard_type: hazardType,
          severity: parseFloat(severity) || 1.5,
          radius_meters: 500,
        },
        scenario_id: scenario.id,
      });

      setIsSaving(false);
      if (session) {
        navigation.navigate("LiveEvacuation");
      }
    } catch (e) {
      setIsSaving(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.headerRow}>
        <View style={{ flex: 1 }}>
          <Text style={styles.title}>SCENARIO BUILDER</Text>
          <Text style={styles.subtitle}>Configure parametric multi-hazard benchmark & stress scenarios</Text>
        </View>
        <GlassPill label="Offline-to-Online MARL" variant="ai" />
      </View>

      <GlassCard variant="standard" style={styles.card}>
        {/* Basic Meta */}
        <View style={styles.inputGroup}>
          <Text style={styles.label}>SCENARIO TITLE</Text>
          <TextInput
            style={styles.input}
            value={name}
            onChangeText={setName}
            placeholder="Scenario Title"
            placeholderTextColor={LiquidGlassTokens.text.muted}
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>DESCRIPTION / OPERATIONAL NOTES</Text>
          <TextInput
            style={[styles.input, { height: 60, textAlignVertical: "top" }]}
            value={description}
            onChangeText={setDescription}
            multiline
            placeholder="Brief operational context"
            placeholderTextColor={LiquidGlassTokens.text.muted}
          />
        </View>

        {/* Network Topology */}
        <View style={styles.inputGroup}>
          <Text style={styles.label}>NETWORK TOPOLOGY</Text>
          <View style={styles.btnRow}>
            {[
              { id: "336_nodes", title: "336-Node Multi-District" },
              { id: "84_nodes", title: "84-Node Baseline" },
            ].map((net) => {
              const isSelected = network === net.id;
              return (
                <TouchableOpacity
                  key={net.id}
                  style={[
                    styles.toggleBtn,
                    isSelected && {
                      borderColor: LiquidGlassTokens.accent.primary,
                      backgroundColor: "rgba(0, 210, 255, 0.15)",
                    },
                  ]}
                  onPress={() => setNetwork(net.id)}
                >
                  <Text
                    style={[
                      styles.toggleBtnText,
                      isSelected && { color: LiquidGlassTokens.accent.primary, fontWeight: "800" },
                    ]}
                  >
                    {net.title}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </View>

        {/* Hazard Type */}
        <View style={styles.inputGroup}>
          <Text style={styles.label}>HAZARD REGIME</Text>
          <View style={styles.btnRow}>
            {(["flood", "fire", "combined"] as const).map((hz) => {
              const isSelected = hazardType === hz;
              const color =
                hz === "flood"
                  ? LiquidGlassTokens.accent.primary
                  : hz === "fire"
                  ? LiquidGlassTokens.accent.danger
                  : LiquidGlassTokens.accent.warning;
              return (
                <TouchableOpacity
                  key={hz}
                  style={[
                    styles.toggleBtn,
                    isSelected && {
                      borderColor: color,
                      backgroundColor: "rgba(255, 255, 255, 0.08)",
                    },
                  ]}
                  onPress={() => setHazardType(hz)}
                >
                  <Text
                    style={[
                      styles.toggleBtnText,
                      isSelected && { color: color, fontWeight: "800" },
                    ]}
                  >
                    {hz.toUpperCase()}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </View>

        {/* Severity & Traffic */}
        <View style={{ flexDirection: "row", gap: 12 }}>
          <View style={[styles.inputGroup, { flex: 1 }]}>
            <Text style={styles.label}>SEVERITY MULTIPLIER</Text>
            <TextInput
              style={styles.input}
              value={severity}
              onChangeText={setSeverity}
              keyboardType="numeric"
              placeholder="1.5"
              placeholderTextColor={LiquidGlassTokens.text.muted}
            />
          </View>

          <View style={[styles.inputGroup, { flex: 1 }]}>
            <Text style={styles.label}>AGENT COHORT</Text>
            <TextInput
              style={styles.input}
              value={agentCount}
              onChangeText={setAgentCount}
              keyboardType="numeric"
              placeholder="4"
              placeholderTextColor={LiquidGlassTokens.text.muted}
            />
          </View>
        </View>

        {/* Traffic Regime */}
        <View style={styles.inputGroup}>
          <Text style={styles.label}>TRAFFIC CONGESTION PROFILE</Text>
          <View style={styles.btnRow}>
            {(["low", "medium", "high"] as const).map((tr) => {
              const isSelected = trafficRegime === tr;
              return (
                <TouchableOpacity
                  key={tr}
                  style={[
                    styles.toggleBtn,
                    isSelected && {
                      borderColor: LiquidGlassTokens.accent.secondary,
                      backgroundColor: "rgba(56, 189, 248, 0.15)",
                    },
                  ]}
                  onPress={() => setTrafficRegime(tr)}
                >
                  <Text
                    style={[
                      styles.toggleBtnText,
                      isSelected && { color: LiquidGlassTokens.accent.secondary, fontWeight: "800" },
                    ]}
                  >
                    {tr.toUpperCase()}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </View>

        {/* Random Seed */}
        <View style={styles.inputGroup}>
          <Text style={styles.label}>STOCHASTIC SEED</Text>
          <TextInput
            style={styles.input}
            value={seed}
            onChangeText={setSeed}
            keyboardType="numeric"
            placeholder="42"
            placeholderTextColor={LiquidGlassTokens.text.muted}
          />
        </View>

        <GlassButton
          title={isSaving ? "CONFIGURING & STARTING..." : "SAVE & LAUNCH SCENARIO"}
          onPress={handleSaveAndRun}
          loading={isSaving}
          variant="primary"
          icon={<Ionicons name="rocket-outline" size={18} color="#000" style={{ marginRight: 6 }} />}
          style={{ marginTop: 10 }}
        />
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
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 16,
    marginTop: 8,
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
  card: {
    padding: 18,
  },
  inputGroup: {
    marginBottom: 14,
  },
  label: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 0.6,
    marginBottom: 6,
  },
  input: {
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: LiquidGlassTokens.text.primary,
    fontSize: 13,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.12)",
  },
  btnRow: {
    flexDirection: "row",
    gap: 8,
  },
  toggleBtn: {
    flex: 1,
    paddingVertical: 10,
    alignItems: "center",
    borderRadius: 8,
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.12)",
  },
  toggleBtnText: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 11,
    fontWeight: "700",
  },
});
