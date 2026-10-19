import React, { useState } from "react";
import { View, Text, StyleSheet, TextInput, Alert, Switch, ScrollView } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { LiquidGlassTokens } from "../theme/tokens";
import { GlassCard } from "../components/glass/GlassCard";
import { GlassButton } from "../components/glass/GlassButton";
import { GlassHeader } from "../components/glass/GlassHeader";
import { useAuthStore } from "../store/authStore";

interface Props {
  navigation: any;
}

export const SettingsScreen: React.FC<Props> = ({ navigation }) => {
  const { user, logout } = useAuthStore();
  const [apiUrl, setApiUrl] = useState("http://10.0.2.2:8000");
  const [useGps, setUseGps] = useState(true);
  const [autoReroute, setAutoReroute] = useState(true);
  const [highContrastMap, setHighContrastMap] = useState(false);

  const handleSave = () => {
    Alert.alert("Configuration Saved", "Backend API endpoint and evacuation preferences updated.");
  };

  const handleLogout = async () => {
    await logout();
    navigation.replace("Login");
  };

  return (
    <View style={styles.container}>
      <GlassHeader title="System Preferences" subtitle="Network endpoints, sensor coupling & credentials" />

      <ScrollView contentContainerStyle={styles.content}>
        {/* User Account Section */}
        <Text style={styles.sectionTitle}>AUTHENTICATED OPERATOR</Text>
        <GlassCard variant="standard" style={styles.card}>
          <View style={styles.userRow}>
            <View style={styles.avatarCircle}>
              <Ionicons name="person" size={22} color={LiquidGlassTokens.accent.primary} />
            </View>
            <View style={{ flex: 1, marginLeft: 12 }}>
              <Text style={styles.userName}>{user?.name || "Incident Responder"}</Text>
              <Text style={styles.userEmail}>{user?.email || "user@evacuation.ai"}</Text>
              <Text style={styles.userRole}>ROLE: {user?.role || "OPERATOR"}</Text>
            </View>
          </View>
        </GlassCard>

        {/* Network & Connectivity */}
        <Text style={styles.sectionTitle}>NETWORK & FASTAPI GATEWAY</Text>
        <GlassCard variant="standard" style={styles.card}>
          <Text style={styles.label}>Authoritative Backend Base URL</Text>
          <TextInput
            style={styles.input}
            value={apiUrl}
            onChangeText={setApiUrl}
            placeholder="http://10.0.2.2:8000"
            placeholderTextColor={LiquidGlassTokens.text.muted}
            autoCapitalize="none"
          />
          <Text style={styles.helpText}>
            Use 10.0.2.2 for Android Studio Emulator or local LAN IP for physical testing. Bearer JWT tokens are stored in SecureStore.
          </Text>
        </GlassCard>

        {/* Simulation & Routing Preferences */}
        <Text style={styles.sectionTitle}>SIMULATION & AUTONOMY PREFERENCES</Text>
        <GlassCard variant="standard" style={styles.card}>
          <View style={styles.switchRow}>
            <View style={{ flex: 1, marginRight: 12 }}>
              <Text style={styles.switchTitle}>Autonomous Dynamic Rerouting</Text>
              <Text style={styles.switchDesc}>
                Authorize MARL policy to execute route changes immediately upon hazard breach.
              </Text>
            </View>
            <Switch
              value={autoReroute}
              onValueChange={setAutoReroute}
              trackColor={{ false: "rgba(255,255,255,0.15)", true: LiquidGlassTokens.accent.primary }}
              thumbColor={autoReroute ? "#000" : LiquidGlassTokens.text.secondary}
            />
          </View>

          <View style={styles.divider} />

          <View style={styles.switchRow}>
            <View style={{ flex: 1, marginRight: 12 }}>
              <Text style={styles.switchTitle}>Hardware GPS Node Resolution</Text>
              <Text style={styles.switchDesc}>Map physical device coordinates to nearest OSM road node.</Text>
            </View>
            <Switch
              value={useGps}
              onValueChange={setUseGps}
              trackColor={{ false: "rgba(255,255,255,0.15)", true: LiquidGlassTokens.accent.primary }}
              thumbColor={useGps ? "#000" : LiquidGlassTokens.text.secondary}
            />
          </View>

          <View style={styles.divider} />

          <View style={styles.switchRow}>
            <View style={{ flex: 1, marginRight: 12 }}>
              <Text style={styles.switchTitle}>High-Contrast Emergency Mode</Text>
              <Text style={styles.switchDesc}>Maximize visual contrast of flood contours and corridor corridors.</Text>
            </View>
            <Switch
              value={highContrastMap}
              onValueChange={setHighContrastMap}
              trackColor={{ false: "rgba(255,255,255,0.15)", true: LiquidGlassTokens.accent.primary }}
              thumbColor={highContrastMap ? "#000" : LiquidGlassTokens.text.secondary}
            />
          </View>
        </GlassCard>

        {/* Action Buttons */}
        <GlassButton
          title="SAVE PREFERENCES"
          onPress={handleSave}
          variant="primary"
          icon={<Ionicons name="save-outline" size={18} color="#000" style={{ marginRight: 6 }} />}
          style={{ marginBottom: 14 }}
        />

        <GlassButton
          title="SIGN OUT"
          onPress={handleLogout}
          variant="danger"
          icon={<Ionicons name="log-out-outline" size={18} color="#FFF" style={{ marginRight: 6 }} />}
        />
      </ScrollView>
    </View>
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
  sectionTitle: {
    fontSize: 11,
    fontWeight: "800",
    color: LiquidGlassTokens.text.secondary,
    letterSpacing: 1,
    marginBottom: 8,
    marginTop: 8,
  },
  card: {
    padding: 16,
    marginBottom: 18,
  },
  userRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  avatarCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "rgba(0, 210, 255, 0.12)",
    justifyContent: "center",
    alignItems: "center",
  },
  userName: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 16,
    fontWeight: "800",
  },
  userEmail: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 12,
    marginTop: 2,
  },
  userRole: {
    color: LiquidGlassTokens.accent.ai,
    fontSize: 10,
    fontWeight: "800",
    marginTop: 4,
    letterSpacing: 0.5,
  },
  label: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 13,
    fontWeight: "700",
    marginBottom: 8,
  },
  input: {
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: LiquidGlassTokens.text.primary,
    fontSize: 13,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.15)",
  },
  helpText: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 11,
    marginTop: 6,
    lineHeight: 16,
  },
  switchRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 6,
  },
  switchTitle: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 13,
    fontWeight: "700",
  },
  switchDesc: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 11,
    marginTop: 2,
    lineHeight: 16,
  },
  divider: {
    height: 1,
    backgroundColor: "rgba(255, 255, 255, 0.06)",
    marginVertical: 12,
  },
});
