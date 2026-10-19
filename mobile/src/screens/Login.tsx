import React, { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, StyleSheet, ActivityIndicator } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { LiquidGlassTokens } from "../theme/tokens";
import { GlassCard } from "../components/glass/GlassCard";
import { GlassButton } from "../components/glass/GlassButton";
import { useAuthStore } from "../store/authStore";

export const LoginScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const [email, setEmail] = useState("user@evacuation.ai");
  const [password, setPassword] = useState("user123");
  const { login, isLoading, error } = useAuthStore();

  const handleLogin = async () => {
    const success = await login(email, password);
    if (success) {
      navigation.replace("MainTabs");
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.brandContainer}>
        <View style={styles.logoBadge}>
          <Ionicons name="shield-checkmark" size={32} color={LiquidGlassTokens.accent.primary} />
        </View>
        <Text style={styles.brandTitle}>AEGIS EVACUATION</Text>
        <Text style={styles.brandSubtitle}>Production Offline-to-Online MARL Emergency Network</Text>
      </View>

      <GlassCard variant="standard" style={styles.card}>
        <Text style={styles.cardHeading}>Sign In</Text>
        <Text style={styles.cardSub}>Authenticate to receive live telemetry & dispatch commands</Text>

        {error && (
          <View style={styles.errorBox}>
            <Ionicons name="alert-circle" size={16} color={LiquidGlassTokens.accent.danger} style={{ marginRight: 6 }} />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        <View style={styles.inputGroup}>
          <Text style={styles.label}>OPERATOR EMAIL</Text>
          <View style={styles.inputWrapper}>
            <Ionicons name="mail-outline" size={18} color={LiquidGlassTokens.text.muted} style={{ marginLeft: 12, marginRight: 8 }} />
            <TextInput
              style={styles.input}
              value={email}
              onChangeText={setEmail}
              placeholder="user@evacuation.ai"
              placeholderTextColor={LiquidGlassTokens.text.muted}
              autoCapitalize="none"
              keyboardType="email-address"
            />
          </View>
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>PASSWORD</Text>
          <View style={styles.inputWrapper}>
            <Ionicons name="lock-closed-outline" size={18} color={LiquidGlassTokens.text.muted} style={{ marginLeft: 12, marginRight: 8 }} />
            <TextInput
              style={styles.input}
              value={password}
              onChangeText={setPassword}
              placeholder="••••••••"
              placeholderTextColor={LiquidGlassTokens.text.muted}
              secureTextEntry
            />
          </View>
        </View>

        <GlassButton
          title={isLoading ? "AUTHENTICATING..." : "SIGN IN TO NETWORK"}
          onPress={handleLogin}
          loading={isLoading}
          variant="primary"
          icon={<Ionicons name="log-in-outline" size={18} color="#000" style={{ marginRight: 6 }} />}
          style={{ marginTop: 8 }}
        />

        <TouchableOpacity
          style={styles.registerLink}
          onPress={() => navigation.navigate("Register")}
          accessibilityRole="link"
        >
          <Text style={styles.registerText}>
            Need access credentials? <Text style={{ color: LiquidGlassTokens.accent.primary, fontWeight: "700" }}>Register</Text>
          </Text>
        </TouchableOpacity>
      </GlassCard>

      <GlassCard variant="ultraThin" style={styles.demoNote}>
        <Text style={styles.demoTitle}>DEVELOPMENT TEST ACCOUNTS</Text>
        <Text style={styles.demoText}>Operator: user@evacuation.ai / user123</Text>
        <Text style={styles.demoText}>Research Lead: researcher@evacuation.ai / researcher123</Text>
      </GlassCard>
    </View>
  );
};
export const Login = LoginScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: LiquidGlassTokens.background.primary,
    padding: 20,
    justifyContent: "center",
  },
  brandContainer: {
    alignItems: "center",
    marginBottom: 20,
  },
  logoBadge: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "rgba(0, 210, 255, 0.12)",
    borderWidth: 1,
    borderColor: "rgba(0, 210, 255, 0.3)",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 10,
  },
  brandTitle: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 18,
    fontWeight: "900",
    letterSpacing: 2,
  },
  brandSubtitle: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 11,
    marginTop: 4,
    textAlign: "center",
  },
  card: {
    padding: 20,
  },
  cardHeading: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 20,
    fontWeight: "800",
  },
  cardSub: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 12,
    marginTop: 4,
    marginBottom: 18,
  },
  errorBox: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(239, 68, 68, 0.15)",
    borderWidth: 1,
    borderColor: "rgba(239, 68, 68, 0.3)",
    borderRadius: 8,
    padding: 10,
    marginBottom: 14,
  },
  errorText: {
    color: LiquidGlassTokens.accent.danger,
    fontSize: 12,
    flex: 1,
  },
  inputGroup: {
    marginBottom: 14,
  },
  label: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 9,
    fontWeight: "800",
    marginBottom: 6,
    letterSpacing: 0.6,
  },
  inputWrapper: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.12)",
  },
  input: {
    flex: 1,
    paddingVertical: 12,
    paddingRight: 12,
    color: LiquidGlassTokens.text.primary,
    fontSize: 13,
  },
  registerLink: {
    alignItems: "center",
    marginTop: 18,
  },
  registerText: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 12,
  },
  demoNote: {
    marginTop: 16,
    padding: 14,
  },
  demoTitle: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 9,
    fontWeight: "800",
    letterSpacing: 1,
    marginBottom: 4,
  },
  demoText: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 11,
    marginTop: 2,
  },
});
