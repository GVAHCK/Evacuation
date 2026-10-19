import React, { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, StyleSheet } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { LiquidGlassTokens } from "../theme/tokens";
import { GlassCard } from "../components/glass/GlassCard";
import { GlassButton } from "../components/glass/GlassButton";
import { useAuthStore } from "../store/authStore";

export const RegisterScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("USER");
  const { register, isLoading, error } = useAuthStore();

  const handleRegister = async () => {
    if (!name || !email || !password) return;
    const success = await register(name, email, password, role);
    if (success) {
      navigation.replace("MainTabs");
    }
  };

  return (
    <View style={styles.container}>
      <GlassCard variant="standard" style={styles.card}>
        <Text style={styles.title}>Register Account</Text>
        <Text style={styles.subtitle}>Join the multi-agent emergency routing intelligence network</Text>

        {error && (
          <View style={styles.errorBox}>
            <Ionicons name="alert-circle" size={16} color={LiquidGlassTokens.accent.danger} style={{ marginRight: 6 }} />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        <View style={styles.inputGroup}>
          <Text style={styles.label}>FULL NAME</Text>
          <View style={styles.inputWrapper}>
            <Ionicons name="person-outline" size={18} color={LiquidGlassTokens.text.muted} style={{ marginLeft: 12, marginRight: 8 }} />
            <TextInput
              style={styles.input}
              value={name}
              onChangeText={setName}
              placeholder="Jane Doe"
              placeholderTextColor={LiquidGlassTokens.text.muted}
            />
          </View>
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>EMAIL ADDRESS</Text>
          <View style={styles.inputWrapper}>
            <Ionicons name="mail-outline" size={18} color={LiquidGlassTokens.text.muted} style={{ marginLeft: 12, marginRight: 8 }} />
            <TextInput
              style={styles.input}
              value={email}
              onChangeText={setEmail}
              placeholder="jane@evacuation.ai"
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

        <View style={styles.inputGroup}>
          <Text style={styles.label}>OPERATOR ROLE</Text>
          <View style={styles.roleRow}>
            {["USER", "RESEARCHER"].map((r) => {
              const isSelected = role === r;
              return (
                <TouchableOpacity
                  key={r}
                  style={[
                    styles.roleBtn,
                    isSelected && {
                      borderColor: LiquidGlassTokens.accent.primary,
                      backgroundColor: "rgba(0, 210, 255, 0.15)",
                    },
                  ]}
                  onPress={() => setRole(r)}
                >
                  <Text
                    style={[
                      styles.roleBtnText,
                      isSelected && { color: LiquidGlassTokens.accent.primary, fontWeight: "800" },
                    ]}
                  >
                    {r}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </View>

        <GlassButton
          title={isLoading ? "REGISTERING..." : "REGISTER & INITIALIZE"}
          onPress={handleRegister}
          loading={isLoading}
          variant="primary"
          icon={<Ionicons name="person-add-outline" size={18} color="#000" style={{ marginRight: 6 }} />}
          style={{ marginTop: 8 }}
        />

        <TouchableOpacity style={styles.loginLink} onPress={() => navigation.navigate("Login")}>
          <Text style={styles.loginText}>
            Already registered? <Text style={{ color: LiquidGlassTokens.accent.primary, fontWeight: "700" }}>Sign in</Text>
          </Text>
        </TouchableOpacity>
      </GlassCard>
    </View>
  );
};
export const Register = RegisterScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: LiquidGlassTokens.background.primary,
    padding: 20,
    justifyContent: "center",
  },
  card: {
    padding: 20,
  },
  title: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 22,
    fontWeight: "800",
  },
  subtitle: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 12,
    marginTop: 4,
    marginBottom: 16,
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
  roleRow: {
    flexDirection: "row",
    gap: 10,
  },
  roleBtn: {
    flex: 1,
    paddingVertical: 10,
    alignItems: "center",
    borderRadius: 8,
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.12)",
  },
  roleBtnText: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 12,
    fontWeight: "700",
  },
  loginLink: {
    alignItems: "center",
    marginTop: 18,
  },
  loginText: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 12,
  },
});
