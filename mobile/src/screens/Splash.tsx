import React, { useEffect } from "react";
import { View, Text, StyleSheet, ActivityIndicator } from "react-native";
import { Colors } from "../theme/colors";

export const SplashScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      navigation.replace("MainTabs");
    }, 1500);
    return () => clearTimeout(timer);
  }, [navigation]);

  return (
    <View style={styles.container}>
      <View style={styles.logoBadge}>
        <Text style={styles.logoIcon}>🛡️</Text>
      </View>
      <Text style={styles.title}>FOODTAILOR EVACUATION</Text>
      <Text style={styles.subtitle}>Multi-Hazard Dynamic Routing Intelligence</Text>

      <View style={styles.marlPill}>
        <Text style={styles.marlText}>Offline-to-Online MARL Engine</Text>
      </View>

      <ActivityIndicator size="small" color={Colors.primary} style={styles.loader} />
    </View>
  );
};
export const Splash = SplashScreen;


const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  logoBadge: {
    width: 80,
    height: 80,
    borderRadius: 24,
    backgroundColor: Colors.card,
    borderWidth: 1.5,
    borderColor: Colors.border,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 20,
  },
  logoIcon: {
    fontSize: 40,
  },
  title: {
    color: Colors.text,
    fontSize: 22,
    fontWeight: "900",
    letterSpacing: 1.5,
    textAlign: "center",
  },
  subtitle: {
    color: Colors.textSecondary,
    fontSize: 13,
    marginTop: 6,
    textAlign: "center",
  },
  marlPill: {
    marginTop: 24,
    backgroundColor: Colors.marlPillSurface,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: Colors.marlPill,
  },
  marlText: {
    color: Colors.text,
    fontSize: 11,
    fontWeight: "700",
  },
  loader: {
    marginTop: 40,
  },
});
