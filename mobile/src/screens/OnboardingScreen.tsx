import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LiquidGlassTokens } from '../theme/tokens';
import { GlassCard } from '../components/glass/GlassCard';
import { GlassButton } from '../components/glass/GlassButton';
import { GlassBadge } from '../components/glass/GlassBadge';
import { useTheme } from '../theme/ThemeContext';

interface Props {
  navigation: any;
}

export const OnboardingScreen: React.FC<Props> = ({ navigation }) => {
  const { haptic } = useTheme();

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        {/* Top Operational Badge */}
        <View style={styles.badgeRow}>
          <GlassBadge
            label="RESEARCH TESTBED v2.0"
            variant="ai"
            icon="school-outline"
          />
        </View>

        <Text style={styles.title}>Dynamic Multi-Hazard Evacuation</Text>
        <Text style={styles.subtitle}>
          Coordinated multi-agent route optimization under coupled hydrodynamics, thermal propagation, and urban traffic congestion.
        </Text>

        {/* Feature Grid with Glass Panels */}
        <View style={styles.features}>
          <GlassCard variant="standard" style={styles.featureItem}>
            <View style={[styles.iconCircle, { backgroundColor: 'rgba(56, 189, 248, 0.15)' }]}>
              <Ionicons name="water-outline" size={24} color={LiquidGlassTokens.accent.primary} />
            </View>
            <View style={styles.featureTextContainer}>
              <Text style={styles.featureTitle}>Coupled Hydro & Thermal Hazards</Text>
              <Text style={styles.featureDesc}>
                Simulates cellular 2D water depth progression and thermal radiation diffusion across OSM road graphs.
              </Text>
            </View>
          </GlassCard>

          <GlassCard variant="standard" style={styles.featureItem}>
            <View style={[styles.iconCircle, { backgroundColor: 'rgba(139, 92, 246, 0.15)' }]}>
              <Ionicons name="hardware-chip-outline" size={24} color={LiquidGlassTokens.accent.ai} />
            </View>
            <View style={styles.featureTextContainer}>
              <Text style={styles.featureTitle}>Offline-to-Online MARL Policy</Text>
              <Text style={styles.featureDesc}>
                Conservative Q-Learning pre-training with hard topological action masking to avoid dangerous road corridors.
              </Text>
            </View>
          </GlassCard>

          <GlassCard variant="standard" style={styles.featureItem}>
            <View style={[styles.iconCircle, { backgroundColor: 'rgba(16, 185, 129, 0.15)' }]}>
              <Ionicons name="flash-outline" size={24} color={LiquidGlassTokens.accent.success} />
            </View>
            <View style={styles.featureTextContainer}>
              <Text style={styles.featureTitle}>Closed-Loop Dynamic Rerouting</Text>
              <Text style={styles.featureDesc}>
                Sub-second trajectory replanning when downstream road segments exceed safety thresholds or suffer closures.
              </Text>
            </View>
          </GlassCard>
        </View>

        {/* Research Scope & Non-Emergency Certification Disclosure */}
        <GlassCard variant="ultraThin" style={styles.disclaimerCard}>
          <View style={styles.disclaimerHeader}>
            <Ionicons name="alert-circle-outline" size={18} color={LiquidGlassTokens.text.muted} />
            <Text style={styles.disclaimerTitle}>ACADEMIC RESEARCH PLATFORM</Text>
          </View>
          <Text style={styles.disclaimerText}>
            This application is an engineering dissertation research evaluation platform. It does not possess governmental emergency service certification and should not replace official civil protection dispatch advisories.
          </Text>
        </GlassCard>

        {/* CTA Button */}
        <GlassButton
          title="ENTER COMMAND PLATFORM"
          variant="primary"
          icon="arrow-forward"
          onPress={() => {
            haptic('medium');
            navigation.navigate('Login');
          }}
          style={styles.ctaButton}
        />
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: LiquidGlassTokens.background.primary,
  },
  content: {
    padding: 20,
    paddingBottom: 40,
  },
  badgeRow: {
    marginBottom: 16,
    alignItems: 'flex-start',
  },
  title: {
    fontSize: 28,
    fontWeight: '800',
    color: LiquidGlassTokens.text.primary,
    letterSpacing: 0.3,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: LiquidGlassTokens.text.secondary,
    lineHeight: 22,
    marginBottom: 24,
  },
  features: {
    marginBottom: 20,
    gap: 12,
  },
  featureItem: {
    flexDirection: 'row',
    padding: 16,
    alignItems: 'center',
  },
  iconCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 14,
  },
  featureTextContainer: {
    flex: 1,
  },
  featureTitle: {
    color: LiquidGlassTokens.text.primary,
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 4,
  },
  featureDesc: {
    color: LiquidGlassTokens.text.secondary,
    fontSize: 12,
    lineHeight: 17,
  },
  disclaimerCard: {
    padding: 14,
    marginBottom: 24,
  },
  disclaimerHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
    gap: 6,
  },
  disclaimerTitle: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  disclaimerText: {
    color: LiquidGlassTokens.text.muted,
    fontSize: 11,
    lineHeight: 16,
  },
  ctaButton: {
    marginTop: 8,
  },
});
