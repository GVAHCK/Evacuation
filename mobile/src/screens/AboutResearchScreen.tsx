import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LiquidGlassTokens } from '../theme/tokens';
import { GlassCard } from '../components/glass/GlassCard';
import { GlassBadge } from '../components/glass/GlassBadge';
import { GlassHeader } from '../components/glass/GlassHeader';
import { useTheme } from '../theme/ThemeContext';

export const AboutResearchScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();

  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
      <GlassHeader
        title="About & Research Scope"
        subtitle="Academic dissertation methodology & safety limitations"
        showBack
        onBackPress={() => {
          haptic('light');
          navigation.goBack();
        }}
        badge={
          <GlassBadge
            label="FINAL YEAR THESIS"
            variant="ai"
            icon="school-outline"
          />
        }
      />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Thesis Title Card */}
        <GlassCard variant="highlight" style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="book-outline" size={24} color={LiquidGlassTokens.accent.primary} />
            <View style={{ flex: 1 }}>
              <Text style={styles.cardTitle}>FINAL YEAR ENGINEERING DISSERTATION</Text>
              <Text style={styles.thesisTitle}>
                Reinforcement-Learning-Based Multi-Hazard Evacuation Routing
              </Text>
            </View>
          </View>
          <Text style={styles.cardDesc}>
            An investigation into coordinating multi-agent cohort evacuations across flood-inundated and fire-threatened metropolitan road networks using Offline-to-Online Reinforcement Learning.
          </Text>
        </GlassCard>

        {/* Mathematical Problem Formulation */}
        <Text style={styles.sectionHeader}>MULTI-AGENT POMDP FORMULATION</Text>
        <GlassCard variant="standard" style={styles.card}>
          <Text style={styles.para}>
            The evacuation environment is modeled as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) defined by the tuple &lang;S, {`{A_i}`}, P, {`{R_i}`}, {`{Ω_i}`}, O, γ&rang;:
          </Text>

          <View style={styles.mathItem}>
            <Text style={styles.mathTerm}>• State Space (S):</Text>
            <Text style={styles.mathDef}>
              336-node road network topology with dynamic 2D flood depth contours, fire propagation fronts, and Bureau of Public Roads (BPR) edge vehicular densities.
            </Text>
          </View>

          <View style={styles.mathItem}>
            <Text style={styles.mathTerm}>• Action Space (A_i):</Text>
            <Text style={styles.mathDef}>
              Discrete topological successor edge selection constrained by a hard safety shield preventing entry into impassable flooded links.
            </Text>
          </View>

          <View style={styles.mathItem}>
            <Text style={styles.mathTerm}>• Reward Formulation (R_i):</Text>
            <Text style={styles.mathDef}>
              R_i = +10.0 · 1_{'{arrival}'} - 0.1 · Δt - 5.0 · H(e_t) - 2.0 · C(a_t), penalizing delay, hazard encroachment, and bottleneck congestion.
            </Text>
          </View>
        </GlassCard>

        {/* Offline-to-Online MARL Architecture */}
        <Text style={styles.sectionHeader}>OFFLINE CQL + ONLINE KL FINE-TUNING</Text>
        <GlassCard variant="standard" style={styles.card}>
          <Text style={styles.para}>
            To overcome the sample inefficiency and catastrophic real-world failures of pure online reinforcement learning, our framework operates in two distinct phases:
          </Text>

          <View style={styles.phaseRow}>
            <View style={styles.phaseBadge}>
              <Text style={styles.phaseNum}>1</Text>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.phaseTitle}>Offline Conservative Q-Learning (CQL)</Text>
              <Text style={styles.phaseDesc}>
                Pre-trained on 50,000 diverse historical flood/fire episodes. CQL adds a conservative regularization term that lower-bounds Q-values for out-of-distribution actions.
              </Text>
            </View>
          </View>

          <View style={styles.divider} />

          <View style={styles.phaseRow}>
            <View style={styles.phaseBadge}>
              <Text style={styles.phaseNum}>2</Text>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.phaseTitle}>KL-Regularized Online Fine-Tuning</Text>
              <Text style={styles.phaseDesc}>
                Adapts in real-time to active simulation events while maintaining a Kullback-Leibler penalty against the pre-trained policy to prevent catastrophic policy collapse.
              </Text>
            </View>
          </View>
        </GlassCard>

        {/* Academic Limitations & Disclaimers */}
        <Text style={styles.sectionHeader}>RESEARCH LIMITATIONS & SAFETY NOTICE</Text>
        <GlassCard variant="standard" style={[styles.card, { borderColor: 'rgba(255, 177, 66, 0.3)' }]}>
          <View style={styles.warningRow}>
            <Ionicons name="alert-circle" size={24} color={LiquidGlassTokens.status.warning} />
            <Text style={styles.warningTitle}>Non-Operational Emergency Dispatch</Text>
          </View>
          <Text style={styles.warningDesc}>
            This application is an academic software prototype developed solely for final-year engineering research validation.
          </Text>
          <Text style={styles.bulletItem}>
            • Results are validated via microscopic SUMO/TraCI and calibrated BPR digital twins; they do not represent real-world physical human trial deployments.
          </Text>
          <Text style={styles.bulletItem}>
            • The platform must NOT be used as an authoritative 911 or civil defense evacuation dispatch system.
          </Text>
          <Text style={styles.bulletItem}>
            • Real emergency responses require certified municipal disaster command authority, emergency radio integration, and field inspection.
          </Text>
        </GlassCard>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: LiquidGlassTokens.background.canvas,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 40,
  },
  card: {
    padding: 16,
    marginBottom: 8,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 8,
  },
  cardTitle: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.8,
    color: LiquidGlassTokens.accent.primary,
  },
  thesisTitle: {
    fontSize: 15,
    fontWeight: '800',
    color: LiquidGlassTokens.text.primary,
    marginTop: 2,
    lineHeight: 20,
  },
  cardDesc: {
    fontSize: 13,
    lineHeight: 19,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 6,
  },
  sectionHeader: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1.0,
    color: LiquidGlassTokens.text.muted,
    marginTop: 20,
    marginBottom: 10,
  },
  para: {
    fontSize: 13,
    lineHeight: 19,
    color: LiquidGlassTokens.text.secondary,
    marginBottom: 10,
  },
  mathItem: {
    marginBottom: 8,
  },
  mathTerm: {
    fontSize: 12,
    fontWeight: '700',
    color: LiquidGlassTokens.accent.primary,
  },
  mathDef: {
    fontSize: 12,
    lineHeight: 17,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 2,
  },
  phaseRow: {
    flexDirection: 'row',
    gap: 12,
    alignItems: 'flex-start',
  },
  phaseBadge: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: 'rgba(0, 210, 255, 0.15)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  phaseNum: {
    fontSize: 12,
    fontWeight: '800',
    color: LiquidGlassTokens.accent.primary,
  },
  phaseTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  phaseDesc: {
    fontSize: 12,
    lineHeight: 17,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 3,
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    marginVertical: 12,
  },
  warningRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  warningTitle: {
    fontSize: 13,
    fontWeight: '800',
    color: LiquidGlassTokens.status.warning,
  },
  warningDesc: {
    fontSize: 12,
    fontWeight: '600',
    color: LiquidGlassTokens.text.primary,
    marginBottom: 8,
    lineHeight: 17,
  },
  bulletItem: {
    fontSize: 12,
    lineHeight: 18,
    color: LiquidGlassTokens.text.secondary,
    marginBottom: 6,
  },
});
