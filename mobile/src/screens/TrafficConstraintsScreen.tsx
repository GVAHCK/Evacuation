import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LiquidGlassTokens } from '../theme/tokens';
import { GlassCard } from '../components/glass/GlassCard';
import { GlassButton } from '../components/glass/GlassButton';
import { GlassBadge } from '../components/glass/GlassBadge';
import { GlassHeader } from '../components/glass/GlassHeader';
import { useTheme } from '../theme/ThemeContext';

export const TrafficConstraintsScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();

  const [density, setDensity] = useState<'low' | 'moderate' | 'severe'>('moderate');
  const [engineMode, setEngineMode] = useState<'auto' | 'digital_twin' | 'sumo'>('auto');

  const densities = [
    {
      id: 'low' as const,
      label: 'Low Background Flow',
      val: '0.20 Density',
      desc: 'Free-flow arterial conditions with minimal intersection delay.',
      color: LiquidGlassTokens.accent.success,
    },
    {
      id: 'moderate' as const,
      label: 'Peak Hour Congestion',
      val: '0.50 Density',
      desc: 'Typical urban peak congestion with moderate bottleneck formation.',
      color: LiquidGlassTokens.accent.warning,
    },
    {
      id: 'severe' as const,
      label: 'Gridlock / Panic Evacuation',
      val: '0.85 Density',
      desc: 'Severe bottleneck queuing; critical intersections saturated.',
      color: LiquidGlassTokens.accent.danger,
    },
  ];

  return (
    <SafeAreaView style={styles.safeArea}>
      <GlassHeader
        title="Traffic & Congestion Setup"
        showBack
        onBackPress={() => navigation.goBack()}
        badge={<GlassBadge status="warning" label="BPR / SUMO" size="sm" />}
      />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Background Traffic Density */}
        <Text style={styles.sectionHeader}>Background Traffic Density</Text>
        {densities.map((item) => {
          const isSelected = density === item.id;
          return (
            <TouchableOpacity
              key={item.id}
              activeOpacity={0.8}
              onPress={() => {
                haptic.selection();
                setDensity(item.id);
              }}
            >
              <GlassCard
                style={[styles.card, isSelected && styles.cardSelected]}
                variant={isSelected ? 'standard' : 'ultraThin'}
              >
                <View style={styles.row}>
                  <View style={[styles.densityIndicator, { backgroundColor: item.color }]} />
                  <View style={styles.textCol}>
                    <Text style={styles.itemTitle}>{item.label}</Text>
                    <Text style={styles.itemDesc}>{item.desc}</Text>
                  </View>
                  <GlassBadge
                    status={item.id === 'low' ? 'healthy' : item.id === 'moderate' ? 'warning' : 'danger'}
                    label={item.val}
                    size="sm"
                  />
                </View>
              </GlassCard>
            </TouchableOpacity>
          );
        })}

        {/* Engine Selection Truthful Mode */}
        <Text style={styles.sectionHeader}>Traffic Simulation Engine</Text>
        <GlassCard style={styles.card}>
          <View style={styles.engineOption}>
            <View style={styles.engineLeft}>
              <Ionicons name="hardware-chip-outline" size={22} color={LiquidGlassTokens.accent.primary} />
              <View style={{ marginLeft: 10 }}>
                <Text style={styles.engineTitle}>Automatic Engine Selection</Text>
                <Text style={styles.engineDesc}>
                  Uses SUMO TraCI if binary is active on server, otherwise transparently activates BPR Macroscopic Digital Twin fallback.
                </Text>
              </View>
            </View>
            <Ionicons name="radio-button-on" size={20} color={LiquidGlassTokens.accent.primary} />
          </View>

          <View style={styles.divider} />

          <View style={styles.formulaBox}>
            <Text style={styles.formulaLabel}>BPR Travel Time Formula:</Text>
            <Text style={styles.formulaText}>t = t₀ × [ 1 + 0.15 × (V / C)⁴ ]</Text>
            <Text style={styles.formulaSub}>
              Where t₀ is free-flow travel time, V is current volume, and C is road capacity.
            </Text>
          </View>
        </GlassCard>

        <GlassButton
          title="Save Traffic Constraints"
          variant="primary"
          onPress={() => navigation.goBack()}
          style={{ marginTop: 16 }}
        />
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: LiquidGlassTokens.background.primary,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 40,
  },
  sectionHeader: {
    fontSize: 13,
    fontWeight: '700',
    color: LiquidGlassTokens.text.secondary,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 10,
    marginTop: 10,
  },
  card: {
    padding: 16,
    marginBottom: 12,
  },
  cardSelected: {
    borderColor: 'rgba(0, 210, 255, 0.40)',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  densityIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 12,
  },
  textCol: {
    flex: 1,
    marginRight: 10,
  },
  itemTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  itemDesc: {
    fontSize: 12,
    color: LiquidGlassTokens.text.muted,
    marginTop: 3,
    lineHeight: 16,
  },
  engineOption: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 4,
  },
  engineLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    marginRight: 10,
  },
  engineTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  engineDesc: {
    fontSize: 12,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 2,
    lineHeight: 16,
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    marginVertical: 12,
  },
  formulaBox: {
    backgroundColor: 'rgba(0, 0, 0, 0.35)',
    borderRadius: LiquidGlassTokens.radius.md,
    padding: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
  },
  formulaLabel: {
    fontSize: 11,
    fontWeight: '700',
    color: LiquidGlassTokens.text.secondary,
    textTransform: 'uppercase',
  },
  formulaText: {
    fontSize: 13,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    color: LiquidGlassTokens.accent.primary,
    fontWeight: '700',
    marginTop: 4,
  },
  formulaSub: {
    fontSize: 11,
    color: LiquidGlassTokens.text.muted,
    marginTop: 4,
    lineHeight: 15,
  },
});
