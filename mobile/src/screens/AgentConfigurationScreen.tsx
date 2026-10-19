import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
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

export const AgentConfigurationScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();

  const [cohortSize, setCohortSize] = useState(4);
  const [selectedTypes, setSelectedTypes] = useState<string[]>([
    'type_a',
    'type_b',
    'type_c',
    'type_d',
  ]);

  const agentArchetypes = [
    {
      id: 'type_a',
      name: 'Type A (Standard Civilian)',
      desc: 'Balanced evacuation routing optimizing path distance and standard flow velocity.',
      icon: 'car-outline' as const,
      color: LiquidGlassTokens.accent.primary,
    },
    {
      id: 'type_b',
      name: 'Type B (Congestion-Sensitive)',
      desc: 'High sensitivity to road occupancy and queue buildup; avoids bottlenecks.',
      icon: 'speedometer-outline' as const,
      color: LiquidGlassTokens.accent.warning,
    },
    {
      id: 'type_c',
      name: 'Type C (Hazard-Sensitive)',
      desc: 'Prioritizes safety buffer; diverts immediately upon water depth >0.25m or radiant heat.',
      icon: 'shield-outline' as const,
      color: LiquidGlassTokens.accent.danger,
    },
    {
      id: 'type_d',
      name: 'Type D (Perimeter Corridor)',
      desc: 'Utilizes peripheral arterial links to bypass urban choke-points.',
      icon: 'navigate-outline' as const,
      color: LiquidGlassTokens.accent.ai,
    },
  ];

  const handleAdjustCohort = (delta: number) => {
    const next = cohortSize + delta;
    if (next >= 1 && next <= 8) {
      haptic.selection();
      setCohortSize(next);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <GlassHeader
        title="Multi-Agent Cohort Setup"
        showBack
        onBackPress={() => navigation.goBack()}
        badge={<GlassBadge status="ai" label="MARL Agents" size="sm" />}
      />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Cohort Size Stepper */}
        <Text style={styles.sectionHeader}>Cohort Size (1 to 8 Agents)</Text>
        <GlassCard style={styles.card}>
          <View style={styles.stepperRow}>
            <View style={styles.stepperLeft}>
              <Text style={styles.stepperTitle}>Concurrent Evacuees</Text>
              <Text style={styles.stepperSub}>
                Authoritative Offline-to-Online policy simulates coordinated interactions.
              </Text>
            </View>

            <View style={styles.counterBox}>
              <TouchableOpacity
                style={[styles.counterBtn, cohortSize <= 1 && styles.counterBtnDisabled]}
                onPress={() => handleAdjustCohort(-1)}
                disabled={cohortSize <= 1}
                accessibilityRole="button"
                accessibilityLabel="Decrease agent count"
              >
                <Ionicons name="remove" size={20} color={cohortSize <= 1 ? '#64748B' : '#FFFFFF'} />
              </TouchableOpacity>

              <Text style={styles.counterVal}>{cohortSize}</Text>

              <TouchableOpacity
                style={[styles.counterBtn, cohortSize >= 8 && styles.counterBtnDisabled]}
                onPress={() => handleAdjustCohort(1)}
                disabled={cohortSize >= 8}
                accessibilityRole="button"
                accessibilityLabel="Increase agent count"
              >
                <Ionicons name="add" size={20} color={cohortSize >= 8 ? '#64748B' : '#FFFFFF'} />
              </TouchableOpacity>
            </View>
          </View>
        </GlassCard>

        {/* Agent Behavioral Archetypes */}
        <Text style={styles.sectionHeader}>Agent Behavioral Archetypes</Text>
        {agentArchetypes.map((type) => {
          const isSelected = selectedTypes.includes(type.id);
          return (
            <TouchableOpacity
              key={type.id}
              activeOpacity={0.8}
              onPress={() => {
                haptic.selection();
                setSelectedTypes((prev) =>
                  isSelected ? prev.filter((t) => t !== type.id) : [...prev, type.id]
                );
              }}
            >
              <GlassCard
                style={[styles.card, isSelected && styles.cardSelected]}
                variant={isSelected ? 'standard' : 'ultraThin'}
              >
                <View style={styles.typeHeader}>
                  <View style={styles.typeLeft}>
                    <View style={[styles.typeIconBox, { backgroundColor: 'rgba(255, 255, 255, 0.08)' }]}>
                      <Ionicons name={type.icon} size={22} color={type.color} />
                    </View>
                    <View style={styles.typeTextCol}>
                      <Text style={styles.typeName}>{type.name}</Text>
                      <Text style={styles.typeDesc}>{type.desc}</Text>
                    </View>
                  </View>
                  <Ionicons
                    name={isSelected ? 'checkbox' : 'square-outline'}
                    size={22}
                    color={isSelected ? LiquidGlassTokens.accent.primary : LiquidGlassTokens.text.muted}
                  />
                </View>
              </GlassCard>
            </TouchableOpacity>
          );
        })}

        <GlassButton
          title="Save Agent Configuration"
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
    borderColor: 'rgba(0, 210, 255, 0.35)',
  },
  stepperRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  stepperLeft: {
    flex: 1,
    marginRight: 10,
  },
  stepperTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  stepperSub: {
    fontSize: 12,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 4,
    lineHeight: 16,
  },
  counterBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.40)',
    borderRadius: LiquidGlassTokens.radius.md,
    padding: 4,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
  },
  counterBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  counterBtnDisabled: {
    opacity: 0.3,
  },
  counterVal: {
    fontSize: 18,
    fontWeight: '800',
    color: LiquidGlassTokens.text.primary,
    paddingHorizontal: 16,
  },
  typeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  typeLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    marginRight: 10,
  },
  typeIconBox: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  typeTextCol: {
    flex: 1,
  },
  typeName: {
    fontSize: 14,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  typeDesc: {
    fontSize: 12,
    color: LiquidGlassTokens.text.muted,
    marginTop: 3,
    lineHeight: 16,
  },
});
