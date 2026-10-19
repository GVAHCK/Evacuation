import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Switch,
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

export const HazardLayerControlsScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();

  const [layers, setLayers] = useState({
    flood: true,
    fire: true,
    blockages: true,
    congestion: true,
    riskHeatmap: false,
    shelters: true,
  });

  const toggleLayer = (key: keyof typeof layers) => {
    haptic.selection();
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <GlassHeader
        title="Geospatial Layer Controls"
        showBack
        onBackPress={() => navigation.goBack()}
        badge={<GlassBadge status="combined" label="Multi-Hazard" size="sm" />}
      />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.sectionHeader}>Active Environmental Overlays</Text>

        {/* Flood Layer */}
        <GlassCard style={styles.card}>
          <View style={styles.row}>
            <View style={styles.leftInfo}>
              <View style={[styles.iconCircle, { backgroundColor: 'rgba(56, 189, 248, 0.16)' }]}>
                <Ionicons name="water" size={22} color="#38BDF8" />
              </View>
              <View style={styles.textCol}>
                <Text style={styles.layerTitle}>Flood Water Depth</Text>
                <Text style={styles.layerSub}>Hydrodynamic cellular depth (&gt;0.30m speed reduction, &gt;0.65m closure)</Text>
              </View>
            </View>
            <Switch
              value={layers.flood}
              onValueChange={() => toggleLayer('flood')}
              trackColor={{ false: '#1e293b', true: '#0284c7' }}
              thumbColor={layers.flood ? '#38bdf8' : '#64748b'}
            />
          </View>
        </GlassCard>

        {/* Fire Layer */}
        <GlassCard style={styles.card}>
          <View style={styles.row}>
            <View style={styles.leftInfo}>
              <View style={[styles.iconCircle, { backgroundColor: 'rgba(239, 68, 68, 0.16)' }]}>
                <Ionicons name="flame" size={22} color="#EF4444" />
              </View>
              <View style={styles.textCol}>
                <Text style={styles.layerTitle}>Thermal Fire Diffusion</Text>
                <Text style={styles.layerSub}>Thermal spread &amp; smoke plume (&gt;0.70 intensity blocks traversability)</Text>
              </View>
            </View>
            <Switch
              value={layers.fire}
              onValueChange={() => toggleLayer('fire')}
              trackColor={{ false: '#1e293b', true: '#dc2626' }}
              thumbColor={layers.fire ? '#ef4444' : '#64748b'}
            />
          </View>
        </GlassCard>

        {/* Road Blockages */}
        <GlassCard style={styles.card}>
          <View style={styles.row}>
            <View style={styles.leftInfo}>
              <View style={[styles.iconCircle, { backgroundColor: 'rgba(244, 63, 94, 0.16)' }]}>
                <Ionicons name="close-circle" size={22} color="#F43F5E" />
              </View>
              <View style={styles.textCol}>
                <Text style={styles.layerTitle}>Road Blockages (Action Masked)</Text>
                <Text style={styles.layerSub}>Impassable links automatically masked from MARL policy</Text>
              </View>
            </View>
            <Switch
              value={layers.blockages}
              onValueChange={() => toggleLayer('blockages')}
              trackColor={{ false: '#1e293b', true: '#be123c' }}
              thumbColor={layers.blockages ? '#f43f5e' : '#64748b'}
            />
          </View>
        </GlassCard>

        {/* Traffic Congestion */}
        <GlassCard style={styles.card}>
          <View style={styles.row}>
            <View style={styles.leftInfo}>
              <View style={[styles.iconCircle, { backgroundColor: 'rgba(245, 158, 11, 0.16)' }]}>
                <Ionicons name="car" size={22} color="#F59E0B" />
              </View>
              <View style={styles.textCol}>
                <Text style={styles.layerTitle}>Traffic Impedance &amp; Density</Text>
                <Text style={styles.layerSub}>BPR macroscopic congestion &amp; SUMO vehicle flow</Text>
              </View>
            </View>
            <Switch
              value={layers.congestion}
              onValueChange={() => toggleLayer('congestion')}
              trackColor={{ false: '#1e293b', true: '#d97706' }}
              thumbColor={layers.congestion ? '#f59e0b' : '#64748b'}
            />
          </View>
        </GlassCard>

        {/* Shelters */}
        <GlassCard style={styles.card}>
          <View style={styles.row}>
            <View style={styles.leftInfo}>
              <View style={[styles.iconCircle, { backgroundColor: 'rgba(16, 185, 129, 0.16)' }]}>
                <Ionicons name="shield-checkmark" size={22} color="#10B981" />
              </View>
              <View style={styles.textCol}>
                <Text style={styles.layerTitle}>Safe Evacuation Shelters</Text>
                <Text style={styles.layerSub}>5 Designated regional safe haven nodes with capacity metrics</Text>
              </View>
            </View>
            <Switch
              value={layers.shelters}
              onValueChange={() => toggleLayer('shelters')}
              trackColor={{ false: '#1e293b', true: '#059669' }}
              thumbColor={layers.shelters ? '#10b981' : '#64748b'}
            />
          </View>
        </GlassCard>

        {/* Visual Map Legend */}
        <Text style={styles.sectionHeader}>Visual Legend &amp; Color Coding</Text>
        <GlassCard style={styles.legendCard}>
          <View style={styles.legendRow}>
            <View style={[styles.legendIndicator, { backgroundColor: '#00D2FF' }]} />
            <Text style={styles.legendLabel}>Active Evacuation Corridor (MARL)</Text>
          </View>
          <View style={styles.legendRow}>
            <View style={[styles.legendIndicator, { backgroundColor: '#38BDF8' }]} />
            <Text style={styles.legendLabel}>Flood Water Extent (&gt;0.3m)</Text>
          </View>
          <View style={styles.legendRow}>
            <View style={[styles.legendIndicator, { backgroundColor: '#EF4444' }]} />
            <Text style={styles.legendLabel}>Thermal Fire Boundary (&gt;0.7)</Text>
          </View>
          <View style={styles.legendRow}>
            <View style={[styles.legendIndicator, { backgroundColor: '#F59E0B' }]} />
            <Text style={styles.legendLabel}>Severe Congestion Delay</Text>
          </View>
          <View style={styles.legendRow}>
            <View style={[styles.legendIndicator, { backgroundColor: '#10B981' }]} />
            <Text style={styles.legendLabel}>Safe Evacuation Center</Text>
          </View>
        </GlassCard>

        <GlassButton
          title="Apply & Return to Map"
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
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  leftInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    marginRight: 10,
  },
  iconCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  textCol: {
    flex: 1,
  },
  layerTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  layerSub: {
    fontSize: 12,
    color: LiquidGlassTokens.text.muted,
    marginTop: 2,
    lineHeight: 16,
  },
  legendCard: {
    padding: 16,
  },
  legendRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
  },
  legendIndicator: {
    width: 14,
    height: 14,
    borderRadius: 7,
    marginRight: 12,
  },
  legendLabel: {
    fontSize: 13,
    color: LiquidGlassTokens.text.primary,
  },
});
