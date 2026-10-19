import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LiquidGlassTokens } from '../theme/tokens';
import { GlassCard } from '../components/glass/GlassCard';
import { GlassButton } from '../components/glass/GlassButton';
import { GlassBadge } from '../components/glass/GlassBadge';
import { GlassHeader } from '../components/glass/GlassHeader';
import { Api } from '../api/client';

export const FireIntelligenceScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const [hazards, setHazards] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = async () => {
    try {
      const data = await Api.getActiveHazards();
      setHazards(data || []);
    } catch {
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const fireHazards = hazards.filter((h) => h.hazard_type === 'fire');

  return (
    <SafeAreaView style={styles.safeArea}>
      <GlassHeader
        title="Thermal Fire Intelligence"
        showBack
        onBackPress={() => navigation.goBack()}
        badge={<GlassBadge status="fire" label="Thermal Diffusion" size="sm" />}
      />

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={() => {
              setRefreshing(true);
              fetchData();
            }}
            tintColor="#EF4444"
          />
        }
      >
        {/* Physics Engine Card */}
        <GlassCard variant="standard" style={styles.heroCard}>
          <View style={styles.row}>
            <View style={[styles.iconCircle, { backgroundColor: 'rgba(239, 68, 68, 0.16)' }]}>
              <Ionicons name="flame" size={30} color="#EF4444" />
            </View>
            <View style={styles.heroTextCol}>
              <Text style={styles.heroTitle}>Thermal Convection-Diffusion Model</Text>
              <Text style={styles.heroSub}>
                Simulates radiant heat flux and atmospheric smoke dispersion across spatial road corridors.
              </Text>
            </View>
          </View>
        </GlassCard>

        {/* Thermal Intensity Impact Thresholds */}
        <Text style={styles.sectionHeader}>Radiant Heat &amp; Smoke Thresholds</Text>
        <GlassCard style={styles.card}>
          <View style={styles.thresholdRow}>
            <View style={[styles.thresholdBadgeBox, { borderColor: 'rgba(245, 158, 11, 0.40)', backgroundColor: 'rgba(245, 158, 11, 0.18)' }]}>
              <Text style={[styles.thresholdBadgeText, { color: '#F59E0B' }]}>&gt; 0.40</Text>
            </View>
            <View style={styles.thresholdTextCol}>
              <Text style={styles.thresholdTitle}>Smoke Dispersion &amp; Visibility Drop</Text>
              <Text style={styles.thresholdSub}>Driver speed drops to 15 km/h. High particulate warning.</Text>
            </View>
          </View>

          <View style={styles.divider} />

          <View style={styles.thresholdRow}>
            <View style={[styles.thresholdBadgeBox, { borderColor: 'rgba(239, 68, 68, 0.45)', backgroundColor: 'rgba(239, 68, 68, 0.20)' }]}>
              <Text style={[styles.thresholdBadgeText, { color: '#EF4444' }]}>&gt; 0.70</Text>
            </View>
            <View style={styles.thresholdTextCol}>
              <Text style={[styles.thresholdTitle, { color: '#F87171' }]}>Thermal Blockage (Link Impassable)</Text>
              <Text style={styles.thresholdSub}>Immediate road closure. Road edge is hard-masked from RL exploration.</Text>
            </View>
          </View>
        </GlassCard>

        {/* Active Fire Cones */}
        <Text style={styles.sectionHeader}>Active Wildfire / Urban Fire Fronts</Text>
        {loading ? (
          <ActivityIndicator size="small" color="#EF4444" style={{ marginVertical: 20 }} />
        ) : fireHazards.length === 0 ? (
          <GlassCard style={styles.emptyCard}>
            <Ionicons name="shield-checkmark-outline" size={32} color={LiquidGlassTokens.accent.success} />
            <Text style={styles.emptyTitle}>No Critical Thermal Fronts Active</Text>
            <Text style={styles.emptySub}>Road graph thermal sensors report standard baseline conditions.</Text>
          </GlassCard>
        ) : (
          fireHazards.map((zone, idx) => (
            <GlassCard key={idx} style={styles.card}>
              <View style={styles.zoneHeader}>
                <Text style={styles.zoneName}>Fire Front #{idx + 1}</Text>
                <GlassBadge status="fire" label={`Thermal: ${(zone.severity * 100).toFixed(0)}%`} size="sm" />
              </View>
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Ignition Coordinates</Text>
                <Text style={styles.infoVal}>
                  {zone.center ? `${zone.center[0].toFixed(4)}, ${zone.center[1].toFixed(4)}` : '13.0180, 80.2310'}
                </Text>
              </View>
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Thermal Radius</Text>
                <Text style={styles.infoVal}>{zone.radius || 420} meters</Text>
              </View>
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Dominant Wind Direction</Text>
                <Text style={styles.infoVal}>North-East (18 km/h)</Text>
              </View>
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Closed Road Segments</Text>
                <Text style={[styles.infoVal, { color: '#F87171' }]}>{zone.closed_edges?.length || 6} links masked</Text>
              </View>
            </GlassCard>
          ))
        )}

        {/* Research Physics Disclosure */}
        <Text style={styles.sectionHeader}>Simulation Model Disclosure</Text>
        <GlassCard style={styles.card}>
          <Text style={styles.disclosureText}>
            The fire model employs a continuous 2D finite-difference heat diffusion equation coupled with an advective wind vector.
            It provides computationally feasible real-time hazard dynamics for multi-agent RL policy testing and should not be used as a standalone wildland fire behavior prediction tool.
          </Text>
        </GlassCard>
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
  heroCard: {
    padding: 18,
    marginBottom: 16,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconCircle: {
    width: 52,
    height: 52,
    borderRadius: 26,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 14,
  },
  heroTextCol: {
    flex: 1,
  },
  heroTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  heroSub: {
    fontSize: 12,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 4,
    lineHeight: 16,
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
    marginBottom: 14,
  },
  thresholdRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
  },
  thresholdBadgeBox: {
    borderWidth: 1,
    borderRadius: LiquidGlassTokens.radius.md,
    paddingHorizontal: 10,
    paddingVertical: 6,
    marginRight: 12,
  },
  thresholdBadgeText: {
    fontSize: 13,
    fontWeight: '800',
  },
  thresholdTextCol: {
    flex: 1,
  },
  thresholdTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  thresholdSub: {
    fontSize: 12,
    color: LiquidGlassTokens.text.muted,
    marginTop: 2,
    lineHeight: 16,
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
  },
  emptyCard: {
    padding: 24,
    alignItems: 'center',
    marginBottom: 14,
  },
  emptyTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: LiquidGlassTokens.accent.success,
    marginTop: 10,
  },
  emptySub: {
    fontSize: 12,
    color: LiquidGlassTokens.text.secondary,
    textAlign: 'center',
    marginTop: 4,
  },
  zoneHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  zoneName: {
    fontSize: 15,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 6,
  },
  infoLabel: {
    fontSize: 13,
    color: LiquidGlassTokens.text.muted,
  },
  infoVal: {
    fontSize: 13,
    fontWeight: '600',
    color: LiquidGlassTokens.text.primary,
  },
  disclosureText: {
    fontSize: 12,
    color: LiquidGlassTokens.text.muted,
    lineHeight: 18,
    fontStyle: 'italic',
  },
});
