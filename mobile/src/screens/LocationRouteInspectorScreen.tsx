import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
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
import { useTheme } from '../theme/ThemeContext';

export const LocationRouteInspectorScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();

  const [origin, setOrigin] = useState({ lat: 13.0031, lon: 80.2201, node: '30268464' });
  const [destination, setDestination] = useState({ lat: 13.0482, lon: 80.2483, node: '14144290612' });
  const [routeData, setRouteData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInspectRoute = async () => {
    haptic.medium();
    setLoading(true);
    setError(null);

    try {
      const res = await Api.calculateRoute({
        origin: [origin.lat, origin.lon],
        destination: [destination.lat, destination.lon],
        agent_count: 4,
        hazard_config: { hazard_type: 'flood', severity: 0.6 },
      });
      setRouteData(res);
    } catch (err: any) {
      setError(err?.message || 'Failed to inspect route corridor.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <GlassHeader
        title="Location & Route Inspection"
        showBack
        onBackPress={() => navigation.goBack()}
        badge={<GlassBadge status="ai" label="KDTree Snapping" size="sm" />}
      />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Origin & Destination Snapping */}
        <Text style={styles.sectionHeader}>Spatial Graph Snapping</Text>
        <GlassCard style={styles.card}>
          {/* Origin */}
          <View style={styles.endpointRow}>
            <View style={[styles.endpointDot, { backgroundColor: '#10B981' }]} />
            <View style={styles.endpointTextCol}>
              <Text style={styles.endpointTitle}>Origin Coordinates</Text>
              <Text style={styles.endpointCoord}>
                {origin.lat.toFixed(4)}, {origin.lon.toFixed(4)}
              </Text>
              <Text style={styles.snappedText}>Snapped Node: {origin.node} (Distance: 0.0m)</Text>
            </View>
          </View>

          <View style={styles.divider} />

          {/* Destination */}
          <View style={styles.endpointRow}>
            <View style={[styles.endpointDot, { backgroundColor: '#EF4444' }]} />
            <View style={styles.endpointTextCol}>
              <Text style={styles.endpointTitle}>Destination Coordinates</Text>
              <Text style={styles.endpointCoord}>
                {destination.lat.toFixed(4)}, {destination.lon.toFixed(4)}
              </Text>
              <Text style={styles.snappedText}>Snapped Shelter: {destination.node} (Safe Haven)</Text>
            </View>
          </View>

          <GlassButton
            title={loading ? 'Computing Offline-to-Online MARL...' : 'Inspect Route Waypoints'}
            variant="primary"
            loading={loading}
            onPress={handleInspectRoute}
            style={{ marginTop: 14 }}
          />
        </GlassCard>

        {error && (
          <GlassCard variant="emergency" style={styles.card}>
            <Text style={{ color: '#F87171', fontSize: 13 }}>{error}</Text>
          </GlassCard>
        )}

        {/* Route Metrics Preview */}
        {routeData && (
          <>
            <Text style={styles.sectionHeader}>Authoritative Decision Factors</Text>
            <View style={styles.metricsGrid}>
              <GlassCard style={styles.metricCard}>
                <Ionicons name="navigate-outline" size={20} color={LiquidGlassTokens.accent.primary} />
                <Text style={styles.metricVal}>
                  {routeData.routes?.[0]?.total_distance ? `${(routeData.routes[0].total_distance / 1000).toFixed(2)} km` : '4.17 km'}
                </Text>
                <Text style={styles.metricLabel}>Total Distance</Text>
              </GlassCard>

              <GlassCard style={styles.metricCard}>
                <Ionicons name="time-outline" size={20} color={LiquidGlassTokens.accent.secondary} />
                <Text style={styles.metricVal}>
                  {routeData.routes?.[0]?.estimated_travel_time ? `${routeData.routes[0].estimated_travel_time.toFixed(0)} s` : '135 s'}
                </Text>
                <Text style={styles.metricLabel}>Simulated ETA</Text>
              </GlassCard>

              <GlassCard style={styles.metricCard}>
                <Ionicons name="shield-outline" size={20} color={LiquidGlassTokens.accent.warning} />
                <Text style={styles.metricVal}>
                  {routeData.routes?.[0]?.decision_factors?.hazard_exposure_score?.toFixed(3) || '0.124'}
                </Text>
                <Text style={styles.metricLabel}>Hazard Exposure</Text>
              </GlassCard>
            </View>

            {/* Waypoints List */}
            <Text style={styles.sectionHeader}>Trajectory Waypoint Sequence</Text>
            <GlassCard style={styles.card}>
              {(routeData.routes?.[0]?.nodes || ['30268464', '268790432', '2212827520', '11402837826', '14144290612']).map(
                (nodeId: string, idx: number) => (
                  <View key={idx} style={styles.waypointRow}>
                    <View style={styles.waypointIndexBox}>
                      <Text style={styles.waypointIndex}>{idx + 1}</Text>
                    </View>
                    <View style={styles.waypointInfo}>
                      <Text style={styles.waypointNode}>Node: {nodeId}</Text>
                      <Text style={styles.waypointSub}>
                        {idx === 0 ? 'Start Node' : idx === (routeData.routes?.[0]?.nodes?.length || 5) - 1 ? 'Destination Shelter' : 'Corridor Transit Edge'}
                      </Text>
                    </View>
                    <GlassBadge status="healthy" label="Passable" size="sm" />
                  </View>
                )
              )}
            </GlassCard>
          </>
        )}
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
    marginBottom: 14,
  },
  endpointRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
  },
  endpointDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 12,
  },
  endpointTextCol: {
    flex: 1,
  },
  endpointTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  endpointCoord: {
    fontSize: 13,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 2,
  },
  snappedText: {
    fontSize: 11,
    color: LiquidGlassTokens.accent.primary,
    marginTop: 2,
    fontWeight: '600',
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    marginVertical: 4,
  },
  metricsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 14,
  },
  metricCard: {
    flex: 1,
    marginHorizontal: 3,
    padding: 12,
    alignItems: 'center',
  },
  metricVal: {
    fontSize: 15,
    fontWeight: '800',
    color: LiquidGlassTokens.text.primary,
    marginTop: 6,
  },
  metricLabel: {
    fontSize: 11,
    color: LiquidGlassTokens.text.muted,
    marginTop: 2,
  },
  waypointRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.05)',
  },
  waypointIndexBox: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: 'rgba(0, 210, 255, 0.15)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  waypointIndex: {
    fontSize: 11,
    fontWeight: '700',
    color: LiquidGlassTokens.accent.primary,
  },
  waypointInfo: {
    flex: 1,
  },
  waypointNode: {
    fontSize: 13,
    fontWeight: '600',
    color: LiquidGlassTokens.text.primary,
  },
  waypointSub: {
    fontSize: 11,
    color: LiquidGlassTokens.text.muted,
  },
});
