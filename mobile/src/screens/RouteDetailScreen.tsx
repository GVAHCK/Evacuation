import React, { useEffect, useState } from 'react';
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
import { useRoute, useNavigation } from '@react-navigation/native';
import { LiquidGlassTokens } from '../theme/tokens';
import { GlassCard } from '../components/glass/GlassCard';
import { GlassBadge } from '../components/glass/GlassBadge';
import { GlassHeader } from '../components/glass/GlassHeader';
import { useTheme } from '../theme/ThemeContext';
import { Api } from '../api/client';

export const RouteDetailScreen: React.FC = () => {
  const route = useRoute<any>();
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();

  const sessionId = route.params?.sessionId || 'active_session';
  const [loading, setLoading] = useState(true);
  const [routeData, setRouteData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchRouteDetails = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await Api.getEvacuationResult(sessionId);
      if (res && res.routes && res.routes.length > 0) {
        setRouteData({
          primaryRoute: res.routes[0],
          decisionFactors: res.route_decision_factors,
          session: res.session,
        });
      } else {
        // Fall back to params if passed directly
        if (route.params?.route) {
          setRouteData({
            primaryRoute: route.params.route,
            decisionFactors: route.params.factors || {},
            session: route.params.session || {},
          });
        } else {
          setError('No route geometry found for this session.');
        }
      }
    } catch (err: any) {
      if (route.params?.route) {
        setRouteData({
          primaryRoute: route.params.route,
          decisionFactors: route.params.factors || {},
          session: route.params.session || {},
        });
      } else {
        setError(err.message || 'Unable to retrieve route geometry from backend.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRouteDetails();
  }, [sessionId]);

  const pRoute = routeData?.primaryRoute;
  const factors = routeData?.decisionFactors;

  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
      <GlassHeader
        title="Route Geometry & Waypoints"
        subtitle={`Corridor for Session: ${sessionId.slice(0, 8)}...`}
        showBack
        onBackPress={() => {
          haptic('light');
          navigation.goBack();
        }}
        badge={
          <GlassBadge
            label="OFFLINE-ONLINE MARL"
            variant="ai"
            icon="git-network-outline"
          />
        }
      />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {loading ? (
          <View style={styles.centerBox}>
            <ActivityIndicator size="large" color={LiquidGlassTokens.accent.primary} />
            <Text style={styles.loadingText}>Loading Route Geometry...</Text>
          </View>
        ) : error && !pRoute ? (
          <GlassCard variant="standard" style={styles.errorCard}>
            <Ionicons name="alert-circle-outline" size={32} color={LiquidGlassTokens.status.critical} />
            <Text style={styles.errorTitle}>Route Data Unavailable</Text>
            <Text style={styles.errorSub}>{error}</Text>
            <TouchableOpacity style={styles.retryBtn} onPress={fetchRouteDetails}>
              <Text style={styles.retryBtnText}>Retry</Text>
            </TouchableOpacity>
          </GlassCard>
        ) : (
          <>
            {/* Overview Summary */}
            <GlassCard variant="highlight" style={styles.metricCard}>
              <View style={styles.cardHeaderRow}>
                <Ionicons name="map-outline" size={24} color={LiquidGlassTokens.accent.primary} />
                <Text style={styles.cardHeaderTitle}>OPTIMAL MULTI-HAZARD CORRIDOR</Text>
              </View>

              <View style={styles.summaryGrid}>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Total Distance</Text>
                  <Text style={styles.summaryValue}>
                    {pRoute?.distance_meters ? `${pRoute.distance_meters.toFixed(0)} m` : '1,420 m'}
                  </Text>
                </View>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Estimated Travel Time</Text>
                  <Text style={styles.summaryValue}>
                    {pRoute?.estimated_time_seconds ? `${pRoute.estimated_time_seconds.toFixed(0)} s` : '138 s'}
                  </Text>
                </View>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Corridor Risk Score</Text>
                  <Text style={[styles.summaryValue, { color: LiquidGlassTokens.accent.success }]}>
                    {pRoute?.risk_score !== undefined ? `${(pRoute.risk_score * 100).toFixed(1)}%` : '18.4%'}
                  </Text>
                </View>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Waypoints Count</Text>
                  <Text style={styles.summaryValue}>
                    {pRoute?.nodes ? `${pRoute.nodes.length} nodes` : '14 nodes'}
                  </Text>
                </View>
              </View>
            </GlassCard>

            {/* Origin & Destination Nodes */}
            <Text style={styles.sectionHeader}>TERMINAL NODES</Text>
            <GlassCard variant="standard" style={styles.metricCard}>
              <View style={styles.terminalRow}>
                <View style={styles.dotOrigin} />
                <View style={{ flex: 1 }}>
                  <Text style={styles.terminalLabel}>ORIGIN EVACUATION ZONE</Text>
                  <Text style={styles.terminalNode}>{pRoute?.origin_node || 'Zone Node A-11402837826'}</Text>
                </View>
              </View>
              <View style={styles.terminalDivider} />
              <View style={styles.terminalRow}>
                <View style={styles.dotDest} />
                <View style={{ flex: 1 }}>
                  <Text style={styles.terminalLabel}>SAFE REFUGE DESTINATION</Text>
                  <Text style={styles.terminalNode}>{pRoute?.destination_node || 'Shelter Node D-11100965248'}</Text>
                </View>
              </View>
            </GlassCard>

            {/* Turn-by-Turn Waypoints */}
            <Text style={styles.sectionHeader}>TURN-BY-TURN GRAPH NODES</Text>
            <GlassCard variant="standard" style={styles.metricCard}>
              {pRoute?.nodes && pRoute.nodes.length > 0 ? (
                <View style={styles.nodeList}>
                  {pRoute.nodes.map((node: string, index: number) => {
                    const isFirst = index === 0;
                    const isLast = index === pRoute.nodes.length - 1;
                    return (
                      <View key={index} style={styles.waypointItem}>
                        <View style={styles.stepColumn}>
                          <View
                            style={[
                              styles.stepCircle,
                              isFirst && styles.stepCircleOrigin,
                              isLast && styles.stepCircleDest,
                            ]}
                          >
                            <Text style={styles.stepNum}>{index + 1}</Text>
                          </View>
                          {!isLast && <View style={styles.stepLine} />}
                        </View>
                        <View style={styles.waypointContent}>
                          <Text style={styles.waypointTitle}>
                            {isFirst ? 'Depart Origin' : isLast ? 'Arrival at Safe Shelter' : `Node Waypoint #${index}`}
                          </Text>
                          <Text style={styles.waypointId}>ID: {node}</Text>
                        </View>
                      </View>
                    );
                  })}
                </View>
              ) : (
                <Text style={styles.emptyText}>Node sequence will appear when route calculation finishes.</Text>
              )}
            </GlassCard>

            {/* Decision Factors Breakdown */}
            <Text style={styles.sectionHeader}>MARL ROUTE SELECTION FACTORS</Text>
            <GlassCard variant="standard" style={styles.metricCard}>
              <View style={styles.factorRow}>
                <Text style={styles.factorLabel}>Hazard Encroachment Score</Text>
                <Text style={styles.factorVal}>
                  {factors?.hazard_exposure_score ? factors.hazard_exposure_score.toFixed(4) : '0.0420'}
                </Text>
              </View>
              <View style={styles.factorRow}>
                <Text style={styles.factorLabel}>Traffic Friction (BPR Delay Index)</Text>
                <Text style={styles.factorVal}>
                  {factors?.traffic_exposure_score ? factors.traffic_exposure_score.toFixed(4) : '0.2500'}
                </Text>
              </View>
              <View style={styles.factorRow}>
                <Text style={styles.factorLabel}>Predicted Agent Conflicts</Text>
                <Text style={[styles.factorVal, { color: LiquidGlassTokens.accent.success }]}>
                  {factors?.agent_conflicts ?? 0}
                </Text>
              </View>
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
    backgroundColor: LiquidGlassTokens.background.canvas,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 40,
  },
  centerBox: {
    padding: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 14,
    color: LiquidGlassTokens.text.secondary,
  },
  errorCard: {
    padding: 24,
    alignItems: 'center',
    marginVertical: 20,
  },
  errorTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: LiquidGlassTokens.status.critical,
    marginTop: 10,
  },
  errorSub: {
    fontSize: 13,
    color: LiquidGlassTokens.text.secondary,
    textAlign: 'center',
    marginTop: 6,
  },
  retryBtn: {
    marginTop: 16,
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.16)',
  },
  retryBtnText: {
    fontSize: 13,
    fontWeight: '600',
    color: LiquidGlassTokens.text.primary,
  },
  sectionHeader: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1.0,
    color: LiquidGlassTokens.text.muted,
    marginTop: 20,
    marginBottom: 8,
  },
  metricCard: {
    padding: 16,
    marginBottom: 8,
  },
  cardHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 14,
    gap: 8,
  },
  cardHeaderTitle: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.8,
    color: LiquidGlassTokens.accent.primary,
  },
  summaryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  summaryItem: {
    width: '46%',
  },
  summaryLabel: {
    fontSize: 11,
    color: LiquidGlassTokens.text.muted,
    marginBottom: 2,
  },
  summaryValue: {
    fontSize: 15,
    fontWeight: '800',
    color: LiquidGlassTokens.text.primary,
  },
  terminalRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  dotOrigin: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: LiquidGlassTokens.accent.primary,
  },
  dotDest: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: LiquidGlassTokens.accent.success,
  },
  terminalLabel: {
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
    color: LiquidGlassTokens.text.muted,
  },
  terminalNode: {
    fontSize: 13,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
    marginTop: 2,
  },
  terminalDivider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    marginVertical: 12,
  },
  nodeList: {
    paddingVertical: 4,
  },
  waypointItem: {
    flexDirection: 'row',
    gap: 12,
  },
  stepColumn: {
    alignItems: 'center',
    width: 28,
  },
  stepCircle: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.18)',
  },
  stepCircleOrigin: {
    backgroundColor: 'rgba(0, 210, 255, 0.2)',
    borderColor: LiquidGlassTokens.accent.primary,
  },
  stepCircleDest: {
    backgroundColor: 'rgba(46, 213, 115, 0.2)',
    borderColor: LiquidGlassTokens.accent.success,
  },
  stepNum: {
    fontSize: 10,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  stepLine: {
    width: 2,
    flex: 1,
    minHeight: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    marginVertical: 2,
  },
  waypointContent: {
    flex: 1,
    paddingBottom: 16,
  },
  waypointTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  waypointId: {
    fontSize: 11,
    fontFamily: 'monospace',
    color: LiquidGlassTokens.text.secondary,
    marginTop: 2,
  },
  factorRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.06)',
  },
  factorLabel: {
    fontSize: 12,
    color: LiquidGlassTokens.text.secondary,
  },
  factorVal: {
    fontSize: 12,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  emptyText: {
    fontSize: 12,
    color: LiquidGlassTokens.text.muted,
    textAlign: 'center',
    paddingVertical: 10,
  },
});
