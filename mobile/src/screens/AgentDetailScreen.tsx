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

export const AgentDetailScreen: React.FC = () => {
  const route = useRoute<any>();
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();

  const sessionId = route.params?.sessionId || 'active_session';
  const agentId = route.params?.agentId ?? 0;

  const [loading, setLoading] = useState(true);
  const [agentData, setAgentData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchAgentDetail = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await Api.getAgentDetail(sessionId, agentId);
      setAgentData(data);
    } catch (err: any) {
      // If error, check if route params provided initial agent
      if (route.params?.agent) {
        setAgentData(route.params.agent);
      } else {
        setError(err.message || 'Unable to retrieve agent telemetry from backend');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAgentDetail();
  }, [sessionId, agentId]);

  const isEvacuated = agentData?.status === 'evacuated';

  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
      <GlassHeader
        title={`Agent #${agentId + 1} Telemetry`}
        subtitle={`Session ID: ${sessionId.slice(0, 8)}...`}
        showBack
        onBackPress={() => {
          haptic('light');
          navigation.goBack();
        }}
        badge={
          <GlassBadge
            label={isEvacuated ? 'EVACUATED' : 'IN TRANSIT'}
            variant={isEvacuated ? 'success' : 'warning'}
            icon={isEvacuated ? 'checkmark-circle' : 'navigate'}
          />
        }
      />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {loading ? (
          <View style={styles.centerBox}>
            <ActivityIndicator size="large" color={LiquidGlassTokens.accent.primary} />
            <Text style={styles.loadingText}>Fetching Agent Telemetry...</Text>
          </View>
        ) : error && !agentData ? (
          <GlassCard variant="standard" style={styles.errorCard}>
            <Ionicons name="alert-circle-outline" size={32} color={LiquidGlassTokens.status.critical} />
            <Text style={styles.errorTitle}>Telemetry Unavailable</Text>
            <Text style={styles.errorSub}>{error}</Text>
            <TouchableOpacity style={styles.retryBtn} onPress={fetchAgentDetail}>
              <Text style={styles.retryBtnText}>Retry Connection</Text>
            </TouchableOpacity>
          </GlassCard>
        ) : (
          <>
            {/* Cohort Spec Card */}
            <GlassCard variant="highlight" style={styles.metricCard}>
              <View style={styles.cardHeaderRow}>
                <Ionicons name="person-circle-outline" size={24} color={LiquidGlassTokens.accent.primary} />
                <Text style={styles.cardHeaderTitle}>MARL COHORT PROFILE</Text>
              </View>
              <View style={styles.specGrid}>
                <View style={styles.specItem}>
                  <Text style={styles.specLabel}>Agent ID</Text>
                  <Text style={styles.specValue}>#{agentId}</Text>
                </View>
                <View style={styles.specItem}>
                  <Text style={styles.specLabel}>Cohort Type</Text>
                  <Text style={styles.specValue}>Standard Evacuee</Text>
                </View>
                <View style={styles.specItem}>
                  <Text style={styles.specLabel}>Policy Engine</Text>
                  <Text style={styles.specValue}>Offline-to-Online MARL</Text>
                </View>
                <View style={styles.specItem}>
                  <Text style={styles.specLabel}>Action Space</Text>
                  <Text style={styles.specValue}>4 Directed Successors</Text>
                </View>
              </View>
            </GlassCard>

            {/* Telemetry Metrics */}
            <Text style={styles.sectionHeader}>REAL-TIME PROGRESS</Text>
            <View style={styles.rowTwo}>
              <GlassCard variant="standard" style={styles.halfCard}>
                <Ionicons name="speedometer-outline" size={20} color={LiquidGlassTokens.accent.primary} />
                <Text style={styles.miniLabel}>Distance Covered</Text>
                <Text style={styles.metricValue}>
                  {agentData?.distance_travelled ? `${agentData.distance_travelled.toFixed(0)} m` : '0 m'}
                </Text>
              </GlassCard>

              <GlassCard variant="standard" style={styles.halfCard}>
                <Ionicons name="time-outline" size={20} color={LiquidGlassTokens.accent.primary} />
                <Text style={styles.miniLabel}>Time Elapsed</Text>
                <Text style={styles.metricValue}>
                  {agentData?.time_elapsed ? `${agentData.time_elapsed.toFixed(1)} s` : '0.0 s'}
                </Text>
              </GlassCard>
            </View>

            <View style={styles.rowTwo}>
              <GlassCard variant="standard" style={styles.halfCard}>
                <Ionicons name="shield-checkmark-outline" size={20} color={LiquidGlassTokens.accent.success} />
                <Text style={styles.miniLabel}>Hazard Exposure</Text>
                <Text style={[styles.metricValue, { color: LiquidGlassTokens.accent.success }]}>
                  {agentData?.hazard_exposure ? agentData.hazard_exposure.toFixed(3) : '0.000'}
                </Text>
              </GlassCard>

              <GlassCard variant="standard" style={styles.halfCard}>
                <Ionicons name="locate-outline" size={20} color={LiquidGlassTokens.accent.primary} />
                <Text style={styles.miniLabel}>Current Node</Text>
                <Text style={styles.metricValue} numberOfLines={1}>
                  {agentData?.current_node || 'Initial Node'}
                </Text>
              </GlassCard>
            </View>

            {/* Decision Factors */}
            <Text style={styles.sectionHeader}>MULTI-CRITERIA DECISION WEIGHTS</Text>
            <GlassCard variant="standard" style={styles.metricCard}>
              <View style={styles.factorRow}>
                <Text style={styles.factorLabel}>Travel Time Weight (Delay Penalty)</Text>
                <Text style={styles.factorVal}>0.45</Text>
              </View>
              <View style={styles.factorRow}>
                <Text style={styles.factorLabel}>Hazard Encroachment Avoidance</Text>
                <Text style={styles.factorVal}>0.35</Text>
              </View>
              <View style={styles.factorRow}>
                <Text style={styles.factorLabel}>Bottleneck Inter-Agent Congestion</Text>
                <Text style={styles.factorVal}>0.20</Text>
              </View>
              <View style={styles.factorRow}>
                <Text style={styles.factorLabel}>Hard Action Masking Violations</Text>
                <Text style={[styles.factorVal, { color: LiquidGlassTokens.accent.success }]}>0 (Constrained)</Text>
              </View>
            </GlassCard>

            {/* Route Waypoints Sequence */}
            <Text style={styles.sectionHeader}>WAYPOINT NODE SEQUENCE</Text>
            <GlassCard variant="standard" style={styles.metricCard}>
              {agentData?.nodes && agentData.nodes.length > 0 ? (
                <View style={styles.nodeList}>
                  {agentData.nodes.slice(0, 8).map((node: string, idx: number) => (
                    <View key={idx} style={styles.nodeRow}>
                      <View style={styles.nodeIndexBadge}>
                        <Text style={styles.nodeIndexText}>{idx + 1}</Text>
                      </View>
                      <Text style={styles.nodeText}>Node ID: {node}</Text>
                    </View>
                  ))}
                  {agentData.nodes.length > 8 && (
                    <Text style={styles.moreNodesText}>
                      + {agentData.nodes.length - 8} additional intermediate waypoints
                    </Text>
                  )}
                </View>
              ) : (
                <Text style={styles.emptyText}>Waypoints stream actively in live simulation mode.</Text>
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
  specGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  specItem: {
    width: '46%',
  },
  specLabel: {
    fontSize: 11,
    color: LiquidGlassTokens.text.muted,
    marginBottom: 2,
  },
  specValue: {
    fontSize: 13,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  rowTwo: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 10,
  },
  halfCard: {
    flex: 1,
    padding: 14,
  },
  miniLabel: {
    fontSize: 11,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 8,
  },
  metricValue: {
    fontSize: 16,
    fontWeight: '800',
    color: LiquidGlassTokens.text.primary,
    marginTop: 4,
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
  nodeList: {
    gap: 8,
  },
  nodeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  nodeIndexBadge: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: 'rgba(0, 210, 255, 0.15)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  nodeIndexText: {
    fontSize: 10,
    fontWeight: '800',
    color: LiquidGlassTokens.accent.primary,
  },
  nodeText: {
    fontSize: 12,
    fontFamily: 'monospace',
    color: LiquidGlassTokens.text.primary,
  },
  moreNodesText: {
    fontSize: 11,
    fontStyle: 'italic',
    color: LiquidGlassTokens.text.muted,
    marginTop: 4,
  },
  emptyText: {
    fontSize: 12,
    color: LiquidGlassTokens.text.muted,
    textAlign: 'center',
    paddingVertical: 10,
  },
});
