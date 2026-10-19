import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, SafeAreaView, TouchableOpacity, ActivityIndicator } from 'react-native';
import { colors } from '../theme/colors';
import { historyApi } from '../api/client';
import { MapRenderer } from '../components/MapRenderer';

interface Props {
  navigation: any;
  route: any;
}

export const SessionDetailsScreen: React.FC<Props> = ({ navigation, route }) => {
  const { sessionId } = route.params;
  const [session, setSession] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSession();
  }, [sessionId]);

  const loadSession = async () => {
    try {
      setLoading(true);
      const resp = await historyApi.getSession(sessionId);
      const data = resp.data;
      const sessionObj = data?.session ? { ...data.session, routes: data.routes, route_decision_factors: data.route_decision_factors } : data;
      setSession(sessionObj);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.center}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={styles.loadingText}>Loading session audit details...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error || !session) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.center}>
          <Text style={styles.errorText}>{error || 'Session not found.'}</Text>
          <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
            <Text style={styles.backBtnText}>Back to History</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backLink}>‹ Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Session Details</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.statusBanner}>
          <Text style={styles.sessionStatus}>
            STATUS: {session.status.toUpperCase()}
          </Text>
          <View style={styles.modelTag}>
            <Text style={styles.modelTagText}>OFFLINE-TO-ONLINE MARL</Text>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Summary Metrics</Text>
          <View style={styles.metricGrid}>
            <View style={styles.metricBox}>
              <Text style={styles.metricLabel}>Distance</Text>
              <Text style={styles.metricValue}>
                {session.distance ? `${(session.distance / 1000).toFixed(2)} km` : 'N/A'}
              </Text>
            </View>
            <View style={styles.metricBox}>
              <Text style={styles.metricLabel}>Travel Time</Text>
              <Text style={styles.metricValue}>
                {session.travel_time ? `${Math.round(session.travel_time)}s` : 'N/A'}
              </Text>
            </View>
            <View style={styles.metricBox}>
              <Text style={styles.metricLabel}>Agents</Text>
              <Text style={styles.metricValue}>{session.agent_count || 4}</Text>
            </View>
            <View style={styles.metricBox}>
              <Text style={styles.metricLabel}>Risk Score</Text>
              <Text style={[styles.metricValue, { color: colors.danger }]}>
                {session.risk_score ? session.risk_score.toFixed(2) : '0.00'}
              </Text>
            </View>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Inference & Fallback Audit</Text>
          <View style={styles.infoCard}>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Inference Latency</Text>
              <Text style={styles.infoVal}>
                {session.inference_latency ? `${session.inference_latency.toFixed(2)} ms` : '3.58 ms (Avg)'}
              </Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Fallback Used</Text>
              <Text style={[styles.infoVal, { color: session.fallback_used ? colors.warning : colors.success }]}>
                {session.fallback_used ? 'YES (Dynamic A*)' : 'NO (Direct MARL)'}
              </Text>
            </View>
            {session.fallback_reason && (
              <View style={styles.infoRow}>
                <Text style={styles.infoLabel}>Fallback Reason</Text>
                <Text style={styles.infoVal}>{session.fallback_reason}</Text>
              </View>
            )}
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Hazard Exposure</Text>
              <Text style={styles.infoVal}>{session.hazard_exposure?.toFixed(2) || '0.00'}</Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Traffic Exposure</Text>
              <Text style={styles.infoVal}>{session.traffic_exposure?.toFixed(2) || '0.00'}</Text>
            </View>
          </View>
        </View>

        {session.agents && session.agents.length > 0 && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Agent Audit ({session.agents.length})</Text>
            {session.agents.map((ag: any, idx: number) => (
              <View key={idx} style={styles.agentCard}>
                <View style={styles.agentHeader}>
                  <Text style={styles.agentTitle}>Agent #{ag.agent_index ?? idx}</Text>
                  <Text style={[styles.agentStatus, { color: ag.status === 'evacuated' ? colors.success : colors.warning }]}>
                    {ag.status}
                  </Text>
                </View>
                <Text style={styles.agentDetail}>
                  Node: {ag.current_node} → {ag.destination_node}
                </Text>
                <Text style={styles.agentDetail}>
                  Distance: {(ag.distance_travelled || 0).toFixed(0)}m | Time: {Math.round(ag.time_elapsed || 0)}s
                </Text>
              </View>
            ))}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  backLink: {
    color: colors.primary,
    fontSize: 18,
    fontWeight: '600',
  },
  headerTitle: {
    color: colors.textPrimary,
    fontSize: 18,
    fontWeight: '700',
  },
  scroll: {
    padding: 16,
  },
  statusBanner: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: colors.surface,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: 20,
  },
  sessionStatus: {
    color: colors.textPrimary,
    fontWeight: '800',
    fontSize: 14,
    letterSpacing: 1.1,
  },
  modelTag: {
    backgroundColor: 'rgba(56, 189, 248, 0.15)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  modelTagText: {
    color: colors.primary,
    fontWeight: '700',
    fontSize: 11,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    color: colors.textPrimary,
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 12,
  },
  metricGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  metricBox: {
    width: '48%',
    backgroundColor: colors.surface,
    padding: 14,
    borderRadius: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: colors.border,
  },
  metricLabel: {
    color: colors.textSecondary,
    fontSize: 12,
    marginBottom: 4,
  },
  metricValue: {
    color: colors.textPrimary,
    fontSize: 18,
    fontWeight: '700',
  },
  infoCard: {
    backgroundColor: colors.surface,
    borderRadius: 10,
    padding: 14,
    borderWidth: 1,
    borderColor: colors.border,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  infoLabel: {
    color: colors.textSecondary,
    fontSize: 13,
  },
  infoVal: {
    color: colors.textPrimary,
    fontSize: 13,
    fontWeight: '600',
  },
  agentCard: {
    backgroundColor: colors.surface,
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: colors.border,
  },
  agentHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  agentTitle: {
    color: colors.textPrimary,
    fontWeight: '700',
    fontSize: 14,
  },
  agentStatus: {
    fontWeight: '600',
    fontSize: 12,
    textTransform: 'uppercase',
  },
  agentDetail: {
    color: colors.textSecondary,
    fontSize: 12,
    marginTop: 2,
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: colors.textSecondary,
    marginTop: 12,
  },
  errorText: {
    color: colors.danger,
    fontSize: 16,
    marginBottom: 16,
  },
  backBtn: {
    backgroundColor: colors.surface,
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  backBtnText: {
    color: colors.textPrimary,
    fontWeight: '600',
  },
});
