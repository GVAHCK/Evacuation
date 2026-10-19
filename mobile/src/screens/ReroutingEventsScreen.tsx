import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
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

export const ReroutingEventsScreen: React.FC = () => {
  const route = useRoute<any>();
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();

  const sessionId = route.params?.sessionId || 'active_session';
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [events, setEvents] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchEvents = async () => {
    setError(null);
    try {
      const data = await Api.getSessionEvents(sessionId);
      setEvents(data || []);
    } catch (err: any) {
      setError(err.message || 'Unable to retrieve rerouting event audit log.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchEvents();
  }, [sessionId]);

  const onRefresh = () => {
    haptic('light');
    setRefreshing(true);
    fetchEvents();
  };

  const rerouteEvents = events.filter((e) => e.type === 'reroute' || e.type === 'reroute_requested');

  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
      <GlassHeader
        title="Rerouting & Event Audit"
        subtitle={`Session ID: ${sessionId.slice(0, 8)}...`}
        showBack
        onBackPress={() => {
          haptic('light');
          navigation.goBack();
        }}
        badge={
          <GlassBadge
            label={`${rerouteEvents.length} REROUTES`}
            variant={rerouteEvents.length > 0 ? 'warning' : 'neutral'}
            icon="git-branch-outline"
          />
        }
      />

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={LiquidGlassTokens.accent.primary}
          />
        }
      >
        {/* Summary Card */}
        <GlassCard variant="highlight" style={styles.summaryCard}>
          <View style={styles.cardHeaderRow}>
            <Ionicons name="shield-checkmark" size={22} color={LiquidGlassTokens.accent.primary} />
            <Text style={styles.summaryTitle}>AUTONOMOUS REROUTING LOG</Text>
          </View>
          <Text style={styles.summaryDesc}>
            Offline-to-Online MARL continually evaluates upstream edges against advancing flood & fire hazard fronts.
            When an edge's risk exceeds 0.65 or becomes impassable, an immediate closed-loop recalculation triggers.
          </Text>
        </GlassCard>

        <Text style={styles.sectionHeader}>EVENT AUDIT TIMELINE</Text>

        {loading ? (
          <View style={styles.centerBox}>
            <ActivityIndicator size="large" color={LiquidGlassTokens.accent.primary} />
            <Text style={styles.loadingText}>Fetching Rerouting Events...</Text>
          </View>
        ) : error && events.length === 0 ? (
          <GlassCard variant="standard" style={styles.errorCard}>
            <Ionicons name="alert-circle-outline" size={32} color={LiquidGlassTokens.status.critical} />
            <Text style={styles.errorTitle}>Audit Log Error</Text>
            <Text style={styles.errorSub}>{error}</Text>
            <TouchableOpacity style={styles.retryBtn} onPress={fetchEvents}>
              <Text style={styles.retryBtnText}>Retry Connection</Text>
            </TouchableOpacity>
          </GlassCard>
        ) : events.length === 0 ? (
          <GlassCard variant="standard" style={styles.emptyCard}>
            <Ionicons name="checkmark-done-circle-outline" size={40} color={LiquidGlassTokens.accent.success} />
            <Text style={styles.emptyTitle}>Optimal Path Maintained</Text>
            <Text style={styles.emptySub}>
              No dynamic route recalculations were required for this session. The initial Offline-to-Online MARL corridor remained clear of advancing hazard fronts.
            </Text>
          </GlassCard>
        ) : (
          <View style={styles.timelineList}>
            {events.map((evt, idx) => {
              const isReroute = evt.type === 'reroute' || evt.type === 'reroute_requested';
              const isArrival = evt.type === 'agent_arrival' || evt.type === 'session_completed';
              const isPause = evt.type === 'pause' || evt.type === 'resume';

              let iconName: any = 'information-circle-outline';
              let iconColor = LiquidGlassTokens.accent.primary;
              if (isReroute) {
                iconName = 'git-branch-outline';
                iconColor = LiquidGlassTokens.status.warning;
              } else if (isArrival) {
                iconName = 'checkmark-circle-outline';
                iconColor = LiquidGlassTokens.status.completed;
              } else if (isPause) {
                iconName = 'pause-circle-outline';
                iconColor = LiquidGlassTokens.text.secondary;
              }

              return (
                <GlassCard key={evt.id || idx} variant={isReroute ? 'standard' : 'subtle'} style={styles.eventCard}>
                  <View style={styles.eventHeader}>
                    <View style={styles.eventIconWrapper}>
                      <Ionicons name={iconName} size={20} color={iconColor} />
                    </View>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.eventType}>{evt.type.toUpperCase()}</Text>
                      <Text style={styles.eventTime}>
                        {evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : 'Recent step'}
                      </Text>
                    </View>
                    {isReroute && (
                      <GlassBadge label="TRIGGERED" variant="warning" />
                    )}
                  </View>

                  <Text style={styles.eventMessage}>{evt.message}</Text>

                  {evt.metadata && Object.keys(evt.metadata).length > 0 && (
                    <View style={styles.metaBox}>
                      {Object.entries(evt.metadata).map(([k, v]: [string, any]) => (
                        <View key={k} style={styles.metaRow}>
                          <Text style={styles.metaKey}>{k}:</Text>
                          <Text style={styles.metaVal}>{typeof v === 'number' ? v.toFixed(2) : String(v)}</Text>
                        </View>
                      ))}
                    </View>
                  )}
                </GlassCard>
              );
            })}
          </View>
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
  summaryCard: {
    padding: 16,
    marginBottom: 8,
  },
  cardHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 10,
  },
  summaryTitle: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.8,
    color: LiquidGlassTokens.accent.primary,
  },
  summaryDesc: {
    fontSize: 13,
    lineHeight: 19,
    color: LiquidGlassTokens.text.secondary,
  },
  sectionHeader: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1.0,
    color: LiquidGlassTokens.text.muted,
    marginTop: 20,
    marginBottom: 12,
  },
  centerBox: {
    padding: 40,
    alignItems: 'center',
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
  emptyCard: {
    padding: 28,
    alignItems: 'center',
    marginVertical: 20,
  },
  emptyTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
    marginTop: 12,
  },
  emptySub: {
    fontSize: 13,
    lineHeight: 19,
    color: LiquidGlassTokens.text.secondary,
    textAlign: 'center',
    marginTop: 8,
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
  timelineList: {
    gap: 12,
  },
  eventCard: {
    padding: 14,
  },
  eventHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 8,
  },
  eventIconWrapper: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  eventType: {
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.6,
    color: LiquidGlassTokens.text.primary,
  },
  eventTime: {
    fontSize: 11,
    color: LiquidGlassTokens.text.muted,
    marginTop: 2,
  },
  eventMessage: {
    fontSize: 13,
    lineHeight: 18,
    color: LiquidGlassTokens.text.secondary,
  },
  metaBox: {
    marginTop: 10,
    padding: 8,
    borderRadius: 8,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    gap: 4,
  },
  metaRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  metaKey: {
    fontSize: 11,
    color: LiquidGlassTokens.text.muted,
  },
  metaVal: {
    fontSize: 11,
    fontFamily: 'monospace',
    fontWeight: '600',
    color: LiquidGlassTokens.text.primary,
  },
});
