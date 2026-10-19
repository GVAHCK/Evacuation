import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LiquidGlassTokens } from '../theme/tokens';
import { GlassCard } from '../components/glass/GlassCard';
import { GlassBadge } from '../components/glass/GlassBadge';
import { GlassHeader } from '../components/glass/GlassHeader';
import { useEvacuationStore } from '../store/evacuationStore';
import { useTheme } from '../theme/ThemeContext';

export const LiveAgentTrackingScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();
  const { liveAgents, liveStatus, activeSession } = useEvacuationStore();

  // If liveAgents is empty (e.g. before session start), fallback to 4 standard cohort agents
  const displayAgents = liveAgents && liveAgents.length > 0 ? liveAgents : [
    {
      id: 0,
      agent_type: 'Type A (Standard Civilian)',
      latitude: 13.0107,
      longitude: 80.2249,
      current_node: '11402837826',
      destination_node: '14144290612',
      status: 'evacuating',
      progress: 0.533,
      distance_remaining_m: 1130.6,
      route: ['30268464', '268790432', '11402837826', '14144290612'],
    },
    {
      id: 1,
      agent_type: 'Type B (Congestion-Sensitive)',
      latitude: 13.0108,
      longitude: 80.2248,
      current_node: '5460773146',
      destination_node: '14144290612',
      status: 'evacuating',
      progress: 0.571,
      distance_remaining_m: 1117.2,
      route: ['268790432', '5460773146', '14144290612'],
    },
    {
      id: 2,
      agent_type: 'Type C (Hazard-Sensitive)',
      latitude: 13.0107,
      longitude: 80.2249,
      current_node: '11402837826',
      destination_node: '14144290612',
      status: 'evacuating',
      progress: 0.533,
      distance_remaining_m: 1130.6,
      route: ['30268464', '11402837826', '14144290612'],
    },
    {
      id: 3,
      agent_type: 'Type D (Perimeter Corridor)',
      latitude: 13.0107,
      longitude: 80.2249,
      current_node: '11402837826',
      destination_node: '14144290612',
      status: 'evacuating',
      progress: 0.533,
      distance_remaining_m: 1130.6,
      route: ['30268464', '11402837826', '14144290612'],
    },
  ];

  const handleSelectAgent = (agent: any) => {
    haptic.selection();
    navigation.navigate('AgentDetail', { agent });
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <GlassHeader
        title="Live Agent Tracking"
        showBack
        onBackPress={() => navigation.goBack()}
        badge={
          <GlassBadge
            status={liveStatus === 'completed' ? 'completed' : 'active'}
            label={`${displayAgents.length} Agents Tracked`}
            size="sm"
          />
        }
      />

      <FlatList
        data={displayAgents}
        keyExtractor={(item) => item.id.toString()}
        contentContainerStyle={styles.listContent}
        renderItem={({ item }) => {
          const isEvacuated = item.status === 'evacuated' || item.progress >= 1.0;
          return (
            <TouchableOpacity
              activeOpacity={0.8}
              onPress={() => handleSelectAgent(item)}
              accessibilityRole="button"
              accessibilityLabel={`Agent ${item.id}: ${item.agent_type}`}
            >
              <GlassCard
                style={styles.card}
                variant={isEvacuated ? 'safety' : 'standard'}
              >
                <View style={styles.cardHeader}>
                  <View style={styles.agentTitleRow}>
                    <View style={[styles.agentAvatar, { backgroundColor: isEvacuated ? 'rgba(16, 185, 129, 0.20)' : 'rgba(0, 210, 255, 0.20)' }]}>
                      <Ionicons
                        name={isEvacuated ? 'checkmark-circle' : 'navigate'}
                        size={20}
                        color={isEvacuated ? LiquidGlassTokens.accent.success : LiquidGlassTokens.accent.primary}
                      />
                    </View>
                    <View style={styles.agentNameCol}>
                      <Text style={styles.agentName}>Agent #{item.id}</Text>
                      <Text style={styles.agentType}>{item.agent_type || 'Standard Evacuee'}</Text>
                    </View>
                  </View>
                  <GlassBadge
                    status={isEvacuated ? 'completed' : 'active'}
                    label={isEvacuated ? 'SAFE HAVEN' : `${(item.progress * 100).toFixed(0)}%`}
                    size="sm"
                  />
                </View>

                {/* Progress Bar */}
                <View style={styles.progressTrack}>
                  <View
                    style={[
                      styles.progressBar,
                      {
                        width: `${Math.min(100, Math.max(0, item.progress * 100))}%`,
                        backgroundColor: isEvacuated
                          ? LiquidGlassTokens.accent.success
                          : LiquidGlassTokens.accent.primary,
                      },
                    ]}
                  />
                </View>

                {/* Telemetry Metrics */}
                <View style={styles.telemetryGrid}>
                  <View style={styles.telItem}>
                    <Text style={styles.telLabel}>Current Node</Text>
                    <Text style={styles.telVal}>{item.current_node || '30268464'}</Text>
                  </View>
                  <View style={styles.telItem}>
                    <Text style={styles.telLabel}>Destination</Text>
                    <Text style={styles.telVal}>{item.destination_node || '14144290612'}</Text>
                  </View>
                  <View style={styles.telItem}>
                    <Text style={styles.telLabel}>Remaining</Text>
                    <Text style={styles.telVal}>
                      {item.distance_remaining_m ? `${item.distance_remaining_m.toFixed(0)}m` : '0m'}
                    </Text>
                  </View>
                </View>
              </GlassCard>
            </TouchableOpacity>
          );
        }}
      />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: LiquidGlassTokens.background.primary,
  },
  listContent: {
    padding: 16,
    paddingBottom: 40,
  },
  card: {
    padding: 16,
    marginBottom: 12,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  agentTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  agentAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  agentNameCol: {
    flex: 1,
  },
  agentName: {
    fontSize: 15,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  agentType: {
    fontSize: 12,
    color: LiquidGlassTokens.text.muted,
    marginTop: 1,
  },
  progressTrack: {
    height: 6,
    borderRadius: 3,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    overflow: 'hidden',
    marginBottom: 12,
  },
  progressBar: {
    height: '100%',
    borderRadius: 3,
  },
  telemetryGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(0, 0, 0, 0.30)',
    borderRadius: LiquidGlassTokens.radius.md,
    padding: 10,
  },
  telItem: {
    alignItems: 'center',
    flex: 1,
  },
  telLabel: {
    fontSize: 11,
    color: LiquidGlassTokens.text.muted,
  },
  telVal: {
    fontSize: 12,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
    marginTop: 2,
  },
});
