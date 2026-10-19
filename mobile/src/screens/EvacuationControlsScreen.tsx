import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useRoute, useNavigation } from '@react-navigation/native';
import { LiquidGlassTokens } from '../theme/tokens';
import { GlassCard } from '../components/glass/GlassCard';
import { GlassButton } from '../components/glass/GlassButton';
import { GlassBadge } from '../components/glass/GlassBadge';
import { GlassHeader } from '../components/glass/GlassHeader';
import { useTheme } from '../theme/ThemeContext';
import { Api } from '../api/client';

export const EvacuationControlsScreen: React.FC = () => {
  const route = useRoute<any>();
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();

  const sessionId = route.params?.sessionId || 'active_session';

  const [isPaused, setIsPaused] = useState(false);
  const [speedFactor, setSpeedFactor] = useState(1.0);
  const [processing, setProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const handleTogglePause = async () => {
    haptic('medium');
    setProcessing(true);
    setStatusMessage(null);
    try {
      if (isPaused) {
        await Api.resumeEvacuation(sessionId);
        setIsPaused(false);
        setStatusMessage('Simulation resumed successfully.');
      } else {
        await Api.pauseEvacuation(sessionId);
        setIsPaused(true);
        setStatusMessage('Simulation paused. Agent state held.');
      }
    } catch (err: any) {
      Alert.alert('Control Error', err.message || 'Unable to update pause state.');
    } finally {
      setProcessing(false);
    }
  };

  const handleSpeedChange = async (speed: number) => {
    haptic('selection');
    setSpeedFactor(speed);
    try {
      await Api.setEvacuationSpeed(sessionId, speed);
      setStatusMessage(`Simulation velocity adjusted to ${speed}x.`);
    } catch (err: any) {
      Alert.alert('Speed Error', err.message || 'Unable to adjust speed factor.');
    }
  };

  const handleTriggerReroute = async () => {
    haptic('heavy');
    setProcessing(true);
    setStatusMessage(null);
    try {
      await Api.rerouteEvacuation(sessionId);
      setStatusMessage('Autonomous MARL recalculation dispatched.');
    } catch (err: any) {
      Alert.alert('Reroute Error', err.message || 'Unable to trigger reroute.');
    } finally {
      setProcessing(false);
    }
  };

  const handleStopEvacuation = () => {
    haptic('heavy');
    Alert.alert(
      'Terminate Evacuation Session',
      'Are you sure you want to stop this live evacuation session? All in-transit agents will halt at their current nodes and final outcomes will be saved to history.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Confirm Stop',
          style: 'destructive',
          onPress: async () => {
            setProcessing(true);
            try {
              await Api.stopEvacuation(sessionId);
              haptic('heavy');
              navigation.replace('EvacuationResult', { sessionId });
            } catch (err: any) {
              Alert.alert('Stop Error', err.message || 'Unable to terminate session.');
              setProcessing(false);
            }
          },
        },
      ]
    );
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
      <GlassHeader
        title="Mission Control Cockpit"
        subtitle={`Session ID: ${sessionId.slice(0, 8)}...`}
        showBack
        onBackPress={() => {
          haptic('light');
          navigation.goBack();
        }}
        badge={
          <GlassBadge
            label={isPaused ? 'PAUSED' : 'SIMULATING'}
            variant={isPaused ? 'warning' : 'success'}
            icon={isPaused ? 'pause-circle' : 'play-circle'}
          />
        }
      />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Active Cockpit Status */}
        <GlassCard variant="highlight" style={styles.statusCard}>
          <View style={styles.statusHeader}>
            <Ionicons
              name={isPaused ? 'pause-circle' : 'pulse'}
              size={24}
              color={isPaused ? LiquidGlassTokens.status.warning : LiquidGlassTokens.accent.primary}
            />
            <Text style={styles.statusTitle}>OPERATIONAL COMMAND OVERVIEW</Text>
          </View>
          <Text style={styles.statusDesc}>
            Direct intervention controls for active multi-hazard evacuation simulation. Commands execute with sub-second latency across WebSocket subscribers.
          </Text>

          {statusMessage && (
            <View style={styles.feedbackBanner}>
              <Ionicons name="checkmark-circle" size={16} color={LiquidGlassTokens.accent.success} />
              <Text style={styles.feedbackText}>{statusMessage}</Text>
            </View>
          )}
        </GlassCard>

        {/* Primary Lifecycle Controls */}
        <Text style={styles.sectionHeader}>SIMULATION LIFECYCLE</Text>
        <GlassCard variant="standard" style={styles.controlCard}>
          <View style={styles.actionRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.actionTitle}>{isPaused ? 'Resume Simulation' : 'Pause Simulation'}</Text>
              <Text style={styles.actionDesc}>
                {isPaused
                  ? 'Reactivate step loop and resume agent movement.'
                  : 'Temporarily freeze agent positions and hazard propagation.'}
              </Text>
            </View>
            <GlassButton
              title={isPaused ? 'RESUME' : 'PAUSE'}
              variant={isPaused ? 'primary' : 'secondary'}
              size="small"
              onPress={handleTogglePause}
              disabled={processing}
              loading={processing}
              style={{ minWidth: 100 }}
            />
          </View>
        </GlassCard>

        {/* Speed Factor Selector */}
        <Text style={styles.sectionHeader}>SIMULATION TEMPO & SPEED FACTOR</Text>
        <GlassCard variant="standard" style={styles.controlCard}>
          <Text style={styles.speedDesc}>
            Scale wall-clock time per simulation step. Higher factors accelerate TraCI and BPR digital twin advancement.
          </Text>
          <View style={styles.speedButtonRow}>
            {[0.5, 1.0, 2.0, 5.0].map((spd) => {
              const isSelected = speedFactor === spd;
              return (
                <TouchableOpacity
                  key={spd}
                  style={[styles.speedBtn, isSelected && styles.speedBtnSelected]}
                  onPress={() => handleSpeedChange(spd)}
                  accessibilityRole="button"
                  accessibilityLabel={`Set simulation speed to ${spd}x`}
                >
                  <Text style={[styles.speedBtnText, isSelected && styles.speedBtnTextSelected]}>
                    {spd}x
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </GlassCard>

        {/* Dynamic Reroute Trigger */}
        <Text style={styles.sectionHeader}>TACTICAL REROUTING OVERRIDE</Text>
        <GlassCard variant="standard" style={styles.controlCard}>
          <View style={styles.actionRow}>
            <View style={{ flex: 1, marginRight: 12 }}>
              <Text style={styles.actionTitle}>Force MARL Reroute</Text>
              <Text style={styles.actionDesc}>
                Send an immediate interrupt signal to recalculate joint paths from current agent coordinates.
              </Text>
            </View>
            <GlassButton
              title="REROUTE"
              variant="outline"
              size="small"
              onPress={handleTriggerReroute}
              disabled={processing}
              style={{ minWidth: 100 }}
            />
          </View>
        </GlassCard>

        {/* Destructive Abort Button */}
        <Text style={[styles.sectionHeader, { color: LiquidGlassTokens.status.critical }]}>
          DESTRUCTIVE MISSION ABORT
        </Text>
        <GlassCard variant="standard" style={[styles.controlCard, { borderColor: 'rgba(255, 71, 87, 0.3)' }]}>
          <Text style={styles.dangerDesc}>
            Concludes simulation immediately. Persists final trajectories and generates complete post-incident evacuation report.
          </Text>
          <GlassButton
            title="TERMINATE EVACUATION"
            variant="destructive"
            size="large"
            onPress={handleStopEvacuation}
            disabled={processing}
            style={{ marginTop: 14 }}
          />
        </GlassCard>
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
  statusCard: {
    padding: 16,
    marginBottom: 8,
  },
  statusHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  statusTitle: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.8,
    color: LiquidGlassTokens.accent.primary,
  },
  statusDesc: {
    fontSize: 13,
    lineHeight: 19,
    color: LiquidGlassTokens.text.secondary,
  },
  feedbackBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 12,
    padding: 10,
    borderRadius: 8,
    backgroundColor: 'rgba(46, 213, 115, 0.12)',
    borderWidth: 1,
    borderColor: 'rgba(46, 213, 115, 0.25)',
  },
  feedbackText: {
    fontSize: 12,
    fontWeight: '600',
    color: LiquidGlassTokens.accent.success,
  },
  sectionHeader: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1.0,
    color: LiquidGlassTokens.text.muted,
    marginTop: 20,
    marginBottom: 8,
  },
  controlCard: {
    padding: 16,
    marginBottom: 8,
  },
  actionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  actionTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  actionDesc: {
    fontSize: 12,
    lineHeight: 17,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 3,
  },
  speedDesc: {
    fontSize: 12,
    color: LiquidGlassTokens.text.secondary,
    marginBottom: 12,
    lineHeight: 17,
  },
  speedButtonRow: {
    flexDirection: 'row',
    gap: 10,
  },
  speedBtn: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 44,
  },
  speedBtnSelected: {
    backgroundColor: 'rgba(0, 210, 255, 0.15)',
    borderColor: LiquidGlassTokens.accent.primary,
  },
  speedBtnText: {
    fontSize: 13,
    fontWeight: '700',
    color: LiquidGlassTokens.text.secondary,
  },
  speedBtnTextSelected: {
    color: LiquidGlassTokens.accent.primary,
  },
  dangerDesc: {
    fontSize: 12,
    lineHeight: 17,
    color: LiquidGlassTokens.text.secondary,
  },
});
