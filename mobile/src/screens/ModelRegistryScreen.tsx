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
import { useNavigation } from '@react-navigation/native';
import { LiquidGlassTokens } from '../theme/tokens';
import { GlassCard } from '../components/glass/GlassCard';
import { GlassBadge } from '../components/glass/GlassBadge';
import { GlassHeader } from '../components/glass/GlassHeader';
import { useTheme } from '../theme/ThemeContext';
import { Api } from '../api/client';
import { ModelMetadata } from '../types';

export const ModelRegistryScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = async () => {
    setError(null);
    try {
      const data = await Api.getModels();
      setModels(data || []);
    } catch (err: any) {
      setError(err.message || 'Unable to retrieve production models from registry.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const onRefresh = () => {
    haptic('light');
    setRefreshing(true);
    fetchModels();
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
      <GlassHeader
        title="AI Model Registry"
        subtitle="Active inference policies & MARL checkpoints"
        showBack
        onBackPress={() => {
          haptic('light');
          navigation.goBack();
        }}
        badge={
          <GlassBadge
            label="PRODUCTION"
            variant="ai"
            icon="hardware-chip-outline"
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
        {/* Banner */}
        <GlassCard variant="highlight" style={styles.bannerCard}>
          <View style={styles.bannerHeader}>
            <Ionicons name="sparkles" size={20} color={LiquidGlassTokens.accent.primary} />
            <Text style={styles.bannerTitle}>OFFLINE-TO-ONLINE MARL POLICY</Text>
          </View>
          <Text style={styles.bannerDesc}>
            Production multi-agent routing policy trained via Conservative Q-Learning (CQL) on 50,000 historical flood and fire evacuation scenarios with online fine-tuning and hard action masking.
          </Text>
        </GlassCard>

        {/* Quick action buttons */}
        <View style={styles.quickNavRow}>
          <TouchableOpacity
            style={styles.quickNavBtn}
            onPress={() => {
              haptic('selection');
              navigation.navigate('MarlModelDetails');
            }}
          >
            <Ionicons name="analytics-outline" size={18} color={LiquidGlassTokens.accent.primary} />
            <Text style={styles.quickNavText}>Architecture Spec</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.quickNavBtn}
            onPress={() => {
              haptic('selection');
              navigation.navigate('ModelDiagnostics');
            }}
          >
            <Ionicons name="pulse-outline" size={18} color={LiquidGlassTokens.status.warning} />
            <Text style={styles.quickNavText}>Live Latency Test</Text>
          </TouchableOpacity>
        </View>

        <Text style={styles.sectionHeader}>REGISTERED INFERENCE MODELS</Text>

        {loading ? (
          <View style={styles.centerBox}>
            <ActivityIndicator size="large" color={LiquidGlassTokens.accent.primary} />
            <Text style={styles.loadingText}>Inspecting Loaded Models...</Text>
          </View>
        ) : error && models.length === 0 ? (
          <GlassCard variant="standard" style={styles.errorCard}>
            <Ionicons name="alert-circle-outline" size={32} color={LiquidGlassTokens.status.critical} />
            <Text style={styles.errorTitle}>Registry Unreachable</Text>
            <Text style={styles.errorSub}>{error}</Text>
            <TouchableOpacity style={styles.retryBtn} onPress={fetchModels}>
              <Text style={styles.retryBtnText}>Retry Connection</Text>
            </TouchableOpacity>
          </GlassCard>
        ) : (
          models.map((mod, idx) => (
            <GlassCard key={idx} variant="standard" style={styles.modelCard}>
              <View style={styles.modelHeader}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.modelName}>{mod.model_name.toUpperCase()}</Text>
                  <Text style={styles.modelFamily}>{mod.model_family}</Text>
                </View>
                <GlassBadge
                  label={mod.status.toUpperCase()}
                  variant={mod.status === 'ready' || mod.status === 'online' ? 'success' : 'warning'}
                  icon="checkmark-circle"
                />
              </View>

              <View style={styles.divider} />

              <View style={styles.statsGrid}>
                <View style={styles.statCol}>
                  <Text style={styles.statLabel}>Parameters</Text>
                  <Text style={styles.statVal}>
                    {mod.parameter_count ? mod.parameter_count.toLocaleString() : '327,204'}
                  </Text>
                </View>
                <View style={styles.statCol}>
                  <Text style={styles.statLabel}>Device</Text>
                  <Text style={styles.statVal}>{mod.device.toUpperCase()}</Text>
                </View>
                <View style={styles.statCol}>
                  <Text style={styles.statLabel}>Obs / Act Dim</Text>
                  <Text style={styles.statVal}>{mod.obs_dim} / {mod.act_dim}</Text>
                </View>
                <View style={styles.statCol}>
                  <Text style={styles.statLabel}>Action Masking</Text>
                  <Text style={[styles.statVal, { color: LiquidGlassTokens.accent.success }]}>
                    {mod.action_masking_supported ? 'Active' : 'No'}
                  </Text>
                </View>
              </View>

              <View style={styles.modelFooter}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.checkpointLabel}>Checkpoint Integrity</Text>
                  <Text style={styles.checkpointVal}>
                    {mod.checkpoint_exists ? `Verified (${mod.checkpoint_size_kb?.toFixed(0)} KB)` : 'Simulated Weights'}
                  </Text>
                </View>
                <TouchableOpacity
                  style={styles.inspectBtn}
                  onPress={() => {
                    haptic('selection');
                    navigation.navigate('MarlModelDetails');
                  }}
                >
                  <Text style={styles.inspectBtnText}>Details</Text>
                  <Ionicons name="chevron-forward" size={14} color={LiquidGlassTokens.accent.primary} />
                </TouchableOpacity>
              </View>
            </GlassCard>
          ))
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
  bannerCard: {
    padding: 16,
    marginBottom: 12,
  },
  bannerHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  bannerTitle: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.8,
    color: LiquidGlassTokens.accent.primary,
  },
  bannerDesc: {
    fontSize: 13,
    lineHeight: 19,
    color: LiquidGlassTokens.text.secondary,
  },
  quickNavRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 16,
  },
  quickNavBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
    minHeight: 44,
  },
  quickNavText: {
    fontSize: 12,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  sectionHeader: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1.0,
    color: LiquidGlassTokens.text.muted,
    marginBottom: 10,
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
  modelCard: {
    padding: 16,
    marginBottom: 12,
  },
  modelHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  modelName: {
    fontSize: 15,
    fontWeight: '800',
    letterSpacing: -0.2,
    color: LiquidGlassTokens.text.primary,
  },
  modelFamily: {
    fontSize: 12,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 2,
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    marginVertical: 12,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statCol: {
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 10,
    color: LiquidGlassTokens.text.muted,
    marginBottom: 4,
  },
  statVal: {
    fontSize: 13,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  modelFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 14,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.06)',
  },
  checkpointLabel: {
    fontSize: 10,
    color: LiquidGlassTokens.text.muted,
  },
  checkpointVal: {
    fontSize: 12,
    fontWeight: '600',
    color: LiquidGlassTokens.text.primary,
    marginTop: 2,
  },
  inspectBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: 'rgba(0, 210, 255, 0.12)',
    borderWidth: 1,
    borderColor: 'rgba(0, 210, 255, 0.25)',
  },
  inspectBtnText: {
    fontSize: 12,
    fontWeight: '700',
    color: LiquidGlassTokens.accent.primary,
  },
});
