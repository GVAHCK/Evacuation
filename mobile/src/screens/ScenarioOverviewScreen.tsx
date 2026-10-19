import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
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
import { useTheme } from '../theme/ThemeContext';

export const ScenarioOverviewScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();
  const [scenarios, setScenarios] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchScenarios = async () => {
    try {
      const data = await Api.listScenarios();
      setScenarios(data || []);
    } catch {
      // Fallback sample
      setScenarios([
        {
          id: 'scen_demo_1',
          name: 'Flash Flood Adyar River Basin',
          description: 'Monsoon rapid river inundation scenario near Guindy / Kotturpuram',
          hazard_type: 'flood',
          agent_count: 4,
          created_at: new Date().toISOString(),
        },
        {
          id: 'scen_demo_2',
          name: 'Industrial Fire Corridor',
          description: 'Thermal spread along southern transit spine with wind dispersion',
          hazard_type: 'fire',
          agent_count: 4,
          created_at: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchScenarios();
  }, []);

  const handleLaunchScenario = (scen: any) => {
    haptic.medium();
    navigation.navigate('EvacuationPreview', {
      scenarioId: scen.id,
      hazardType: scen.hazard_type,
      agentCount: scen.agent_count || 4,
    });
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <GlassHeader
        title="Scenario Catalog"
        showBack
        onBackPress={() => navigation.goBack()}
        badge={
          <GlassBadge
            status="healthy"
            label={`${scenarios.length} Scenarios`}
            size="sm"
          />
        }
      />

      <View style={styles.topActionRow}>
        <GlassButton
          title="Create New Scenario"
          icon={<Ionicons name="add-circle" size={18} color="#FFFFFF" />}
          onPress={() => navigation.navigate('ScenarioBuilder')}
          style={{ flex: 1 }}
        />
      </View>

      {loading ? (
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color={LiquidGlassTokens.accent.primary} />
          <Text style={styles.loadingText}>Loading persisted scenarios...</Text>
        </View>
      ) : (
        <FlatList
          data={scenarios}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContent}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={() => {
                setRefreshing(true);
                fetchScenarios();
              }}
              tintColor="#00D2FF"
            />
          }
          renderItem={({ item }) => (
            <GlassCard style={styles.card}>
              <View style={styles.cardHeader}>
                <View style={styles.titleCol}>
                  <Text style={styles.scenarioName}>{item.name}</Text>
                  <Text style={styles.scenarioDesc}>{item.description || 'Standard evacuation profile'}</Text>
                </View>
                <GlassBadge
                  status={item.hazard_type === 'flood' ? 'flood' : item.hazard_type === 'fire' ? 'fire' : 'combined'}
                  label={item.hazard_type?.toUpperCase() || 'FLOOD'}
                  size="sm"
                />
              </View>

              <View style={styles.infoRow}>
                <View style={styles.infoItem}>
                  <Ionicons name="people-outline" size={16} color={LiquidGlassTokens.text.muted} />
                  <Text style={styles.infoText}>{item.agent_count || 4} Evacuees</Text>
                </View>
                <View style={styles.infoItem}>
                  <Ionicons name="hardware-chip-outline" size={16} color={LiquidGlassTokens.accent.ai} />
                  <Text style={styles.infoText}>MARL Offline-Online</Text>
                </View>
              </View>

              <View style={styles.cardActions}>
                <TouchableOpacity
                  style={styles.outlineBtn}
                  onPress={() => navigation.navigate('ScenarioBuilder', { scenarioId: item.id })}
                >
                  <Text style={styles.outlineBtnText}>Configure</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.launchBtn}
                  onPress={() => handleLaunchScenario(item)}
                >
                  <Ionicons name="play" size={14} color="#FFFFFF" style={{ marginRight: 6 }} />
                  <Text style={styles.launchBtnText}>Review &amp; Launch</Text>
                </TouchableOpacity>
              </View>
            </GlassCard>
          )}
        />
      )}
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: LiquidGlassTokens.background.primary,
  },
  topActionRow: {
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  listContent: {
    padding: 16,
    paddingTop: 4,
    paddingBottom: 40,
  },
  card: {
    padding: 16,
    marginBottom: 14,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  titleCol: {
    flex: 1,
    marginRight: 10,
  },
  scenarioName: {
    fontSize: 16,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  scenarioDesc: {
    fontSize: 12,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 4,
    lineHeight: 16,
  },
  infoRow: {
    flexDirection: 'row',
    paddingVertical: 10,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    marginBottom: 12,
  },
  infoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 20,
  },
  infoText: {
    fontSize: 12,
    color: LiquidGlassTokens.text.secondary,
    marginLeft: 6,
    fontWeight: '500',
  },
  cardActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  outlineBtn: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: LiquidGlassTokens.radius.md,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.20)',
    marginRight: 10,
    minHeight: 44,
    justifyContent: 'center',
  },
  outlineBtnText: {
    fontSize: 13,
    color: LiquidGlassTokens.text.primary,
    fontWeight: '600',
  },
  launchBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0284C7',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: LiquidGlassTokens.radius.md,
    minHeight: 44,
  },
  launchBtnText: {
    fontSize: 13,
    color: '#FFFFFF',
    fontWeight: '700',
  },
  centerContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    color: LiquidGlassTokens.text.secondary,
    marginTop: 12,
  },
});
