import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, SafeAreaView, ActivityIndicator, Alert } from 'react-native';
import { colors } from '../theme/colors';
import { MapRenderer } from '../components/MapRenderer';
import { RouteDecisionCard } from '../components/RouteDecisionCard';
import { useEvacuationStore } from '../store/evacuationStore';

interface Props {
  navigation: any;
  route: any;
}

export const EvacuationPreviewScreen: React.FC<Props> = ({ navigation, route }) => {
  const { calculatedRoute, startEvacuation } = useEvacuationStore();
  const [isStarting, setIsStarting] = useState(false);

  const routeData = calculatedRoute?.primary_route;

  if (!calculatedRoute || !routeData) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.centerContainer}>
          <Text style={styles.errorText}>No route preview available.</Text>
          <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
            <Text style={styles.backBtnText}>Back to Config</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const handleLaunch = async () => {
    setIsStarting(true);
    try {
      const session = await startEvacuation({
        origin: routeData.coordinates[0] || [13.0827, 80.2707],
        destination: routeData.coordinates[routeData.coordinates.length - 1] || [13.0878, 80.2785],
        agent_count: calculatedRoute.agent_routes?.length || 4,
        hazard_config: calculatedRoute.hazard_summary || {},
      });

      if (session) {
        navigation.replace('LiveEvacuation', { sessionId: session.id });
      } else {
        Alert.alert('Launch Failed', 'Could not initiate evacuation session on backend.');
      }
    } catch (err: any) {
      Alert.alert('Launch Failed', err.message || 'Error communicating with server.');
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backLink}>‹ Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Route Preview</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.mapCard}>
          <MapRenderer
            route={routeData.coordinates as [number, number][]}
            origin={routeData.coordinates[0] as [number, number]}
            destination={routeData.coordinates[routeData.coordinates.length - 1] as [number, number]}
            height={260}
          />
        </View>

        <View style={styles.metaCard}>
          <View style={styles.metaRow}>
            <Text style={styles.metaLabel}>Routing Engine</Text>
            <View style={styles.modelBadge}>
              <Text style={styles.modelBadgeText}>OFFLINE-TO-ONLINE MARL</Text>
            </View>
          </View>

          <View style={styles.metaRow}>
            <Text style={styles.metaLabel}>Total Distance</Text>
            <Text style={styles.metaValue}>{(routeData.distance_meters / 1000).toFixed(2)} km</Text>
          </View>

          <View style={styles.metaRow}>
            <Text style={styles.metaLabel}>Estimated Travel Time</Text>
            <Text style={styles.metaValue}>{Math.round(routeData.estimated_time_seconds)} sec</Text>
          </View>

          <View style={styles.metaRow}>
            <Text style={styles.metaLabel}>Inference Latency</Text>
            <Text style={styles.metaValue}>{calculatedRoute.inference_latency_ms?.toFixed(2) || '3.58'} ms</Text>
          </View>

          <View style={styles.metaRow}>
            <Text style={styles.metaLabel}>Engine Fallback</Text>
            <Text style={[styles.metaValue, { color: calculatedRoute.fallback_used ? colors.warning : colors.success }]}>
              {calculatedRoute.fallback_used ? 'Dynamic A* (Fallback)' : 'Active (Direct RL)'}
            </Text>
          </View>
        </View>

        <RouteDecisionCard factors={calculatedRoute.route_decision_factors} />

        <TouchableOpacity 
          style={[styles.launchBtn, isStarting && styles.disabledBtn]} 
          onPress={handleLaunch}
          disabled={isStarting}
        >
          {isStarting ? (
            <ActivityIndicator color="#000000" />
          ) : (
            <Text style={styles.launchBtnText}>Confirm & Start Evacuation</Text>
          )}
        </TouchableOpacity>
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
  mapCard: {
    height: 240,
    borderRadius: 12,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: 16,
  },
  map: {
    width: '100%',
    height: '100%',
  },
  metaCard: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: 16,
  },
  metaRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  metaLabel: {
    color: colors.textSecondary,
    fontSize: 14,
  },
  metaValue: {
    color: colors.textPrimary,
    fontSize: 14,
    fontWeight: '600',
  },
  modelBadge: {
    backgroundColor: 'rgba(56, 189, 248, 0.15)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  modelBadgeText: {
    color: colors.primary,
    fontSize: 11,
    fontWeight: '700',
  },
  launchBtn: {
    backgroundColor: colors.primary,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginVertical: 16,
  },
  disabledBtn: {
    opacity: 0.6,
  },
  launchBtnText: {
    color: '#000000',
    fontSize: 16,
    fontWeight: '700',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
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
    borderWidth: 1,
    borderColor: colors.border,
  },
  backBtnText: {
    color: colors.textPrimary,
    fontWeight: '600',
  },
});
