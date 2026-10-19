import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LiquidGlassTokens } from '../theme/tokens';
import { GlassCard } from '../components/glass/GlassCard';
import { GlassButton } from '../components/glass/GlassButton';
import { GlassBadge } from '../components/glass/GlassBadge';
import { GlassHeader } from '../components/glass/GlassHeader';
import { GlassInput } from '../components/glass/GlassInput';
import { useTheme } from '../theme/ThemeContext';

export const HazardConfigurationScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const { haptic } = useTheme();

  const [hazardType, setHazardType] = useState<'flood' | 'fire' | 'combined'>('flood');
  const [severity, setSeverity] = useState('0.8');
  const [radius, setRadius] = useState('450');
  const [centerLat, setCenterLat] = useState('13.0107');
  const [centerLon, setCenterLon] = useState('80.2248');

  const hazardOptions = [
    {
      id: 'flood' as const,
      label: 'Flood Inundation',
      icon: 'water' as const,
      color: '#38BDF8',
      desc: 'Cellular hydraulic depth. Blocks roads with depth >0.65m.',
    },
    {
      id: 'fire' as const,
      label: 'Wildfire / Urban Fire',
      icon: 'flame' as const,
      color: '#EF4444',
      desc: 'Thermal convection diffusion. Impassable at intensity >0.70.',
    },
    {
      id: 'combined' as const,
      label: 'Combined Hazards',
      icon: 'thunderstorm' as const,
      color: '#C084FC',
      desc: 'Simultaneous flood and fire with compound risk aggregation.',
    },
  ];

  return (
    <SafeAreaView style={styles.safeArea}>
      <GlassHeader
        title="Hazard Engine Setup"
        showBack
        onBackPress={() => navigation.goBack()}
        badge={<GlassBadge status={hazardType} label={hazardType.toUpperCase()} size="sm" />}
      />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Hazard Type Selector */}
        <Text style={styles.sectionHeader}>Hazard Physics Selection</Text>
        {hazardOptions.map((opt) => {
          const isSelected = hazardType === opt.id;
          return (
            <TouchableOpacity
              key={opt.id}
              activeOpacity={0.8}
              onPress={() => {
                haptic.selection();
                setHazardType(opt.id);
              }}
            >
              <GlassCard
                style={[styles.card, isSelected && styles.cardSelected]}
                variant={isSelected ? 'standard' : 'ultraThin'}
              >
                <View style={styles.optRow}>
                  <View style={[styles.iconBox, { backgroundColor: `${opt.color}20` }]}>
                    <Ionicons name={opt.icon} size={24} color={opt.color} />
                  </View>
                  <View style={styles.optTextCol}>
                    <Text style={styles.optLabel}>{opt.label}</Text>
                    <Text style={styles.optDesc}>{opt.desc}</Text>
                  </View>
                  <Ionicons
                    name={isSelected ? 'radio-button-on' : 'radio-button-off'}
                    size={22}
                    color={isSelected ? LiquidGlassTokens.accent.primary : LiquidGlassTokens.text.muted}
                  />
                </View>
              </GlassCard>
            </TouchableOpacity>
          );
        })}

        {/* Hazard Parameters */}
        <Text style={styles.sectionHeader}>Spatial Extent &amp; Severity Bounds</Text>
        <GlassCard style={styles.card}>
          <GlassInput
            label="Hazard Severity [0.0 - 1.0]"
            value={severity}
            onChangeText={setSeverity}
            keyboardType="numeric"
            helperText="Pydantic validation strictly enforces values between 0.0 and 1.0"
          />

          <GlassInput
            label="Inundation / Thermal Radius (Meters)"
            value={radius}
            onChangeText={setRadius}
            keyboardType="numeric"
            helperText="Default standard research radius: 350m to 500m"
          />

          <View style={styles.coordRow}>
            <View style={{ flex: 1, marginRight: 8 }}>
              <GlassInput
                label="Center Latitude"
                value={centerLat}
                onChangeText={setCenterLat}
                keyboardType="numeric"
              />
            </View>
            <View style={{ flex: 1, marginLeft: 8 }}>
              <GlassInput
                label="Center Longitude"
                value={centerLon}
                onChangeText={setCenterLon}
                keyboardType="numeric"
              />
            </View>
          </View>
        </GlassCard>

        <GlassButton
          title="Save Hazard Configuration"
          variant="primary"
          onPress={() => navigation.goBack()}
          style={{ marginTop: 16 }}
        />
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
    marginBottom: 12,
  },
  cardSelected: {
    borderColor: 'rgba(0, 210, 255, 0.40)',
  },
  optRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconBox: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  optTextCol: {
    flex: 1,
    marginRight: 8,
  },
  optLabel: {
    fontSize: 15,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  optDesc: {
    fontSize: 12,
    color: LiquidGlassTokens.text.muted,
    marginTop: 2,
    lineHeight: 16,
  },
  coordRow: {
    flexDirection: 'row',
  },
});
