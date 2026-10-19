import React from 'react';
import { View, Text, StyleSheet, ViewStyle, StyleProp } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LiquidGlassTokens } from '../../theme/tokens';

export type BadgeStatus =
  | 'healthy'
  | 'active'
  | 'warning'
  | 'danger'
  | 'critical'
  | 'completed'
  | 'offline'
  | 'degraded'
  | 'unknown'
  | 'flood'
  | 'fire'
  | 'combined'
  | 'ai';

export interface GlassBadgeProps {
  status?: BadgeStatus;
  variant?: string;
  icon?: any;
  label?: string;
  style?: StyleProp<ViewStyle>;
  size?: 'sm' | 'md' | 'lg';
}

export const GlassBadge: React.FC<GlassBadgeProps> = ({
  status,
  variant,
  icon,
  label,
  style,
  size = 'md',
}) => {
  const mapVariantToStatus = (v?: string): BadgeStatus => {
    if (!v) return 'active';
    if (v === 'ai' || v === 'aiPill') return 'ai';
    if (v === 'success' || v === 'healthy' || v === 'completed') return 'completed';
    if (v === 'warning' || v === 'degraded' || v === 'caution') return 'warning';
    if (v === 'critical' || v === 'danger') return 'critical';
    if (v === 'offline') return 'offline';
    if (v === 'flood') return 'flood';
    if (v === 'fire') return 'fire';
    return 'active';
  };

  const effectiveStatus = status || mapVariantToStatus(variant);

  const getBadgeConfig = () => {
    switch (effectiveStatus) {
      case 'healthy':
      case 'completed':
      case 'active':
        return {
          icon: 'checkmark-circle' as const,
          color: LiquidGlassTokens.accent.success,
          bg: 'rgba(16, 185, 129, 0.16)',
          border: 'rgba(16, 185, 129, 0.35)',
          defaultLabel: effectiveStatus === 'completed' ? 'Completed' : effectiveStatus === 'active' ? 'Active' : 'Healthy',
        };
      case 'warning':
      case 'degraded':
        return {
          icon: 'warning' as const,
          color: LiquidGlassTokens.accent.warning,
          bg: 'rgba(245, 158, 11, 0.16)',
          border: 'rgba(245, 158, 11, 0.35)',
          defaultLabel: effectiveStatus === 'degraded' ? 'Degraded' : 'Caution',
        };
      case 'danger':
      case 'critical':
        return {
          icon: 'alert-circle' as const,
          color: LiquidGlassTokens.accent.danger,
          bg: 'rgba(239, 68, 68, 0.16)',
          border: 'rgba(239, 68, 68, 0.40)',
          defaultLabel: effectiveStatus === 'critical' ? 'Critical' : 'Danger',
        };
      case 'flood':
        return {
          icon: 'water' as const,
          color: '#38BDF8',
          bg: 'rgba(56, 189, 248, 0.16)',
          border: 'rgba(56, 189, 248, 0.35)',
          defaultLabel: 'Flood Active',
        };
      case 'fire':
        return {
          icon: 'flame' as const,
          color: '#F97316',
          bg: 'rgba(249, 115, 22, 0.16)',
          border: 'rgba(249, 115, 22, 0.35)',
          defaultLabel: 'Fire Active',
        };
      case 'combined':
        return {
          icon: 'thunderstorm' as const,
          color: '#C084FC',
          bg: 'rgba(192, 132, 252, 0.16)',
          border: 'rgba(192, 132, 252, 0.35)',
          defaultLabel: 'Combined Hazard',
        };
      case 'ai':
        return {
          icon: 'hardware-chip' as const,
          color: LiquidGlassTokens.accent.ai,
          bg: LiquidGlassTokens.glass.aiPill.backgroundColor,
          border: LiquidGlassTokens.glass.aiPill.borderColor,
          defaultLabel: 'MARL AI',
        };
      case 'offline':
        return {
          icon: 'cloud-offline' as const,
          color: '#94A3B8',
          bg: 'rgba(148, 163, 184, 0.14)',
          border: 'rgba(148, 163, 184, 0.30)',
          defaultLabel: 'Offline',
        };
      case 'unknown':
      default:
        return {
          icon: 'help-circle' as const,
          color: '#94A3B8',
          bg: 'rgba(148, 163, 184, 0.14)',
          border: 'rgba(148, 163, 184, 0.30)',
          defaultLabel: 'Unknown',
        };
    }
  };

  const config = getBadgeConfig();
  const iconSize = size === 'sm' ? 12 : size === 'lg' ? 18 : 14;
  const fontSize = size === 'sm' ? 11 : size === 'lg' ? 14 : 12;

  return (
    <View
      style={[
        styles.badge,
        {
          backgroundColor: config.bg,
          borderColor: config.border,
        },
        size === 'sm' && styles.badgeSm,
        size === 'lg' && styles.badgeLg,
        style,
      ]}
      accessibilityRole="text"
      accessibilityLabel={`Status: ${label || config.defaultLabel}`}
    >
      <Ionicons name={icon || config.icon} size={iconSize} color={config.color} style={styles.icon} />
      <Text style={[styles.label, { color: config.color, fontSize }]}>
        {label || config.defaultLabel}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: LiquidGlassTokens.radius.pill,
    borderWidth: 1,
    alignSelf: 'flex-start',
  },
  badgeSm: {
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  badgeLg: {
    paddingHorizontal: 14,
    paddingVertical: 7,
  },
  icon: {
    marginRight: 5,
  },
  label: {
    fontWeight: '700',
    letterSpacing: 0.2,
  },
});
