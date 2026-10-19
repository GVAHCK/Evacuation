import React from 'react';
import { View, Text, StyleSheet, ViewStyle, StyleProp } from 'react-native';
import { LiquidGlassTokens } from '../../theme/tokens';

export interface GlassPillProps {
  label: string;
  variant?: 'cyan' | 'emergency' | 'safety' | 'warning' | 'ai' | 'neutral';
  icon?: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  pulsing?: boolean;
}

export const GlassPill: React.FC<GlassPillProps> = ({
  label,
  variant = 'cyan',
  icon,
  style,
}) => {
  const getTheme = () => {
    switch (variant) {
      case 'emergency':
        return {
          bg: 'rgba(239, 68, 68, 0.16)',
          border: 'rgba(239, 68, 68, 0.40)',
          text: LiquidGlassTokens.accent.danger,
          dot: LiquidGlassTokens.accent.danger,
        };
      case 'safety':
        return {
          bg: 'rgba(16, 185, 129, 0.16)',
          border: 'rgba(16, 185, 129, 0.40)',
          text: LiquidGlassTokens.accent.success,
          dot: LiquidGlassTokens.accent.success,
        };
      case 'warning':
        return {
          bg: 'rgba(245, 158, 11, 0.16)',
          border: 'rgba(245, 158, 11, 0.40)',
          text: LiquidGlassTokens.accent.warning,
          dot: LiquidGlassTokens.accent.warning,
        };
      case 'ai':
        return {
          bg: 'rgba(139, 92, 246, 0.18)',
          border: 'rgba(139, 92, 246, 0.45)',
          text: '#C4B5FD',
          dot: LiquidGlassTokens.accent.ai,
        };
      case 'neutral':
        return {
          bg: 'rgba(255, 255, 255, 0.06)',
          border: 'rgba(255, 255, 255, 0.15)',
          text: LiquidGlassTokens.text.secondary,
          dot: LiquidGlassTokens.text.muted,
        };
      default:
        return {
          bg: 'rgba(0, 210, 255, 0.14)',
          border: 'rgba(0, 210, 255, 0.40)',
          text: LiquidGlassTokens.accent.primary,
          dot: LiquidGlassTokens.accent.primary,
        };
    }
  };

  const theme = getTheme();

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: theme.bg,
          borderColor: theme.border,
        },
        style,
      ]}
      accessibilityRole="text"
      accessibilityLabel={label}
    >
      <View style={[styles.indicatorDot, { backgroundColor: theme.dot }]} />
      {icon && <View style={styles.iconContainer}>{icon}</View>}
      <Text style={[styles.text, { color: theme.text }]}>{label}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 4.5,
    borderRadius: LiquidGlassTokens.radius.pill,
    borderWidth: 1,
    alignSelf: 'flex-start',
  },
  indicatorDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: 6,
  },
  iconContainer: {
    marginRight: 4,
  },
  text: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.4,
    textTransform: 'uppercase',
  },
});
