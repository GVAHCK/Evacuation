import React from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
  StyleProp,
  View,
} from 'react-native';
import { LiquidGlassTokens } from '../../theme/tokens';
import { useTheme } from '../../theme/ThemeContext';

import { Ionicons } from '@expo/vector-icons';

export interface GlassButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'danger' | 'success' | 'secondary' | 'glass' | 'destructive' | 'outline';
  size?: 'small' | 'medium' | 'large';
  loading?: boolean;
  disabled?: boolean;
  icon?: any;
  style?: StyleProp<ViewStyle>;
  textStyle?: StyleProp<TextStyle>;
  accessibilityLabel?: string;
  accessibilityHint?: string;
}

export const GlassButton: React.FC<GlassButtonProps> = ({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  loading = false,
  disabled = false,
  icon,
  style,
  textStyle,
  accessibilityLabel,
  accessibilityHint,
}) => {
  const getButtonStyles = () => {
    switch (variant) {
      case 'danger':
      case 'destructive':
        return {
          bg: '#DC2626',
          border: 'rgba(254, 202, 202, 0.4)',
          textColor: '#FFFFFF',
          topRim: 'rgba(255, 255, 255, 0.4)',
        };
      case 'success':
        return {
          bg: '#059669',
          border: 'rgba(167, 243, 208, 0.4)',
          textColor: '#FFFFFF',
          topRim: 'rgba(255, 255, 255, 0.4)',
        };
      case 'secondary':
        return {
          bg: 'rgba(30, 41, 59, 0.85)',
          border: 'rgba(255, 255, 255, 0.16)',
          textColor: LiquidGlassTokens.text.primary,
          topRim: 'rgba(255, 255, 255, 0.25)',
        };
      case 'outline':
        return {
          bg: 'transparent',
          border: 'rgba(56, 189, 248, 0.45)',
          textColor: LiquidGlassTokens.accent.primary,
          topRim: 'transparent',
        };
      case 'glass':
        return {
          bg: 'rgba(255, 255, 255, 0.08)',
          border: 'rgba(255, 255, 255, 0.20)',
          textColor: LiquidGlassTokens.text.primary,
          topRim: 'rgba(255, 255, 255, 0.35)',
        };
      default: // primary
        return {
          bg: '#0284C7',
          border: 'rgba(56, 189, 248, 0.5)',
          textColor: '#FFFFFF',
          topRim: 'rgba(255, 255, 255, 0.45)',
        };
    }
  };

  const theme = getButtonStyles();
  const { haptic } = useTheme();

  const handlePress = () => {
    if (variant === 'danger') {
      haptic.heavy();
    } else {
      haptic.light();
    }
    onPress();
  };

  return (
    <TouchableOpacity
      activeOpacity={0.78}
      onPress={handlePress}
      disabled={disabled || loading}
      style={[
        styles.button,
        {
          backgroundColor: theme.bg,
          borderColor: theme.border,
          opacity: disabled ? 0.45 : 1,
          minHeight: size === 'small' ? 38 : size === 'large' ? 54 : 48,
          paddingVertical: size === 'small' ? 8 : size === 'large' ? 14 : 12,
        },
        style,
      ]}
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel || title}
      accessibilityHint={accessibilityHint}
      accessibilityState={{ disabled: disabled || loading, busy: loading }}
    >
      {/* Refraction highlight rim */}
      <View
        pointerEvents="none"
        style={[styles.topRim, { borderTopColor: theme.topRim }]}
      />

      {loading ? (
        <ActivityIndicator color={theme.textColor} size="small" />
      ) : (
        <View style={styles.contentRow}>
          {typeof icon === 'string' ? (
            <Ionicons name={icon as any} size={18} color={theme.textColor} style={styles.iconMargin} />
          ) : (
            icon && <View style={styles.iconMargin}>{icon}</View>
          )}
          <Text style={[styles.text, { color: theme.textColor }, textStyle]}>
            {title}
          </Text>
        </View>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    minHeight: 48,
    borderRadius: LiquidGlassTokens.radius.md,
    borderWidth: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 12,
    overflow: 'hidden',
    position: 'relative',
    ...LiquidGlassTokens.shadows.subtle,
  },
  topRim: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 1.5,
    borderTopWidth: 1,
  },
  contentRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconMargin: {
    marginRight: 8,
  },
  text: {
    fontSize: 15,
    fontWeight: '700',
    letterSpacing: 0.2,
  },
});
