import React from 'react';
import { View, StyleSheet, ViewStyle, StyleProp } from 'react-native';
import { BlurView } from 'expo-blur';
import { LiquidGlassTokens } from '../../theme/tokens';
import { useTheme } from '../../theme/ThemeContext';

export interface GlassCardProps {
  children?: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  variant?: 'standard' | 'emergency' | 'safety' | 'ultraThin' | 'aiPill' | 'modal' | 'highlight' | 'subtle';
  elevated?: boolean;
}

export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  style,
  variant = 'standard',
  elevated = true,
}) => {
  const { isBlurSupported } = useTheme();
  const effectiveVariant = variant === 'highlight' ? 'aiPill' : variant === 'subtle' ? 'ultraThin' : variant;
  const glassConfig = LiquidGlassTokens.glass[effectiveVariant] || LiquidGlassTokens.glass.standard;


  return (
    <View
      style={[
        styles.base,
        {
          backgroundColor: isBlurSupported ? 'rgba(16, 24, 44, 0.40)' : glassConfig.backgroundColor,
          borderColor: glassConfig.borderColor,
          borderTopColor: glassConfig.borderTopColor,
        },
        elevated && styles.elevated,
        style,
      ]}
    >
      {isBlurSupported && (
        <BlurView
          intensity={30}
          tint="dark"
          style={StyleSheet.absoluteFill}
        />
      )}
      {/* Specular inner highlight rim */}
      <View
        pointerEvents="none"
        style={[
          styles.specularRim,
          {
            borderTopColor: glassConfig.borderTopColor,
          },
        ]}
      />
      {children}
    </View>
  );
};

const styles = StyleSheet.create({
  base: {
    borderRadius: LiquidGlassTokens.radius.lg,
    borderWidth: 1,
    overflow: 'hidden',
    position: 'relative',
  },
  elevated: {
    ...LiquidGlassTokens.shadows.subtle,
  },
  specularRim: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 1.5,
    borderTopWidth: 1,
    opacity: 0.85,
  },
});
