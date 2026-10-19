import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, StyleProp, ViewStyle } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LiquidGlassTokens } from '../../theme/tokens';
import { GlassPill } from './GlassPill';

export interface GlassHeaderProps {
  title: string;
  subtitle?: string;
  onBack?: () => void;
  onBackPress?: () => void;
  showBack?: boolean;
  rightAction?: React.ReactNode;
  badge?: React.ReactNode;
  modelBadge?: boolean;
  style?: StyleProp<ViewStyle>;
}

export const GlassHeader: React.FC<GlassHeaderProps> = ({
  title,
  subtitle,
  onBack,
  onBackPress,
  showBack,
  rightAction,
  badge,
  modelBadge = true,
  style,
}) => {
  const handleBack = onBackPress || onBack;
  const shouldShowBack = showBack !== undefined ? showBack : !!handleBack;

  return (
    <View style={[styles.headerContainer, style]}>
      <View style={styles.specularRim} />
      <View style={styles.topRow}>
        <View style={styles.leftCol}>
          {shouldShowBack && handleBack && (
            <TouchableOpacity
              onPress={handleBack}
              style={styles.backButton}
              accessibilityRole="button"
              accessibilityLabel="Navigate back"
              hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
            >
              <Ionicons name="chevron-back" size={24} color={LiquidGlassTokens.accent.primary} />
            </TouchableOpacity>
          )}
          <View style={styles.titleCol}>
            <Text style={styles.titleText} numberOfLines={1}>{title}</Text>
            {subtitle && <Text style={styles.subtitleText}>{subtitle}</Text>}
          </View>
        </View>

        <View style={styles.rightCol}>
          {badge}
          {!badge && modelBadge && (
            <GlassPill label="Offline-to-Online MARL" variant="ai" />
          )}
          {rightAction}
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  headerContainer: {
    backgroundColor: 'rgba(11, 17, 32, 0.88)',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.12)',
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 14,
    position: 'relative',
    ...LiquidGlassTokens.shadows.subtle,
  },
  specularRim: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.18)',
  },
  topRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  leftCol: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  backButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.12)',
  },
  titleCol: {
    flex: 1,
  },
  titleText: {
    fontSize: 18,
    fontWeight: '800',
    color: LiquidGlassTokens.text.primary,
    letterSpacing: -0.3,
  },
  subtitleText: {
    fontSize: 12,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 2,
  },
  rightCol: {
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: 10,
  },
});
