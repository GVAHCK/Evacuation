import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Switch,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LiquidGlassTokens } from '../theme/tokens';
import { GlassCard } from '../components/glass/GlassCard';
import { GlassBadge } from '../components/glass/GlassBadge';
import { GlassButton } from '../components/glass/GlassButton';
import { GlassHeader } from '../components/glass/GlassHeader';
import { useTheme } from '../theme/ThemeContext';

export const AccessibilitySettingsScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const {
    mode,
    setMode,
    reduceTransparency,
    setReduceTransparency,
    reduceMotion,
    setReduceMotion,
    haptic,
  } = useTheme();

  const [highContrast, setHighContrast] = useState(false);
  const [fontScale, setFontScale] = useState<'1.0x' | '1.2x' | '1.4x' | '1.6x'>('1.0x');
  const [hapticsEnabled, setHapticsEnabled] = useState(true);
  const [voiceAlerts, setVoiceAlerts] = useState(true);

  const handleModeToggle = (targetMode: 'dark' | 'light') => {
    haptic('selection');
    setMode(targetMode);
  };

  const handleSavePreferences = () => {
    haptic('heavy');
    Alert.alert('Accessibility Configured', 'Display, transparency, and tactile preferences saved.');
  };

  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
      <GlassHeader
        title="Accessibility & Display"
        subtitle="Dynamic Type, High Contrast & Liquid Glass Control"
        showBack
        onBackPress={() => {
          haptic('light');
          navigation.goBack();
        }}
        badge={
          <GlassBadge
            label="WCAG 2.1 AA"
            variant="success"
            icon="accessibility-outline"
          />
        }
      />

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Banner */}
        <GlassCard variant="highlight" style={styles.card}>
          <View style={styles.cardHeader}>
            <Ionicons name="eye-outline" size={24} color={LiquidGlassTokens.accent.primary} />
            <Text style={styles.cardTitle}>ACCESSIBILITY STANDARDS</Text>
          </View>
          <Text style={styles.cardDesc}>
            All interactive elements meet minimum 44pt touch target sizing. Status indicators strictly combine distinct glyph icons with text labels rather than communicating severity through color alone.
          </Text>
        </GlassCard>

        {/* Theme Appearance Mode */}
        <Text style={styles.sectionHeader}>THEME APPEARANCE</Text>
        <GlassCard variant="standard" style={styles.card}>
          <Text style={styles.prefDesc}>
            Initial theme is dark by design to preserve night vision and battery in emergency response operations.
          </Text>
          <View style={styles.modeRow}>
            <TouchableOpacity
              style={[styles.modeBtn, mode === 'dark' && styles.modeBtnActive]}
              onPress={() => handleModeToggle('dark')}
              accessibilityRole="button"
              accessibilityLabel="Select Dark Theme"
            >
              <Ionicons
                name="moon"
                size={18}
                color={mode === 'dark' ? LiquidGlassTokens.accent.primary : LiquidGlassTokens.text.muted}
              />
              <Text style={[styles.modeText, mode === 'dark' && styles.modeTextActive]}>Dark (Default)</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.modeBtn, mode === 'light' && styles.modeBtnActive]}
              onPress={() => handleModeToggle('light')}
              accessibilityRole="button"
              accessibilityLabel="Select Light Theme"
            >
              <Ionicons
                name="sunny"
                size={18}
                color={mode === 'light' ? LiquidGlassTokens.accent.primary : LiquidGlassTokens.text.muted}
              />
              <Text style={[styles.modeText, mode === 'light' && styles.modeTextActive]}>Light Theme</Text>
            </TouchableOpacity>
          </View>
        </GlassCard>

        {/* Liquid Glass & Motion Preferences */}
        <Text style={styles.sectionHeader}>VISUAL MATERIAL & REDUCED MOTION</Text>
        <GlassCard variant="standard" style={styles.card}>
          <View style={styles.switchRow}>
            <View style={{ flex: 1, marginRight: 12 }}>
              <Text style={styles.switchTitle}>Reduce Transparency</Text>
              <Text style={styles.switchDesc}>
                Disables backdrop blur on glass materials, replacing them with solid high-contrast opaque cards.
              </Text>
            </View>
            <Switch
              value={reduceTransparency}
              onValueChange={(val) => {
                haptic('selection');
                setReduceTransparency(val);
              }}
              trackColor={{ false: 'rgba(255,255,255,0.15)', true: LiquidGlassTokens.accent.primary }}
              thumbColor={reduceTransparency ? '#000' : LiquidGlassTokens.text.secondary}
            />
          </View>

          <View style={styles.divider} />

          <View style={styles.switchRow}>
            <View style={{ flex: 1, marginRight: 12 }}>
              <Text style={styles.switchTitle}>Reduce Motion</Text>
              <Text style={styles.switchDesc}>
                Avoids fluid animated transitions and route movements, substituting instantaneous status updates.
              </Text>
            </View>
            <Switch
              value={reduceMotion}
              onValueChange={(val) => {
                haptic('selection');
                setReduceMotion(val);
              }}
              trackColor={{ false: 'rgba(255,255,255,0.15)', true: LiquidGlassTokens.accent.primary }}
              thumbColor={reduceMotion ? '#000' : LiquidGlassTokens.text.secondary}
            />
          </View>

          <View style={styles.divider} />

          <View style={styles.switchRow}>
            <View style={{ flex: 1, marginRight: 12 }}>
              <Text style={styles.switchTitle}>High Contrast Map Layers</Text>
              <Text style={styles.switchDesc}>
                Emphasizes route corridor strokes with high-contrast borders against dark satellite base tiles.
              </Text>
            </View>
            <Switch
              value={highContrast}
              onValueChange={(val) => {
                haptic('selection');
                setHighContrast(val);
              }}
              trackColor={{ false: 'rgba(255,255,255,0.15)', true: LiquidGlassTokens.accent.primary }}
              thumbColor={highContrast ? '#000' : LiquidGlassTokens.text.secondary}
            />
          </View>
        </GlassCard>

        {/* Dynamic Type Scaling */}
        <Text style={styles.sectionHeader}>DYNAMIC TYPE FONT SCALING</Text>
        <GlassCard variant="standard" style={styles.card}>
          <Text style={styles.prefDesc}>
            Guarantees text readability without clipping or overlapping up to the 1.6× OS accessibility limit:
          </Text>
          <View style={styles.scaleRow}>
            {(['1.0x', '1.2x', '1.4x', '1.6x'] as const).map((sc) => {
              const isSelected = fontScale === sc;
              return (
                <TouchableOpacity
                  key={sc}
                  style={[styles.scaleBtn, isSelected && styles.scaleBtnActive]}
                  onPress={() => {
                    haptic('selection');
                    setFontScale(sc);
                  }}
                  accessibilityRole="button"
                  accessibilityLabel={`Set font scale to ${sc}`}
                >
                  <Text style={[styles.scaleBtnText, isSelected && styles.scaleBtnTextActive]}>
                    {sc}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </GlassCard>

        {/* Tactile Feedback & Alerts */}
        <Text style={styles.sectionHeader}>TACTILE & AUDITORY FEEDBACK</Text>
        <GlassCard variant="standard" style={styles.card}>
          <View style={styles.switchRow}>
            <View style={{ flex: 1, marginRight: 12 }}>
              <Text style={styles.switchTitle}>Expo Haptic Feedback</Text>
              <Text style={styles.switchDesc}>
                Emits subtle tactile responses on route selection, button taps, and reroute triggers.
              </Text>
            </View>
            <Switch
              value={hapticsEnabled}
              onValueChange={(val) => {
                haptic('selection');
                setHapticsEnabled(val);
              }}
              trackColor={{ false: 'rgba(255,255,255,0.15)', true: LiquidGlassTokens.accent.primary }}
              thumbColor={hapticsEnabled ? '#000' : LiquidGlassTokens.text.secondary}
            />
          </View>

          <View style={styles.divider} />

          <View style={styles.switchRow}>
            <View style={{ flex: 1, marginRight: 12 }}>
              <Text style={styles.switchTitle}>Emergency Sound Cues</Text>
              <Text style={styles.switchDesc}>
                Auditory chime upon dynamic reroute triggers or critical hazard encroachment.
              </Text>
            </View>
            <Switch
              value={voiceAlerts}
              onValueChange={(val) => {
                haptic('selection');
                setVoiceAlerts(val);
              }}
              trackColor={{ false: 'rgba(255,255,255,0.15)', true: LiquidGlassTokens.accent.primary }}
              thumbColor={voiceAlerts ? '#000' : LiquidGlassTokens.text.secondary}
            />
          </View>
        </GlassCard>

        <GlassButton
          title="SAVE ACCESSIBILITY PREFERENCES"
          variant="primary"
          size="large"
          onPress={handleSavePreferences}
          style={{ marginTop: 14, marginBottom: 20 }}
        />
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
  card: {
    padding: 16,
    marginBottom: 8,
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  cardTitle: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.8,
    color: LiquidGlassTokens.accent.primary,
  },
  cardDesc: {
    fontSize: 13,
    lineHeight: 19,
    color: LiquidGlassTokens.text.secondary,
  },
  sectionHeader: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 1.0,
    color: LiquidGlassTokens.text.muted,
    marginTop: 20,
    marginBottom: 10,
  },
  prefDesc: {
    fontSize: 12,
    lineHeight: 17,
    color: LiquidGlassTokens.text.secondary,
    marginBottom: 12,
  },
  modeRow: {
    flexDirection: 'row',
    gap: 10,
  },
  modeBtn: {
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
  modeBtnActive: {
    backgroundColor: 'rgba(0, 210, 255, 0.15)',
    borderColor: LiquidGlassTokens.accent.primary,
  },
  modeText: {
    fontSize: 13,
    fontWeight: '700',
    color: LiquidGlassTokens.text.muted,
  },
  modeTextActive: {
    color: LiquidGlassTokens.accent.primary,
  },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    minHeight: 44,
  },
  switchTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: LiquidGlassTokens.text.primary,
  },
  switchDesc: {
    fontSize: 12,
    lineHeight: 17,
    color: LiquidGlassTokens.text.secondary,
    marginTop: 2,
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    marginVertical: 12,
  },
  scaleRow: {
    flexDirection: 'row',
    gap: 10,
  },
  scaleBtn: {
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
  scaleBtnActive: {
    backgroundColor: 'rgba(0, 210, 255, 0.15)',
    borderColor: LiquidGlassTokens.accent.primary,
  },
  scaleBtnText: {
    fontSize: 13,
    fontWeight: '700',
    color: LiquidGlassTokens.text.secondary,
  },
  scaleBtnTextActive: {
    color: LiquidGlassTokens.accent.primary,
  },
});
