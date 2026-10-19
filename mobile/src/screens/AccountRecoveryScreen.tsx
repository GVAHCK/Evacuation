import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { LiquidGlassTokens } from '../theme/tokens';
import { GlassCard } from '../components/glass/GlassCard';
import { GlassButton } from '../components/glass/GlassButton';
import { GlassInput } from '../components/glass/GlassInput';
import { GlassBadge } from '../components/glass/GlassBadge';
import { GlassHeader } from '../components/glass/GlassHeader';
import { Api } from '../api/client';

export const AccountRecoveryScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ message: string; instructions: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRecover = async () => {
    if (!email || !email.includes('@')) {
      setError('Please provide a valid emergency account email address.');
      return;
    }
    setError(null);
    setLoading(true);

    try {
      const resp = await Api.forgotPassword(email.trim());
      setResult({
        message: resp.message || 'Recovery token generated successfully.',
        instructions: resp.instructions || 'Contact administrator with verification code.',
      });
    } catch (err: any) {
      setError(err?.message || 'Failed to dispatch recovery request. Please retry.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <GlassHeader
        title="Account Recovery"
        showBack
        onBackPress={() => navigation.goBack()}
        badge={<GlassBadge status="healthy" label="Security Portal" size="sm" />}
      />

      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={styles.keyboardView}
      >
        <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
          <View style={styles.headerHero}>
            <View style={styles.iconCircle}>
              <Ionicons name="key-outline" size={36} color={LiquidGlassTokens.accent.primary} />
            </View>
            <Text style={styles.title}>Credential Reset</Text>
            <Text style={styles.subtitle}>
              Reinforcement-Learning Evacuation Command Platform account verification.
            </Text>
          </View>

          {result ? (
            <GlassCard variant="safety" style={styles.resultCard}>
              <View style={styles.resultHeader}>
                <Ionicons name="checkmark-circle" size={28} color={LiquidGlassTokens.accent.success} />
                <Text style={styles.resultTitle}>Request Dispatched</Text>
              </View>
              <Text style={styles.resultMessage}>{result.message}</Text>
              <Text style={styles.instructions}>{result.instructions}</Text>
              <GlassButton
                title="Return to Sign In"
                onPress={() => navigation.navigate('Login')}
                style={styles.actionBtn}
              />
            </GlassCard>
          ) : (
            <GlassCard style={styles.formCard}>
              <Text style={styles.formInstruction}>
                Enter the operational email associated with your dispatch or research profile:
              </Text>

              <GlassInput
                label="Registered Email"
                placeholder="name@evacuation.ai"
                value={email}
                onChangeText={(val) => {
                  setEmail(val);
                  setError(null);
                }}
                keyboardType="email-address"
                autoCapitalize="none"
                autoCorrect={false}
                error={error || undefined}
                icon={<Ionicons name="mail-outline" size={18} color={LiquidGlassTokens.text.secondary} />}
              />

              <GlassButton
                title={loading ? 'Verifying...' : 'Request Recovery Instructions'}
                onPress={handleRecover}
                loading={loading}
                style={styles.actionBtn}
              />

              <View style={styles.noticeBox}>
                <Ionicons name="information-circle-outline" size={18} color={LiquidGlassTokens.text.muted} />
                <Text style={styles.noticeText}>
                  For immediate operational priority or field system access, contact the platform administrator directly at admin@evacuation.ai.
                </Text>
              </View>
            </GlassCard>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: LiquidGlassTokens.background.primary,
  },
  keyboardView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 40,
  },
  headerHero: {
    alignItems: 'center',
    marginVertical: 20,
  },
  iconCircle: {
    width: 68,
    height: 68,
    borderRadius: 34,
    backgroundColor: 'rgba(0, 210, 255, 0.12)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0, 210, 255, 0.30)',
    marginBottom: 14,
  },
  title: {
    fontSize: 22,
    fontWeight: '800',
    color: LiquidGlassTokens.text.primary,
    letterSpacing: 0.3,
  },
  subtitle: {
    fontSize: 13,
    color: LiquidGlassTokens.text.secondary,
    textAlign: 'center',
    marginTop: 6,
    paddingHorizontal: 20,
    lineHeight: 18,
  },
  formCard: {
    padding: 20,
  },
  formInstruction: {
    fontSize: 14,
    color: LiquidGlassTokens.text.secondary,
    marginBottom: 16,
    lineHeight: 20,
  },
  actionBtn: {
    marginTop: 8,
  },
  noticeBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
    borderRadius: LiquidGlassTokens.radius.md,
    padding: 12,
    marginTop: 18,
  },
  noticeText: {
    fontSize: 12,
    color: LiquidGlassTokens.text.muted,
    marginLeft: 8,
    flex: 1,
    lineHeight: 16,
  },
  resultCard: {
    padding: 20,
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: LiquidGlassTokens.accent.success,
    marginLeft: 10,
  },
  resultMessage: {
    fontSize: 14,
    color: LiquidGlassTokens.text.primary,
    marginBottom: 10,
    lineHeight: 20,
  },
  instructions: {
    fontSize: 13,
    color: LiquidGlassTokens.text.secondary,
    marginBottom: 20,
    lineHeight: 18,
  },
});
