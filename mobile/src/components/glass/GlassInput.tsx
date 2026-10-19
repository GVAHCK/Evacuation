import React from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  TextInputProps,
  StyleProp,
  ViewStyle,
} from 'react-native';
import { LiquidGlassTokens } from '../../theme/tokens';

export interface GlassInputProps extends TextInputProps {
  label?: string;
  error?: string;
  containerStyle?: StyleProp<ViewStyle>;
  helperText?: string;
  icon?: React.ReactNode;
}

export const GlassInput: React.FC<GlassInputProps> = ({
  label,
  error,
  containerStyle,
  helperText,
  icon,
  style,
  ...props
}) => {
  return (
    <View style={[styles.container, containerStyle]}>
      {label && <Text style={styles.label}>{label}</Text>}
      <View
        style={[
          styles.inputWrapper,
          error ? styles.inputWrapperError : null,
        ]}
      >
        {icon && <View style={styles.iconMargin}>{icon}</View>}
        <TextInput
          placeholderTextColor={LiquidGlassTokens.text.muted}
          style={[styles.input, style]}
          accessibilityLabel={label || props.placeholder}
          {...props}
        />
      </View>
      {error ? (
        <Text style={styles.errorText}>{error}</Text>
      ) : helperText ? (
        <Text style={styles.helperText}>{helperText}</Text>
      ) : null}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
  },
  label: {
    fontSize: 13,
    fontWeight: '600',
    color: LiquidGlassTokens.text.secondary,
    marginBottom: 6,
    letterSpacing: 0.2,
  },
  inputWrapper: {
    minHeight: 48,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(16, 24, 44, 0.70)',
    borderRadius: LiquidGlassTokens.radius.md,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.14)',
    borderTopColor: 'rgba(255, 255, 255, 0.24)',
    paddingHorizontal: 14,
  },
  inputWrapperError: {
    borderColor: LiquidGlassTokens.accent.danger,
  },
  iconMargin: {
    marginRight: 10,
  },
  input: {
    flex: 1,
    color: LiquidGlassTokens.text.primary,
    fontSize: 15,
    paddingVertical: 10,
  },
  errorText: {
    fontSize: 12,
    color: LiquidGlassTokens.accent.danger,
    marginTop: 4,
    fontWeight: '500',
  },
  helperText: {
    fontSize: 12,
    color: LiquidGlassTokens.text.muted,
    marginTop: 4,
  },
});
