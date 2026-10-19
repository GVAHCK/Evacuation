import React, { createContext, useContext, useState, useEffect } from 'react';
import { Platform, AccessibilityInfo } from 'react-native';
import * as Haptics from 'expo-haptics';
import { LiquidGlassTokens } from './tokens';

export type HapticHandler = ((type?: 'light' | 'medium' | 'heavy' | 'selection') => void) & {
  light: () => void;
  medium: () => void;
  heavy: () => void;
  selection: () => void;
  notification: (type?: Haptics.NotificationFeedbackType) => void;
};

export interface ThemeContextValue {
  mode: 'dark' | 'light';
  setMode: (mode: 'dark' | 'light') => void;
  tokens: typeof LiquidGlassTokens;
  reduceTransparency: boolean;
  setReduceTransparency: (val: boolean) => void;
  reduceMotion: boolean;
  setReduceMotion: (val: boolean) => void;
  fontScale: number;
  setFontScale: (scale: number) => void;
  isBlurSupported: boolean;
  haptic: HapticHandler;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export interface ThemeProviderProps {
  children: React.ReactNode;
  initialMode?: 'dark' | 'light';
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({
  children,
  initialMode = 'dark',
}) => {
  const [mode, setMode] = useState<'dark' | 'light'>(initialMode);
  const [reduceTransparency, setReduceTransparency] = useState<boolean>(false);
  const [reduceMotion, setReduceMotion] = useState<boolean>(false);
  const [fontScale, setFontScale] = useState<number>(1.0);

  // Check system accessibility settings
  useEffect(() => {
    AccessibilityInfo.isReduceMotionEnabled?.().then((enabled) => {
      if (enabled !== undefined) setReduceMotion(enabled);
    });
    AccessibilityInfo.isReduceTransparencyEnabled?.().then((enabled) => {
      if (enabled !== undefined) setReduceTransparency(enabled);
    });

    const motionSub = AccessibilityInfo.addEventListener(
      'reduceMotionChanged',
      (enabled) => setReduceMotion(enabled)
    );
    const transparencySub = AccessibilityInfo.addEventListener(
      'reduceTransparencyChanged',
      (enabled) => setReduceTransparency(enabled)
    );

    return () => {
      motionSub?.remove?.();
      transparencySub?.remove?.();
    };
  }, []);

  const isBlurSupported = Platform.OS === 'ios' && !reduceTransparency;

  const hapticObj = {
    light: () => {
      try {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
      } catch {}
    },
    medium: () => {
      try {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
      } catch {}
    },
    heavy: () => {
      try {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy).catch(() => {});
      } catch {}
    },
    selection: () => {
      try {
        Haptics.selectionAsync().catch(() => {});
      } catch {}
    },
    notification: (type = Haptics.NotificationFeedbackType.Success) => {
      try {
        Haptics.notificationAsync(type).catch(() => {});
      } catch {}
    },
  };

  const hapticFn = ((type: 'light' | 'medium' | 'heavy' | 'selection' = 'light') => {
    if (type === 'heavy') hapticObj.heavy();
    else if (type === 'medium') hapticObj.medium();
    else if (type === 'selection') hapticObj.selection();
    else hapticObj.light();
  }) as HapticHandler;

  Object.assign(hapticFn, hapticObj);

  return (
    <ThemeContext.Provider
      value={{
        mode,
        setMode,
        tokens: LiquidGlassTokens,
        reduceTransparency,
        setReduceTransparency,
        reduceMotion,
        setReduceMotion,
        fontScale,
        setFontScale,
        isBlurSupported,
        haptic: hapticFn,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
};

const fallbackHaptic = (() => {}) as HapticHandler;
fallbackHaptic.light = () => {};
fallbackHaptic.medium = () => {};
fallbackHaptic.heavy = () => {};
fallbackHaptic.selection = () => {};
fallbackHaptic.notification = () => {};

export const useTheme = (): ThemeContextValue => {
  const context = useContext(ThemeContext);
  if (!context) {
    return {
      mode: 'dark',
      setMode: () => {},
      tokens: LiquidGlassTokens,
      reduceTransparency: false,
      setReduceTransparency: () => {},
      reduceMotion: false,
      setReduceMotion: () => {},
      fontScale: 1.0,
      setFontScale: () => {},
      isBlurSupported: Platform.OS === 'ios',
      haptic: fallbackHaptic,
    };
  }
  return context;
};

