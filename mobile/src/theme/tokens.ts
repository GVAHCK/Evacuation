/**
 * Apple Liquid Glass-inspired centralized Design System tokens.
 * Crafted specifically for emergency response and multi-hazard routing interfaces.
 */

export const LiquidGlassTokens = {
  // Base Surface Backgrounds
  background: {
    primary: "#070B14",       // Deepest abyss obsidian
    secondary: "#0B1120",     // Midnight navy base
    tertiary: "#111A2E",      // Elevated dark surface
    canvas: "#04070F",        // Map / viewport background
  },

  // Glass Material System (Simulated Translucent Layering)
  glass: {
    // Ultra-thin surface for badges and small floating chips
    ultraThin: {
      backgroundColor: "rgba(255, 255, 255, 0.04)",
      borderColor: "rgba(255, 255, 255, 0.10)",
      borderTopColor: "rgba(255, 255, 255, 0.22)", // Specular top edge highlight
    },
    // Standard card surface for panels and information containers
    standard: {
      backgroundColor: "rgba(16, 24, 44, 0.72)",
      borderColor: "rgba(56, 189, 248, 0.18)",
      borderTopColor: "rgba(255, 255, 255, 0.28)", // Refraction highlight
    },
    // Strong modal / bottom sheet surface
    modal: {
      backgroundColor: "rgba(11, 17, 32, 0.90)",
      borderColor: "rgba(255, 255, 255, 0.16)",
      borderTopColor: "rgba(255, 255, 255, 0.35)",
    },
    // High-visibility emergency glass container
    emergency: {
      backgroundColor: "rgba(239, 68, 68, 0.14)",
      borderColor: "rgba(239, 68, 68, 0.35)",
      borderTopColor: "rgba(254, 202, 202, 0.50)",
    },
    // Verified safety glass container
    safety: {
      backgroundColor: "rgba(16, 185, 129, 0.14)",
      borderColor: "rgba(16, 185, 129, 0.35)",
      borderTopColor: "rgba(167, 243, 208, 0.50)",
    },
    // AI MARL specialized intelligence surface
    aiPill: {
      backgroundColor: "rgba(139, 92, 246, 0.16)",
      borderColor: "rgba(139, 92, 246, 0.38)",
      borderTopColor: "rgba(221, 214, 254, 0.55)",
    }
  },

  // High-Contrast Emergency & Operational Accents
  accent: {
    primary: "#00D2FF",       // Cool Cyan (Active route / Primary CTAs)
    primaryGlow: "rgba(0, 210, 255, 0.35)",
    secondary: "#38BDF8",     // Sky Blue
    danger: "#EF4444",        // Fire / Road Blockage / Imminent Danger
    dangerGlow: "rgba(239, 68, 68, 0.40)",
    warning: "#F59E0B",       // Congestion / Flood Warning / Caution
    warningGlow: "rgba(245, 158, 11, 0.35)",
    success: "#10B981",       // Evacuated / Safe Haven / Normal Flow
    successGlow: "rgba(16, 185, 129, 0.35)",
    ai: "#8B5CF6",            // MARL Offline-to-Online intelligence
    aiGlow: "rgba(139, 92, 246, 0.35)",
  },

  // Operational Severity Status Map
  status: {
    critical: "#EF4444",
    danger: "#EF4444",
    warning: "#F59E0B",
    completed: "#10B981",
    healthy: "#10B981",
    degraded: "#F59E0B",
    unknown: "#64748B",
  },

  // Typography
  text: {
    primary: "#F8FAFC",       // High-contrast crisp white
    secondary: "#94A3B8",     // Slate secondary
    muted: "#64748B",         // Dimmed assistive text
    inverse: "#020617",       // Dark text for light chips
    accent: "#38BDF8",        // Primary cyan text
    danger: "#F87171",        // Red alert text
    warning: "#FBBF24",       // Amber warning text
    success: "#34D399",       // Green confirmation text
  },

  // Geometric Curvature (Apple Continuity Corners)
  radius: {
    sm: 8,
    md: 14,
    lg: 20,
    xl: 26,
    pill: 9999,
  },

  // Spatial Shadows & Depth
  shadows: {
    subtle: {
      shadowColor: "#000000",
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.3,
      shadowRadius: 8,
      elevation: 4,
    },
    glassFloating: {
      shadowColor: "#00D2FF",
      shadowOffset: { width: 0, height: 8 },
      shadowOpacity: 0.18,
      shadowRadius: 18,
      elevation: 8,
    },
    modal: {
      shadowColor: "#000000",
      shadowOffset: { width: 0, height: 16 },
      shadowOpacity: 0.55,
      shadowRadius: 28,
      elevation: 16,
    }
  }
};

export const colors = {
  ...LiquidGlassTokens.background,
  ...LiquidGlassTokens.accent,
  ...LiquidGlassTokens.text,
  background: LiquidGlassTokens.background.primary,
  surface: LiquidGlassTokens.background.secondary,
  surfaceElevated: LiquidGlassTokens.background.tertiary,
  border: "rgba(255, 255, 255, 0.12)",
  card: LiquidGlassTokens.glass.standard.backgroundColor,
  cardHover: "rgba(26, 38, 64, 0.85)",
  textPrimary: LiquidGlassTokens.text.primary,
  textSecondary: LiquidGlassTokens.text.secondary,
  textMuted: LiquidGlassTokens.text.muted,
  trafficLow: LiquidGlassTokens.accent.success,
  trafficMedium: LiquidGlassTokens.accent.warning,
  trafficHigh: LiquidGlassTokens.accent.danger,
  trafficClosed: "#881337",
  flood: "#3B82F6",
  floodSurface: "rgba(59, 130, 246, 0.18)",
  fire: "#EF4444",
  fireSurface: "rgba(239, 68, 68, 0.20)",
  marlPill: LiquidGlassTokens.accent.ai,
  marlPillSurface: LiquidGlassTokens.glass.aiPill.backgroundColor,
};
