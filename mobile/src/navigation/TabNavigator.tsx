import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { View, StyleSheet, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LiquidGlassTokens } from '../theme/tokens';

import { HomeScreen } from '../screens/HomeScreen';
import { LiveMapScreen } from '../screens/LiveMapScreen';
import { StartEvacuationScreen } from '../screens/StartEvacuationScreen';
import { HistoryScreen } from '../screens/HistoryScreen';
import { ResearchScreen } from '../screens/ResearchScreen';

const Tab = createBottomTabNavigator();

export const TabNavigator: React.FC = () => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: styles.tabBar,
        tabBarActiveTintColor: LiquidGlassTokens.accent.primary,
        tabBarInactiveTintColor: LiquidGlassTokens.text.muted,
        tabBarLabelStyle: styles.tabBarLabel,
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: keyof typeof Ionicons.glyphMap = 'navigate-outline';

          if (route.name === 'Home') {
            iconName = focused ? 'home' : 'home-outline';
          } else if (route.name === 'LiveMap') {
            iconName = focused ? 'map' : 'map-outline';
          } else if (route.name === 'Evacuate') {
            iconName = focused ? 'navigate-circle' : 'navigate-circle-outline';
          } else if (route.name === 'History') {
            iconName = focused ? 'time' : 'time-outline';
          } else if (route.name === 'Research') {
            iconName = focused ? 'flask' : 'flask-outline';
          }

          return (
            <View style={[styles.iconWrapper, focused && styles.activeIconWrapper]}>
              <Ionicons name={iconName} size={focused ? 23 : 21} color={color} />
            </View>
          );
        },
      })}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          tabBarLabel: 'Dashboard',
          tabBarAccessibilityLabel: 'Dashboard screen',
        }}
      />
      <Tab.Screen
        name="LiveMap"
        component={LiveMapScreen}
        options={{
          tabBarLabel: 'Network',
          tabBarAccessibilityLabel: 'Road network map',
        }}
      />
      <Tab.Screen
        name="Evacuate"
        component={StartEvacuationScreen}
        options={{
          tabBarLabel: 'Evacuate',
          tabBarAccessibilityLabel: 'Start evacuation routing',
        }}
      />
      <Tab.Screen
        name="History"
        component={HistoryScreen}
        options={{
          tabBarLabel: 'History',
          tabBarAccessibilityLabel: 'Evacuation session history',
        }}
      />
      <Tab.Screen
        name="Research"
        component={ResearchScreen}
        options={{
          tabBarLabel: 'MARL AI',
          tabBarAccessibilityLabel: 'Research benchmark and model analytics',
        }}
      />
    </Tab.Navigator>
  );
};

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: 'rgba(11, 17, 32, 0.94)',
    borderTopColor: 'rgba(255, 255, 255, 0.16)',
    borderTopWidth: 1,
    height: Platform.OS === 'ios' ? 86 : 68,
    paddingBottom: Platform.OS === 'ios' ? 24 : 10,
    paddingTop: 8,
    ...LiquidGlassTokens.shadows.subtle,
  },
  tabBarLabel: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.2,
    marginTop: 2,
  },
  iconWrapper: {
    alignItems: 'center',
    justifyContent: 'center',
    width: 38,
    height: 30,
    borderRadius: 15,
  },
  activeIconWrapper: {
    backgroundColor: 'rgba(0, 210, 255, 0.12)',
  },
});

