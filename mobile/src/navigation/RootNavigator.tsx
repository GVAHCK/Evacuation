import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

// Core Flow
import { Splash } from '../screens/Splash';
import { TabNavigator } from './TabNavigator';
import { StartEvacuationScreen } from '../screens/StartEvacuationScreen';

// Group A: Authentication & Onboarding (Screens 1–4)
import { OnboardingScreen } from '../screens/OnboardingScreen'; // Screen 1
import { Login } from '../screens/Login'; // Screen 2
import { Register } from '../screens/Register'; // Screen 3
import { AccountRecoveryScreen } from '../screens/AccountRecoveryScreen'; // Screen 4

// Group B: Main Navigation & System Status (Screens 5–8)
import { HomeScreen } from '../screens/HomeScreen'; // Screen 5 (Dashboard)
import { SystemHealthScreen } from '../screens/SystemHealthScreen'; // Screen 6
import { NotificationInboxScreen } from '../screens/NotificationInboxScreen'; // Screen 7
import { ProfileScreen } from '../screens/ProfileScreen'; // Screen 8

// Group C: Geospatial & Hazard Intelligence (Screens 9–14)
import { LiveMapScreen } from '../screens/LiveMapScreen'; // Screen 9
import { HazardLayerControlsScreen } from '../screens/HazardLayerControlsScreen'; // Screen 10
import { FloodIntelligenceScreen } from '../screens/FloodIntelligenceScreen'; // Screen 11
import { FireIntelligenceScreen } from '../screens/FireIntelligenceScreen'; // Screen 12
import { RoadNetworkExplorerScreen } from '../screens/RoadNetworkExplorerScreen'; // Screen 13
import { LocationRouteInspectorScreen } from '../screens/LocationRouteInspectorScreen'; // Screen 14

// Group D: Scenario Configuration (Screens 15–20)
import { ScenarioOverviewScreen } from '../screens/ScenarioOverviewScreen'; // Screen 15
import { ScenarioBuilderScreen } from '../screens/ScenarioBuilderScreen'; // Screen 16
import { AgentConfigurationScreen } from '../screens/AgentConfigurationScreen'; // Screen 17
import { HazardConfigurationScreen } from '../screens/HazardConfigurationScreen'; // Screen 18
import { TrafficConstraintsScreen } from '../screens/TrafficConstraintsScreen'; // Screen 19
import { EvacuationPreviewScreen } from '../screens/EvacuationPreviewScreen'; // Screen 20

// Group E: Real-time Operations (Screens 21–27)
import { LiveEvacuationScreen } from '../screens/LiveEvacuationScreen'; // Screen 21
import { LiveAgentTrackingScreen } from '../screens/LiveAgentTrackingScreen'; // Screen 22
import { AgentDetailScreen } from '../screens/AgentDetailScreen'; // Screen 23
import { RouteDetailScreen } from '../screens/RouteDetailScreen'; // Screen 24
import { ReroutingEventsScreen } from '../screens/ReroutingEventsScreen'; // Screen 25
import { EvacuationControlsScreen } from '../screens/EvacuationControlsScreen'; // Screen 26
import { EvacuationResultScreen } from '../screens/EvacuationResultScreen'; // Screen 27

// Group F: Routing Algorithms & Model Intelligence (Screens 28–33)
import { ModelRegistryScreen } from '../screens/ModelRegistryScreen'; // Screen 28
import { MarlModelDetailsScreen } from '../screens/MarlModelDetailsScreen'; // Screen 29
import { AlgorithmComparisonScreen } from '../screens/AlgorithmComparisonScreen'; // Screen 30
import { ResearchScreen } from '../screens/ResearchScreen'; // Screen 31
import { AlgorithmPerformanceScreen } from '../screens/AlgorithmPerformanceScreen'; // Screen 32
import { ModelDiagnosticsScreen } from '../screens/ModelDiagnosticsScreen'; // Screen 33

// Group G: Analytics & Historical Evidence (Screens 34–38)
import { HistoryScreen } from '../screens/HistoryScreen'; // Screen 34
import { SessionDetailsScreen } from '../screens/SessionDetailsScreen'; // Screen 35
import { PerformanceAnalyticsScreen } from '../screens/PerformanceAnalyticsScreen'; // Screen 36
import { HazardSafetyAnalyticsScreen } from '../screens/HazardSafetyAnalyticsScreen'; // Screen 37
import { ResearchReportsScreen } from '../screens/ResearchReportsScreen'; // Screen 38

// Group H: Reliability, Diagnostics & Settings (Screens 39–42)
import { ConnectionDiagnosticsScreen } from '../screens/ConnectionDiagnosticsScreen'; // Screen 39
import { SimulationDiagnosticsScreen } from '../screens/SimulationDiagnosticsScreen'; // Screen 40
import { AccessibilitySettingsScreen } from '../screens/AccessibilitySettingsScreen'; // Screen 41
import { SettingsScreen } from '../screens/SettingsScreen'; // Screen 41 fallback
import { AboutResearchScreen } from '../screens/AboutResearchScreen'; // Screen 42

const Stack = createNativeStackNavigator();

export const RootNavigator: React.FC = () => {
  return (
    <Stack.Navigator
      initialRouteName="Splash"
      screenOptions={{
        headerShown: false,
        animation: 'fade',
      }}
    >
      {/* Root Entry & Tab Shell */}
      <Stack.Screen name="Splash" component={Splash} />
      <Stack.Screen name="Main" component={TabNavigator} />
      <Stack.Screen name="MainTabs" component={TabNavigator} />
      <Stack.Screen name="StartEvacuation" component={StartEvacuationScreen} />

      {/* Group A: Authentication and onboarding (Screens 1–4) */}
      <Stack.Screen name="Onboarding" component={OnboardingScreen} />
      <Stack.Screen name="Login" component={Login} />
      <Stack.Screen name="Register" component={Register} />
      <Stack.Screen name="AccountRecovery" component={AccountRecoveryScreen} />

      {/* Group B: Main navigation and system status (Screens 5–8) */}
      <Stack.Screen name="Home" component={HomeScreen} />
      <Stack.Screen name="SystemHealth" component={SystemHealthScreen} />
      <Stack.Screen name="NotificationInbox" component={NotificationInboxScreen} />
      <Stack.Screen name="Profile" component={ProfileScreen} />

      {/* Group C: Geospatial and hazard intelligence (Screens 9–14) */}
      <Stack.Screen name="LiveMap" component={LiveMapScreen} />
      <Stack.Screen name="HazardLayerControls" component={HazardLayerControlsScreen} />
      <Stack.Screen name="FloodIntelligence" component={FloodIntelligenceScreen} />
      <Stack.Screen name="FireIntelligence" component={FireIntelligenceScreen} />
      <Stack.Screen name="RoadNetworkExplorer" component={RoadNetworkExplorerScreen} />
      <Stack.Screen name="LocationRouteInspector" component={LocationRouteInspectorScreen} />

      {/* Group D: Scenario configuration (Screens 15–20) */}
      <Stack.Screen name="ScenarioOverview" component={ScenarioOverviewScreen} />
      <Stack.Screen name="ScenarioBuilder" component={ScenarioBuilderScreen} />
      <Stack.Screen name="AgentConfiguration" component={AgentConfigurationScreen} />
      <Stack.Screen name="HazardConfiguration" component={HazardConfigurationScreen} />
      <Stack.Screen name="TrafficConstraints" component={TrafficConstraintsScreen} />
      <Stack.Screen name="EvacuationPreview" component={EvacuationPreviewScreen} />

      {/* Group E: Evacuation execution and real-time operations (Screens 21–27) */}
      <Stack.Screen name="LiveEvacuation" component={LiveEvacuationScreen} />
      <Stack.Screen name="LiveAgentTracking" component={LiveAgentTrackingScreen} />
      <Stack.Screen name="AgentDetail" component={AgentDetailScreen} />
      <Stack.Screen name="RouteDetail" component={RouteDetailScreen} />
      <Stack.Screen name="ReroutingEvents" component={ReroutingEventsScreen} />
      <Stack.Screen name="EvacuationControls" component={EvacuationControlsScreen} />
      <Stack.Screen name="EvacuationResult" component={EvacuationResultScreen} />

      {/* Group F: Routing algorithms and model intelligence (Screens 28–33) */}
      <Stack.Screen name="ModelRegistry" component={ModelRegistryScreen} />
      <Stack.Screen name="MarlModelDetails" component={MarlModelDetailsScreen} />
      <Stack.Screen name="AlgorithmComparison" component={AlgorithmComparisonScreen} />
      <Stack.Screen name="Research" component={ResearchScreen} />
      <Stack.Screen name="AlgorithmPerformance" component={AlgorithmPerformanceScreen} />
      <Stack.Screen name="ModelDiagnostics" component={ModelDiagnosticsScreen} />

      {/* Group G: Analytics and historical evidence (Screens 34–38) */}
      <Stack.Screen name="History" component={HistoryScreen} />
      <Stack.Screen name="SessionDetails" component={SessionDetailsScreen} />
      <Stack.Screen name="PerformanceAnalytics" component={PerformanceAnalyticsScreen} />
      <Stack.Screen name="HazardSafetyAnalytics" component={HazardSafetyAnalyticsScreen} />
      <Stack.Screen name="ResearchReports" component={ResearchReportsScreen} />

      {/* Group H: Reliability, diagnostics, and settings (Screens 39–42) */}
      <Stack.Screen name="ConnectionDiagnostics" component={ConnectionDiagnosticsScreen} />
      <Stack.Screen name="SimulationDiagnostics" component={SimulationDiagnosticsScreen} />
      <Stack.Screen name="AccessibilitySettings" component={AccessibilitySettingsScreen} />
      <Stack.Screen name="Settings" component={SettingsScreen} />
      <Stack.Screen name="AboutResearch" component={AboutResearchScreen} />
    </Stack.Navigator>
  );
};
