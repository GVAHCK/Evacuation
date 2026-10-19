import React from "react";
import { View, Text, StyleSheet, Dimensions } from "react-native";
import Svg, { Circle, Polyline, Line, Rect, G, Text as SvgText } from "react-native-svg";
import { Colors } from "../theme/colors";
import { AgentLiveState, HazardLiveState, RouteLiveState } from "../types";

interface MapRendererProps {
  route?: [number, number][]; // [[lat, lon], ...]
  altRoute?: [number, number][];
  agents?: AgentLiveState[];
  hazards?: HazardLiveState[];
  origin?: [number, number];
  destination?: [number, number];
  destinationName?: string;
  height?: number;
}

export const MapRenderer: React.FC<MapRendererProps> = ({
  route = [],
  altRoute = [],
  agents = [],
  hazards = [],
  origin,
  destination,
  destinationName = "Central Evacuation Shelter",
  height = 360,
}) => {
  const { width } = Dimensions.get("window");
  const mapWidth = width - 32;

  // Compute spatial bounding box from all coordinates
  const allCoords: [number, number][] = [
    ...route,
    ...altRoute,
    ...(origin ? [origin] : []),
    ...(destination ? [destination] : []),
  ];

  let minLat = 12.98, maxLat = 13.06, minLon = 80.18, maxLon = 80.26;
  if (allCoords.length > 0) {
    const lats = allCoords.map((c) => c[0]);
    const lons = allCoords.map((c) => c[1]);
    minLat = Math.min(...lats);
    maxLat = Math.max(...lats);
    minLon = Math.min(...lons);
    maxLon = Math.max(...lons);
  }

  // Padding
  const padLat = Math.max((maxLat - minLat) * 0.15, 0.005);
  const padLon = Math.max((maxLon - minLon) * 0.15, 0.005);
  minLat -= padLat;
  maxLat += padLat;
  minLon -= padLon;
  maxLon += padLon;

  const latRange = Math.max(maxLat - minLat, 1e-5);
  const lonRange = Math.max(maxLon - minLon, 1e-5);

  const toX = (lon: number) => ((lon - minLon) / lonRange) * (mapWidth - 40) + 20;
  const toY = (lat: number) => (1.0 - (lat - minLat) / latRange) * (height - 40) + 20;

  const pointsToString = (pts: [number, number][]) =>
    pts.map(([lat, lon]) => `${toX(lon).toFixed(1)},${toY(lat).toFixed(1)}`).join(" ");

  return (
    <View style={[styles.container, { height }]}>
      <Svg width={mapWidth} height={height} style={styles.svg}>
        {/* Dark Tactical Map Grid Lines */}
        <G opacity={0.15}>
          {[0.2, 0.4, 0.6, 0.8].map((f, i) => (
            <Line key={`h-${i}`} x1={0} y1={height * f} x2={mapWidth} y2={height * f} stroke={Colors.borderLight} strokeWidth={1} />
          ))}
          {[0.2, 0.4, 0.6, 0.8].map((f, i) => (
            <Line key={`v-${i}`} x1={mapWidth * f} y1={0} x2={mapWidth * f} y2={height} stroke={Colors.borderLight} strokeWidth={1} />
          ))}
        </G>

        {/* Hazard Areas */}
        {hazards.map((h, idx) => {
          const c = h.center || [minLat + latRange * 0.5, minLon + lonRange * 0.5];
          const cx = toX(c[1]);
          const cy = toY(c[0]);
          const isFlood = h.type.toLowerCase().includes("flood");
          const color = isFlood ? Colors.flood : Colors.fire;
          const r = Math.max(25, (h.radius || 350) / 10);

          return (
            <G key={`haz-${idx}`}>
              <Circle cx={cx} cy={cy} r={r} fill={color} fillOpacity={0.2} stroke={color} strokeWidth={2} strokeDasharray="4 2" />
              <Circle cx={cx} cy={cy} r={r * 0.4} fill={color} fillOpacity={0.4} />
              <SvgText x={cx} y={cy - r - 4} fill={color} fontSize={10} fontWeight="bold" textAnchor="middle">
                {isFlood ? "FLOOD INUNDATION" : "WILDFIRE PERIMETER"}
              </SvgText>
            </G>
          );
        })}

        {/* Alternative Route Polyline */}
        {altRoute.length > 1 && (
          <Polyline points={pointsToString(altRoute)} fill="none" stroke="#64748B" strokeWidth={3} strokeDasharray="6 4" opacity={0.6} />
        )}

        {/* Recommended Offline-to-Online MARL Route */}
        {route.length > 1 && (
          <G>
            {/* Glowing outer trace */}
            <Polyline points={pointsToString(route)} fill="none" stroke={Colors.primary} strokeWidth={6} strokeOpacity={0.3} strokeLinecap="round" strokeLinejoin="round" />
            {/* Core route */}
            <Polyline points={pointsToString(route)} fill="none" stroke={Colors.primary} strokeWidth={3.5} strokeLinecap="round" strokeLinejoin="round" />
          </G>
        )}

        {/* Origin / User Marker */}
        {origin && (
          <G>
            <Circle cx={toX(origin[1])} cy={toY(origin[0])} r={8} fill={Colors.accent} />
            <Circle cx={toX(origin[1])} cy={toY(origin[0])} r={14} stroke={Colors.accent} strokeWidth={2} fill="none" opacity={0.7} />
          </G>
        )}

        {/* Destination / Evacuation Shelter Marker */}
        {destination && (
          <G>
            <Rect x={toX(destination[1]) - 10} y={toY(destination[0]) - 10} width={20} height={20} rx={4} fill={Colors.success} />
            <SvgText x={toX(destination[1])} y={toY(destination[0]) + 4} fill="#000" fontSize={11} fontWeight="bold" textAnchor="middle">
              H
            </SvgText>
            <SvgText x={toX(destination[1])} y={toY(destination[0]) + 22} fill={Colors.success} fontSize={10} fontWeight="bold" textAnchor="middle">
              SHELTER
            </SvgText>
          </G>
        )}

        {/* Multi-Agent Markers */}
        {agents.map((ag) => {
          const ax = toX(ag.longitude);
          const ay = toY(ag.latitude);
          const isEvacuated = ag.status === "evacuated";
          const agColor = isEvacuated ? Colors.success : ag.id === 0 ? Colors.primary : Colors.warning;

          return (
            <G key={`agent-${ag.id}`}>
              <Circle cx={ax} cy={ay} r={9} fill={agColor} stroke="#FFF" strokeWidth={1.5} />
              <SvgText x={ax} y={ay + 3} fill="#000" fontSize={9} fontWeight="bold" textAnchor="middle">
                {ag.id}
              </SvgText>
            </G>
          );
        })}
      </Svg>

      {/* Floating Status Badges */}
      <View style={styles.legendContainer}>
        <View style={styles.legendItem}>
          <View style={[styles.dot, { backgroundColor: Colors.primary }]} />
          <Text style={styles.legendText}>Offline-to-Online MARL</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.dot, { backgroundColor: Colors.success }]} />
          <Text style={styles.legendText}>Evacuation Center</Text>
        </View>
        {hazards.length > 0 && (
          <View style={styles.legendItem}>
            <View style={[styles.dot, { backgroundColor: Colors.fire }]} />
            <Text style={styles.legendText}>Hazard Zone</Text>
          </View>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: "#0A101D",
    borderRadius: 16,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: Colors.border,
    position: "relative",
  },
  svg: {
    width: "100%",
    height: "100%",
  },
  legendContainer: {
    position: "absolute",
    bottom: 8,
    left: 8,
    right: 8,
    flexDirection: "row",
    flexWrap: "wrap",
    backgroundColor: "rgba(10, 16, 29, 0.88)",
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderRadius: 8,
    gap: 12,
  },
  legendItem: {
    flexDirection: "row",
    alignItems: "center",
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  legendText: {
    color: Colors.textSecondary,
    fontSize: 10,
    fontWeight: "500",
  },
});
