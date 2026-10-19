import math
from typing import Dict, Any, List, Set, Optional, Tuple
import networkx as nx

from backend.app.core.logging import logger
from backend.app.routing.graph_manager import graph_manager
from backend.app.schemas.hazard import HazardConfig, HazardArea
from src.hazard.flood_engine import FloodEngine
from src.hazard.fire_engine import FireEngine
from src.hazard.hazard_engine import HazardEngine

class HazardManager:
    """Production Hazard Manager coordinating flood, wildfire, and multi-hazard simulation."""

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.graph = graph if graph is not None else graph_manager.graph
        self.config = {
            "hazard": {
                "flood": {
                    "spread_rate": 0.08,
                    "propagation_prob": 0.65,
                    "elevation_sensitivity": 0.5,
                    "degradation_threshold": 0.35,
                    "closure_threshold": 0.70
                },
                "fire": {
                    "spread_rate": 0.12,
                    "wind_speed": 5.0,
                    "wind_direction": 45.0,
                    "humidity": 0.30,
                    "degradation_threshold": 0.30,
                    "closure_threshold": 0.60
                }
            }
        }

        self.flood_engine = FloodEngine(self.graph, self.config)
        self.fire_engine = FireEngine(self.graph, self.config)

        self.current_hazard_type = "normal"
        self.active_engine: Optional[HazardEngine] = None
        self.severity: float = 1.0
        self.center_lat: Optional[float] = None
        self.center_lon: Optional[float] = None
        self.radius_meters: float = 350.0

    def configure(
        self,
        hazard_config: Optional[HazardConfig] = None,
        seed: int = 42,
        origin_node: Optional[str] = None
    ):
        """Initializes or resets hazard state with given configuration."""
        if not hazard_config:
            self.current_hazard_type = "normal"
            self.active_engine = None
            return

        self.current_hazard_type = hazard_config.hazard_type.lower()
        self.severity = hazard_config.severity
        self.center_lat = hazard_config.center_lat
        self.center_lon = hazard_config.center_lon
        self.radius_meters = hazard_config.radius_meters

        # If custom center lat/lon provided, resolve origin node
        resolved_origin = origin_node
        if self.center_lat is not None and self.center_lon is not None:
            resolved_origin = graph_manager.resolve_to_node([self.center_lat, self.center_lon])

        if self.current_hazard_type == "flood":
            self.flood_engine.reset(seed=seed, origin_node=resolved_origin, spread_rate_mult=self.severity)
            self.active_engine = self.flood_engine
        elif self.current_hazard_type == "fire":
            self.fire_engine.reset(seed=seed, origin_node=resolved_origin, spread_rate_mult=self.severity)
            self.active_engine = self.fire_engine
        elif self.current_hazard_type in ("combined", "flood+fire", "flood_fire"):
            self.flood_engine.reset(seed=seed, origin_node=resolved_origin, spread_rate_mult=self.severity)
            self.fire_engine.reset(seed=seed + 1, origin_node=resolved_origin, spread_rate_mult=self.severity)
            self.active_engine = None  # Handled combined in getters
        else:
            self.current_hazard_type = "normal"
            self.active_engine = None

        logger.info(f"HazardManager configured for '{self.current_hazard_type}' (Severity: {self.severity:.2f}, Origin: {resolved_origin})")

    def step(self, dt: float = 1.0):
        """Advances physical hazard propagation by dt seconds."""
        if self.current_hazard_type == "flood" and self.flood_engine:
            self.flood_engine.update(dt)
        elif self.current_hazard_type == "fire" and self.fire_engine:
            self.fire_engine.update(dt)
        elif self.current_hazard_type in ("combined", "flood+fire", "flood_fire"):
            self.flood_engine.update(dt)
            self.fire_engine.update(dt)

    def get_intensity(self, node_or_edge_id: str) -> float:
        """Returns normalized hazard intensity [0.0, 1.0]."""
        if self.current_hazard_type == "normal":
            return 0.0
        if self.current_hazard_type == "flood":
            return self.flood_engine.get_intensity(node_or_edge_id)
        if self.current_hazard_type == "fire":
            return self.fire_engine.get_intensity(node_or_edge_id)
        if self.current_hazard_type in ("combined", "flood+fire", "flood_fire"):
            return max(
                self.flood_engine.get_intensity(node_or_edge_id),
                self.fire_engine.get_intensity(node_or_edge_id)
            )
        return 0.0

    def get_edge_status(self, edge_id: str) -> int:
        """Returns 0 (safe), 1 (degraded), or 2 (closed)."""
        if self.current_hazard_type == "normal":
            return HazardEngine.STATE_SAFE
        if self.current_hazard_type == "flood":
            return self.flood_engine.get_edge_status(edge_id)
        if self.current_hazard_type == "fire":
            return self.fire_engine.get_edge_status(edge_id)
        if self.current_hazard_type in ("combined", "flood+fire", "flood_fire"):
            s1 = self.flood_engine.get_edge_status(edge_id)
            s2 = self.fire_engine.get_edge_status(edge_id)
            return max(s1, s2)
        return HazardEngine.STATE_SAFE

    def get_closed_edges(self) -> Set[str]:
        if self.current_hazard_type == "normal":
            return set()
        if self.current_hazard_type == "flood":
            return self.flood_engine.get_closed_edges()
        if self.current_hazard_type == "fire":
            return self.fire_engine.get_closed_edges()
        if self.current_hazard_type in ("combined", "flood+fire", "flood_fire"):
            return self.flood_engine.get_closed_edges() | self.fire_engine.get_closed_edges()
        return set()

    def get_degraded_edges(self) -> Set[str]:
        if self.current_hazard_type == "normal":
            return set()
        if self.current_hazard_type == "flood":
            return self.flood_engine.get_degraded_edges()
        if self.current_hazard_type == "fire":
            return self.fire_engine.get_degraded_edges()
        if self.current_hazard_type in ("combined", "flood+fire", "flood_fire"):
            return self.flood_engine.get_degraded_edges() | self.fire_engine.get_degraded_edges()
        return set()

    def get_active_hazard_areas(self) -> List[HazardArea]:
        """Constructs spatial hazard areas with centers and radii for mobile map rendering."""
        if self.current_hazard_type == "normal":
            return []

        areas = []
        if self.current_hazard_type in ("flood", "combined", "flood+fire", "flood_fire"):
            origin = self.flood_engine.origin_node or graph_manager.nodes[0]
            lat, lon = graph_manager.get_node_coordinates(origin)
            affected_nodes = [n for n, intensity in self.flood_engine.node_intensities.items() if intensity > 0.2]
            areas.append(HazardArea(
                type="flood",
                severity=self.severity,
                center=[lat, lon],
                radius=self.radius_meters,
                affected_nodes=affected_nodes,
                affected_edges=list(self.flood_engine.get_closed_edges() | self.flood_engine.get_degraded_edges())
            ))

        if self.current_hazard_type in ("fire", "combined", "flood+fire", "flood_fire"):
            origin = self.fire_engine.origin_node or graph_manager.nodes[-1]
            lat, lon = graph_manager.get_node_coordinates(origin)
            affected_nodes = [n for n, intensity in self.fire_engine.node_intensities.items() if intensity > 0.2]
            areas.append(HazardArea(
                type="fire",
                severity=self.severity,
                center=[lat, lon],
                radius=self.radius_meters * 0.85,
                affected_nodes=affected_nodes,
                affected_edges=list(self.fire_engine.get_closed_edges() | self.fire_engine.get_degraded_edges())
            ))

        return areas
