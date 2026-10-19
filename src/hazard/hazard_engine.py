"""Abstract Base Class for Hazard Simulation Engines."""

from abc import ABC, abstractmethod
from typing import Dict, Set, Tuple, Any, Optional
import networkx as nx


class HazardEngine(ABC):
    """Unified interface for dynamic hazard propagation (Flood and Fire)

    affecting road networks.
    """

    STATE_SAFE = 0
    STATE_DEGRADED = 1
    STATE_CLOSED = 2

    def __init__(self, graph: nx.DiGraph, config: Dict[str, Any]):
        self.graph = graph
        self.config = config
        self.current_time = 0.0
        self.node_intensities: Dict[str, float] = {}
        self.edge_intensities: Dict[str, float] = {}
        self.edge_states: Dict[str, int] = {}
        self.closed_edges: Set[str] = set()
        self.degraded_edges: Set[str] = set()

    @abstractmethod
    def reset(
        self,
        seed: int = 42,
        origin_node: Optional[str] = None,
        spread_rate_mult: float = 1.0,
        **kwargs
    ) -> None:
        """Resets hazard state with new initial conditions."""
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        """Advances hazard propagation by dt seconds."""
        pass

    def get_intensity(self, node_or_edge_id: str) -> float:
        """Returns hazard intensity in range [0.0, 1.0]."""
        if node_or_edge_id in self.node_intensities:
            return self.node_intensities[node_or_edge_id]
        return self.edge_intensities.get(node_or_edge_id, 0.0)

    def get_edge_status(self, edge_id: str) -> int:
        """Returns 0 (safe), 1 (degraded), or 2 (closed)."""
        return self.edge_states.get(edge_id, self.STATE_SAFE)

    def get_closed_edges(self) -> Set[str]:
        """Returns set of all currently closed edge IDs."""
        return set(self.closed_edges)

    def get_degraded_edges(self) -> Set[str]:
        """Returns set of all currently degraded edge IDs."""
        return set(self.degraded_edges)

    def to_dict(self) -> Dict[str, Any]:
        """Returns serializable snapshot of hazard state."""
        return {
            "time": self.current_time,
            "closed_count": len(self.closed_edges),
            "degraded_count": len(self.degraded_edges),
            "mean_intensity": (
                sum(self.edge_intensities.values()) / max(len(self.edge_intensities), 1)
            )
        }
