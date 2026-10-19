from fastapi import APIRouter, Depends
from backend.app.schemas.route import RouteCalculationRequest, RouteCalculationResponse
from backend.app.hazards.hazard_manager import HazardManager
from backend.app.routing.route_builder import RouteBuilder
from backend.app.routing.graph_manager import graph_manager
from backend.app.core.rate_limiter import compute_rate_limiter

router = APIRouter(prefix="/routes", tags=["Routing"])

@router.post("/calculate", response_model=RouteCalculationResponse, summary="Calculate multi-agent evacuation routes using Offline-to-Online MARL", dependencies=[Depends(compute_rate_limiter)])
def calculate_route(req: RouteCalculationRequest):
    """Calculates synchronized multi-agent evacuation routes using Offline-to-Online MARL.
    Automatically applies hard action masking, dynamic hazard states, and traffic congestion.
    Returns transparent numerical route decision factors.
    """
    # 1. Setup session hazard manager
    hazard_mgr = HazardManager()
    origin_node = graph_manager.resolve_to_node(req.origin)
    hazard_mgr.configure(req.hazard_config, origin_node=origin_node)

    # 2. Build routes with Offline-to-Online MARL
    route_builder = RouteBuilder(hazard_mgr)
    response = route_builder.calculate_routes(
        origin_node=origin_node,
        destination_node=graph_manager.resolve_to_node(req.destination),
        agent_count=req.agent_count,
        traffic_config=req.traffic_config
    )

    return response

@router.get("/graph-summary", summary="Get road network graph topology summary and sample nodes")
def get_graph_summary():
    return {
        "nodes_count": len(graph_manager.nodes),
        "edges_count": graph_manager.graph.number_of_edges() if graph_manager.graph else 0,
        "max_distance_m": round(graph_manager.max_distance, 1),
        "evacuation_centers": graph_manager.evacuation_centers,
        "sample_nodes": [
            {
                "id": str(n),
                "latitude": graph_manager.nodes[n]["lat"],
                "longitude": graph_manager.nodes[n]["lon"],
                "neighbors_count": len(list(graph_manager.graph.neighbors(n))) if graph_manager.graph and n in graph_manager.graph else 0
            }
            for n in list(graph_manager.nodes.keys())[:25]
        ]
    }
