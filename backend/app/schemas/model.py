from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ModelMetadataResponse(BaseModel):
    model_name: str
    model_family: str
    status: str  # online, ready, degraded, unavailable
    checkpoint_path: str
    checkpoint_exists: bool
    checkpoint_size_kb: float
    device: str
    n_agents: int
    obs_dim: int
    act_dim: int
    parameter_count: int
    action_masking_supported: bool
    warmup_completed: bool

class BenchmarkMetrics(BaseModel):
    joint_success_rate: float
    joint_success_std: float
    individual_success_rate: float
    mean_evacuation_time: float
    mean_evacuation_time_std: float
    hei_hazard_exposure: float
    hei_std: float
    routing_f1: float
    inference_latency_ms: float
    parameter_count: int
    evaluation_network: str
    scenarios_evaluated: int

class ResearchBenchmarkResponse(BaseModel):
    model_name: str
    benchmark_metrics: BenchmarkMetrics
    benchmark_summary_small: Optional[Dict[str, Any]] = None
    benchmark_summary_large: Optional[Dict[str, Any]] = None
    disclaimer: str = "These metrics reflect the completed 500-scenario peer-reviewed benchmark evaluation and are distinct from live real-time simulation session results."
