import os
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Multi-Hazard Evacuation Intelligence Platform"
    VERSION: str = "2.0.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/evacuation_db"
    )
    SQLITE_FALLBACK_URL: str = "sqlite:///./evacuation.db"

    # Security & JWT
    JWT_SECRET: str = os.getenv("JWT_SECRET", "super-secret-marl-evacuation-key-2026-secure-token")
    JWT_REFRESH_SECRET: str = os.getenv("JWT_REFRESH_SECRET", "super-secret-refresh-marl-key-2026-secure-token")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Machine Learning & MARL
    MODEL_NAME: str = "offline_to_online"
    MODEL_CHECKPOINT: str = os.getenv(
        "MODEL_CHECKPOINT",
        os.path.join("models", "marl_six", "offline_online", "seed_42", "best_model.pt")
    )
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "cuda")  # Fallbacks to cpu automatically if cuda unavailable

    # Spatial Road Graph
    OSM_GRAPH_PATH: str = os.getenv(
        "OSM_GRAPH_PATH",
        os.path.join("data", "processed", "road_graph_large.json")
    )
    OSM_GRAPH_FALLBACK_PATH: str = os.path.join("data", "processed", "road_graph.json")

    # Simulation & SUMO
    SUMO_HOME: str = os.getenv("SUMO_HOME", "")
    SUMO_BINARY: str = os.getenv("SUMO_BINARY", "sumo")

    # CORS
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8081",
        "http://localhost:19000",
        "http://localhost:19006",
        "http://10.0.2.2:8000",
        "http://10.0.2.2:8081",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8081",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    parsed = json.loads(v)
                    return [o for o in parsed if o != "*"]
                except Exception:
                    pass
            origins = [origin.strip() for origin in v.split(",") if origin.strip() and origin.strip() != "*"]
            if not origins:
                origins = [
                    "http://localhost:8081",
                    "http://localhost:19000",
                    "http://localhost:3000",
                    "http://10.0.2.2:8081"
                ]
            return origins
        if isinstance(v, list):
            return [o for o in v if o != "*"]
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

settings = Settings()

