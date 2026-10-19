# Database Architecture & Entity Specifications

## Dual Resilient Storage Strategy

The platform features an enterprise storage layer with dual-engine resilience:
1. **Primary Engine**: **PostgreSQL 16** via `postgresql+psycopg2`.
2. **Resilient Local Fallback Engine**: **SQLite 3** (`evacuation.db`) via native Python standard library.

If PostgreSQL is temporarily offline or Docker is unstarted during local evaluation, `backend/app/database/session.py` transparently shifts to SQLite without raising unhandled exceptions or disrupting running simulations.

---

## Entity Relational Schema

```mermaid
erDiagram
    USERS ||--o{ EVACUATION_SESSIONS : "conducts"
    USERS ||--o{ SCENARIOS : "creates"
    USERS ||--o{ AUDIT_LOGS : "generates"
    EVACUATION_SESSIONS ||--o{ AGENTS : "contains"
    EVACUATION_SESSIONS ||--o{ ROUTES : "generates"
    EVACUATION_SESSIONS ||--o{ HAZARDS : "records"
    EVACUATION_SESSIONS ||--o{ TRAFFIC_SNAPSHOTS : "records"
    ROUTES ||--o{ ROUTE_STEPS : "comprises"

    USERS {
        uuid id PK
        string name
        string email UK
        string password_hash
        string role "USER | RESEARCHER | ADMIN"
        timestamp created_at
        timestamp updated_at
    }

    EVACUATION_SESSIONS {
        uuid id PK
        uuid user_id FK
        uuid scenario_id FK
        string model "offline_to_online"
        string status "active | completed | stopped | error"
        timestamp started_at
        timestamp completed_at
        string origin
        string destination
        int agent_count
        float distance
        float travel_time
        float hazard_exposure
        float traffic_exposure
        float risk_score
        float inference_latency
        bool success
        bool fallback_used
        string fallback_reason
    }

    AGENTS {
        uuid id PK
        uuid session_id FK
        int agent_index
        string current_node
        string destination_node
        string status "evacuating | evacuated | trapped"
        json route
        float distance_travelled
        float time_elapsed
        float hazard_exposure
    }

    ROUTE_STEPS {
        uuid id PK
        uuid route_id FK
        int sequence
        string node_id
        float latitude
        float longitude
        float timestamp
        float hazard_level
        float traffic_level
    }
```
