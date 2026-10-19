# Production Deployment & Containerization Guide

## Docker Compose Orchestration

The platform provides a production-ready `docker-compose.yml` deploying PostgreSQL and FastAPI.

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: evac-postgres
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: evacuation_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  fastapi:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: evac-fastapi
    restart: always
    environment:
      - DATABASE_URL=postgresql+psycopg2://postgres:postgres@postgres:5432/evacuation_db
      - JWT_SECRET=super-secret-marl-evacuation-key-2026-secure-token
      - MODEL_CHECKPOINT=models/marl_six/offline_online/seed_42/best_model.pt
      - MODEL_DEVICE=cpu
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
```

---

## Launching Services

```bash
docker compose up -d --build
```

To view logs:
```bash
docker compose logs -f fastapi
```

To stop all services:
```bash
docker compose down
```
