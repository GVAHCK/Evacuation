"""
Comprehensive Security Test Matrix for Aegis Multi-Hazard Evacuation Platform.

Validates:
1. Authentication Security:
   - Invalid credentials rejection (401)
   - Missing Authorization header on protected route (401)
   - Tampered/invalid JWT signature (401)
   - Expired token handling (401)
2. Authorization & Tenant Isolation:
   - Cross-user session access prevention (403 Forbidden)
   - Cross-user WebSocket subscription rejection (Closed / Disconnected)
3. Input Validation & Injection Resistance:
   - SQL injection payload neutralization via parameterized SQLAlchemy queries
   - Negative hazard severity validation (422)
   - Excessive agent count rejection (422)
   - Malformed payload handling (422)
4. Rate Limiting & Abuse Prevention:
   - Rapid authentication flood triggers 429 Too Many Requests
5. CORS Origin Enforcement:
   - Explicit allowlist verification
"""

import pytest
import time
import jwt
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from backend.app.core.security import create_access_token
from backend.app.core.rate_limiter import auth_rate_limiter
from backend.app.database.session import SessionLocal
from backend.app.models.user import User
from backend.app.models.evacuation_session import EvacuationSession
import uuid

client = TestClient(app)


def test_auth_invalid_credentials():
    """Test that invalid credentials return HTTP 401 with normalized error schema."""
    res = client.post("/api/v1/auth/login", json={
        "email": "user@evacuation.ai",
        "password": "wrong_password_999"
    })
    assert res.status_code == 401
    data = res.json()
    assert data.get("code") == "INVALID_CREDENTIALS" or data.get("detail", {}).get("code") == "INVALID_CREDENTIALS"


def test_auth_missing_token_on_protected_route():
    """Test accessing protected /api/v1/users/me without a token returns 401."""
    res = client.get("/api/v1/users/me")
    assert res.status_code == 401
    data = res.json()
    assert data.get("code") == "AUTHENTICATION_REQUIRED" or data.get("detail", {}).get("code") == "AUTHENTICATION_REQUIRED"


def test_auth_tampered_jwt_signature():
    """Test that a JWT with an invalid signature is rejected."""
    forged_token = jwt.encode(
        {"sub": "fake-user-id", "role": "ADMIN", "type": "access", "exp": time.time() + 3600},
        "forged_secret_key_123456789012345678901234567890",
        algorithm="HS256"
    )
    res = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {forged_token}"})
    assert res.status_code == 401
    data = res.json()
    assert data.get("code") == "INVALID_TOKEN" or data.get("detail", {}).get("code") == "INVALID_TOKEN"


def test_auth_expired_jwt():
    """Test that an expired JWT token is rejected."""
    expired_token = jwt.encode(
        {"sub": "some-id", "role": "USER", "type": "access", "exp": time.time() - 3600},
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM
    )
    res = client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert res.status_code == 401
    data = res.json()
    assert data.get("code") == "INVALID_TOKEN" or data.get("detail", {}).get("code") == "INVALID_TOKEN"


def test_cross_user_session_isolation():
    """Test that User A cannot read or access User B's private evacuation session."""
    db = SessionLocal()
    try:
        user_a_id = f"user_a_{uuid.uuid4().hex[:8]}"
        user_b_id = f"user_b_{uuid.uuid4().hex[:8]}"

        # Insert User A and User B
        user_a = User(id=user_a_id, name="User A", email=f"{user_a_id}@test.ai", password_hash="dummy", role="USER")
        user_b = User(id=user_b_id, name="User B", email=f"{user_b_id}@test.ai", password_hash="dummy", role="USER")
        db.add(user_a)
        db.add(user_b)
        db.commit()

        # Create session owned by User B
        session_b = EvacuationSession(
            id=str(uuid.uuid4()),
            user_id=user_b_id,
            model="offline_online",
            status="active",
            origin="node_0",
            destination="node_335",
            agent_count=4,
            distance=4169.2,
            travel_time=135.0,
            hazard_exposure=0.1,
            traffic_exposure=0.2,
            risk_score=0.15,
            inference_latency=3.5,
            success=True,
            fallback_used=False
        )
        db.add(session_b)
        db.commit()

        # Token for User A (regular USER role)
        token_a = create_access_token({"sub": user_a_id, "role": "USER"})

        # User A attempts to access User B's session
        res = client.get(
            f"/api/v1/evacuation/{session_b.id}",
            headers={"Authorization": f"Bearer {token_a}"}
        )
        assert res.status_code == 403
        data = res.json()
        assert data.get("code") == "FORBIDDEN" or data.get("detail", {}).get("code") == "FORBIDDEN"
    finally:
        db.close()


def test_cross_user_websocket_rejection():
    """Test that an unauthorized user is rejected when connecting to another user's WebSocket."""
    db = SessionLocal()
    try:
        user_a_id = f"user_a_{uuid.uuid4().hex[:8]}"
        user_b_id = f"user_b_{uuid.uuid4().hex[:8]}"

        user_a = User(id=user_a_id, name="User A", email=f"{user_a_id}@test.ai", password_hash="dummy", role="USER")
        user_b = User(id=user_b_id, name="User B", email=f"{user_b_id}@test.ai", password_hash="dummy", role="USER")
        db.add(user_a)
        db.add(user_b)
        db.commit()

        # Create session owned by User B
        sess_id = str(uuid.uuid4())
        session_b = EvacuationSession(
            id=sess_id,
            user_id=user_b_id,
            model="offline_online",
            status="active",
            origin="node_0",
            destination="node_335",
            agent_count=4,
            distance=4169.2,
            travel_time=135.0,
            hazard_exposure=0.1,
            traffic_exposure=0.2,
            risk_score=0.15,
            inference_latency=3.5,
            success=True,
            fallback_used=False
        )
        db.add(session_b)
        db.commit()

        # Token for User A
        token_a = create_access_token({"sub": user_a_id, "role": "USER"})

        # Attempt WebSocket connection with User A's token -> should be closed/disconnected
        with pytest.raises(Exception):
            with client.websocket_connect(f"/ws/evacuation/{sess_id}?token={token_a}") as ws:
                ws.receive_json()
    finally:
        db.close()


def test_sql_injection_resilience():
    """Test SQL injection payloads in scenario filters and parameters."""
    sqli_payload = "'; DROP TABLE users; --"
    res = client.get(f"/api/v1/history?status={sqli_payload}")
    assert res.status_code == 200  # Returns empty list safely, ORM parameterizes query
    assert isinstance(res.json(), list)


def test_input_validation_negative_severity():
    """Test that negative hazard severity is rejected with HTTP 422."""
    res = client.post("/api/v1/routes/calculate", json={
        "origin": [13.002, 80.205],
        "destination": [13.048, 80.248],
        "hazard_config": {
            "hazard_type": "flood",
            "severity": -5.0  # Invalid negative severity
        }
    })
    assert res.status_code == 422


def test_input_validation_excessive_agent_count():
    """Test that excessively large agent cohort (e.g. 500 agents) is rejected with 422."""
    res = client.post("/api/v1/routes/calculate", json={
        "origin": [13.002, 80.205],
        "destination": [13.048, 80.248],
        "agent_count": 500  # Limit is 8
    })
    assert res.status_code == 422


def test_cors_origin_headers():
    """Test CORS headers for allowed and disallowed origins."""
    res_allowed = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:8081",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert res_allowed.headers.get("access-control-allow-origin") == "http://localhost:8081"

    res_disallowed = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://malicious-attacker-domain.xyz",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert res_disallowed.headers.get("access-control-allow-origin") != "http://malicious-attacker-domain.xyz"


def test_rate_limiter_flood_prevention():
    """Test that rapid authentication requests trigger rate limiter."""
    auth_rate_limiter.history.clear()

    # Rate limit for auth is 20 requests per minute
    # Send 25 requests in rapid succession
    status_codes = []
    for _ in range(25):
        r = client.post("/api/v1/auth/login", json={
            "email": "user@evacuation.ai",
            "password": "wrongpassword"
        })
        status_codes.append(r.status_code)

    assert 429 in status_codes
    auth_rate_limiter.history.clear()
