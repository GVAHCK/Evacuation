import time
from typing import Dict, List, Tuple
from fastapi import Request, HTTPException, status
from backend.app.core.logging import logger

class RateLimiter:
    """In-memory sliding-window rate limiter for sensitive API endpoints."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        # client_ip -> list of epoch timestamps
        self.history: Dict[str, List[float]] = {}

    def __call__(self, request: Request):
        # Extract client IP; support forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else "127.0.0.1")

        now = time.time()
        window_start = now - 60.0

        timestamps = self.history.get(client_ip, [])
        # Prune older than 60 seconds
        valid_timestamps = [t for t in timestamps if t > window_start]

        if len(valid_timestamps) >= self.requests_per_minute:
            retry_after = int(60.0 - (now - valid_timestamps[0]))
            logger.warning(f"Rate limit exceeded for client {client_ip} on {request.url.path} (Limit: {self.requests_per_minute}/min)")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit of {self.requests_per_minute} requests per minute exceeded. Please retry after {max(1, retry_after)}s.",
                    "retry_after_seconds": max(1, retry_after)
                },
                headers={"Retry-After": str(max(1, retry_after))}
            )

        valid_timestamps.append(now)
        self.history[client_ip] = valid_timestamps

# Rate limiter instances for different route sensitivity levels
auth_rate_limiter = RateLimiter(requests_per_minute=20)       # 20 auth attempts / min
compute_rate_limiter = RateLimiter(requests_per_minute=30)    # 30 route calculations / min
general_rate_limiter = RateLimiter(requests_per_minute=120)   # 120 requests / min
