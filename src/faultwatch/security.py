"""Optional bearer access and per-identity token buckets for telemetry routes."""

import math
import os
from collections import OrderedDict
from threading import Lock
from time import monotonic, time

import jwt
from fastapi import HTTPException, Request

AUTH_ENV = "FAULTWATCH_JWT_SECRET"
RATE_ENV = "FAULTWATCH_RATE_PER_SECOND"
BURST_ENV = "FAULTWATCH_RATE_BURST"
ISSUER = "faultwatch"
AUDIENCE = "faultwatch-api"
ROLES = {"read": {"read"}, "ingest": {"ingest"}, "admin": {"read", "ingest"}}
MAX_TOKEN_SECONDS = 3600


def validate_secret(secret: str | None) -> None:
    if secret is not None and len(secret.encode()) < 32:
        raise ValueError(f"{AUTH_ENV} must contain at least 32 bytes")


def issue_token(secret: str, subject: str, role: str, lifetime: int = 900) -> str:
    validate_secret(secret)
    if not 1 <= len(subject) <= 128 or role not in ROLES or not 1 <= lifetime <= MAX_TOKEN_SECONDS:
        raise ValueError("Invalid token subject, role or lifetime")
    issued = int(time())
    return jwt.encode(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": subject,
            "role": role,
            "iat": issued,
            "exp": issued + lifetime,
        },
        secret,
        algorithm="HS256",
    )


class RateLimiter:
    def __init__(self, rate: float, burst: int, max_identities: int = 4096) -> None:
        if not math.isfinite(rate) or rate <= 0 or burst < 1 or max_identities < 1:
            raise ValueError("Rate, burst and identity capacity must be positive")
        self.rate = rate
        self.burst = burst
        self.max_identities = max_identities
        self.buckets: OrderedDict[tuple[str, str], tuple[float, float]] = OrderedDict()
        self.lock = Lock()

    def retry_after(self, key: tuple[str, str]) -> int | None:
        with self.lock:
            now = monotonic()
            previous = self.buckets.pop(key, None)
            if previous is None and len(self.buckets) == self.max_identities:
                oldest, (_, timestamp) = next(iter(self.buckets.items()))
                available_after = self.burst / self.rate - (now - timestamp)
                if available_after > 0:
                    return max(1, math.ceil(available_after))
                del self.buckets[oldest]
            tokens, timestamp = previous if previous is not None else (float(self.burst), now)
            tokens = min(self.burst, tokens + (now - timestamp) * self.rate)
            self.buckets[key] = (tokens - 1 if tokens >= 1 else tokens, now)
            return None if tokens >= 1 else max(1, math.ceil((1 - tokens) / self.rate))


class AccessPolicy:
    def __init__(self) -> None:
        self.secret = os.getenv(AUTH_ENV) or None
        validate_secret(self.secret)
        rate = os.getenv(RATE_ENV)
        self.limiter = RateLimiter(float(rate), int(os.getenv(BURST_ENV) or "30")) if rate else None

    async def check(self, request: Request) -> None:
        operation = "ingest" if request.method == "POST" else "read"
        subject = request.client.host if request.client else "unknown"
        if self.secret is not None:
            claims = self._claims(request)
            subject = claims["sub"]
            if operation not in ROLES[claims["role"]]:
                raise HTTPException(403, "This token lacks permission for the operation")
        if self.limiter is not None:
            retry = self.limiter.retry_after((subject, operation))
            if retry is not None:
                raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": str(retry)})

    def _claims(self, request: Request) -> dict:
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        try:
            if scheme.lower() != "bearer" or not 1 <= len(token) <= 4096:
                raise ValueError("Missing bearer token")
            claims = jwt.decode(
                token,
                self.secret,
                algorithms=["HS256"],
                issuer=ISSUER,
                audience=AUDIENCE,
                options={"require": ["sub", "role", "iat", "exp"], "strict_aud": True},
            )
            subject, role = claims["sub"], claims["role"]
            issued, expires = claims["iat"], claims["exp"]
            if (
                not isinstance(subject, str)
                or not 1 <= len(subject) <= 128
                or not isinstance(role, str)
                or role not in ROLES
                or type(issued) is not int
                or type(expires) is not int
                or not 1 <= expires - issued <= MAX_TOKEN_SECONDS
            ):
                raise ValueError("Invalid token claims")
        except (jwt.PyJWTError, ValueError, TypeError, OverflowError) as error:
            raise HTTPException(
                401, "Valid bearer token required", headers={"WWW-Authenticate": "Bearer"}
            ) from error
        return claims


class TelemetryBodyLimit:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/v1/telemetry":
            await self.app(scope, receive, send)
            return
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > 16_384:
                raise HTTPException(413, "Telemetry body exceeds 16 KiB")
            return message

        await self.app(scope, bounded_receive, send)
