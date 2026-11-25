import json
from time import time

import jwt
import pytest
from fastapi.testclient import TestClient

from faultwatch.api import create_app
from faultwatch.security import AUDIENCE, ISSUER, RateLimiter, issue_token


@pytest.fixture
def signing_secret(monkeypatch):
    secret = "test fixture " * 4
    monkeypatch.setenv("FAULTWATCH_JWT_SECRET", secret)
    return secret


def test_bad_credentials_and_roles_fail_before_database(tmp_path, signing_secret):
    now = int(time())
    standard = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "device",
        "role": "ingest",
        "iat": now - 700,
        "exp": now + 200,
    }
    invalid = [
        jwt.encode(standard | {"exp": now - 1}, signing_secret, algorithm="HS256"),
        jwt.encode(standard | {"aud": "different-api"}, signing_secret, algorithm="HS256"),
        jwt.encode(standard | {"role": []}, signing_secret, algorithm="HS256"),
        jwt.encode(standard, signing_secret, algorithm="HS384"),
        jwt.encode(standard, "a different fixture signing key" * 3, algorithm="HS256"),
    ]
    with TestClient(create_app(model_dir=tmp_path)) as client:
        unauthenticated = client.get("/v1/engines")
        assert unauthenticated.status_code == 401
        assert unauthenticated.headers["WWW-Authenticate"] == "Bearer"
        for token in invalid:
            response = client.get("/v1/engines", headers={"Authorization": f"Bearer {token}"})
            assert response.status_code == 401
        writer = issue_token(signing_secret, "device", "ingest")
        response = client.get("/v1/engines", headers={"Authorization": f"Bearer {writer}"})
        assert response.status_code == 403
        assert client.get("/healthz").status_code == 200
        assert client.get("/").status_code == 200


def test_rate_limit_is_per_subject_and_has_retry_header(tmp_path, signing_secret, monkeypatch):
    monkeypatch.setenv("FAULTWATCH_RATE_PER_SECOND", "0.01")
    monkeypatch.setenv("FAULTWATCH_RATE_BURST", "1")
    first = issue_token(signing_secret, "dashboard-a", "read")
    second = issue_token(signing_secret, "dashboard-b", "read")
    with TestClient(create_app(model_dir=tmp_path)) as client:
        header = {"Authorization": f"Bearer {first}"}
        assert client.get("/v1/engines", headers=header).status_code == 503
        response = client.get("/v1/engines", headers=header)
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) > 0
        assert not response.headers.get("WWW-Authenticate")
        assert (
            client.get("/v1/engines", headers={"Authorization": f"Bearer {second}"}).status_code
            == 503
        )


def test_rate_limiter_does_not_evict_an_active_bucket(monkeypatch):
    clock = [1.0]
    monkeypatch.setattr("faultwatch.security.monotonic", lambda: clock[0])
    limiter = RateLimiter(2, 2, max_identities=1)
    user, newcomer = ("reader", "read"), ("new", "read")
    assert limiter.retry_after(user) is None
    assert limiter.retry_after(user) is None
    assert limiter.retry_after(newcomer) == 1
    assert limiter.retry_after(user) == 1
    clock[0] += 1
    assert limiter.retry_after(newcomer) is None


def test_telemetry_request_size_is_bounded(tmp_path):
    with TestClient(create_app(model_dir=tmp_path)) as client:
        response = client.post(
            "/v1/telemetry",
            content=json.dumps({"sensors": [0] * 17_000}),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413
