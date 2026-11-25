"""Optional Redis cache for complete prediction windows."""

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from math import isfinite
from struct import Struct
from typing import Any

from opentelemetry.trace import SpanKind
from prometheus_client import Counter
from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

from faultwatch.data import OPERATING_SETTINGS, SENSORS
from faultwatch.tracing import span

CACHE_ENV = "FAULTWATCH_REDIS_URL"
TTL_SECONDS = 120
_ROW = Struct(f"!ii{len(OPERATING_SETTINGS) + len(SENSORS)}d")


def prediction_key(
    model_version: str, artifact_sha256: str, rows: Sequence[Mapping[str, Any]]
) -> str:
    digest = sha256()
    for row in rows:
        digest.update(_ROW.pack(row["engine_id"], row["cycle"], *row["settings"], *row["sensors"]))
    return f"faultwatch:prediction:{model_version}:{artifact_sha256}:{digest.hexdigest()}"


class PredictionCache:
    def __init__(self, url: str, requests: Counter, tracer=None) -> None:
        self.client = Redis.from_url(
            url,
            socket_connect_timeout=0.075,
            socket_timeout=0.075,
            retry=Retry(NoBackoff(), 0),
            max_connections=32,
        )
        self.requests = requests
        self.tracer = tracer

    def _span(self, name: str):
        return span(self.tracer, name, kind=SpanKind.CLIENT, attributes={"db.system.name": "redis"})

    def get_or_score(self, key: str, score: Callable[[], float]) -> float:
        try:
            with self._span("redis.get"):
                cached = self.client.get(key)
        except RedisError:
            self.requests.labels(outcome="error").inc()
            return score()

        if cached is not None:
            try:
                value = float(cached)
            except (TypeError, ValueError, OverflowError):
                value = -1
            if isfinite(value) and value >= 0:
                self.requests.labels(outcome="hit").inc()
                return value

        value = score()
        try:
            with self._span("redis.set"):
                self.client.set(key, repr(value), ex=TTL_SECONDS)
        except RedisError:
            self.requests.labels(outcome="error").inc()
        else:
            self.requests.labels(outcome="miss").inc()
        return value

    def close(self) -> None:
        self.client.close()
