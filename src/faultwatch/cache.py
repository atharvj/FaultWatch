"""Optional Redis cache for complete prediction windows."""

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from math import isfinite
from struct import Struct
from typing import Any

from prometheus_client import Counter
from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

from faultwatch.data import OPERATING_SETTINGS, SENSORS

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
    def __init__(self, url: str, requests: Counter) -> None:
        self.client = Redis.from_url(
            url,
            socket_connect_timeout=0.075,
            socket_timeout=0.075,
            retry=Retry(NoBackoff(), 0),
            max_connections=32,
        )
        self.requests = requests

    def get_or_score(self, key: str, score: Callable[[], float]) -> float:
        try:
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
            self.client.set(key, repr(value), ex=TTL_SECONDS)
        except RedisError:
            self.requests.labels(outcome="error").inc()
        else:
            self.requests.labels(outcome="miss").inc()
        return value

    def close(self) -> None:
        self.client.close()
