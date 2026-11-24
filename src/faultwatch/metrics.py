"""Prometheus counters and duration histograms for a single API process."""

from time import perf_counter

from prometheus_client import CollectorRegistry, Counter, Histogram, Info
from starlette.types import ASGIApp, Message, Receive, Scope, Send

DURATION_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)
METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
PROBE_PATHS = {"/metrics", "/healthz", "/readyz"}


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.model = Info("faultwatch_model", "Active inference runtime", registry=self.registry)
        self.requests = Counter(
            "faultwatch_http_requests",
            "HTTP requests by method, route template and status.",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "faultwatch_http_request_duration_seconds",
            "HTTP response duration by method and route template.",
            ("method", "route"),
            buckets=DURATION_BUCKETS,
            registry=self.registry,
        )
        self.inference_duration = Histogram(
            "faultwatch_inference_duration_seconds",
            "Duration of feature extraction and model prediction.",
            buckets=DURATION_BUCKETS,
            registry=self.registry,
        )
        self.telemetry = Counter(
            "faultwatch_telemetry_events",
            "Telemetry ingest outcomes after a database transaction.",
            ("outcome",),
            registry=self.registry,
        )
        self.cache = Counter(
            "faultwatch_prediction_cache_requests",
            "Redis prediction lookups by outcome.",
            ("outcome",),
            registry=self.registry,
        )
        for outcome in ("hit", "miss", "error"):
            self.cache.labels(outcome=outcome)
        for outcome in ("created", "duplicate", "conflict"):
            self.telemetry.labels(outcome=outcome)


class RequestMetrics:
    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in PROBE_PATHS:
            await self.app(scope, receive, send)
            return

        start = perf_counter()
        status = 500

        async def record_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, record_status)
        finally:
            method = scope["method"] if scope["method"] in METHODS else "OTHER"
            route = getattr(scope.get("route"), "path", "unmatched")
            self.metrics.requests.labels(method=method, route=route, status=str(status)).inc()
            self.metrics.request_duration.labels(method=method, route=route).observe(
                perf_counter() - start
            )
