"""OpenTelemetry configuration and spans with bounded, non-payload attributes."""

import os
from contextlib import contextmanager
from urllib.parse import urlsplit

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from starlette.datastructures import Headers

OTLP_ENV = "FAULTWATCH_OTLP_ENDPOINT"
CONSOLE_ENV = "FAULTWATCH_TRACE_CONSOLE"
PROPAGATOR = TraceContextTextMapPropagator()


def configure_tracing() -> TracerProvider | None:
    endpoint = os.getenv(OTLP_ENV)
    console = os.getenv(CONSOLE_ENV) == "1"
    if not endpoint and not console:
        return None
    provider = TracerProvider(resource=Resource.create({"service.name": "faultwatch"}))
    if endpoint:
        address = urlsplit(endpoint)
        if (
            address.scheme not in {"http", "https"}
            or not address.hostname
            or address.username is not None
        ):
            raise ValueError(f"{OTLP_ENV} must be an HTTP(S) trace endpoint without credentials")
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, timeout=3))
        )
    if console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    return provider


def trace_context(scope):
    return PROPAGATOR.extract(Headers(scope=scope))


@contextmanager
def span(tracer, name: str, *, kind=SpanKind.INTERNAL, attributes=None, context=None):
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=attributes,
        context=context,
        record_exception=False,
        set_status_on_exception=False,
    ) as active:
        try:
            yield active
        except Exception:
            active.set_status(StatusCode.ERROR)
            raise
