import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import make_url
from sqlalchemy.schema import CreateSchema, DropSchema

from faultwatch.api import create_app
from faultwatch.data import CYCLE, ENGINE, TELEMETRY_COLUMNS
from faultwatch.db import connect_database
from faultwatch.features import WINDOW_SIZE
from faultwatch.inference import load_predictor


def event(engine=1, cycle=1):
    return {"engine_id": engine, "cycle": cycle, "settings": [0, 0, 100], "sensors": [1] * 21}


@pytest.fixture
def without_services(tmp_path, monkeypatch):
    monkeypatch.delenv("FAULTWATCH_DATABASE_URL", raising=False)
    with TestClient(create_app(model_dir=tmp_path)) as client:
        yield client


@pytest.mark.parametrize(
    "invalid",
    [
        {"engine_id": "1"},
        {"engine_id": True},
        {"cycle": 0},
        {"sensors": [1] * 20},
        {"sensors": [1] * 20 + [float("nan")]},
        {"sensors": [1] * 20 + [float("inf")]},
        {"sensors": [1] * 20 + [True]},
        {"sensors": [1] * 20 + ["1"]},
        {"unexpected": 1},
    ],
)
def test_malformed_telemetry_is_rejected_before_storage(without_services, invalid):
    response = without_services.post(
        "/v1/telemetry",
        content=json.dumps(event() | invalid),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]


def test_liveness_without_db_or_model(without_services):
    assert without_services.get("/healthz").status_code == 200
    response = without_services.get("/readyz")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "database": False, "model": False}


def test_dashboard_is_served_before_telemetry_arrives(without_services):
    response = without_services.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Simulated C-MAPSS FD001 telemetry" in response.text


@pytest.fixture(scope="module")
def postgres():
    value = os.getenv("FAULTWATCH_TEST_DATABASE_URL")
    if not value:
        pytest.skip("Set FAULTWATCH_TEST_DATABASE_URL to run PostgreSQL tests")
    url = make_url(value)
    schema = f"faultwatch_test_{uuid4().hex}"
    engine = connect_database(url)
    config = Config(Path(__file__).parents[1] / "alembic.ini")
    config.attributes["database_url"] = url.update_query_dict(
        {"options": f"-csearch_path={schema}"}
    )
    with engine.begin() as connection:
        connection.execute(CreateSchema(schema))
    try:
        command.upgrade(config, "head")
        yield config.attributes["database_url"]
        command.downgrade(config, "base")
    finally:
        with engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        engine.dispose()


def test_duplicate_delivery_and_persistence(postgres, model_bundle):
    bundle, _ = model_bundle
    payload = event(engine=40)
    with TestClient(create_app(postgres, bundle)) as client:
        assert client.get("/readyz").json() == {"status": "ready", "database": True, "model": True}
        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = [pool.submit(client.post, "/v1/telemetry", json=payload) for _ in range(2)]
            responses = [attempt.result() for attempt in attempts]
        assert sorted(response.status_code for response in responses) == [200, 201]
        assert {response.json()["status"] for response in responses} == {"created", "duplicate"}
        conflict = client.post("/v1/telemetry", json=payload | {"settings": [0, 0, 99]})
        assert conflict.status_code == 409
        metrics = client.get("/metrics").text
        for outcome in ("created", "duplicate", "conflict"):
            assert f'faultwatch_telemetry_events_total{{outcome="{outcome}"}} 1.0' in metrics
    with TestClient(create_app(postgres, bundle)) as restarted:
        rows = restarted.get("/v1/engines/40/history").json()
        assert len(rows) == 1
        assert rows[0]["settings"] == payload["settings"]
        assert datetime.fromisoformat(rows[0]["received_at"]).utcoffset() is not None
        assert {"engine_id": 40, "latest_cycle": 1} in restarted.get("/v1/engines").json()


def test_out_of_order_readings_require_contiguous_history(postgres, model_bundle, trajectories):
    bundle, _ = model_bundle
    frame = trajectories.loc[(trajectories[ENGINE] == 2) & (trajectories[CYCLE] <= WINDOW_SIZE)]
    readings = frame.loc[:, list(TELEMETRY_COLUMNS)].itertuples(index=False, name=None)
    with TestClient(create_app(postgres, bundle)) as client:
        url = "/v1/engines/2/prediction"
        assert client.get(url).status_code == 404
        missing = None
        for engine, cycle, *values in reversed(list(readings)):
            payload = event(engine, cycle) | {"settings": values[:3], "sensors": values[3:]}
            if cycle == 3:
                missing = payload
            else:
                assert client.post("/v1/telemetry", json=payload).status_code == 201
        result = client.get(url).json()
        assert result["status"] == "insufficient_history"
        assert result["available_cycles"] == WINDOW_SIZE - 3
        assert result["predicted_rul_cycles"] is None
        assert client.post("/v1/telemetry", json=missing).status_code == 201

        response = client.get(url)
        assert response.status_code == 200
        prediction = response.json()
        predictor = load_predictor(bundle)
        assert prediction["status"] == "ready"
        assert prediction["cycle"] == WINDOW_SIZE
        assert prediction["model_version"] == predictor.model_version
        assert prediction["predicted_rul_cycles"] == pytest.approx(
            predictor.predict_latest(frame).iloc[0]
        )
        last = client.get("/v1/engines/2/history?limit=3").json()
        assert [row["cycle"] for row in last] == [18, 19, 20]
        assert "faultwatch_inference_duration_seconds_count 1.0" in client.get("/metrics").text


def test_missing_model_does_not_block_ingest(postgres, tmp_path):
    with TestClient(create_app(postgres, tmp_path)) as client:
        assert client.get("/readyz").json() == {
            "status": "unavailable",
            "database": True,
            "model": False,
        }
        assert client.post("/v1/telemetry", json=event(engine=99)).status_code == 201
        assert client.get("/v1/engines/99/prediction").status_code == 503


def test_metrics_use_route_templates_and_exclude_probes(without_services):
    assert without_services.get("/v1/engines/731/history").status_code == 503
    assert without_services.get("/missing-924").status_code == 404
    assert without_services.get("/healthz").status_code == 200
    response = without_services.get("/metrics")
    assert response.status_code == 200
    assert (
        'faultwatch_http_requests_total{method="GET",route="/v1/engines/{engine_id}/history",status="503"} 1.0'
        in response.text
    )
    assert (
        'faultwatch_http_requests_total{method="GET",route="unmatched",status="404"} 1.0'
        in response.text
    )
    assert 'route="/healthz"' not in response.text
    assert 'route="/v1/engines/731/history"' not in response.text
    assert 'route="/missing-924"' not in response.text


def test_native_predictions_use_the_existing_api(
    postgres, model_bundle, trajectories, tmp_path, monkeypatch
):
    pytest.importorskip("_faultwatch_native")
    pytest.importorskip("onnxscript")
    from shutil import copy2

    from faultwatch.features import causal_features
    from faultwatch.onnx_export import export_and_validate

    source, _ = model_bundle
    copy2(source / "manifest.json", tmp_path)
    model = tmp_path / "model.onnx"
    export_and_validate(source, model, causal_features(trajectories, latest_only=True), repeats=2)
    monkeypatch.setenv("FAULTWATCH_BACKEND", "native")
    with TestClient(create_app(postgres, tmp_path)) as client:
        assert client.get("/readyz").status_code == 200
        assert (
            'faultwatch_model_info{backend="native",version="fd001-v1"}'
            in client.get("/metrics").text
        )
        frame = trajectories.loc[(trajectories[ENGINE] == 3) & (trajectories[CYCLE] <= WINDOW_SIZE)]
        for engine, cycle, *values in frame.itertuples(index=False, name=None):
            payload = {
                "engine_id": engine,
                "cycle": cycle,
                "settings": values[:3],
                "sensors": values[3:],
            }
            assert client.post("/v1/telemetry", json=payload).status_code == 201
            if cycle == WINDOW_SIZE - 1:
                assert (
                    client.get("/v1/engines/3/prediction").json()["status"]
                    == "insufficient_history"
                )
        prediction = client.get("/v1/engines/3/prediction").json()
        assert prediction["status"] == "ready"
        expected = load_predictor(source).predict_latest(frame).iloc[0]
        assert prediction["predicted_rul_cycles"] == pytest.approx(expected, abs=1e-4)
    model.with_suffix(".onnx.sha256").write_text("incorrect\n")
    with TestClient(create_app(postgres, tmp_path)) as client:
        assert client.get("/readyz").json()["model"] is False


def test_authenticated_concurrency_and_tracing(postgres, model_bundle, monkeypatch):
    from types import SimpleNamespace

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from faultwatch.security import issue_token

    secret = "independent test secret for telemetry" * 2
    monkeypatch.setenv("FAULTWATCH_JWT_SECRET", secret)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    store = {}
    monkeypatch.setattr(
        "faultwatch.cache.Redis.from_url",
        lambda *args, **kwargs: SimpleNamespace(
            get=store.get,
            set=lambda key, value, **options: store.update({key: value}),
            close=lambda: None,
        ),
    )
    reader = {"Authorization": f"Bearer {issue_token(secret, 'console', 'read')}"}
    writer = {"Authorization": f"Bearer {issue_token(secret, 'gateway', 'ingest')}"}
    url = "/v1/engines/77/prediction"
    bundle, _ = model_bundle
    with TestClient(create_app(postgres, bundle, "redis://127.0.0.1", provider)) as client:
        assert client.post("/v1/telemetry", json=event(77), headers=reader).status_code == 403
        assert client.get(url, headers=writer).status_code == 403

        def run(cycle):
            response = client.post("/v1/telemetry", json=event(77, cycle), headers=writer)
            assert response.status_code == 201
            prediction = client.get(url, headers=reader)
            assert prediction.json()["status"] in {"insufficient_history", "ready"}

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(run, range(1, WINDOW_SIZE + 1)))
        headers = reader | {
            "traceparent": "00-87654321876543218765432187654321-1234567890123456-01"
        }
        store.clear()
        assert client.get(url, headers=headers).json()["status"] == "ready"
        assert client.get(url, headers=reader).json()["status"] == "ready"
        trace_id = int("87654321876543218765432187654321", 16)
        traced = [
            item for item in exporter.get_finished_spans() if item.context.trace_id == trace_id
        ]
        assert {item.name for item in traced} == {
            "GET /v1/engines/{engine_id}/prediction",
            "postgres.telemetry.window",
            "redis.get",
            "redis.set",
            "inference.predict",
        }
        assert all(not item.events for item in traced)
        assert secret not in str([item.attributes for item in traced])
