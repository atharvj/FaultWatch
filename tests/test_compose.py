import os
import subprocess
from pathlib import Path
from uuid import uuid4

import httpx2
import pytest

from faultwatch.cache import TTL_SECONDS
from faultwatch.data import CYCLE, ENGINE
from faultwatch.features import WINDOW_SIZE
from faultwatch.inference import load_predictor

pytestmark = pytest.mark.skipif(
    os.getenv("FAULTWATCH_COMPOSE_SMOKE") != "1",
    reason="Set FAULTWATCH_COMPOSE_SMOKE=1 to build and test the container stack",
)


@pytest.mark.parametrize("cached", [False, True], ids=["direct", "redis"])
def test_compose_dashboard_and_prediction(model_bundle, trajectories, cached):
    bundle, _ = model_bundle
    bundle.chmod(0o755)
    project = f"faultwatch-smoke-{uuid4().hex[:10]}"
    env = {
        **os.environ,
        "FAULTWATCH_PORT": "0",
        "FAULTWATCH_DB_PORT": "0",
        "FAULTWATCH_MODEL_DIR": str(bundle),
        "FAULTWATCH_REDIS_URL": "redis://redis:6379/0" if cached else "",
    }

    profiles = ["--profile", "cache"] if cached else []

    def compose(*arguments, check=True, capture_output=False):
        return subprocess.run(
            ["docker", "compose", "--project-name", project, *profiles, *arguments],
            cwd=Path(__file__).parents[1],
            env=env,
            check=check,
            capture_output=capture_output,
            text=True,
            timeout=360,
        )

    try:
        compose("up", "--build", "--wait", "--wait-timeout", "120")
        endpoint = compose("port", "api", "8000", capture_output=True).stdout.strip()
        frame = trajectories.loc[(trajectories[ENGINE] == 1) & (trajectories[CYCLE] <= WINDOW_SIZE)]
        events = []
        for engine_id, cycle, *values in frame.itertuples(index=False, name=None):
            events.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "settings": values[:3],
                    "sensors": values[3:],
                }
            )
        with httpx2.Client(base_url=f"http://{endpoint}", trust_env=False, timeout=8) as client:
            assert client.get("/readyz").json() == {
                "status": "ready",
                "database": True,
                "model": True,
            }
            assert "Simulated C-MAPSS FD001 telemetry" in client.get("/").text
            url = "/v1/engines/1/prediction"
            for event in events:
                if cached and event["cycle"] == 3:
                    continue
                assert client.post("/v1/telemetry", json=event).status_code == 201
            if cached:
                assert client.get(url).json()["status"] == "insufficient_history"
                assert client.post("/v1/telemetry", json=events[2]).status_code == 201
            assert client.post("/v1/telemetry", json=events[-1]).status_code == 200
            prediction = client.get(url).json()
            assert prediction["status"] == "ready"
            assert prediction["cycle"] == WINDOW_SIZE
            predictor = load_predictor(bundle)
            expected = predictor.predict_latest(frame).iloc[0]
            assert prediction["predicted_rul_cycles"] == pytest.approx(expected, abs=1e-4)
            if cached:
                check_redis(client, compose, url, prediction, trajectories, predictor)
            else:
                compose("stop", "db")
                assert client.get(url).status_code == 503
                compose("up", "--wait", "db")
                compose("restart", "api")
                compose("up", "--wait", "api")
                endpoint = compose("port", "api", "8000", capture_output=True).stdout.strip()
                with httpx2.Client(base_url=f"http://{endpoint}", trust_env=False) as restarted:
                    assert restarted.get(url).json() == prediction
    except Exception:
        compose("logs", "--tail", "30", check=False)
        raise
    finally:
        compose("down", "--volumes")


def check_redis(client, compose, url, first, trajectories, predictor):
    def redis(*arguments):
        return compose(
            "exec", "-T", "redis", "redis-cli", *arguments, capture_output=True
        ).stdout.strip()

    assert client.get(url).json() == first
    metrics = client.get("/metrics").text
    assert 'faultwatch_prediction_cache_requests_total{outcome="hit"} 1.0' in metrics
    assert "faultwatch_inference_duration_seconds_count 1.0" in metrics
    (key,) = redis("--scan", "--pattern", "faultwatch:prediction:*").splitlines()
    assert 0 < int(redis("TTL", key)) <= TTL_SECONDS

    assert redis("SET", key, "nan", "EX", str(TTL_SECONDS)) == "OK"
    assert client.get(url).json() == first
    assert redis("EXPIRE", key, "0") == "1"
    assert client.get(url).json() == first

    frame = trajectories.loc[(trajectories[ENGINE] == 1) & (trajectories[CYCLE] <= WINDOW_SIZE + 1)]
    engine, cycle, *values = frame.iloc[-1].tolist()
    event = {
        "engine_id": int(engine),
        "cycle": int(cycle),
        "settings": values[:3],
        "sensors": values[3:],
    }
    assert client.post("/v1/telemetry", json=event).status_code == 201
    second = client.get(url).json()
    assert second["cycle"] == WINDOW_SIZE + 1
    assert second["predicted_rul_cycles"] == pytest.approx(
        predictor.predict_latest(frame).iloc[0], abs=1e-4
    )
    metrics = client.get("/metrics").text
    assert 'faultwatch_prediction_cache_requests_total{outcome="miss"} 4.0' in metrics

    compose("stop", "redis")
    assert client.get("/readyz").status_code == 200
    assert client.get(url).json() == second
    assert (
        'faultwatch_prediction_cache_requests_total{outcome="error"} 1.0'
        in client.get("/metrics").text
    )
