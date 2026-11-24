import os
import subprocess
from pathlib import Path
from uuid import uuid4

import httpx2
import pytest

from faultwatch.data import CYCLE, ENGINE
from faultwatch.features import WINDOW_SIZE
from faultwatch.inference import load_predictor

pytestmark = pytest.mark.skipif(
    os.getenv("FAULTWATCH_COMPOSE_SMOKE") != "1",
    reason="Set FAULTWATCH_COMPOSE_SMOKE=1 to build and test the container stack",
)


def test_compose_dashboard_and_prediction(model_bundle, trajectories):
    bundle, _ = model_bundle
    bundle.chmod(0o755)
    project = f"faultwatch-smoke-{uuid4().hex[:10]}"
    env = {
        **os.environ,
        "FAULTWATCH_PORT": "0",
        "FAULTWATCH_DB_PORT": "0",
        "FAULTWATCH_MODEL_DIR": str(bundle),
    }

    def compose(*arguments, check=True, capture_output=False):
        return subprocess.run(
            ["docker", "compose", "--project-name", project, *arguments],
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
        with httpx2.Client(base_url=f"http://{endpoint}", trust_env=False) as client:
            assert client.get("/readyz").json() == {
                "status": "ready",
                "database": True,
                "model": True,
            }
            assert "Simulated C-MAPSS FD001 telemetry" in client.get("/").text
            for engine_id, cycle, *values in frame.itertuples(index=False, name=None):
                payload = {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "settings": values[:3],
                    "sensors": values[3:],
                }
                assert client.post("/v1/telemetry", json=payload).status_code == 201
            assert client.post("/v1/telemetry", json=payload).status_code == 200
            prediction = client.get("/v1/engines/1/prediction").json()
            assert prediction["status"] == "ready"
            assert prediction["cycle"] == WINDOW_SIZE
            expected = load_predictor(bundle).predict_latest(frame).iloc[0]
            assert prediction["predicted_rul_cycles"] == pytest.approx(expected, abs=1e-4)
    except Exception:
        compose("logs", "--tail", "30", check=False)
        raise
    finally:
        compose("down", "--volumes")
