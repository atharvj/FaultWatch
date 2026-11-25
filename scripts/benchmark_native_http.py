"""Compare Python ONNX and native scoring through the same local HTTP mix."""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryFile
from time import monotonic, sleep

import httpx2
from sqlalchemy import make_url

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("onnx", "native", "native", "onnx")
DEFAULT_DATABASE = "postgresql+psycopg://faultwatch:faultwatch-local@127.0.0.1:55433/faultwatch"


def readiness(client, process, backend: str):
    deadline = monotonic() + 30
    while monotonic() < deadline and process.poll() is None:
        try:
            if client.get("/readyz").status_code == 200:
                model = client.get("/metrics").text
                if f'faultwatch_model_info{{backend="{backend}",' not in model:
                    raise ValueError(f"Unexpected runtime for {backend} trial")
                return
        except httpx2.HTTPError:
            pass
        sleep(0.25)
    raise RuntimeError(f"The {backend} API did not reach readiness")


def trial(args, backend: str) -> dict:
    env = {
        **os.environ,
        "FAULTWATCH_BACKEND": backend,
        "FAULTWATCH_DATABASE_URL": args.database,
        "FAULTWATCH_MODEL_DIR": str(args.bundle.resolve()),
        "FAULTWATCH_REDIS_URL": "",
        "FAULTWATCH_JWT_SECRET": "",
        "FAULTWATCH_RATE_PER_SECOND": "",
        "FAULTWATCH_OTLP_ENDPOINT": "",
        "FAULTWATCH_TRACE_CONSOLE": "",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "ORT_DISABLE_TELEMETRY": "1",
    }
    with socket.socket() as listener, TemporaryFile(mode="w+t") as log:
        listener.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        fd = listener.fileno()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "faultwatch.api:create_app",
                "--factory",
                "--fd",
                str(fd),
                "--no-access-log",
            ],
            env=env,
            cwd=ROOT,
            pass_fds=(fd,),
            stdout=log,
            stderr=log,
        )
        try:
            with httpx2.Client(base_url=url, timeout=2, trust_env=False) as client:
                readiness(client, process, backend)
                scores = []
                for engine in (1, 2, 3):
                    response = client.get(f"/v1/engines/{engine}/prediction")
                    if not response.is_success or response.json().get("status") != "ready":
                        raise ValueError("Replay FD001 engines 1–3 before running this benchmark")
                    scores.append(response.json())
            timed = subprocess.check_output(
                [
                    sys.executable,
                    str(ROOT / "scripts/benchmark_api.py"),
                    "--url",
                    url,
                    "--workers",
                    str(args.workers),
                    "--seconds",
                    str(args.seconds),
                ],
                text=True,
                timeout=args.seconds + 30,
            )
            rss = subprocess.check_output(["ps", "-o", "rss=", "-p", str(process.pid)], text=True)
            return {
                "backend": backend,
                "rss_mib_after_trial": int(rss.strip()) / 1024,
                "preflight_predictions": scores,
                **json.loads(timed),
            }
        except Exception:
            log.seek(0)
            print(log.read()[-4000:], file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", default=os.getenv("FAULTWATCH_DATABASE_URL", DEFAULT_DATABASE)
    )
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/fd001-v1"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seconds", type=int, default=20)
    args = parser.parse_args()
    if make_url(args.database).host not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("Use a loopback PostgreSQL database")
    if not 1 <= args.workers <= 32 or not 1 <= args.seconds <= 120:
        parser.error("Use 1–32 workers and 1–120 seconds")
    trials = [trial(args, backend) for backend in ORDER]
    reference = trials[0]["preflight_predictions"]
    differences = []
    score_key = "predicted_rul_cycles"
    for t in trials:
        for recorded, predicted in zip(reference, t["preflight_predictions"], strict=True):
            if {k: v for k, v in recorded.items() if k != score_key} != {
                k: v for k, v in predicted.items() if k != score_key
            }:
                raise ValueError("Telemetry changed during the benchmark")
            differences.append(abs(recorded[score_key] - predicted[score_key]))
    if max(differences) > 1e-4:
        raise ValueError("Native HTTP scores drifted from Python ONNX")
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "redis_enabled": False,
                "onnx_threads": 2,
                "max_score_drift_cycles": max(differences),
                "trials": trials,
            },
            indent=2,
        )
    )
