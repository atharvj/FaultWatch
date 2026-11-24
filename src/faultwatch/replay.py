"""Replay real FD001 test telemetry through the HTTP ingest endpoint."""

import math
from pathlib import Path
from time import monotonic, sleep

import httpx2

from faultwatch.data import ENGINE, read_fd001_telemetry, verify_fd001

DEFAULT_ENGINES = (1, 2, 3)


def replay_fd001(data: Path, base_url: str, engines: list[int] | None, rate: float) -> dict:
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("--rate must be finite and nonnegative")
    test_file = "test_FD001.txt"
    verify_fd001(data, test_file)
    frame = read_fd001_telemetry(data / test_file)
    selected = set(DEFAULT_ENGINES if engines is None else engines)
    missing = selected.difference(frame[ENGINE])
    if missing:
        raise ValueError(f"FD001 test engines not found: {sorted(missing)}")
    rows = frame.loc[frame[ENGINE].isin(selected)]

    counts = {"created": 0, "duplicate": 0}
    interval = 1 / rate if rate else 0
    next_send = monotonic()
    try:
        with httpx2.Client(base_url=base_url, timeout=5, trust_env=False) as client:
            for engine_id, cycle, *values in rows.itertuples(index=False, name=None):
                wait = next_send - monotonic()
                if wait > 0:
                    sleep(wait)
                next_send = monotonic() + interval
                response = client.post(
                    "/v1/telemetry",
                    json={
                        "engine_id": engine_id,
                        "cycle": cycle,
                        "settings": values[:3],
                        "sensors": values[3:],
                    },
                )
                response.raise_for_status()
                status = response.json().get("status")
                if status not in counts:
                    raise ValueError(
                        f"Unexpected ingest response for engine {engine_id}, cycle {cycle}"
                    )
                counts[status] += 1
    except (httpx2.HTTPError, httpx2.InvalidURL) as error:
        raise ValueError(f"Replay stopped: {error}") from error
    return {"engines": sorted(selected), **counts}
