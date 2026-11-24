"""Measure the local demo with a fixed, read-only request mix."""

import argparse
import itertools
import json
import math
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from urllib.parse import urlsplit

import httpx2

PREDICTION = "/v1/engines/{engine}/prediction"
REQUESTS = (
    PREDICTION,
    PREDICTION,
    PREDICTION,
    "/v1/engines/{engine}/history?limit=20",
    "/v1/engines",
)


def run_worker(url, deadline, index):
    observations = []
    with httpx2.Client(base_url=url, trust_env=False, timeout=5) as client:
        for turn in itertools.count(index):
            start = perf_counter()
            if start >= deadline:
                break
            template = REQUESTS[turn % len(REQUESTS)]
            try:
                response = client.get(template.format(engine=turn % 3 + 1))
                ok = response.is_success
                if template == PREDICTION and ok:
                    ok = response.json().get("status") == "ready"
            except (httpx2.HTTPError, ValueError):
                ok = False
            observations.append((template, (perf_counter() - start) * 1000, ok))
    return observations


def summarize(observations, elapsed):
    latencies = sorted(latency for _, latency, ok in observations if ok)
    errors = len(observations) - len(latencies)
    return {
        "requests": len(observations),
        "errors": errors,
        "error_rate": round(errors / len(observations), 6) if observations else None,
        "requests_per_second": round(len(observations) / elapsed, 2),
        "p50_ms": round(latencies[math.ceil(len(latencies) * 0.5) - 1], 3) if latencies else None,
        "p95_ms": round(latencies[math.ceil(len(latencies) * 0.95) - 1], 3) if latencies else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()
    address = urlsplit(args.url)
    if address.scheme != "http" or address.hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("--url must be an HTTP loopback address")
    if not 1 <= args.workers <= 32 or not 1 <= args.seconds <= 120:
        parser.error("Use 1–32 workers and 1–120 seconds")

    with httpx2.Client(base_url=args.url, trust_env=False, timeout=5) as client:
        for engine in (1, 2, 3):
            response = client.get(PREDICTION.format(engine=engine))
            if not response.is_success or response.json().get("status") != "ready":
                parser.error("Start the demo and replay test engines 1–3 first")

    start = perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(run_worker, args.url, start + args.seconds, worker)
            for worker in range(args.workers)
        ]
        observations = [sample for job in jobs for sample in job.result()]
    elapsed = perf_counter() - start
    total = summarize(observations, elapsed)
    print(
        json.dumps(
            {
                "workers": args.workers,
                "target_seconds": args.seconds,
                "elapsed_seconds": round(elapsed, 3),
                "total": total,
                "routes": {
                    template: summarize(
                        [sample for sample in observations if sample[0] == template], elapsed
                    )
                    for template in dict.fromkeys(REQUESTS)
                },
            },
            indent=2,
        )
    )
    if total["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
