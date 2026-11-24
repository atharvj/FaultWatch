# Local performance

## API load test

Run `make observe`, `make replay`, then `make benchmark` on a trained checkout.
The benchmark uses eight concurrent HTTP clients for 30 seconds. The request
mix is 60% prediction, 20% recent history and 20% engine listing, cycling
through FD001 test engines 1–3. It measures client-side latency for successful
requests and treats an unscored prediction as an error.

Measured November 24, 2026 on an Apple M2 Pro with 16 GB RAM, using native
ARM64 Docker Desktop, one Uvicorn process, the local PostgreSQL volume and
Prometheus scraping every five seconds. The database already held 206 readings.
The run used a warm database and no concurrent ingest.

| Route | Requests | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|
| Prediction | 5,494 | 28.71 | 43.70 |
| Recent history | 1,831 | 20.27 | 33.63 |
| Engine listing | 1,831 | 19.18 | 31.85 |
| All requests | 9,156 | 25.26 | 41.44 |

The run completed in 30.027 seconds at 304.93 requests/s with zero failures.
See the [benchmark JSON](../reports/api-benchmark.json).
