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

## Model runtime

`uv run --locked --group experiments faultwatch export-onnx` exports the
MLP and scaler as a 40,742-byte ONNX graph. It takes a dynamic batch of 64
ordered features per row and emits remaining cycles. Feature names and the
model version are embedded in the graph.

ONNX Runtime and PyTorch agreed within 0.000031 cycles across all 100
official test endpoints. Timings start from precomputed features and include
scaling and graph execution. Both runtimes used two CPU threads on the same
M2 Pro. Each batch size and runtime used 30 warmup calls and 300 timed calls.

| Batch | PyTorch p50 / p95 (µs) | ONNX Runtime p50 / p95 (µs) |
|---|---:|---:|
| 1 | 20.83 / 21.21 | 5.63 / 5.92 |
| 100 | 34.17 / 38.67 | 25.77 / 49.43 |

See the [ONNX benchmark JSON](../reports/onnx-benchmark.json) for versions,
tolerances, and the graph checksum.

## Redis cache

`make cache` enables the optional Redis service and the cache panel in Grafana.
PostgreSQL supplies the current 20-cycle window on every prediction request.
Redis keys include the model artifact checksum and the exact ordered readings.
New telemetry gets a different key and older entries expire after two minutes.
A hit skips feature extraction and inference. Redis errors fall back to direct
scoring. Incomplete windows return insufficient history.

The same 30-second workload ran four times on the M2 Pro in off/on/on/off
order. Both settings used the same Compose image, 206 readings, eight
clients and three warm prediction windows. The Redis container was running
for every trial. Preflight filled entries before the timed cached runs.

| Trial | Redis | All requests/s | Prediction p95 (ms) | Failures |
|---|---|---:|---:|---:|
| 1 | Off | 286.31 | 48.78 | 0 |
| 2 | On | 535.66 | 26.09 | 0 |
| 3 | On | 556.93 | 24.63 | 0 |
| 4 | Off | 289.43 | 46.58 | 0 |

The mean of the two trial p95s fell 46.8%, from 47.68 to 25.36 ms. This
workload repeatedly reads the same three engines. Cache tests cover expiry,
invalid entries, a new cycle, gap backfill and a stopped Redis container.
[Full trial results](../reports/redis-benchmark.json)

## Terraform S3 lab

Terraform 1.17.2 with the AWS provider 5.100.0 provisioned three resources
against Moto 5.2.6. All four public-access blocks and AES256 configuration
were verified through the AWS SDK. Two artifacts were restored with
matching checksums and the bucket count returned to zero after destroy.
[Lab result](../reports/s3-lab.json)
