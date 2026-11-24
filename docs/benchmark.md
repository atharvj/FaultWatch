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

## Live AWS S3 round trip

Terraform created one private bucket in `us-west-1` and set all four
public-access blocks and AES256 default encryption. The AWS SDK uploaded
and restored two artifacts totaling 28,228 bytes, verified both checksums,
and loaded the restored model. The test used a 15-minute session scoped to
that bucket and those two keys. Terraform destroyed all three resources.
AWS confirmed the bucket and temporary identity were gone and the account
was on the Free plan before and after the test. [Live result](../reports/aws-s3-live.json)

## Kafka replay correctness

The KRaft broker accepted 80 real FD001 messages. A consumer stopped after
25 commits, and a new consumer in the same group resumed for the remaining
55. A second publication returned 80 duplicates and zero inserts. PostgreSQL
retained 31 and 49 cycles in order for engines 1 and 2. This lab measures
ordering and repeated delivery on a single broker.
[Replay results](../reports/kafka-replay.json)

## Spark and Parquet

Spark 4.3.0 running with two local workers generated and verified 64 causal
features for 18,731 train and 11,196 test rows in 10.33 seconds. Four Parquet
files across two split partitions took 3,934,010 bytes. The largest feature
error against pandas was 5.46e-12. Only the training partition contains the
true remaining-life label. [Round-trip results](../reports/spark-parquet.json)

## Documentation retrieval and Q&A

On a small fixed check of five questions with answers in the public
documentation, pgvector retrieved the expected source in the top three
for all five. FLAN-T5 returned five correct answers with citations and
abstained on both unrelated questions. This checks the local workflow
on seven examples. [Questions and results](../reports/rag-evaluation.json)

## Helm lifecycle

On the local kind cluster, Helm 4.3.2 installed the API chart and upgraded its
CPU request to 300m. Both revisions returned HTTP 200 from the readiness
probe. The test released its resources and both original workloads stayed
ready. [Release verification](../reports/helm-lab.json)

## XGBoost

XGBoost 3.2.0 trained a two-thread CPU histogram regressor on the original
FD001 splits and 64 causal features. Across 300 validation cutoffs,
MAE/RMSE were 25.24/34.52 cycles. Across 100 official test endpoints,
MAE/RMSE were 18.58/25.55. The MLP remains the serving model based on its
23.28-cycle validation MAE. Reloading the XGBoost UBJSON model reproduced
every test prediction exactly. [Scores and parameters](../reports/xgboost-fd001.json)

## ONNX Runtime over HTTP

The same Linux arm64 Compose image ran four 30-second trials on an M2 Pro
with eight clients, 206 PostgreSQL readings and no concurrent ingest.
The client sent 60% prediction requests and split the rest between history
and engine listings. Redis was disabled for every trial. Only the configured
prediction backend changed.

| Trial | Runtime | All requests/s | Prediction p95 (ms) | Failures |
|---|---|---:|---:|---:|
| 1 | PyTorch | 201.68 | 69.56 | 0 |
| 2 | ONNX Runtime | 222.10 | 61.71 | 0 |
| 3 | ONNX Runtime | 228.76 | 59.43 | 0 |
| 4 | PyTorch | 207.09 | 68.21 | 0 |

Mean trial prediction p95 fell 12.1%, from 68.89 to 60.57 ms. Mean throughput
rose 10.3%, from 204.39 to 225.43 requests/s. Predictions for the three
engines agreed within 1.53e-05 cycles over HTTP. The requests include
database reads and feature extraction. These results describe one local
workload. [All trials](../reports/onnx-serving.json)

To repeat after exporting the serving graph and replaying engines 1–3,
run this sequence once for each backend in the table.

```sh
FAULTWATCH_BACKEND=onnx FAULTWATCH_REDIS_URL= docker compose up --wait api
make benchmark
```
