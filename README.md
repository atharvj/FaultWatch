# FaultWatch

FaultWatch estimates how many cycles a turbofan engine has left from its latest
20 cycles of sensor data. It trains and runs locally on NASA's C-MAPSS FD001
simulation dataset. Replay streams held-out telemetry to a local API and
dashboard, with PostgreSQL storing one reading per engine and cycle.

![Remaining-life prediction and sensor history in the FaultWatch dashboard](docs/dashboard.png)

## Run it

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv sync --locked
uv run --locked faultwatch fetch
uv run --locked faultwatch train
uv run --locked faultwatch predict data/fd001/test_FD001.txt --engine 1
```

The prediction command reports estimated remaining cycles at engine 1's last
recorded cycle. Run `uv run --locked faultwatch evaluate` to score the saved
models or `uv run --locked pytest -q` to run the tests.

## Live dashboard

With Docker running and the model trained, launch the stack and replay test
engines 1–3.

```sh
make demo
make replay
curl -fsS http://127.0.0.1:8000/v1/engines/1/prediction
```

Open the [dashboard](http://127.0.0.1:8000) or [API docs](http://127.0.0.1:8000/docs).
Compose runs migrations before starting the API. `make stop` stops the stack
and keeps the database volume.

Replay accepts `--engine` and `--rate` through `faultwatch replay`.
`POST /v1/telemetry` returns `201` for new readings, `200` for identical retries
and `409` if the same engine and cycle already has different readings.
Predictions require 20 consecutive cycles.

Run `make test` for lint and pytest. With Compose running, set the local
database URL to include PostgreSQL tests.

```sh
FAULTWATCH_TEST_DATABASE_URL=postgresql+psycopg://faultwatch:faultwatch-local@127.0.0.1:55433/faultwatch make test
```

## Monitoring

`make observe` adds Prometheus and a provisioned [Grafana dashboard](http://127.0.0.1:3000/d/faultwatch/faultwatch-operations).
Replay the test engines, then run `make benchmark` to exercise the metrics.
`make cache` enables Redis for repeated predictions.
See the [local performance measurements](docs/benchmark.md).

## Kubernetes

For a local Kubernetes run, install [kind](https://kind.sigs.k8s.io/docs/user/quick-start/)
and kubectl. With the model trained and Docker running, use:

```sh
make kind-up
make kind-forward
```

In another terminal, run `make replay FAULTWATCH_PORT=18000` and open the
[dashboard](http://127.0.0.1:18000). The single-node cluster uses a project-local
kubeconfig, a read-only model mount and a PostgreSQL volume that survives pod
replacement. `make kind-down` deletes the cluster and its local database.

## Model tools

```sh
uv run --locked --group experiments faultwatch track
uv run --locked --group experiments faultwatch runs
uv run --locked --group experiments faultwatch export-onnx
```

MLflow keeps the fitted candidates, parameters and scores in a local store.
ONNX export checks the graph against PyTorch on the official test endpoints.

## Artifact storage

The [Terraform lab](infra/terraform/s3) provisions an S3-compatible bucket on
loopback, transfers the fitted model and manifest, verifies their checksums,
and removes the bucket.

```sh
uv run --locked --group aws python scripts/s3_lab.py
```

Use Terraform 1.17 and a trained bundle. The lab uses Moto, fixed local
credentials, and a loopback-only endpoint. See the [apply and teardown results](reports/s3-lab.json).
A live AWS deployment requires an eligible account and separate approval.

## Kafka replay

Start the optional single-node broker with `docker compose --profile kafka up --wait kafka`.
With the Compose database running and the [FD001 data](docs/dataset.md) fetched:

```sh
export FAULTWATCH_DATABASE_URL=postgresql+psycopg://faultwatch:faultwatch-local@127.0.0.1:55433/faultwatch
uv run --locked --group kafka python -m faultwatch.kafka_replay publish
uv run --locked --group kafka python -m faultwatch.kafka_replay consume
```

Engine IDs are partition keys. The worker commits offsets after the database
transaction and accepts identical retries. The direct HTTP replay remains
available. Run `uv run --locked --group kafka python scripts/kafka_lab.py`
for the [restart and repeat-delivery check](reports/kafka-replay.json).

## Spark feature export

With Java 21 and the FD001 files available, run
`uv run --locked --group spark python scripts/spark_features.py` to write
train and test Parquet partitions to `artifacts/fd001-parquet`. The job
reads them back and checks every feature against the pandas pipeline.
[Verification results](reports/spark-parquet.json)

## Ask the documentation

The local Q&A tool retrieves public project passages from PostgreSQL with
pgvector. It generates a short answer with MiniLM embeddings and FLAN-T5,
and returns the source passage or abstains.

```sh
docker compose --profile rag up --wait rag-db
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --locked --group rag python scripts/rag.py fetch
uv run --locked --group rag python scripts/rag.py index
uv run --locked --group rag python scripts/rag.py ask 'How many consecutive readings does inference require?'
uv run --locked --group rag python scripts/rag.py evaluate
```

The one-time fetch downloads public model weights. Indexing and questions
run locally. [Evaluation results](reports/rag-evaluation.json)

## Helm deployment

After `make kind-up`, run `sh scripts/helm_lab.sh` with Helm and kubectl
installed. The chart deploys a separate API release using the demo's existing
PostgreSQL service and model mount. The check installs, upgrades, verifies
readiness and removes its temporary release. [Release results](reports/helm-lab.json)

## XGBoost comparison

Run `uv run --locked --group xgboost python scripts/xgb_compare.py` to
fit a CPU histogram model on the same grouped validation folds and test
endpoints as the original candidates. XGBoost requires OpenMP on macOS
(`brew install libomp`). The experiment saves a UBJSON model and verifies
that reload preserves every test prediction. [Full results](reports/xgboost-fd001.json)

## Results

One prediction per engine on 100 held-out NASA test trajectories. Errors are
in cycles.

| Model | MAE | RMSE |
|---|---:|---:|
| Median baseline | 36.32 | 45.08 |
| Boosted trees | **18.28** | **24.77** |
| PyTorch MLP (default) | 19.35 | 25.76 |
| XGBoost | 18.58 | 25.55 |

The MLP had the lowest MAE across three validation folds split by engine. The
boosted trees had the lowest test error. See the [evaluation protocol](docs/model_card.md)
and [dataset source and checksums](docs/dataset.md).
