# FaultWatch

FaultWatch estimates how many cycles a turbofan engine has left from its latest
20 cycles of sensor data. It trains and runs locally on NASA's C-MAPSS FD001
simulation dataset. Replay streams held-out telemetry to a local API and
dashboard, with PostgreSQL storing one reading per engine and cycle.

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

## Results

One prediction per engine on 100 held-out NASA test trajectories. Errors are
in cycles.

| Model | MAE | RMSE |
|---|---:|---:|
| Median baseline | 36.32 | 45.08 |
| Boosted trees | **18.28** | **24.77** |
| PyTorch MLP (default) | 19.35 | 25.76 |

The MLP had the lowest MAE across three validation folds split by engine. The
boosted trees had the lowest test error. See the [evaluation protocol](docs/model_card.md)
and [dataset source and checksums](docs/dataset.md).
