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

With the model trained, start PostgreSQL and the API. The container uses a
localhost port and a named volume. The password below is for this local demo.

```sh
docker run -d --name faultwatch-postgres \
  -p 127.0.0.1:55432:5432 -v faultwatch-pgdata:/var/lib/postgresql/data \
  -e POSTGRES_USER=faultwatch -e POSTGRES_DB=faultwatch \
  -e POSTGRES_PASSWORD=faultwatch-local postgres:17.12-alpine
export FAULTWATCH_DATABASE_URL=postgresql+psycopg://faultwatch:faultwatch-local@127.0.0.1:55432/faultwatch
uv run --locked alembic upgrade head
uv run --locked faultwatch serve
```

If the container already exists, use `docker start faultwatch-postgres`.
Open <http://127.0.0.1:8000> and run this in a second terminal.

```sh
uv run --locked faultwatch replay
curl -fsS http://127.0.0.1:8000/v1/engines/1/prediction
curl -fsS 'http://127.0.0.1:8000/v1/engines/1/history?limit=20'
```

Replay sends FD001 test engines 1–3 in cycle order. Use `--engine` to select
engines and `--rate` to set events per second. `POST /v1/telemetry` returns
`201` for new readings, `200` for identical retries and `409` for conflicting
readings at an existing engine and cycle. Prediction requires 20 consecutive
cycles. The API schema is at [/docs](http://127.0.0.1:8000/docs).

To include PostgreSQL integration tests, run
`FAULTWATCH_TEST_DATABASE_URL="$FAULTWATCH_DATABASE_URL" uv run --locked pytest -q`.
The tests use a temporary schema.

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
