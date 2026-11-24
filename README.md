# FaultWatch

FaultWatch estimates how many cycles a turbofan engine has left from its latest
20 cycles of sensor data. It trains and runs locally on NASA's C-MAPSS FD001
simulation dataset.

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
