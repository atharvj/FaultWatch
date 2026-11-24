# FaultWatch

FaultWatch estimates remaining useful life (RUL) in cycles for simulated turbofan
engines. The current workflow fetches NASA C-MAPSS FD001, trains and evaluates
three candidates, and predicts from saved models and raw telemetry.

## Development

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Check for a working interpreter with `uv python find 3.11`. If none is
installed, use `uv python install 3.11`.

```sh
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check .
```

## Run the FD001 workflow

```sh
uv run --locked faultwatch fetch
uv run --locked faultwatch train
uv run --locked faultwatch evaluate
uv run --locked faultwatch predict data/fd001/test_FD001.txt --engine 1
```

The model bundle is generated at `artifacts/fd001-v1`. To repeat training, use
`uv run --locked faultwatch train --output artifacts/fd001-rerun`. Predictions require 20
consecutive cycles per engine. Dataset files and model bundles are local.

The validation-selected PyTorch model recorded 19.35 cycles MAE and 25.76
cycles RMSE on 100 NASA test endpoints. See the
[model card](docs/model_card.md) for the full comparison and
[dataset provenance](docs/dataset.md) for the download and checksums.

The ML workflow runs locally on CPU.

This is a simulation research project. Its predictions are unsuitable for
aircraft maintenance decisions.
