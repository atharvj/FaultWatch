# FaultWatch

FaultWatch estimates remaining useful life (RUL) in cycles for simulated turbofan
engines. It uses NASA's C-MAPSS FD001 data. The planned local demo replays
telemetry into an API, stores it, and displays predictions and engine history.

## Development

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Check for a working interpreter with `uv python find 3.11`. If none is
installed, use `uv python install 3.11`.

```sh
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check .
```

The ML workflow runs locally on CPU.

This is a simulation research project. Its predictions are unsuitable for
aircraft maintenance decisions.
