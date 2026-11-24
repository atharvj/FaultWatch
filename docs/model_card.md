# Model card · FD001 v1

## Intended use

Estimate remaining useful life (RUL) in cycles for simulated C-MAPSS FD001
engines with 20 consecutive observations. The validation-selected model is a
PyTorch MLP. The [dataset description](dataset.md) records the archive,
checksums and schema.

## Features and training

- Each window ends at the cycle being scored. Inputs are that cycle number and
  the last value, mean and first-to-last change for each of 21 sensor channels.
  Engine ID only groups observations. FD001 has one operating condition, so
  the model omits the three operating settings.
- Training targets are uncapped cycles to the last row of the engine's
  run-to-failure trajectory. Training and validation use the same feature
  extraction as inference. Fewer than 20 consecutive cycles gives
  insufficient history. Predictions have a lower bound of zero cycles.
- The median baseline predicts the median RUL of the fitting windows.
  The scikit-learn model fits histogram gradient boosting on the same features.
  The MLP uses 64 and 32 hidden units, ReLU, AdamW and a Smooth L1 loss. It fits
  its `StandardScaler` on the fitting fold and scales targets by 100 cycles
  during training. Neither tree nor neural training consumes validation rows.
- Three shuffled `GroupKFold` folds with seed 27 split the 100 official training
  engines. Each validation engine contributes three simulated endpoints at
  40%, 60% and 80% of its trajectory. The percentages use the complete training
  lifespan to construct the evaluation cases. Features at each cutoff use
  telemetry only through that cycle.
- The lowest pooled validation MAE determines selection. All candidates are
  refit on all 18,731 eligible training windows, then scored on the 100
  official test engines. Each test engine contributes its last observed
  cycle, joined to the NASA RUL vector by engine ID.

## Measured results

All error values are in cycles. MAE is the mean absolute error. RMSE takes
the square root of the mean squared error and penalizes large misses more.

| Model | Grouped validation MAE (300 cutoffs) | Grouped validation RMSE | Official test MAE (100 endpoints) | Official test RMSE |
|---|---:|---:|---:|---:|
| Median baseline | 34.15 | 40.72 | 36.32 | 45.08 |
| Histogram gradient boosting | 24.64 | 33.93 | **18.28** | **24.77** |
| PyTorch MLP (selected) | **23.28** | **31.08** | 19.35 | 25.76 |

The selected MLP improved official test MAE by 46.7% relative to the median
baseline. Boosted trees had the lowest official test errors. The MLP remains
the default because its selection preceded official test evaluation.

Run on 2026-11-23 with CPython 3.11.12 on an Apple M2 Pro, native arm64 CPU
with two training threads. Full-precision [metrics](../reports/fd001-v1.metrics.json)
and [split manifest](../reports/fd001-v1.splits.json) are committed.

## Artifact and limits

The local bundle contains the versioned feature schema, artifact hashes and all
three estimators. scikit-learn estimators use skops with a fixed allowlist for
the histogram tree class. The MLP uses a NumPy archive with numeric weights and
scaler arrays loaded with pickle disabled. Training checks that all restored
estimators reproduce predictions from their in-memory versions on the 100
official endpoints. Use bundles generated locally with the locked dependencies.

FD001 is simulated and covers one operating condition and one fault mode.
The official endpoint metrics describe this benchmark and can vary with engine
lifespan and where telemetry ends. Nearby validation cutoffs for an engine
are correlated. This model provides point estimates with no calibrated
uncertainty and is unsuitable for aircraft maintenance decisions.
