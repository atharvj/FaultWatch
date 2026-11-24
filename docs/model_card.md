# FD001 model and evaluation

FaultWatch predicts uncapped remaining useful life in cycles. The model uses
the current cycle number and each of 21 sensors' last value, mean and change
over a trailing 20-cycle window. Engine IDs group readings. FD001 covers one
operating condition, so the model drops the three operating settings.

Training labels are the last cycle of each complete trajectory minus the
current cycle. The median baseline uses the median label over training
windows. Histogram gradient boosting and a PyTorch MLP with 64 and 32 hidden units
use the same inputs. The MLP fits a standard scaler on each fitting fold and trains with
AdamW and Smooth L1 loss.

Model selection used three engine-grouped folds across the 100 NASA training
engines, with seed 27. Each validation engine was scored at 40%, 60%, and 80%
of its lifespan. Features only use readings through each cutoff. The MLP had
the lowest pooled MAE over the 300 validation cutoffs (23.28 cycles), so it is
the default. The candidates were refit on 18,731 windows, then scored once per
engine at the 100 official test endpoints.

The evaluation ran on an Apple M2 Pro CPU with two threads. Full-precision
scores, versions and training parameters are in the
[metrics file](../reports/fd001-v1.metrics.json). The
[split manifest](../reports/fd001-v1.splits.json) records the engine IDs and
validation cutoffs.

The local bundle in `artifacts/fd001-v1` stores the feature schema and model
checksums. The scikit-learn pipelines use skops. The MLP stores weights and
scaler arrays in a NumPy archive loaded with pickle disabled. Inference
requires 20 consecutive readings and returns a nonnegative point estimate.

FD001 covers one simulated compressor degradation mode. The results are
specific to this benchmark. The model has no calibrated uncertainty and is
unsuitable for aircraft maintenance decisions.
