"""Evaluate a CPU XGBoost candidate on the fixed FD001 protocol."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from faultwatch.data import CYCLE, ENGINE, add_training_labels, read_fd001_telemetry, verify_fd001
from faultwatch.features import causal_features
from faultwatch.models import SEED, predict_cycles
from faultwatch.training import grouped_folds, official_test_cases, validation_endpoints

PARAMETERS = {
    "n_estimators": 240,
    "max_depth": 5,
    "learning_rate": 0.05,
    "colsample_bytree": 0.9,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "random_state": SEED,
    "n_jobs": 2,
}


def scores(actual, predicted):
    return {
        "samples": len(actual),
        "mae_cycles": float(mean_absolute_error(actual, predicted)),
        "rmse_cycles": float(root_mean_squared_error(actual, predicted)),
    }


def run(data: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    verify_fd001(data, "train_FD001.txt")
    telemetry = read_fd001_telemetry(data / "train_FD001.txt")
    features = causal_features(telemetry)
    labels = telemetry.assign(rul=add_training_labels(telemetry)).set_index([ENGINE, CYCLE])["rul"]
    labels = labels.loc[features.index]
    endpoints = validation_endpoints(telemetry)
    actual, predicted = [], []
    start = perf_counter()
    for rows, _, validation_ids in grouped_folds(features):
        fold_endpoints = endpoints[endpoints.get_level_values(ENGINE).isin(validation_ids)]
        model = xgb.XGBRegressor(**PARAMETERS).fit(features.iloc[rows], labels.iloc[rows])
        actual.extend(labels.loc[fold_endpoints])
        predicted.extend(predict_cycles(model, features.loc[fold_endpoints]))
    validation = scores(actual, predicted)
    model = xgb.XGBRegressor(**PARAMETERS).fit(features, labels)
    test, target = official_test_cases(data)
    estimate = predict_cycles(model, test)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output)
    restored = xgb.XGBRegressor()
    restored.load_model(output)
    np.testing.assert_array_equal(predict_cycles(restored, test), estimate)
    return {
        "xgboost": xgb.__version__,
        "parameters": PARAMETERS,
        "training_windows": len(features),
        "validation": validation,
        "official_test": scores(target, estimate),
        "reloaded_predictions_identical": True,
        "seconds": round(perf_counter() - start, 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fd001"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fd001-xgboost.ubj"))
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output), indent=2))
