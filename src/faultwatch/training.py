"""Engine-grouped model selection and official endpoint evaluation."""

import json
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from faultwatch.data import (
    CYCLE,
    ENGINE,
    FD001_HASHES,
    add_training_labels,
    read_fd001_telemetry,
    read_test_rul,
    verify_fd001,
)
from faultwatch.features import WINDOW_SIZE, causal_features
from faultwatch.inference import MODEL_FILES, MODEL_VERSION, load_manifest, load_model, save_models
from faultwatch.models import MODEL_PARAMS, SEED, fit_candidates, predict_cycles

FOLD_COUNT = 3
CUTOFF_PERCENTAGES = (40, 60, 80)
CPU_THREADS = 2


def validation_endpoints(telemetry: pd.DataFrame) -> pd.MultiIndex:
    last_cycles = telemetry.groupby(ENGINE)[CYCLE].max()
    cases = [
        (engine, max(WINDOW_SIZE, last * percent // 100))
        for engine, last in last_cycles.items()
        for percent in CUTOFF_PERCENTAGES
    ]
    return pd.MultiIndex.from_tuples(cases, names=(ENGINE, CYCLE))


def grouped_folds(features: pd.DataFrame):
    ids = features.index.get_level_values(ENGINE).to_numpy()
    splitter = GroupKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=SEED)
    for train_rows, validation_rows in splitter.split(features, groups=ids):
        fit_ids = np.unique(ids[train_rows])
        validation_ids = np.unique(ids[validation_rows])
        if np.intersect1d(fit_ids, validation_ids).size:
            raise RuntimeError("An engine appears in both sides of a validation fold")
        yield train_rows, fit_ids, validation_ids


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "sample_count": len(labels),
        "mae_cycles": float(mean_absolute_error(labels, predictions)),
        "rmse_cycles": float(root_mean_squared_error(labels, predictions)),
    }


def _validate_candidates(
    features: pd.DataFrame, labels: pd.Series, endpoints: pd.MultiIndex
) -> tuple[dict, list[dict]]:
    validation_predictions = {name: [] for name in MODEL_FILES}
    validation_labels = []
    folds = []
    for fold, (train_rows, fit_ids, validation_ids) in enumerate(grouped_folds(features), 1):
        fold_endpoints = endpoints[endpoints.get_level_values(ENGINE).isin(validation_ids)]
        fit_labels = labels.iloc[train_rows].to_numpy()
        models = fit_candidates(features.iloc[train_rows], fit_labels)
        validation_labels.append(labels.loc[fold_endpoints].to_numpy())
        for name, model in models.items():
            validation_predictions[name].append(predict_cycles(model, features.loc[fold_endpoints]))
        folds.append(
            {
                "fold": fold,
                "fit_engine_ids": fit_ids.tolist(),
                "validation_engine_ids": validation_ids.tolist(),
                "validation_endpoints": [
                    {"engine_id": int(engine), "cycle": int(cycle)}
                    for engine, cycle in fold_endpoints
                ],
            }
        )
    actual = np.concatenate(validation_labels)
    validation_metrics = {
        name: _metrics(actual, np.concatenate(fold_predictions))
        for name, fold_predictions in validation_predictions.items()
    }
    return validation_metrics, folds


def official_test_cases(directory: Path) -> tuple[pd.DataFrame, np.ndarray]:
    verify_fd001(directory, "test_FD001.txt", "RUL_FD001.txt")
    test = read_fd001_telemetry(directory / "test_FD001.txt")
    labels = read_test_rul(directory / "RUL_FD001.txt")
    features = causal_features(test, latest_only=True)
    ids = features.index.get_level_values(ENGINE)
    if not ids.equals(labels.index):
        raise ValueError("Official test endpoints do not align with the NASA RUL vector")
    return features, labels.loc[ids].to_numpy()


def evaluate(directory: Path, bundle: Path) -> dict:
    features, actual = official_test_cases(directory)
    manifest = load_manifest(bundle)
    with threadpool_limits(CPU_THREADS):
        return {
            name: _metrics(actual, predict_cycles(load_model(bundle, name, manifest), features))
            for name in MODEL_FILES
        }


def train_and_evaluate(directory: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose a new --output directory.")
    verify_fd001(directory, "train_FD001.txt")
    train = read_fd001_telemetry(directory / "train_FD001.txt")
    features = causal_features(train)
    labels = train.assign(rul=add_training_labels(train)).set_index([ENGINE, CYCLE])["rul"]
    labels = labels.loc[features.index]
    endpoints = validation_endpoints(train)

    torch.set_num_threads(CPU_THREADS)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(CPU_THREADS):
        validation, folds = _validate_candidates(features, labels, endpoints)
        selected = min(validation, key=lambda name: validation[name]["mae_cycles"])
        final = fit_candidates(features, labels.to_numpy())

        # Access the official evaluation cases only after selection/refitting.
        test_features, test_actual = official_test_cases(directory)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as staging:
            bundle = Path(staging)
            save_models(bundle, final, selected)
            manifest = load_manifest(bundle)
            test = {}
            for name in MODEL_FILES:
                original = predict_cycles(final[name], test_features)
                restored = predict_cycles(load_model(bundle, name, manifest), test_features)
                np.testing.assert_allclose(restored, original, rtol=1e-7, atol=1e-7)
                test[name] = _metrics(test_actual, restored)

            metrics = {
                "dataset": "NASA C-MAPSS FD001",
                "model_version": MODEL_VERSION,
                "selected_model": selected,
                "selection_metric": "grouped_validation.mae_cycles",
                "target": "uncapped_remaining_cycles",
                "window_cycles": WINDOW_SIZE,
                "seed": SEED,
                "cpu_threads": CPU_THREADS,
                "python": platform.python_version(),
                "platform": f"{platform.system()} {platform.machine()}",
                "versions": {
                    name: version(name)
                    for name in ("numpy", "pandas", "scikit-learn", "skops", "torch")
                },
                "model_parameters": MODEL_PARAMS,
                "training_trajectory_count": int(train[ENGINE].nunique()),
                "training_window_count": len(features),
                "grouped_validation": validation,
                "official_test_endpoints": test,
            }
            splits = {
                "dataset_sha256": FD001_HASHES,
                "seed": SEED,
                "method": "shuffled_group_k_fold",
                "validation_cutoff_percentages": list(CUTOFF_PERCENTAGES),
                "folds": folds,
                "final_fit_nasa_training_engine_ids": sorted(train[ENGINE].unique().tolist()),
                "official_test_engine_ids": test_features.index.get_level_values(ENGINE).tolist(),
            }
            (bundle / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            (bundle / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
            # Inference containers read the published bundle as a nonroot user.
            bundle.chmod(0o755)
            bundle.rename(output)

    return metrics
