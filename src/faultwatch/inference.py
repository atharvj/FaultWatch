"""Load verified local model bundles and predict from raw telemetry."""

import io
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from faultwatch.data import TELEMETRY_COLUMNS
from faultwatch.features import FEATURE_COLUMNS, WINDOW_SIZE, causal_features
from faultwatch.models import Regressor, TorchMLP, mlp_network, predict_cycles

FORMAT_VERSION = 1
MODEL_VERSION = "fd001-v1"
MODEL_FILES = {
    "median": "median.skops",
    "hist_gradient_boosting": "hist_gradient_boosting.skops",
    "torch_mlp": "torch_mlp.npz",
}
TREE_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def feature_schema() -> dict:
    return {
        "telemetry_columns": list(TELEMETRY_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "history_cycles": WINDOW_SIZE,
        "target_unit": "cycles",
    }


def save_models(directory: Path, models: dict[str, Regressor], selected: str) -> None:
    checksums = {}
    for name, model in models.items():
        path = directory / MODEL_FILES[name]
        if isinstance(model, TorchMLP):
            weights = {
                key: tensor.detach().cpu().numpy()
                for key, tensor in model.network.state_dict().items()
            }
            np.savez_compressed(path, center=model.center, scale=model.scale, **weights)
        else:
            skops.io.dump(model, path)
        checksums[name] = sha256(path.read_bytes()).hexdigest()

    manifest = {
        "format_version": FORMAT_VERSION,
        "model_version": MODEL_VERSION,
        "selected_model": selected,
        "schema": feature_schema(),
        "artifact_sha256": checksums,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _load_mlp(content: bytes) -> TorchMLP:
    network = mlp_network()
    expected = network.state_dict()
    with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
        if set(arrays.files) != {"center", "scale", *expected}:
            raise ValueError("Unexpected MLP artifact keys")
        center, scale = arrays["center"], arrays["scale"]
        if (
            center.dtype != np.float64
            or scale.dtype != np.float64
            or center.shape != (len(FEATURE_COLUMNS),)
            or scale.shape != center.shape
            or not np.isfinite(center).all()
            or not np.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("Invalid MLP preprocessing parameters")
        weights = {}
        for key, tensor in expected.items():
            values = arrays[key]
            if (
                values.shape != tuple(tensor.shape)
                or values.dtype != np.float32
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"Invalid MLP weights for {key}")
            weights[key] = torch.from_numpy(values)
    network.load_state_dict(weights)
    return TorchMLP(network.eval(), center, scale)


def load_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Incompatible FaultWatch model manifest")
    checksums = manifest.get("artifact_sha256")
    if (
        manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("model_version") != MODEL_VERSION
        or manifest.get("schema") != feature_schema()
        or manifest.get("selected_model") not in tuple(MODEL_FILES)
        or not isinstance(checksums, dict)
        or set(checksums) != set(MODEL_FILES)
    ):
        raise ValueError("Incompatible FaultWatch model manifest")
    return manifest


def load_model(directory: Path, name: str, manifest: dict) -> Regressor:
    content = (directory / MODEL_FILES[name]).read_bytes()
    if sha256(content).hexdigest() != manifest["artifact_sha256"][name]:
        raise ValueError(f"Artifact checksum mismatch for {name}")
    if name == "torch_mlp":
        return _load_mlp(content)
    is_tree = name == "hist_gradient_boosting"
    model = skops.io.loads(content, trusted=TREE_TYPES if is_tree else [])
    expected = HistGradientBoostingRegressor if is_tree else DummyRegressor
    if not isinstance(model, Pipeline) or not isinstance(model[-1], expected):
        raise ValueError(f"Unexpected estimator in {name} artifact")
    return model


@dataclass(frozen=True)
class Predictor:
    model: Regressor
    model_name: str
    model_version: str
    artifact_sha256: str

    def predict_latest(self, telemetry: pd.DataFrame) -> pd.Series:
        features = causal_features(telemetry, latest_only=True)
        predictions = np.empty(0) if features.empty else predict_cycles(self.model, features)
        return pd.Series(predictions, index=features.index, name="predicted_rul_cycles")

    def predict_window(self, engine_id: int, rows: list) -> float:
        frame = pd.DataFrame(
            [(engine_id, row["cycle"], *row["settings"], *row["sensors"]) for row in rows],
            columns=TELEMETRY_COLUMNS,
        )
        return float(self.predict_latest(frame).iloc[0])


def load_predictor(directory: Path) -> Predictor:
    """Load the validation-selected model from a locally generated artifact bundle."""
    manifest = load_manifest(directory)
    name = manifest["selected_model"]
    return Predictor(
        load_model(directory, name, manifest),
        name,
        manifest["model_version"],
        manifest["artifact_sha256"][name],
    )
