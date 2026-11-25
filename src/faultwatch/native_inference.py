"""Load the checked ONNX graph in the optional C++ runtime."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from faultwatch.features import WINDOW_SIZE
from faultwatch.onnx_inference import expected_metadata, verified_graph


@dataclass(frozen=True)
class NativePredictor:
    engine: Any
    model_version: str
    artifact_sha256: str

    def predict_window(self, _engine_id: int, rows: list) -> float:
        cycles = np.fromiter((row["cycle"] for row in rows), dtype=np.int64, count=WINDOW_SIZE)[
            None, :
        ]
        sensors = np.array([row["sensors"] for row in rows], dtype=np.float64)[None, :, :]
        return float(self.engine.predict(cycles, sensors)[0])


def load_native_predictor(directory: Path) -> NativePredictor:
    try:
        from _faultwatch_native import Engine
    except ImportError as error:
        raise ValueError("Build the FaultWatch native module and add it to PYTHONPATH") from error

    manifest, content, digest = verified_graph(directory)
    engine = Engine(content)
    if any(engine.metadata(key) != value for key, value in expected_metadata(manifest).items()):
        raise ValueError("ONNX model does not match the active bundle or feature schema")
    cache_identity = sha256(f"native:{digest}".encode()).hexdigest()
    return NativePredictor(engine, manifest["model_version"], cache_identity)
