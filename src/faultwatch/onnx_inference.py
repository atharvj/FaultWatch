"""Load the checked ONNX export as an optional prediction backend."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from faultwatch.features import FEATURE_COLUMNS
from faultwatch.inference import Predictor, load_manifest

MODEL_FILE = "model.onnx"
THREADS = 2


def session_options():
    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    return options


class OnnxModel:
    def __init__(self, session):
        self.session = session

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        values = np.ascontiguousarray(features.to_numpy(), dtype=np.float64)
        return self.session.run(["rul_cycles"], {"features": values})[0]


def load_onnx_predictor(directory: Path) -> Predictor:
    manifest = load_manifest(directory)
    if manifest["selected_model"] != "torch_mlp":
        raise ValueError("ONNX serving requires the selected PyTorch MLP")
    path = directory / MODEL_FILE
    content = path.read_bytes()
    digest = sha256(content).hexdigest()
    if digest != path.with_suffix(".onnx.sha256").read_text().strip():
        raise ValueError("ONNX model checksum mismatch")

    ort.disable_telemetry_events()
    try:
        session = ort.InferenceSession(
            content, session_options(), providers=["CPUExecutionProvider"]
        )
    except Exception as error:
        raise ValueError("Invalid ONNX model graph") from error
    metadata = session.get_modelmeta().custom_metadata_map
    inputs = session.get_inputs()
    if (
        metadata.get("model_version") != manifest["model_version"]
        or metadata.get("source_artifact_sha256") != manifest["artifact_sha256"]["torch_mlp"]
        or metadata.get("feature_columns") != json.dumps(FEATURE_COLUMNS)
        or len(inputs) != 1
        or inputs[0].name != "features"
        or inputs[0].type != "tensor(double)"
        or len(inputs[0].shape) != 2
        or inputs[0].shape[1] != len(FEATURE_COLUMNS)
        or not isinstance(inputs[0].shape[0], str)
        or [output.name for output in session.get_outputs()] != ["rul_cycles"]
    ):
        raise ValueError("ONNX model does not match the active bundle or feature schema")
    return Predictor(OnnxModel(session), "torch_mlp", manifest["model_version"], digest)
