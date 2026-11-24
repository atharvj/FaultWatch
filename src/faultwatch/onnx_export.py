"""Export and check the fitted MLP with ONNX Runtime."""

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from torch import nn

from faultwatch.features import FEATURE_COLUMNS
from faultwatch.inference import load_manifest, load_model
from faultwatch.models import TARGET_SCALE, TorchMLP, predict_cycles
from faultwatch.onnx_inference import THREADS, session_options

WARMUP = 30
OPSET = 18
ATOL = 1e-4
RTOL = 1e-5


class PredictionGraph(nn.Module):
    def __init__(self, model: TorchMLP):
        super().__init__()
        self.network = model.network
        self.register_buffer("center", torch.from_numpy(model.center))
        self.register_buffer("scale", torch.from_numpy(model.scale))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        inputs = ((features - self.center) / self.scale).float()
        return (self.network(inputs).squeeze(1) * TARGET_SCALE).clamp_min(0)


def _measure(predict, repeats: int) -> dict:
    for _ in range(WARMUP):
        predict()
    samples = []
    for _ in range(repeats):
        start = perf_counter_ns()
        predict()
        samples.append((perf_counter_ns() - start) / 1000)
    return {
        "p50_us": round(float(np.percentile(samples, 50)), 3),
        "p95_us": round(float(np.percentile(samples, 95)), 3),
    }


def export_and_validate(
    bundle: Path, output: Path, features: pd.DataFrame, repeats: int = 300
) -> dict:
    checksum = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or checksum.exists():
        raise FileExistsError(f"{output} already exists. Choose a new --output file.")
    if repeats < 1 or features.empty:
        raise ValueError("ONNX verification needs at least one repetition and feature row")
    manifest = load_manifest(bundle)
    model = load_model(bundle, "torch_mlp", manifest)
    if not isinstance(model, TorchMLP):
        raise ValueError("Expected the FaultWatch PyTorch MLP")
    ort.disable_telemetry_events()
    torch.set_num_threads(THREADS)
    expected = predict_cycles(model, features)
    values = np.ascontiguousarray(features.to_numpy(), dtype=np.float64)
    graph = PredictionGraph(model).eval()
    example = torch.zeros((2, len(FEATURE_COLUMNS)), dtype=torch.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        path = Path(temp) / "model.onnx"
        torch.onnx.export(
            graph,
            (example,),
            path,
            input_names=["features"],
            output_names=["rul_cycles"],
            dynamic_shapes=({0: torch.export.Dim("batch", min=1)},),
            opset_version=OPSET,
            dynamo=True,
            external_data=False,
            verbose=False,
        )
        proto = onnx.load(path)
        onnx.helper.set_model_props(
            proto,
            {
                "model_version": manifest["model_version"],
                "source_artifact_sha256": manifest["artifact_sha256"]["torch_mlp"],
                "feature_columns": json.dumps(FEATURE_COLUMNS),
                "target_unit": "cycles",
            },
        )
        onnx.checker.check_model(proto)
        onnx.save(proto, path)
        session = ort.InferenceSession(
            str(path), session_options(), providers=["CPUExecutionProvider"]
        )
        observed = session.run(["rul_cycles"], {"features": values})[0]
        np.testing.assert_allclose(observed, expected, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(
            session.run(["rul_cycles"], {"features": values[:1]})[0],
            predict_cycles(model, features.iloc[:1]),
            atol=ATOL,
            rtol=RTOL,
        )

        timings = {}
        with torch.inference_mode():
            for batch in sorted({1, len(values)}):
                inputs = values[:batch]

                def run_torch():
                    return graph(torch.from_numpy(inputs)).numpy()

                def run_ort():
                    return session.run(["rul_cycles"], {"features": inputs})[0]

                timings[str(batch)] = {
                    "pytorch": _measure(run_torch, repeats),
                    "onnxruntime": _measure(run_ort, repeats),
                }

        digest = sha256(path.read_bytes()).hexdigest()
        proof = Path(temp) / "model.sha256"
        proof.write_text(digest + "\n")
        output.hardlink_to(path)
        checksum.hardlink_to(proof)

    return {
        "model": "torch_mlp",
        "input_shape": ["batch", len(FEATURE_COLUMNS)],
        "input_dtype": "float64",
        "opset": OPSET,
        "onnxruntime": ort.__version__,
        "pytorch": torch.__version__,
        "cpu_threads": THREADS,
        "artifact_bytes": output.stat().st_size,
        "artifact_sha256": digest,
        "samples": len(values),
        "max_absolute_difference_cycles": float(np.max(np.abs(observed - expected))),
        "tolerance_cycles": {"atol": ATOL, "rtol": RTOL},
        "warmup_calls": WARMUP,
        "timed_calls": repeats,
        "latency_by_batch": timings,
    }
