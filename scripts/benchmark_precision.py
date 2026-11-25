"""Measure CPU precision variants of the fixed FD001 MLP on held-out endpoints."""

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def measure(model: Path, features: Path, dtype: str) -> dict:
    import resource
    from time import perf_counter_ns

    import numpy as np
    import onnxruntime as ort

    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    values = np.load(features, allow_pickle=False).astype(dtype, copy=False)
    result = session.run(None, {"features": values})[0]
    if result.shape != (len(values),) or not np.isfinite(result).all():
        raise ValueError(f"{model.name} returned invalid predictions")
    timings = {}
    for batch in (1, 32, 100):
        inputs = {"features": values[:batch]}
        for _ in range(50):
            session.run(None, inputs)
        samples = []
        for _ in range(300):
            start = perf_counter_ns()
            session.run(None, inputs)
            samples.append(perf_counter_ns() - start)
        median, tail = np.percentile(samples, (50, 95))
        timings[str(batch)] = {
            "p50_us": float(median / 1000),
            "p95_us": float(tail / 1000),
            "windows_per_second_at_p50": float(batch * 1e9 / median),
        }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "runtime": ort.__version__,
        "peak_process_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024),
        "batches": timings,
        "predictions": result.tolist(),
    }


def compare(data: Path, bundle: Path, output: Path) -> dict:
    import numpy as np
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from torch import nn

    from faultwatch.features import FEATURE_COLUMNS
    from faultwatch.inference import load_manifest, load_model
    from faultwatch.models import TARGET_SCALE
    from faultwatch.onnx_inference import verified_graph
    from faultwatch.training import official_test_cases

    class Graph(nn.Module):
        def __init__(self, model, dtype):
            super().__init__()
            self.network = copy.deepcopy(model.network).to(dtype)
            self.register_buffer("center", torch.as_tensor(model.center, dtype=dtype))
            self.register_buffer("scale", torch.as_tensor(model.scale, dtype=dtype))

        def forward(self, features):
            values = (features - self.center) / self.scale
            return (self.network(values).squeeze(1) * TARGET_SCALE).clamp_min(0)

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model = load_model(bundle, "torch_mlp", load_manifest(bundle))
    features, labels = official_test_cases(data)
    feature_path = output / "features.npy"
    np.save(feature_path, np.ascontiguousarray(features.to_numpy(), dtype=np.float64))
    _, graph, _ = verified_graph(bundle)
    (output / "mixed.onnx").write_bytes(graph)

    for name, dtype in (("fp64", torch.float64), ("fp32", torch.float32)):
        torch.onnx.export(
            Graph(model, dtype).eval(),
            (torch.zeros((2, len(FEATURE_COLUMNS)), dtype=dtype),),
            output / f"{name}.onnx",
            input_names=["features"],
            output_names=["rul_cycles"],
            dynamic_shapes=({0: torch.export.Dim("batch", min=1)},),
            opset_version=18,
            dynamo=True,
            external_data=False,
            verbose=False,
        )
    quantize_dynamic(
        output / "fp32.onnx",
        output / "int8.onnx",
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    variants, reference = {}, None
    for name, dtype in (
        ("mixed", "float64"),
        ("fp64", "float64"),
        ("fp32", "float32"),
        ("int8", "float32"),
    ):
        path = output / f"{name}.onnx"
        nodes = sorted({node.op_type for node in onnx.load(path).graph.node})
        if name == "int8" and not {"MatMulInteger", "QLinearMatMul"}.intersection(nodes):
            raise ValueError("INT8 transformation did not produce a quantized multiplication")
        try:
            trial = json.loads(
                subprocess.check_output(
                    [sys.executable, __file__, "--worker", str(path), str(feature_path), dtype],
                    env={**os.environ, "ORT_DISABLE_TELEMETRY": "1"},
                    text=True,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
            )
        except subprocess.CalledProcessError as error:
            if name != "int8":
                raise
            variants[name] = {
                "supported": False,
                "nodes": nodes,
                "error": error.stderr.splitlines()[-1],
            }
            continue
        scores = np.asarray(trial.pop("predictions"))
        if reference is None:
            reference = scores
        errors = scores - labels
        variants[name] = {
            "supported": True,
            "model_bytes": path.stat().st_size,
            "nodes": nodes,
            "mae_cycles": float(np.mean(np.abs(errors))),
            "rmse_cycles": float(np.sqrt(np.mean(errors**2))),
            "max_drift_from_serving_cycles": float(np.max(np.abs(reference - scores))),
            **trial,
        }
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "endpoints": len(labels),
        "cpu_threads": 2,
        "warmup_calls": 50,
        "timed_calls": 300,
        "variants": variants,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fd001"))
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/fd001-v1"))
    parser.add_argument("--output", type=Path, default=Path(".local/native-precision"))
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        graph, raw, data_type = args.worker
        report = measure(Path(graph), Path(raw), data_type)
    else:
        report = compare(args.data, args.bundle, args.output)
    print(json.dumps(report, indent=2))
