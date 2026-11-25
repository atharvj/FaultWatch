"""Compare native and Python predictions on the checksummed FD001 test set."""

import argparse
import io
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from faultwatch.data import CYCLE, ENGINE, SENSORS, read_fd001_telemetry, verify_fd001
from faultwatch.features import WINDOW_SIZE, causal_features
from faultwatch.inference import load_predictor
from faultwatch.native_inference import load_native_predictor
from faultwatch.onnx_inference import load_onnx_predictor
from faultwatch.training import official_test_cases


def telemetry_windows(telemetry: pd.DataFrame):
    cycles, sensors, index = [], [], []
    for engine, rows in telemetry.groupby(ENGINE, sort=True):
        if len(rows) < WINDOW_SIZE:
            continue
        history = np.lib.stride_tricks.sliding_window_view(rows[CYCLE].to_numpy(), WINDOW_SIZE)
        readings = np.lib.stride_tricks.sliding_window_view(
            rows[list(SENSORS)].to_numpy(), WINDOW_SIZE, axis=0
        )
        cycles.append(history)
        sensors.append(readings.transpose(0, 2, 1))
        index.extend((engine, cycle) for cycle in history[:, -1])
    return (
        np.ascontiguousarray(np.concatenate(cycles), dtype=np.int64),
        np.ascontiguousarray(np.concatenate(sensors), dtype=np.float64),
        pd.MultiIndex.from_tuples(index, names=[ENGINE, CYCLE]),
    )


def verify(data: Path, bundle: Path, executable: Path) -> dict:
    from _faultwatch_native import features as native_features
    from _faultwatch_native import runtime_version

    verify_fd001(data, "test_FD001.txt")
    source = data / "test_FD001.txt"
    telemetry = read_fd001_telemetry(source)
    cycles, sensors, index = telemetry_windows(telemetry)
    reference = causal_features(telemetry)
    pd.testing.assert_index_equal(index, reference.index)
    native = load_native_predictor(bundle)
    observed_features = native_features(cycles, sensors)
    np.testing.assert_allclose(observed_features, reference.to_numpy(), rtol=0, atol=1e-9)
    raw_predictions = native.engine.predict(cycles, sensors)
    # The two macOS packages each embed CoreML classes. Run the reference
    # in its own process so imports cannot change either runtime's behavior.
    encoded = subprocess.check_output(
        [
            sys.executable,
            __file__,
            "--reference-only",
            "--data",
            str(data),
            "--bundle",
            str(bundle),
        ],
        timeout=60,
    )
    expected = np.load(io.BytesIO(encoded), allow_pickle=False)
    np.testing.assert_allclose(raw_predictions, expected, rtol=0, atol=1e-4)

    official, targets = official_test_cases(data)
    latest = pd.Series(raw_predictions, index=index).loc[official.index]
    torch = load_predictor(bundle).predict_latest(telemetry)
    pd.testing.assert_index_equal(latest.index, torch.index)
    np.testing.assert_allclose(latest, torch, rtol=0, atol=1e-4)
    output = subprocess.check_output(
        [str(executable.resolve()), str(bundle / "model.onnx"), str(source)], text=True, timeout=30
    )
    cli = pd.read_csv(io.StringIO(output), index_col=[ENGINE, CYCLE])
    pd.testing.assert_index_equal(cli.index, latest.index)
    np.testing.assert_array_equal(cli.iloc[:, 0].to_numpy(dtype=np.float32), latest.to_numpy())
    error = latest.to_numpy(dtype=np.float64) - targets
    return {
        "architecture": platform.machine(),
        "system": platform.system(),
        "native_onnxruntime": runtime_version(),
        "python_onnxruntime": version("onnxruntime"),
        "telemetry_rows": len(telemetry),
        "held_out_windows": len(reference),
        "official_test_endpoints": len(latest),
        "max_feature_difference": float(np.max(np.abs(observed_features - reference.to_numpy()))),
        "max_native_vs_python_onnx_cycles": float(np.max(np.abs(raw_predictions - expected))),
        "max_native_vs_pytorch_endpoint_cycles": float(np.max(np.abs(latest - torch))),
        "standalone_matches_binding": True,
        "native_endpoint_mae_cycles": float(np.mean(np.abs(error))),
        "native_endpoint_rmse_cycles": float(np.sqrt(np.mean(error**2))),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fd001"))
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/fd001-v1"))
    parser.add_argument("--executable", type=Path, default=Path(".local/native/faultwatch-native"))
    parser.add_argument("--reference-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reference_only:
        verify_fd001(args.data, "test_FD001.txt")
        features = causal_features(read_fd001_telemetry(args.data / "test_FD001.txt"))
        encoded = io.BytesIO()
        np.save(encoded, load_onnx_predictor(args.bundle).model.predict(features))
        sys.stdout.buffer.write(encoded.getvalue())
    else:
        print(json.dumps(verify(args.data, args.bundle, args.executable), indent=2))
