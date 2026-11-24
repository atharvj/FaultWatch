import json

import pytest

from faultwatch.features import FEATURE_COLUMNS, causal_features
from faultwatch.models import MODEL_PARAMS, SEED


def test_onnx_export_matches_torch_for_dynamic_batches(tmp_path, model_bundle, trajectories):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from faultwatch.onnx_export import export_and_validate

    bundle, _ = model_bundle
    features = causal_features(trajectories, latest_only=True)
    path = tmp_path / "model.onnx"
    report = export_and_validate(bundle, path, features, repeats=3)
    assert report["input_shape"] == ["batch", len(FEATURE_COLUMNS)]
    assert report["samples"] == len(features)
    assert report["max_absolute_difference_cycles"] < 1e-4
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape[1] == len(FEATURE_COLUMNS)
    assert isinstance(session.get_inputs()[0].shape[0], str)
    assert json.loads(session.get_modelmeta().custom_metadata_map["feature_columns"]) == list(
        FEATURE_COLUMNS
    )


def test_mlflow_persists_scores_and_artifacts(tmp_path, model_bundle):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    from faultwatch.inference import MODEL_FILES
    from faultwatch.tracking import list_runs, track_bundle

    bundle, _ = model_bundle
    validation = {
        model: {"mae_cycles": 3 - index, "rmse_cycles": 4 - index, "sample_count": 6}
        for index, model in enumerate(MODEL_FILES)
    }
    report = {
        "model_version": "fd001-v1",
        "selected_model": "torch_mlp",
        "dataset": "Synthetic fixture",
        "seed": SEED,
        "window_cycles": 20,
        "model_parameters": MODEL_PARAMS,
        "grouped_validation": validation,
        "official_test_endpoints": validation,
    }
    (bundle / "metrics.json").write_text(json.dumps(report))
    (bundle / "splits.json").write_text('{"fixture": true}')
    store = tmp_path / "mlflow"
    ids = track_bundle(bundle, store)
    runs = list_runs(store)
    assert [run["model"] for run in runs] == list(reversed(MODEL_FILES))
    assert runs[0]["selected"] and runs[0]["validation_mae_cycles"] == 1
    assert len({run["split_sha256"] for run in runs}) == 1
    client = MlflowClient(tracking_uri=f"sqlite:///{store / 'tracking.db'}")
    for name, run in ids.items():
        assert client.get_run(run).data.params["dataset"] == "Synthetic fixture"
        assert {entry.path for entry in client.list_artifacts(run)} == {
            MODEL_FILES[name],
            "manifest.json",
            "metrics.json",
            "splits.json",
        }
