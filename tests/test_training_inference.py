import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from faultwatch.data import CYCLE, ENGINE
from faultwatch.features import WINDOW_SIZE, causal_features
from faultwatch.inference import MODEL_FILES, MODEL_VERSION, load_manifest, load_model
from faultwatch.inference import load_predictor
from faultwatch.models import predict_cycles
from faultwatch.training import grouped_folds, validation_endpoints


def test_grouped_validation_is_seeded_and_holds_out_whole_engines(trajectories):
    features = causal_features(trajectories)
    ids = features.index.get_level_values(ENGINE)
    endpoints = validation_endpoints(trajectories)
    folds = list(grouped_folds(features))
    ids_validated = []
    for (rows, fit_ids, val_ids), (_, _, repeated_val_ids) in zip(folds, grouped_folds(features)):
        assert set(fit_ids) == set(ids[rows])
        assert set(fit_ids).isdisjoint(val_ids)
        np.testing.assert_array_equal(val_ids, repeated_val_ids)
        ids_validated.extend(val_ids)
    assert sorted(ids_validated) == sorted(ids.unique())
    assert endpoints.isin(features.index).all()
    assert endpoints.value_counts().eq(1).all()
    assert len(endpoints) == 3 * len(ids_validated)


def test_all_model_roundtrips_and_live_prediction_match_batch(model_bundle, trajectories):
    bundle, models = model_bundle
    manifest = load_manifest(bundle)
    latest_features = causal_features(trajectories, latest_only=True)
    with threadpool_limits(2):
        for name, original in models.items():
            expected = predict_cycles(original, latest_features)
            restored = predict_cycles(load_model(bundle, name, manifest), latest_features)
            np.testing.assert_allclose(restored, expected)
            assert np.isfinite(restored).all()
            assert (restored >= 0).all()
    predictor = load_predictor(bundle)
    assert predictor.model_version == MODEL_VERSION
    actual = predictor.predict_latest(trajectories)
    batch_features = causal_features(trajectories).loc[actual.index]
    np.testing.assert_allclose(actual, predict_cycles(models["torch_mlp"], batch_features))
    assert predictor.predict_latest(trajectories.loc[trajectories[CYCLE] < WINDOW_SIZE]).empty


def test_bundle_rejects_incompatible_schema_and_corrupted_model(tmp_path, model_bundle):
    bundle, _ = model_bundle
    manifest = load_manifest(bundle)
    manifest["schema"]["history_cycles"] = 19
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incompatible"):
        load_manifest(tmp_path)
    (tmp_path / MODEL_FILES["torch_mlp"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model(tmp_path, "torch_mlp", load_manifest(bundle))


def test_model_rejects_incorrect_feature_schema(model_bundle, trajectories):
    _, models = model_bundle
    features = causal_features(trajectories, latest_only=True)
    with pytest.raises(ValueError, match="feature schema"):
        predict_cycles(models["median"], features.drop(columns="cycle"))
