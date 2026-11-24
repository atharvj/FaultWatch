import numpy as np
import pandas as pd
import pytest
import torch
from threadpoolctl import threadpool_limits

from faultwatch.data import CYCLE, ENGINE, TELEMETRY_COLUMNS, add_training_labels
from faultwatch.features import causal_features
from faultwatch.inference import save_models
from faultwatch.models import fit_candidates


@pytest.fixture(scope="module")
def trajectories():
    records = [
        [engine, cycle, *np.sin(np.arange(24) + (cycle + engine) / 12)]
        for engine in range(1, 7)
        for cycle in range(1, 51 + engine)
    ]
    return pd.DataFrame(records, columns=TELEMETRY_COLUMNS)


@pytest.fixture(scope="module")
def model_bundle(tmp_path_factory, trajectories):
    bundle = tmp_path_factory.mktemp("model")
    features = causal_features(trajectories)
    labels = trajectories.assign(rul=add_training_labels(trajectories)).set_index([ENGINE, CYCLE])
    torch.set_num_threads(2)
    with threadpool_limits(2):
        models = fit_candidates(features, labels.loc[features.index, "rul"].to_numpy())
    save_models(bundle, models, "torch_mlp")
    return bundle, models
