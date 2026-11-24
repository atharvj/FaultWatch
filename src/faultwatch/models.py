"""Candidate RUL regressors that consume the same causal features."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from faultwatch.features import FEATURE_COLUMNS

SEED = 27
TREE_PARAMS = {
    "max_iter": 180,
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": SEED,
}
MLP_WIDTHS = (64, 32)
MLP_EPOCHS = 30
MLP_BATCH_SIZE = 256
MLP_LEARNING_RATE = 0.001
TARGET_SCALE = 100.0
MLP_LOSS_BETA = 0.2
MODEL_PARAMS = {
    "median": {"strategy": "median"},
    "hist_gradient_boosting": TREE_PARAMS,
    "torch_mlp": {
        "widths": list(MLP_WIDTHS),
        "epochs": MLP_EPOCHS,
        "batch_size": MLP_BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": MLP_LEARNING_RATE,
        "loss": "SmoothL1Loss",
        "loss_beta": MLP_LOSS_BETA,
        "target_scale": TARGET_SCALE,
    },
}


class Regressor(Protocol):
    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


def mlp_network() -> nn.Sequential:
    first, second = MLP_WIDTHS
    return nn.Sequential(
        nn.Linear(len(FEATURE_COLUMNS), first),
        nn.ReLU(),
        nn.Linear(first, second),
        nn.ReLU(),
        nn.Linear(second, 1),
    )


@dataclass
class TorchMLP:
    network: nn.Sequential
    center: np.ndarray
    scale: np.ndarray

    def inputs(self, features: pd.DataFrame) -> torch.Tensor:
        scaled = (features.to_numpy() - self.center) / self.scale
        return torch.from_numpy(np.ascontiguousarray(scaled, dtype=np.float32))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        with torch.inference_mode():
            return self.network(self.inputs(features)).squeeze(1).numpy() * TARGET_SCALE


def _fit_mlp(features: pd.DataFrame, labels: np.ndarray) -> TorchMLP:
    scaler = StandardScaler().fit(features)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        model = TorchMLP(mlp_network(), scaler.mean_, scaler.scale_)
    batches = DataLoader(
        TensorDataset(
            model.inputs(features), torch.from_numpy((labels / TARGET_SCALE).astype(np.float32))
        ),
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=MLP_LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss(beta=MLP_LOSS_BETA)
    for _ in range(MLP_EPOCHS):
        for values, targets in batches:
            optimizer.zero_grad()
            loss = loss_fn(model.network(values).squeeze(1), targets)
            loss.backward()
            optimizer.step()
    model.network.eval()
    return model


def fit_candidates(features: pd.DataFrame, labels: np.ndarray) -> dict[str, Regressor]:
    return {
        "median": make_pipeline(DummyRegressor(strategy="median")).fit(features, labels),
        "hist_gradient_boosting": make_pipeline(HistGradientBoostingRegressor(**TREE_PARAMS)).fit(
            features, labels
        ),
        "torch_mlp": _fit_mlp(features, labels),
    }


def predict_cycles(model: Regressor, features: pd.DataFrame) -> np.ndarray:
    if tuple(features.columns) != FEATURE_COLUMNS or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Expected finite inputs matching the feature schema")
    predictions = np.asarray(model.predict(features), dtype=np.float64)
    if predictions.shape != (len(features),) or not np.isfinite(predictions).all():
        raise ValueError("Expected one finite prediction per feature row")
    return predictions.clip(min=0)
