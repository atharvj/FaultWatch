"""Shared feature extraction for training and live telemetry."""

import numpy as np
import pandas as pd

from faultwatch.data import CYCLE, ENGINE, SENSORS, validate_telemetry

WINDOW_SIZE = 20
FEATURE_COLUMNS = (
    CYCLE,
    *(f"{sensor}_{stat}" for stat in ("last", "mean", "delta") for sensor in SENSORS),
)


def causal_features(telemetry: pd.DataFrame, *, latest_only: bool = False) -> pd.DataFrame:
    """Extract features from complete windows ending at their indexed cycle."""
    frame = validate_telemetry(telemetry)
    features = []
    for engine_id, trajectory in frame.groupby(ENGINE, sort=True):
        if len(trajectory) < WINDOW_SIZE:
            continue
        cycles = trajectory[CYCLE].to_numpy()
        readings = trajectory.loc[:, list(SENSORS)].to_numpy()
        if latest_only:
            cycles = cycles[-WINDOW_SIZE:]
            readings = readings[-WINDOW_SIZE:]
        windows = np.lib.stride_tricks.sliding_window_view(readings, WINDOW_SIZE, axis=0)
        endpoint_cycles = cycles[WINDOW_SIZE - 1 :]
        values = np.column_stack(
            (
                endpoint_cycles,
                windows[:, :, -1],
                windows.mean(axis=2),
                windows[:, :, -1] - windows[:, :, 0],
            )
        )
        index = pd.MultiIndex.from_arrays(
            (np.full(len(endpoint_cycles), engine_id, dtype=np.int64), endpoint_cycles),
            names=(ENGINE, CYCLE),
        )
        features.append(pd.DataFrame(values, index=index, columns=FEATURE_COLUMNS))
    if not features:
        index = pd.MultiIndex.from_arrays([[], []], names=(ENGINE, CYCLE))
        return pd.DataFrame(index=index, columns=FEATURE_COLUMNS, dtype="float64")
    return pd.concat(features)
