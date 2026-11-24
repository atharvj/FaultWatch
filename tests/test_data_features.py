import numpy as np
import pandas as pd
import pytest

from faultwatch.data import (
    CYCLE,
    ENGINE,
    TELEMETRY_COLUMNS,
    add_training_labels,
    read_telemetry,
    read_test_rul,
    validate_telemetry,
)
from faultwatch.features import FEATURE_COLUMNS, WINDOW_SIZE, causal_features


@pytest.fixture
def telemetry():
    values = [
        [engine, cycle, *([engine * 1000 + cycle] * 24)]
        for engine in (1, 2)
        for cycle in range(1, WINDOW_SIZE + 3)
    ]
    return pd.DataFrame(values, columns=TELEMETRY_COLUMNS)


def test_layout_and_uncapped_labels(tmp_path, telemetry):
    path = tmp_path / "train.txt"
    np.savetxt(path, telemetry)
    frame = read_telemetry(path)
    assert tuple(frame.columns) == TELEMETRY_COLUMNS
    labels = add_training_labels(frame).to_numpy().reshape(2, -1)
    assert labels[0].tolist() == list(range(WINDOW_SIZE + 1, -1, -1))
    np.testing.assert_array_equal(labels[0], labels[1])


def test_rul_vector_aligns_by_engine_index(tmp_path):
    path = tmp_path / "rul.txt"
    np.savetxt(path, np.arange(99, -1, -1), fmt="%d")
    labels = read_test_rul(path)
    assert labels.index.name == ENGINE
    assert labels.loc[[100, 1, 40]].tolist() == [0, 99, 60]
    path.write_text("3 4\n")
    with pytest.raises(ValueError, match="one nonnegative integer"):
        read_test_rul(path)


def test_windows_are_causal_and_cannot_cross_engines(telemetry):
    first = causal_features(telemetry)
    assert tuple(first.columns) == FEATURE_COLUMNS
    assert first.index.tolist() == [
        (engine, cycle) for engine in (1, 2) for cycle in range(WINDOW_SIZE, WINDOW_SIZE + 3)
    ]
    assert first.loc[(2, WINDOW_SIZE), "sensor_1_mean"] == 2010.5
    assert first.loc[(2, WINDOW_SIZE), "sensor_1_delta"] == 19

    changed = telemetry.sample(frac=1, random_state=4)
    changed.loc[changed[CYCLE] > WINDOW_SIZE, "sensor_1"] = -1000
    second = causal_features(changed)
    pd.testing.assert_frame_equal(
        first.xs(WINDOW_SIZE, level=CYCLE), second.xs(WINDOW_SIZE, level=CYCLE)
    )


def test_latest_window_matches_batch_and_can_start_after_cycle_one(telemetry):
    full = causal_features(telemetry)
    recent = telemetry.loc[telemetry[CYCLE] >= 3]
    expected = full.xs(WINDOW_SIZE + 2, level=CYCLE, drop_level=False)
    pd.testing.assert_frame_equal(causal_features(recent, latest_only=True), expected)
    assert causal_features(recent.loc[recent[CYCLE] > 3], latest_only=True).empty


def test_validation_rejects_gaps_duplicates_and_invalid_sensors(telemetry):
    with pytest.raises(ValueError, match="unique and consecutive"):
        validate_telemetry(telemetry.drop(index=1))
    with pytest.raises(ValueError, match="unique and consecutive"):
        validate_telemetry(pd.concat([telemetry, telemetry.iloc[[0]]]))
    with pytest.raises(ValueError, match="26 FD001 columns"):
        validate_telemetry(telemetry.drop(columns="sensor_21"))
    invalid = telemetry.assign(sensor_21=np.inf)
    with pytest.raises(ValueError, match="finite"):
        validate_telemetry(invalid)
    with pytest.raises(ValueError, match="positive integers"):
        validate_telemetry(telemetry.assign(cycle=1.5))


def test_fetch_rejects_unverified_archive(tmp_path):
    from faultwatch.data import fetch_fd001

    path = tmp_path / "invalid.zip"
    path.write_bytes(b"unverified archive")
    destination = tmp_path / "download"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch_fd001(destination, path)
    assert not destination.exists()
