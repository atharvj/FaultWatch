"""Verified NASA FD001 download and ingest."""

import io
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

NASA_ARCHIVE_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
OUTER_MEMBER = "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"
OUTER_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
INNER_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"
FD001_HASHES = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
}
ENGINE = "engine_id"
CYCLE = "cycle"
OPERATING_SETTINGS = tuple(f"setting_{number}" for number in range(1, 4))
SENSORS = tuple(f"sensor_{number}" for number in range(1, 22))
TELEMETRY_COLUMNS = (ENGINE, CYCLE, *OPERATING_SETTINGS, *SENSORS)
FD001_ENGINE_IDS = list(range(1, 101))


def _verify_hash(content: bytes, expected: str, name: str) -> None:
    if sha256(content).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {name}")


def fetch_fd001(destination: Path, archive_path: Path | None = None) -> None:
    """Download the NASA-listed archive or use a verified local copy."""
    if archive_path is None:
        with urlopen(NASA_ARCHIVE_URL, timeout=60) as response:
            archive_bytes = response.read(13_000_001)
    else:
        archive_bytes = archive_path.read_bytes()

    if sha256(archive_bytes).hexdigest() == OUTER_SHA256:
        with ZipFile(io.BytesIO(archive_bytes)) as outer:
            archive_bytes = outer.read(OUTER_MEMBER)
    _verify_hash(archive_bytes, INNER_SHA256, "CMAPSSData.zip")

    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(io.BytesIO(archive_bytes)) as archive, TemporaryDirectory(dir=destination) as tmp:
        for name, digest in FD001_HASHES.items():
            content = archive.read(name)
            _verify_hash(content, digest, name)
            (Path(tmp) / name).write_bytes(content)
        for name in FD001_HASHES:
            (Path(tmp) / name).replace(destination / name)


def validate_telemetry(frame: pd.DataFrame) -> pd.DataFrame:
    """Return telemetry sorted by engine and cycle after checking its schema."""
    if frame.empty or not frame.columns.is_unique or set(frame.columns) != set(TELEMETRY_COLUMNS):
        raise ValueError("Expected nonempty telemetry with the 26 FD001 columns")

    frame = frame.loc[:, list(TELEMETRY_COLUMNS)].astype("float64")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Telemetry must contain finite readings")

    for name in (ENGINE, CYCLE):
        values = frame[name].to_numpy()
        if ((values < 1) | (values >= 2**53) | (values != np.floor(values))).any():
            raise ValueError(f"{name} must contain positive integers")
        frame[name] = values.astype(np.int64)

    frame = frame.sort_values([ENGINE, CYCLE], ignore_index=True)
    differences = frame.groupby(ENGINE, sort=False)[CYCLE].diff().dropna()
    if not differences.eq(1).all():
        raise ValueError("Cycles must be unique and consecutive within each engine")
    return frame


def read_telemetry(path: Path) -> pd.DataFrame:
    values = np.loadtxt(path, ndmin=2)
    if values.shape[1] != len(TELEMETRY_COLUMNS):
        raise ValueError(f"{path.name} must have {len(TELEMETRY_COLUMNS)} columns")
    return validate_telemetry(pd.DataFrame(values, columns=TELEMETRY_COLUMNS))


def read_fd001_telemetry(path: Path) -> pd.DataFrame:
    frame = read_telemetry(path)
    grouped_cycles = frame.groupby(ENGINE, sort=True)[CYCLE]
    first_cycles = grouped_cycles.first()
    if first_cycles.index.tolist() != FD001_ENGINE_IDS or not first_cycles.eq(1).all():
        raise ValueError(f"{path.name} must contain FD001 engines 1–100 starting at cycle 1")
    return frame


def verify_fd001(directory: Path, *names: str) -> None:
    for name in names:
        _verify_hash((directory / name).read_bytes(), FD001_HASHES[name], name)


def add_training_labels(frame: pd.DataFrame) -> pd.Series:
    """Return uncapped RUL aligned to complete training trajectories."""
    max_cycles = frame.groupby(ENGINE)[CYCLE].transform("max")
    return (max_cycles - frame[CYCLE]).rename("rul")


def read_test_rul(path: Path) -> pd.Series:
    values = np.loadtxt(path, ndmin=2)
    if (
        values.shape != (len(FD001_ENGINE_IDS), 1)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError("Expected one nonnegative integer RUL per FD001 test engine")
    return pd.Series(
        values[:, 0].astype(np.int64),
        index=pd.Index(FD001_ENGINE_IDS, name=ENGINE),
        name="rul",
    )
