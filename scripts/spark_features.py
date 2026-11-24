"""Materialize the FD001 causal feature table to Parquet with local Spark."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from faultwatch.data import (
    CYCLE,
    ENGINE,
    SENSORS,
    TELEMETRY_COLUMNS,
    read_fd001_telemetry,
    verify_fd001,
)
from faultwatch.features import FEATURE_COLUMNS, WINDOW_SIZE, causal_features


def table(spark, path: Path, split: str):
    cells = F.split(F.trim(F.col("value")), r"\s+")
    columns = [
        cells[index].cast("int" if name in (ENGINE, CYCLE) else "double").alias(name)
        for index, name in enumerate(TELEMETRY_COLUMNS)
    ]
    telemetry = spark.read.text(str(path)).select(*columns)
    engine = Window.partitionBy(ENGINE)
    history = engine.orderBy(CYCLE).rowsBetween(1 - WINDOW_SIZE, 0)
    last_cycle = F.max(CYCLE).over(engine)
    features = (
        [F.col(ENGINE), F.col(CYCLE)]
        + [F.col(name).alias(f"{name}_last") for name in SENSORS]
        + [F.avg(name).over(history).alias(f"{name}_mean") for name in SENSORS]
        + [(F.col(name) - F.first(name).over(history)).alias(f"{name}_delta") for name in SENSORS]
    )
    return telemetry.select(
        *features,
        (last_cycle - F.col(CYCLE) if split == "train" else F.lit(None).cast("int")).alias("rul"),
        F.lit(split).alias("split"),
    ).where(F.col(CYCLE) >= WINDOW_SIZE)


def materialize(data: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    names = {split: f"{split}_FD001.txt" for split in ("train", "test")}
    verify_fd001(data, *names.values())
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("FaultWatch FD001")
        .config("spark.driver.memory", "768m")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    start = perf_counter()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as tmp:
            staging = Path(tmp) / "features"
            train = table(spark, data / names["train"], "train")
            test = table(spark, data / names["test"], "test")
            train.unionByName(test).write.partitionBy("split").parquet(str(staging))
            restored = spark.read.parquet(str(staging))
            comparisons = {}
            for split, name in names.items():
                telemetry = read_fd001_telemetry(data / name)
                expected = causal_features(telemetry)
                actual = restored.where(F.col("split") == split).orderBy(ENGINE, CYCLE).toPandas()
                np.testing.assert_array_equal(
                    actual[[ENGINE, CYCLE]], expected.index.to_frame(index=False)
                )
                actual_values = actual[list(FEATURE_COLUMNS)].to_numpy()
                expected_values = expected.to_numpy()
                np.testing.assert_allclose(actual_values, expected_values, atol=1e-9, rtol=0)
                if split == "train":
                    ends = telemetry.groupby(ENGINE)[CYCLE].max()
                    np.testing.assert_array_equal(
                        actual["rul"], actual[ENGINE].map(ends) - actual[CYCLE]
                    )
                else:
                    assert actual["rul"].isna().all()
                comparisons[split] = {
                    "rows": len(actual),
                    "max_absolute_feature_difference": float(
                        np.max(np.abs(actual_values - expected_values))
                    ),
                }
            files = list(staging.glob("split=*/*.parquet"))
            result = {
                "spark": spark.version,
                "master": spark.sparkContext.master,
                "feature_count": len(FEATURE_COLUMNS),
                "partition_column": "split",
                "parquet_files": len(files),
                "parquet_bytes": sum(path.stat().st_size for path in files),
                "splits": comparisons,
                "materialize_and_verify_seconds": round(perf_counter() - start, 3),
            }
            staging.rename(output)
            return result
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fd001"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fd001-parquet"))
    args = parser.parse_args()
    print(json.dumps(materialize(args.data, args.output), indent=2))
