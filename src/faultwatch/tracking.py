"""Record and compare completed FD001 experiments in a local MLflow store."""

import json
from hashlib import sha256
from pathlib import Path
from time import time

from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

from faultwatch.inference import MODEL_FILES, load_manifest

EXPERIMENT = "FaultWatch FD001"


def _client(store: Path) -> MlflowClient:
    store = store.resolve()
    store.mkdir(parents=True, exist_ok=True)
    return MlflowClient(tracking_uri=f"sqlite:///{store / 'tracking.db'}")


def track_bundle(bundle: Path, store: Path) -> dict[str, str]:
    manifest = load_manifest(bundle)
    report = json.loads((bundle / "metrics.json").read_text())
    if any(report[key] != manifest[key] for key in ("model_version", "selected_model")):
        raise ValueError("Metrics do not match the model manifest")
    splits = bundle / "splits.json"
    split_digest = sha256(splits.read_bytes()).hexdigest()
    for name, file in MODEL_FILES.items():
        if sha256((bundle / file).read_bytes()).hexdigest() != manifest["artifact_sha256"][name]:
            raise ValueError(f"Artifact checksum mismatch for {name}")

    client = _client(store)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(
            EXPERIMENT, artifact_location=(store / "runs").resolve().as_uri()
        )
    )
    ids = {}
    for name, file in MODEL_FILES.items():
        params = {
            **report["model_parameters"][name],
            "seed": report["seed"],
            "window_cycles": report["window_cycles"],
            "dataset": report["dataset"],
        }
        values = {
            f"{split}_{metric}": score
            for split, group in (
                ("validation", "grouped_validation"),
                ("test", "official_test_endpoints"),
            )
            for metric, score in report[group][name].items()
        }
        run = client.create_run(
            experiment_id,
            run_name=name,
            tags={
                "split_sha256": split_digest,
                "model_version": manifest["model_version"],
                "selected": str(name == manifest["selected_model"]).lower(),
            },
        )
        run_id = run.info.run_id
        try:
            timestamp_ms = int(time() * 1000)
            client.log_batch(
                run_id,
                metrics=[Metric(key, value, timestamp_ms, 0) for key, value in values.items()],
                params=[Param(key, str(value)) for key, value in params.items()],
            )
            for source in (
                bundle / file,
                splits,
                bundle / "metrics.json",
                bundle / "manifest.json",
            ):
                client.log_artifact(run_id, str(source))
        except Exception:
            client.set_terminated(run_id, status="FAILED")
            raise
        client.set_terminated(run_id)
        ids[name] = run_id
    return ids


def list_runs(store: Path) -> list[dict]:
    client = _client(store)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return []
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.validation_mae_cycles ASC"],
    )
    return [
        {
            "run_id": run.info.run_id,
            "model": run.info.run_name,
            "selected": run.data.tags["selected"] == "true",
            "split_sha256": run.data.tags["split_sha256"],
            **run.data.metrics,
        }
        for run in runs
    ]
