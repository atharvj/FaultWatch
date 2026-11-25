"""FaultWatch command-line entry point."""

import argparse
import json
import os
from pathlib import Path

from faultwatch.data import ENGINE, fetch_fd001, read_telemetry

DEFAULT_DATA = Path("data/fd001")
DEFAULT_BUNDLE = Path("artifacts/fd001-v1")


def main() -> None:
    parser = argparse.ArgumentParser(prog="faultwatch")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Fetch and verify NASA C-MAPSS FD001")
    fetch.add_argument("--data", type=Path, default=DEFAULT_DATA)
    fetch.add_argument("--archive", type=Path, help="Use an existing official ZIP")
    train = commands.add_parser("train", help="Select and evaluate FD001 models")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    evaluate = commands.add_parser("evaluate", help="Evaluate a saved bundle on NASA endpoints")
    evaluate.add_argument("--data", type=Path, default=DEFAULT_DATA)
    evaluate.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    predict = commands.add_parser("predict", help="Predict at the last observed cycle per engine")
    predict.add_argument("telemetry", type=Path, help="Telemetry in FD001 column order")
    predict.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    predict.add_argument("--engine", type=int)
    serve = commands.add_parser("serve", help="Start the local telemetry API")
    serve.add_argument("--port", type=int, default=8000)
    replay = commands.add_parser("replay", help="Send held-out FD001 telemetry to the API")
    replay.add_argument("--data", type=Path, default=DEFAULT_DATA)
    replay.add_argument("--url", default="http://127.0.0.1:8000")
    replay.add_argument(
        "--engine", type=int, action="append", help="Repeat to select multiple engines"
    )
    replay.add_argument(
        "--rate", type=float, default=25, help="Events per second. Use 0 for no delay"
    )
    token = commands.add_parser("token", help="Issue a short-lived telemetry API token")
    token.add_argument("--subject", required=True)
    token.add_argument("--role", choices=("read", "ingest", "admin"), required=True)
    token.add_argument("--seconds", type=int, default=900)
    export = commands.add_parser("export-onnx", help="Export and benchmark the FD001 MLP")
    export.add_argument("--data", type=Path, default=DEFAULT_DATA)
    export.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    export.add_argument("--output", type=Path, default=Path("artifacts/fd001-v1.onnx"))

    track = commands.add_parser("track", help="Log a training bundle to local MLflow")
    track.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    track.add_argument("--store", type=Path, default=Path("artifacts/mlflow"))
    runs = commands.add_parser("runs", help="Compare local MLflow runs by validation MAE")
    runs.add_argument("--store", type=Path, default=Path("artifacts/mlflow"))
    args = parser.parse_args()

    try:
        if args.command == "token":
            from faultwatch.security import AUTH_ENV, issue_token

            secret = os.getenv(AUTH_ENV)
            if not secret:
                raise ValueError(f"Set {AUTH_ENV} before issuing tokens")
            print(issue_token(secret, args.subject, args.role, args.seconds))
        elif args.command == "fetch":
            fetch_fd001(args.data, args.archive)
            print(f"Verified FD001 files in {args.data}")
        elif args.command == "train":
            from faultwatch.training import train_and_evaluate

            print(json.dumps(train_and_evaluate(args.data, args.output), indent=2))
        elif args.command == "evaluate":
            from faultwatch.training import evaluate

            print(json.dumps(evaluate(args.data, args.bundle), indent=2))
        elif args.command == "predict":
            from faultwatch.inference import load_predictor

            telemetry = read_telemetry(args.telemetry)
            if args.engine is not None:
                telemetry = telemetry.loc[telemetry[ENGINE] == args.engine]
                if telemetry.empty:
                    raise ValueError(f"Engine {args.engine} is absent from the telemetry")
            predictor = load_predictor(args.bundle)
            predictions = predictor.predict_latest(telemetry)
            rows = predictions.reset_index().to_dict(orient="records")
            print(
                json.dumps(
                    {
                        "model_version": predictor.model_version,
                        "model": predictor.model_name,
                        "predictions": rows,
                        "insufficient_history_engine_ids": sorted(
                            set(telemetry[ENGINE]) - {row[ENGINE] for row in rows}
                        ),
                    },
                    indent=2,
                )
            )
        elif args.command == "serve":
            import uvicorn

            uvicorn.run("faultwatch.api:create_app", factory=True, host="127.0.0.1", port=args.port)
        elif args.command == "replay":
            from faultwatch.replay import replay_fd001

            print(json.dumps(replay_fd001(args.data, args.url, args.engine, args.rate), indent=2))
        elif args.command == "export-onnx":
            from faultwatch.onnx_export import export_and_validate
            from faultwatch.training import official_test_cases

            features, _ = official_test_cases(args.data)
            print(json.dumps(export_and_validate(args.bundle, args.output, features), indent=2))
        elif args.command in {"track", "runs"}:
            os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
            from faultwatch.tracking import list_runs, track_bundle

            result = (
                track_bundle(args.bundle, args.store)
                if args.command == "track"
                else list_runs(args.store)
            )
            print(json.dumps(result, indent=2))
    except ModuleNotFoundError as error:
        if args.command not in {"export-onnx", "track", "runs"}:
            raise
        parser.exit(1, f"faultwatch: {error}. Run with uv run --group experiments.\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"faultwatch: {error}\n")
