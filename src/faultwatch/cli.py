"""FaultWatch command-line entry point."""

import argparse
import json
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
    args = parser.parse_args()

    try:
        if args.command == "fetch":
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
    except (OSError, ValueError) as error:
        parser.exit(1, f"faultwatch: {error}\n")
