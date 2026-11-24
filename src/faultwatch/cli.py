"""FaultWatch command-line entry point."""

import argparse
from pathlib import Path

from faultwatch.data import fetch_fd001

DEFAULT_DATA = Path("data/fd001")


def main() -> None:
    parser = argparse.ArgumentParser(prog="faultwatch")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Fetch and verify NASA C-MAPSS FD001")
    fetch.add_argument("--data", type=Path, default=DEFAULT_DATA)
    fetch.add_argument("--archive", type=Path, help="Use an existing official ZIP")
    args = parser.parse_args()

    fetch_fd001(args.data, args.archive)
    print(f"Verified FD001 files in {args.data}")
