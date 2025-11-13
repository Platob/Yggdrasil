"""Command line entrypoint for the Yggdrasil Python utilities."""

from __future__ import annotations

import argparse
import json
from typing import Iterable, Sequence

import polars as pl

from .data import DATA_CAST_REGISTRY


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yggdrasil", description=__doc__)
    parser.add_argument("--version", action="version", version="yggdrasil 0.1.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    greet_parser = subparsers.add_parser("greet", help="Print a greeting message.")
    greet_parser.add_argument("name", nargs="?", default="Adventurer")

    data_parser = subparsers.add_parser(
        "data-cast",
        help="Cast a list of integers to a wider integer type using the registry.",
    )
    data_parser.add_argument(
        "values",
        nargs="*",
        type=int,
        default=[1, 2, 3],
        help="Integer values to cast.",
    )

    subparsers.add_parser(
        "demo-table",
        help="Emit a demo Polars table as JSON for quick inspection.",
    )

    return parser


def _run_data_cast(values: Iterable[int]) -> dict[str, object]:
    registry = DATA_CAST_REGISTRY
    source_dtype = pl.Int32
    target_dtype = pl.Int64

    caster = registry.get_or_build(source_dtype, target_dtype, source_name="values", target_name="values")
    series = pl.Series("values", values, dtype=source_dtype)
    cast_series = caster.cast_series(series)

    return {
        "source_type": str(source_dtype),
        "target_type": str(target_dtype),
        "values": cast_series.to_list(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "data-cast":
        payload = _run_data_cast(args.values)
        print(json.dumps(payload, indent=2))
    elif args.command == "greet":
        print(f"Hello, {args.name}!")
    elif args.command == "demo-table":
        # Create a simple demo table
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "value": [10.5, 20.0, 30.5]
        })
        print(df.to_dict())
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2

    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    raise SystemExit(main())