"""Command line entrypoint for the Yggdrasil Python utilities."""

from __future__ import annotations

import argparse
import json
from typing import Iterable, Sequence

import pyarrow as pa

from .data.arrow import ARROW_CAST_REGISTRY


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yggdrasil", description=__doc__)
    parser.add_argument("--version", action="version", version="yggdrasil 0.1.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    greet_parser = subparsers.add_parser("greet", help="Print a greeting message.")
    greet_parser.add_argument("name", nargs="?", default="Adventurer")

    arrow_parser = subparsers.add_parser(
        "arrow-cast",
        help="Cast an Arrow list array of integers to a wider integer type using the registry.",
    )
    arrow_parser.add_argument(
        "values",
        nargs="*",
        type=int,
        default=[1, 2, 3],
        help="Integer values to cast.",
    )

    subparsers.add_parser(
        "demo-table",
        help="Emit the demo Arrow table as JSON for quick inspection.",
    )

    return parser


def _run_arrow_cast(values: Iterable[int]) -> dict[str, object]:
    registry = ARROW_CAST_REGISTRY
    source_field = pa.field("values", pa.list_(pa.int32()))
    target_field = pa.field("values", pa.list_(pa.int64()))

    caster = registry.get_or_build(source_field, target_field)
    array = pa.array([list(values)], type=source_field.type)
    cast_array = caster.cast(array)

    return {
        "source_type": str(source_field.type),
        "target_type": str(target_field.type),
        "values": cast_array.to_pylist(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "arrow-cast":
        payload = _run_arrow_cast(args.values)
        print(json.dumps(payload, indent=2))
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2

    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    raise SystemExit(main())
