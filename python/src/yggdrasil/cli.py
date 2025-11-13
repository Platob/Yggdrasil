"""Command line entrypoint for the Yggdrasil Python utilities."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Iterable, Sequence

import polars as pl

from .data import DATA_CAST_REGISTRY
from .logging import get_logger, setup_logging

# Create module-level logger
logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yggdrasil", description=__doc__)
    parser.add_argument("--version", action="version", version="yggdrasil 0.1.0")
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )

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
    """Cast integer values from Int32 to Int64."""
    logger.debug("Starting data cast operation")
    registry = DATA_CAST_REGISTRY
    source_dtype = pl.Int32
    target_dtype = pl.Int64

    logger.debug(f"Getting caster for {source_dtype} -> {target_dtype}")
    caster = registry.get_or_build(source_dtype, target_dtype, source_name="values", target_name="values")

    logger.debug(f"Creating source series with values: {list(values)}")
    series = pl.Series("values", values, dtype=source_dtype)

    logger.debug("Casting series")
    cast_series = caster.cast_series(series)

    result = {
        "source_type": str(source_dtype),
        "target_type": str(target_dtype),
        "values": cast_series.to_list(),
    }
    logger.debug(f"Data cast completed: {result}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging based on verbosity
    log_level = logging.WARNING  # Default

    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose > 0:
        # Map verbosity level to logging level
        # -v: INFO, -vv: DEBUG
        if args.verbose == 1:
            log_level = logging.INFO
        else:  # args.verbose >= 2
            log_level = logging.DEBUG

    # Set up logging with the appropriate level
    setup_logging(level=log_level)
    logger.debug(f"Log level set to: {logging.getLevelName(log_level)}")

    logger.info(f"Running command: {args.command}")

    try:
        if args.command == "data-cast":
            logger.info(f"Casting values: {args.values}")
            payload = _run_data_cast(args.values)
            logger.debug(f"Cast result: {payload}")
            print(json.dumps(payload, indent=2))
        elif args.command == "greet":
            logger.info(f"Greeting user: {args.name}")
            print(f"Hello, {args.name}!")
        elif args.command == "demo-table":
            logger.info("Creating demo table")
            # Create a simple demo table
            df = pl.DataFrame({
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "value": [10.5, 20.0, 30.5]
            })
            logger.debug(f"Demo table shape: {df.shape}")
            print(df.to_dict())
        else:
            error_msg = f"Unsupported command: {args.command}"
            logger.error(error_msg)
            parser.error(error_msg)
            return 2

        logger.info(f"Command {args.command} completed successfully")
        return 0
    except Exception as e:
        logger.exception(f"Error executing command {args.command}: {e}")
        return 1


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    raise SystemExit(main())