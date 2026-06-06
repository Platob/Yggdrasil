"""``ygg-loki`` — the entry point a deployed DatabricksLoki job runs.

A deployed agent (see :meth:`DatabricksLoki.deploy`) invokes this on
Databricks serverless: it resolves the agent from the runtime, runs the
requested behavior with JSON kwargs, and prints the result.
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(prog="ygg-loki", description="Run a Loki behavior.")
    parser.add_argument("behavior", help="Behavior name, or 'reason' for a prompt.")
    parser.add_argument("--kwargs", default="{}", help="JSON object of keyword args.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from yggdrasil.databricks.loki import DatabricksLoki
    from yggdrasil.pickle import json as yjson

    kwargs = yjson.loads(args.kwargs) if args.kwargs else {}
    loki = DatabricksLoki.current()

    if args.behavior == "reason":
        result = loki.reason(**kwargs)
    else:
        result = loki.run(args.behavior, **kwargs)
    print(result)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
