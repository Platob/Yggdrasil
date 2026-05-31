"""``ygg-job`` — the job CLI a deployed :class:`Flow` task invokes on the cluster.

A small argparse CLI (one entry point, its own subcommands) the python-wheel
task runs as ``ygg-job <command> <args...>`` on the Databricks cluster. Today::

    ygg-job table-async-load <catalog.schema.table>

reconstructs the table from the runtime client and drives the async loader
(:class:`~yggdrasil.databricks.table.async_job.TableJob`). Add subcommands here
as more job kinds appear.
"""
from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["main", "build_parser", "table_async_load"]


# -- commands ---------------------------------------------------------------
def table_async_load(full_name: str, *, wait: bool = True) -> int:
    """Aggregate + load a table's pending async-insert drops. Returns the count."""
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.table.async_job import TableJob

    client = DatabricksClient()              # runtime auth on the cluster
    table = client.tables[full_name]
    processed = TableJob(table).run(wait=wait)
    logger.info("table-async-load %s: %s ops", full_name, processed)
    return int(processed or 0)


# -- CLI --------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ygg-job",
        description="Run ygg job tasks on a Databricks cluster.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    al = sub.add_parser(
        "table-async-load",
        help="Aggregate + load a table's async-insert drops (the file-arrival job).",
    )
    al.add_argument("table", help="Fully-qualified table name (catalog.schema.table).")
    al.set_defaults(handler=lambda a: table_async_load(a.table))

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from yggdrasil.databricks.job.skeleton import ensure_console_logging

    ensure_console_logging()  # surface ygg logs in the job output
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    # Always surface a result line, even if logging is filtered.
    print(f"ygg-job {args.command} -> {result}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
