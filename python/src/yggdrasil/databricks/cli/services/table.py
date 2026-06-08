"""``ygg databricks table`` — the **on-cluster** table data-plane CLI.

This is the dedicated subcommand a deployed serverless Auto Loader job runs *as*
on the cluster — the single python-wheel task built by
:meth:`yggdrasil.databricks.table.table.Table.auto_loader` invokes the ``ygg``
entry point with::

    ygg databricks table autoload --table <catalog.schema.table> --source <path> \\
        --format parquet --available-now --clean-source-retention "8 days"

``autoload`` coerces those flags and calls
:func:`yggdrasil.databricks.table.auto_loader.auto_load` — the Spark Structured
Streaming + ``cloudFiles`` ingestion body — then returns its summary. The
handler stays thin; the logic lives in ``auto_load``.

This is the **data plane** (run the ingestion). The control-plane command that
*creates / deploys* the job is ``ygg databricks tables autoload`` (see
:mod:`~yggdrasil.databricks.cli.services.tables`).
"""
from __future__ import annotations

import sys
from typing import Any


class TableCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "table", help="On-cluster table data plane (Auto Loader ingestion run).")
        sub = parser.add_subparsers(dest="table_action")

        al = sub.add_parser(
            "autoload", aliases=["auto-load", "auto_loader"],
            help="Run a cloudFiles Auto Loader ingestion sweep into a table (on-cluster).")
        al.add_argument("--table", "-t", required=True,
                        help="Target table: catalog.schema.table.")
        al.add_argument("--source", "-s", required=True,
                        help="Cloud/volume input path Auto Loader watches (s3://… or /Volumes/…).")
        al.add_argument("--format", dest="file_format", default="parquet",
                        help="cloudFiles.format (parquet/json/csv/avro/…). Default parquet.")
        al.add_argument("--checkpoint", default="",
                        help="Checkpoint + schema location (default: derived next to the table).")
        al.add_argument("--available-now", dest="available_now", action="store_true",
                        default=True,
                        help="One-shot Trigger.AvailableNow sweep then stop (default).")
        al.add_argument("--no-available-now", dest="available_now", action="store_false",
                        help="Continuous 1-minute micro-batch stream instead of one sweep.")
        al.add_argument("--clean-source", dest="clean_source", action="store_true",
                        help="Delete each staged file once ingested + past retention (self-cleaning).")
        al.add_argument("--clean-source-retention", dest="clean_source_retention", default="8 days",
                        help="Retention window for --clean-source (> 7 days; default '8 days').")
        al.set_defaults(handler=cls._autoload)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    # -- handlers --------------------------------------------------------
    @classmethod
    def _autoload(cls, args: Any, build_client: Any) -> int:
        """Coerce the CLI args and run :func:`auto_load` in-process (on the
        cluster); print its summary. No client is needed — Spark is acquired
        inside ``auto_load`` via the active session."""
        from yggdrasil.databricks.job.skeleton import ensure_console_logging
        from yggdrasil.databricks.table.auto_loader import auto_load

        ensure_console_logging()  # surface ygg logs in the job task output
        summary = auto_load(
            table=args.table,
            source=args.source,
            file_format=args.file_format,
            checkpoint=args.checkpoint,
            available_now=args.available_now,
            clean_source=args.clean_source,
            clean_source_retention=args.clean_source_retention,
        )
        sys.stdout.write(f"{summary}\n")
        return 0
