"""``ygg databricks table`` — table operations (async insert, …)."""
from __future__ import annotations

import sys
from typing import Any


class TablesCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "table", help="Table operations (async insert, …).",
        )
        sub = parser.add_subparsers(dest="table_action")

        ai = sub.add_parser(
            "async_insert",
            help="Stage data for an async (file-arrival) load into a table.",
        )
        ai.add_argument(
            "--table-name", dest="table_name", required=True,
            help="Fully-qualified table name (catalog.schema.table).",
        )
        ai.add_argument(
            "--data", required=True,
            help="Source data path/URL to load — parquet/csv/json/… on a local "
                 "path, /Volumes/…, or s3://…",
        )
        ai.add_argument(
            "--mode", default="append", choices=["append", "overwrite"],
            help="Insert mode (default: append).",
        )
        ai.add_argument(
            "--execute", action="store_true",
            help="Load the staged drop immediately (synchronous) instead of "
                 "leaving it for the file-arrival job.",
        )
        ai.add_argument(
            "--ensure-job", dest="ensure_job", action="store_true",
            help="Also get-or-create the file-arrival loader job so the drop "
                 "is picked up automatically.",
        )
        ai.set_defaults(handler=cls._async_insert)

        ex = sub.add_parser(
            "execute_async_insert",
            help="Run the async loader over pending op-logs (group by table, "
                 "aggregate, load).",
        )
        ex.add_argument(
            "--logs",
            help="A log file or a logs directory to scan for pending op-logs.",
        )
        ex.add_argument(
            "--log-file", dest="log_files", action="append", metavar="PATH",
            help="An explicit op-log file to consume (repeatable); skips the "
                 "directory scan.",
        )
        ex.set_defaults(handler=cls._execute_async_insert)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    @classmethod
    def _async_insert(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.databricks.table.insert import (
            ensure_async_job, load_async, stage_async_insert,
        )

        client = build_client(args)
        # Producer: stage_async_insert reads the path/URL source and stages a
        # Parquet + drops a JSON op-log (full metadata) — the CLI stays thin.
        table = client.tables[args.table_name]
        log_file = stage_async_insert(table, args.data, mode=args.mode)
        sys.stdout.write(f"async insert staged → {log_file.full_path()}\n")

        if args.execute:
            # Loader only needs the log path — the log carries everything.
            n = load_async(client.tables, log_file.full_path(), wait=True)
            sys.stdout.write(f"executed {n} pending operation(s)\n")
        elif args.ensure_job:
            job = ensure_async_job(table)
            sys.stdout.write(f"loader job ready: {getattr(job, 'job_id', job)}\n")
        else:
            sys.stdout.write(
                "run with `--execute` to load now, or `--ensure-job` to let "
                "the file-arrival job pick the drop up.\n"
            )
        return 0

    @classmethod
    def _execute_async_insert(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.databricks.table.insert import load_async

        client = build_client(args)
        # Loader: the logs carry full metadata, so it only needs their path(s).
        n = load_async(
            client.tables, logs=args.logs, log_files=args.log_files or None, wait=True,
        )
        sys.stdout.write(f"executed {n} pending operation(s)\n")
        return 0
