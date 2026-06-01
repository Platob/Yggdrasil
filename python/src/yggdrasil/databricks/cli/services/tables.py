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
            "--ensure-job", dest="ensure_job", action="store_true",
            help="Also get-or-create the file-arrival loader job so the drop "
                 "is picked up automatically.",
        )
        ai.set_defaults(handler=cls._async_insert)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    @classmethod
    def _async_insert(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        # All the work lives in the service — read source + async drop +
        # optional loader-job deploy — so the CLI stays a thin shell.
        log_file = client.tables.async_insert(
            args.table_name,
            args.data,
            mode=args.mode,
            ensure_job=args.ensure_job,
        )
        sys.stdout.write(f"async insert staged → {log_file.full_path()}\n")
        if not args.ensure_job:
            sys.stdout.write(
                "deploy the loader once with `--ensure-job` (or "
                "`table.async_job().ensure()`) so the drop is picked up.\n"
            )
        return 0
