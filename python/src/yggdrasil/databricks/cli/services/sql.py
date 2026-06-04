"""``ygg databricks sql`` — run SQL and export results.

    ygg databricks sql query "SELECT * FROM main.default.orders LIMIT 50"
    ygg databricks sql query "SELECT …" --target out.parquet
    ygg databricks sql export --statement-id 01ef… --target /Volumes/main/default/stg/out.csv
    ygg databricks sql export --query "SELECT …"   --target s3://bucket/out.parquet

``query`` runs a statement on the workspace's SQL warehouse and either
prints a preview or writes the result to ``--target``. It echoes the
``statement_id`` so you can re-fetch the same result later with
``export --statement-id`` — the Statement Execution API keeps a finished
result available for a window, so the export costs no re-run.

The export format is the target's extension (``.csv`` / ``.parquet`` /
``.arrow`` / ``.ndjson`` / ``.json``) unless ``--format`` overrides it.
A target shaped like ``dbfs:/…``, ``/Volumes/…`` or ``/Workspace/…`` is
written into the workspace; anything else (local path, ``s3://…``) is
written through the generic path layer.
"""
from __future__ import annotations

import sys
from typing import Any


def _kv(pairs: "list[str] | None") -> "dict[str, str]":
    out: dict[str, str] = {}
    for item in pairs or []:
        key, _, val = item.partition("=")
        out[key.strip()] = val
    return out


def _add_warehouse_flags(parser: Any) -> None:
    parser.add_argument("--warehouse-id", dest="warehouse_id", default=None,
                        help="Run on a specific warehouse id (default: the workspace default).")
    parser.add_argument("--warehouse-name", dest="warehouse_name", default=None,
                        help="Run on a warehouse by name.")
    parser.add_argument("--param", action="append", default=None,
                        help="Bind parameter k=v (repeatable) — used for :name placeholders.")


class SQLCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser("sql", help="Run SQL and export results.")
        sub = parser.add_subparsers(dest="sql_action")

        query = sub.add_parser("query", aliases=["exec", "run"],
                               help="Run a SQL statement; preview it or write it to --target.")
        query.add_argument("statement", help="SQL text to execute.")
        query.add_argument("--target", default=None,
                           help="Write the result here instead of printing (path/uri; format by extension).")
        query.add_argument("--format", dest="format", default=None,
                           help="Override the export format (csv/parquet/arrow/ndjson/json).")
        query.add_argument("--limit", type=int, default=None,
                           help="Row limit applied to the query (default: a 50-row preview when not exporting).")
        _add_warehouse_flags(query)
        query.set_defaults(handler=cls._query)

        export = sub.add_parser("export",
                                help="Export a statement's result to --target (by --statement-id or --query).")
        export.add_argument("--statement-id", dest="statement_id", default=None,
                            help="Re-attach to an already-executed statement by its id.")
        export.add_argument("--query", dest="query", default=None,
                            help="Run this SQL, then export its result.")
        export.add_argument("--target", required=True,
                            help="Destination path/uri. Format is taken from the extension unless --format is set.")
        export.add_argument("--format", dest="format", default=None,
                            help="Override the export format (csv/parquet/arrow/ndjson/json).")
        _add_warehouse_flags(export)
        export.set_defaults(handler=cls._export)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    # -- handlers --------------------------------------------------------
    @classmethod
    def _query(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        # Preview defaults to a bounded fetch; an export pulls the full result.
        row_limit = args.limit if args.limit is not None else (None if args.target else 50)
        result = client.sql.execute(
            args.statement,
            parameters=_kv(args.param) or None,
            warehouse_id=args.warehouse_id,
            warehouse_name=args.warehouse_name,
            row_limit=row_limit,
        )

        statement_id = getattr(result, "statement_id", None)
        if statement_id:
            style.info(f"statement_id {style.bold(statement_id)}")

        table = result.to_arrow_table()
        if args.target:
            dest, n = cls._write_table(client, table, args.target, args.format)
            style.ok(f"exported {n} row(s) → {style.brand(dest)}")
            return 0

        # Preview to stdout (clean rows, so it stays pipeable).
        if table.num_rows == 0:
            style.info("0 rows")
            return 0
        sys.stdout.write(table.to_pandas().to_string(index=False) + "\n")
        return 0

    @classmethod
    def _export(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        if not args.statement_id and not args.query:
            style.fail("export needs --statement-id or --query")
            return 1
        if args.statement_id and args.query:
            style.fail("pass only one of --statement-id / --query")
            return 1

        client = build_client(args)
        if args.statement_id:
            result = client.sql.statement_result(
                args.statement_id,
                warehouse_id=args.warehouse_id,
                warehouse_name=args.warehouse_name,
            )
        else:
            result = client.sql.execute(
                args.query,
                parameters=_kv(args.param) or None,
                warehouse_id=args.warehouse_id,
                warehouse_name=args.warehouse_name,
            )

        table = result.to_arrow_table()      # waits + materialises
        dest, n = cls._write_table(client, table, args.target, args.format)
        style.ok(f"exported {n} row(s) → {style.brand(dest)}")
        return 0

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _write_table(client: Any, table: Any, target: str, fmt: "str | None"):
        """Write an Arrow table to ``target`` in the format named by ``fmt``
        or inferred from the target's extension. Returns ``(dest, n_rows)``."""
        from yggdrasil.enums.media_type import MediaType

        media = (MediaType.from_(fmt, default=None) if fmt else None) \
            or MediaType.from_(target, default=None)
        if media is None:
            raise ValueError(
                f"cannot infer a format from {target!r}; "
                f"pass --format (csv/parquet/arrow/ndjson/json)"
            )

        s = str(target)
        if s.startswith(("dbfs:", "dbfs+", "/Volumes/", "/Workspace/")):
            path = client.path(s)            # workspace target (Volumes/DBFS/Workspace)
        else:
            from yggdrasil.path import Path as YggPath
            path = YggPath.from_(s)          # local / s3 / http

        with path.open("wb", media_type=media) as bio:
            bio.write_arrow_table(table)
        dest = path.full_path() if hasattr(path, "full_path") else s
        return dest, table.num_rows
