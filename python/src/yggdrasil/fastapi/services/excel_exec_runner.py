"""Subprocess runner for Excel prepare endpoint.

Invoked as::

    python excel_exec_runner.py <request.json> <result.parquet> <manifest.json>

* Reads ``request.json`` for ``code``, ``df_name``, and ``max_rows``.
* Executes the user code in a sandboxed namespace.
* Writes the resulting DataFrame as a Parquet file and a JSON manifest.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <request.json> <result.parquet> <manifest.json>", file=sys.stderr)
        sys.exit(2)

    request_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    manifest_path = Path(sys.argv[3])

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    user_code: str = payload["code"]
    df_name: str = payload.get("df_name", "df")
    max_rows: int | None = payload.get("max_rows")

    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]
        _write_error(manifest_path, "ImportError", "pandas is not installed in this environment")
        sys.exit(1)

    globals_dict: dict = {"__name__": "__main__", "__file__": "<excel-prepare>"}
    locals_dict: dict = {}

    try:
        exec(compile(user_code, "<user-code>", "exec"), globals_dict, locals_dict)  # noqa: S102
        namespace = {**globals_dict, **locals_dict}

        if df_name not in namespace:
            raise KeyError(f"Expected DataFrame '{df_name}' in locals()")

        df = namespace[df_name]
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"locals()['{df_name}'] must be a pandas.DataFrame, got {type(df).__name__}"
            )

        total_rows = len(df.index)
        export_df = df if max_rows is None else df.head(max_rows)

        # Write parquet
        export_df.to_parquet(str(result_path), index=False, engine="pyarrow")

        columns = [str(c) for c in export_df.columns]
        schema = [
            {"name": str(c), "dtype": str(d)}
            for c, d in zip(export_df.columns, export_df.dtypes)
        ]

        manifest = {
            "df_name": df_name,
            "columns": columns,
            "schema": schema,
            "row_count": total_rows,
            "returned_rows": len(export_df.index),
            "truncated": bool(max_rows is not None and total_rows > len(export_df.index)),
            "result_path": str(result_path),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    except Exception as exc:
        _write_error(manifest_path, type(exc).__name__, str(exc))
        traceback.print_exc()
        sys.exit(1)


def _write_error(manifest_path: Path, error_type: str, message: str) -> None:
    manifest_path.write_text(
        json.dumps(
            {
                "ok": False,
                "error": {
                    "type": error_type,
                    "message": message,
                    "traceback": traceback.format_exc(),
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

