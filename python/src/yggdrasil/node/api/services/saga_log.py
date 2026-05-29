"""Per-asset operation log for Saga, on compressed Arrow IPC.

Mutations and SQL runs against a table are appended to an Arrow IPC stream file
partitioned ``{asset}/{YYYY-MM-DD}/{HH}.arrows`` and zstd-compressed. Reads and
list calls are *not* logged — only state changes and queries, with the acting
user. Dropping a table purges its whole log subtree.

Volume is low by design (one row per mutation/query), so an hourly partition is
rewritten in place on append: read it, concat one row, write it back. That keeps
each partition a single self-contained, columnar, compressed artifact that any
node can open with the standard Arrow reader.
"""
from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from threading import Lock

import pyarrow as pa
import pyarrow.ipc as ipc

_SCHEMA = pa.schema([
    ("ts", pa.string()),
    ("op", pa.string()),
    ("user", pa.string()),
    ("node", pa.string()),
    ("statement", pa.string()),
    ("rows", pa.int64()),
    ("detail", pa.string()),
])

_WRITE_OPTS = ipc.IpcWriteOptions(compression="zstd")


def _safe(asset: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in asset) or "_"


class OpLog:
    """Append-only, hour-partitioned, zstd Arrow-IPC log under one root."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = Lock()

    def _partition(self, asset: str, when: dt.datetime) -> Path:
        return (self._root / _safe(asset) / when.strftime("%Y-%m-%d")
                / f"{when.strftime('%H')}.arrows")

    def append(self, asset: str, op: str, *, user: str = "", node: str = "",
               statement: str = "", rows: int | None = None, detail: str = "") -> None:
        when = dt.datetime.now(dt.timezone.utc)
        row = {
            "ts": [when.isoformat()], "op": [op], "user": [user], "node": [node],
            "statement": [statement[:4000]], "rows": [rows], "detail": [detail[:1000]],
        }
        part = self._partition(asset, when)
        with self._lock:
            part.parent.mkdir(parents=True, exist_ok=True)
            new = pa.table(row, schema=_SCHEMA)
            if part.exists():
                try:
                    existing = ipc.open_stream(pa.memory_map(str(part), "r")).read_all()
                    new = pa.concat_tables([existing, new])
                except Exception:
                    pass  # corrupt/partial partition — start it over
            with pa.OSFile(str(part), "wb") as sink:
                with ipc.RecordBatchStreamWriter(sink, _SCHEMA, options=_WRITE_OPTS) as w:
                    w.write_table(new)

    def read(self, asset: str, *, limit: int = 200) -> pa.Table:
        """Most-recent ``limit`` rows across this asset's partitions."""
        base = self._root / _safe(asset)
        if not base.exists():
            return _SCHEMA.empty_table()
        parts = sorted(base.rglob("*.arrows"), reverse=True)  # newest day/hour first
        tables: list[pa.Table] = []
        total = 0
        for p in parts:
            try:
                t = ipc.open_stream(pa.memory_map(str(p), "r")).read_all()
            except Exception:
                continue
            tables.append(t)
            total += t.num_rows
            if total >= limit:
                break
        if not tables:
            return _SCHEMA.empty_table()
        out = pa.concat_tables(tables)
        # newest first, then cap
        order = pa.compute.sort_indices(out, sort_keys=[("ts", "descending")])
        return out.take(order).slice(0, limit)

    def purge(self, asset: str) -> None:
        base = self._root / _safe(asset)
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)

    def rename(self, old_asset: str, new_asset: str) -> None:
        """Move an asset's log subtree — keeps history across a table rename."""
        src = self._root / _safe(old_asset)
        if not src.exists() or old_asset == new_asset:
            return
        dst = self._root / _safe(new_asset)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        src.replace(dst)
