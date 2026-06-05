"""Per-asset operation log for Saga, on compressed Arrow IPC.

Mutations and SQL runs against a table are appended under an hour partition
``{asset}/{YYYY-MM-DD}/{HH}/`` as one tiny zstd Arrow-IPC file per op. Appending
is O(1) — write one file, no read-modify-write — so logging never weighs on the
query path. Reads/list calls are *not* logged; dropping a table purges its log
subtree. ``compact`` folds an hour's files into one when they accumulate.
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path
from threading import Lock

import pyarrow as pa
import pyarrow.compute as pc
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
# Fold an hour's per-op files into one once this many accumulate.
_COMPACT_AT = 256


def _safe(asset: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in asset) or "_"


def _write(path: Path, table: pa.Table) -> None:
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.RecordBatchStreamWriter(sink, _SCHEMA, options=_WRITE_OPTS) as w:
            w.write_table(table)


class OpLog:
    """Hour-partitioned, one-file-per-op, zstd Arrow-IPC log under one root."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = Lock()
        self._seq = 0

    def _hour_dir(self, asset: str, when: dt.datetime) -> Path:
        return (self._root / _safe(asset) / when.strftime("%Y-%m-%d")
                / when.strftime("%H"))

    def append(self, asset: str, op: str, *, user: str = "", node: str = "",
               statement: str = "", rows: int | None = None, detail: str = "") -> None:
        when = dt.datetime.now(dt.timezone.utc)
        row = pa.table({
            "ts": [when.isoformat()], "op": [op], "user": [user], "node": [node],
            "statement": [statement[:4000]], "rows": [rows], "detail": [detail[:1000]],
        }, schema=_SCHEMA)
        hour = self._hour_dir(asset, when)
        with self._lock:
            hour.mkdir(parents=True, exist_ok=True)
            self._seq += 1
            _write(hour / f"{when.strftime('%H%M%S%f')}-{self._seq}.arrows", row)
            try:
                names = os.listdir(hour)
            except OSError:
                names = []
            if len(names) >= _COMPACT_AT:
                self._compact(hour, names)

    def _compact(self, hour: Path, names: list[str]) -> None:
        tables, paths = [], []
        for n in names:
            p = hour / n
            try:
                tables.append(ipc.open_stream(pa.memory_map(str(p), "r")).read_all())
                paths.append(p)
            except Exception:
                continue
        if not tables:
            return
        merged = pa.concat_tables(tables)
        merged = merged.take(pc.sort_indices(merged, sort_keys=[("ts", "ascending")]))
        tmp = hour / "_compact.arrows"
        _write(tmp, merged)
        for p in paths:
            p.unlink(missing_ok=True)
        tmp.replace(hour / "000000000000-0.arrows")

    def read(self, asset: str, *, limit: int = 200) -> pa.Table:
        """Most-recent ``limit`` rows across this asset's partitions."""
        base = self._root / _safe(asset)
        if not base.exists():
            return _SCHEMA.empty_table()
        parts = sorted(base.rglob("*.arrows"), reverse=True)  # newest day/hour/op first
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
        out = out.take(pc.sort_indices(out, sort_keys=[("ts", "descending")]))
        return out.slice(0, limit)

    def recent_all(self, *, limit: int = 50) -> list[dict]:
        """Most-recent ops across *every* asset — the monitoring feed.

        Each returned row carries an extra ``asset`` key recovered from its log
        subtree, so the dashboard can show "what changed across the catalog"
        without asking per asset. Newest-first, capped at ``limit``.
        """
        if not self._root.exists():
            return []
        parts = sorted(self._root.rglob("*.arrows"), reverse=True)
        rows: list[dict] = []
        for p in parts:
            # Layout is <root>/<safe_asset>/<day>/<hour>/<file>.arrows — the asset
            # dir is three levels up from the file.
            try:
                asset = p.relative_to(self._root).parts[0]
            except ValueError:
                asset = ""
            try:
                t = ipc.open_stream(pa.memory_map(str(p), "r")).read_all()
            except Exception:
                continue
            for r in t.to_pylist():
                r["asset"] = asset
                rows.append(r)
            if len(rows) >= limit * 3:  # over-read a little, then sort + slice
                break
        rows.sort(key=lambda r: r.get("ts") or "", reverse=True)
        return rows[:limit]

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
