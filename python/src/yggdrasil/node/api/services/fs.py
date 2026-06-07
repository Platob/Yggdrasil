"""Filesystem service — inspect node-local data files.

Lists files under ``node_home`` (glob-filtered), reads parquet/arrow schemas
without materializing data, and reports aggregate disk usage. All paths are
constrained to ``node_home`` so a client can't probe the host filesystem.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..schemas.base import now_ms

__all__ = ["FsService"]


class FsService:
    def __init__(self, node_home: Path) -> None:
        self._home = node_home

    def _resolve(self, path: str) -> Path:
        resolved = (self._home / path).resolve()
        home = self._home.resolve()
        if home not in resolved.parents and resolved != home:
            raise PermissionError(path)
        return resolved

    async def list_files(self, path: str, glob: str) -> list[dict[str, Any]]:
        root = self._resolve(path or ".")
        if not root.exists():
            return []
        if root.is_file():
            return [self._describe(root)]
        return [self._describe(p) for p in sorted(root.glob(glob)) if p.is_file()]

    def _describe(self, p: Path) -> dict[str, Any]:
        stat = p.stat()
        return {
            "name": p.name,
            "path": str(p.relative_to(self._home.resolve())),
            "size": stat.st_size,
            "modified": int(stat.st_mtime * 1000),
            "suffix": p.suffix.lower(),
        }

    async def read_parquet_schema(self, path: str) -> dict[str, Any]:
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(path)
        meta = pq.read_metadata(resolved)
        schema = meta.schema.to_arrow_schema()
        return {
            "path": path,
            "rows": meta.num_rows,
            "row_groups": meta.num_row_groups,
            "columns": [
                {"name": f.name, "type": str(f.type), "nullable": f.nullable}
                for f in schema
            ],
            "size": resolved.stat().st_size,
        }

    async def get_stats(self) -> dict[str, Any]:
        home = self._home.resolve()
        files = [p for p in home.rglob("*") if p.is_file()] if home.exists() else []
        total_size = sum(p.stat().st_size for p in files)
        by_suffix: dict[str, int] = {}
        for p in files:
            by_suffix[p.suffix.lower() or "(none)"] = by_suffix.get(p.suffix.lower() or "(none)", 0) + 1
        return {
            "home": str(home),
            "file_count": len(files),
            "total_size": total_size,
            "by_suffix": by_suffix,
            "ts": now_ms(),
        }
