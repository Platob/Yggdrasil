"""Tabular service — inspect reads the parquet footer only, never row data.

Opening a file in the UI needs the schema, an exact row count, and an editable
flag. For parquet that all lives in the footer (O(1) regardless of file size),
so ``inspect`` reads metadata + schema and never pulls a single row.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from yggdrasil.exceptions.api import NotFoundError
from yggdrasil.node.api.services.fs import FsService

# Pydantic v2 warns that `schema` shadows BaseModel.schema() — the field name
# is intentional (the node contract is `.schema`), so we filter it at load.
warnings.filterwarnings(
    "ignore",
    message=r"Field name \"schema\" in \"InspectResult\" shadows",
    category=UserWarning,
)


class InspectResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    row_count: int
    editable: bool
    schema: dict


class TabularService:
    """Footer-only parquet inspection rooted at ``settings.node_home``."""

    def __init__(self, settings: object, fs: FsService) -> None:
        self._root = Path(settings.node_home)
        self._fs = fs
        self._preview_rows = settings.tabular_preview_max_rows

    async def inspect(self, path: str) -> InspectResult:
        full = self._root / path
        if not full.is_file():
            raise NotFoundError(f"File {path!r} not found.")
        meta = pq.read_metadata(str(full))
        schema = pq.read_schema(str(full))
        return InspectResult(
            row_count=meta.num_rows,
            editable=False,
            schema={field.name: str(field.type) for field in schema},
        )
