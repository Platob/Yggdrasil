"""Databricks SQL execution service for the Excel router.

Uses PyArrow exclusively (no pandas/polars) and caches results in-memory
with per-query TTL to avoid flooding Databricks with duplicate calls.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any

import pyarrow as pa
from fastapi.concurrency import run_in_threadpool

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.sql.engine import SQLEngine

from ..config import Settings
from ..exceptions import APIError
from ..schemas.databricks import DatabricksSQLRequest, DatabricksSQLResponse
from ..schemas.python import DataFrameColumn, DataFramePayload


# ---------------------------------------------------------------------------
# Lightweight TTL cache entry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _CacheEntry:
    table: pa.Table
    created_at: float
    ttl: int  # seconds

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.created_at) >= self.ttl


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DatabricksExcelService:
    """Execute SQL on Databricks and return DataFramePayload for Excel/Power Query.

    Features
    --------
    * Pure PyArrow pipeline — no pandas/polars dependency.
    * In-memory TTL cache keyed on (statement + connection params) to reduce
      redundant Databricks SQL calls.  Bounded by ``max_size`` and evicted
      by LRU + TTL.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        max_size = 128
        default_ttl = 300
        if settings is not None:
            max_size = settings.databricks_cache_max_size
            default_ttl = settings.databricks_cache_default_ttl

        self._max_size: int = max(1, max_size)
        self._default_ttl: int = max(0, default_ttl)
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Async bridge
    # ------------------------------------------------------------------

    async def _run(self, fn, /, *args, **kwargs):
        return await run_in_threadpool(partial(fn, *args, **kwargs))

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(req: DatabricksSQLRequest) -> str:
        """Deterministic hash of the request's identity fields."""
        blob = json.dumps(
            {
                "statement": req.statement,
                "host": req.host,
                "catalog_name": req.catalog_name,
                "schema_name": req.schema_name,
                "warehouse_id": req.warehouse_id,
                "warehouse_name": req.warehouse_name,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _get_cached(self, key: str) -> pa.Table | None:
        """Return a cached Arrow table if present and not expired (LRU touch)."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.expired:
                self._cache.pop(key, None)
                return None
            # LRU: move to end on access
            self._cache.move_to_end(key)
            return entry.table

    def _put_cached(self, key: str, table: pa.Table, ttl: int) -> None:
        """Store an Arrow table in the cache, evicting the oldest entry if full."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = _CacheEntry(
                    table=table, created_at=time.monotonic(), ttl=ttl,
                )
                return
            # Evict oldest entries until we have room
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = _CacheEntry(
                table=table, created_at=time.monotonic(), ttl=ttl,
            )

    def _evict_expired(self) -> None:
        """Sweep and remove all expired entries (best-effort)."""
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.expired]
            for k in expired_keys:
                self._cache.pop(k, None)

    # ------------------------------------------------------------------
    # Databricks client / engine builders
    # ------------------------------------------------------------------

    def _build_client(self, host: str | None, token: str | None) -> DatabricksClient:
        kwargs: dict[str, Any] = {}
        if host is not None:
            kwargs["host"] = host
        if token is not None:
            kwargs["token"] = token

        if kwargs:
            return DatabricksClient(**kwargs)
        return DatabricksClient.current()

    def _build_engine(
        self,
        client: DatabricksClient,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
    ) -> SQLEngine:
        return SQLEngine(
            client=client,
            catalog_name=catalog_name,
            schema_name=schema_name,
        )

    # ------------------------------------------------------------------
    # Core execution (runs in threadpool)
    # ------------------------------------------------------------------

    @staticmethod
    def _arrow_table_to_payload(
        table: pa.Table,
        *,
        df_name: str,
        max_rows: int | None,
    ) -> tuple[DataFramePayload, int]:
        """Convert a PyArrow Table into a DataFramePayload (no pandas)."""
        total_rows = table.num_rows
        export_table = table if max_rows is None else table.slice(0, max_rows)
        returned_rows = export_table.num_rows

        rows: list[dict[str, Any]] = export_table.to_pylist()
        columns = [f.name for f in export_table.schema]
        schema = [
            DataFrameColumn(name=f.name, dtype=str(f.type))
            for f in export_table.schema
        ]

        payload = DataFramePayload(
            df_name=df_name,
            columns=columns,
            schema=schema,
            rows=rows,
            row_count=total_rows,
            returned_rows=returned_rows,
            truncated=bool(max_rows is not None and total_rows > returned_rows),
        )
        return payload, total_rows

    def _fetch_arrow_table(self, req: DatabricksSQLRequest) -> pa.Table:
        """Execute the SQL and return the raw Arrow table from Databricks."""
        client = self._build_client(req.host, req.token)
        engine = self._build_engine(
            client,
            catalog_name=req.catalog_name,
            schema_name=req.schema_name,
        )

        result = engine.execute(
            req.statement,
            row_limit=req.max_rows,
            warehouse_id=req.warehouse_id,
            warehouse_name=req.warehouse_name,
            wait=True,
            raise_error=True,
        )
        return result.to_arrow_table()

    def _execute_sql(self, req: DatabricksSQLRequest) -> DatabricksSQLResponse:
        ttl = req.cache_ttl if req.cache_ttl is not None else self._default_ttl
        use_cache = ttl > 0
        cache_key = self._cache_key(req) if use_cache else ""
        cache_hit = False

        # --- cache lookup ---
        table: pa.Table | None = None
        if use_cache and not req.force_refresh:
            table = self._get_cached(cache_key)
            if table is not None:
                cache_hit = True

        # --- fetch from Databricks ---
        if table is None:
            table = self._fetch_arrow_table(req)
            if use_cache:
                self._put_cached(cache_key, table, ttl)

        # --- build response ---
        payload, total_rows = self._arrow_table_to_payload(
            table, df_name=req.df_name, max_rows=req.max_rows,
        )

        return DatabricksSQLResponse(
            ok=True,
            data=payload,
            row_count=total_rows,
            truncated=payload.truncated,
            cache_hit=cache_hit,
        )

    # ------------------------------------------------------------------
    # Public async entry-point
    # ------------------------------------------------------------------

    async def execute_sql(self, req: DatabricksSQLRequest) -> DatabricksSQLResponse:
        try:
            return await self._run(self._execute_sql, req)
        except Exception as exc:
            if isinstance(exc, APIError):
                raise
            raise APIError(
                detail=f"{type(exc).__name__}: {exc}",
                status_code=500,
            ) from exc

