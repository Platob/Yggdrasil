"""Databricks-specific ``AsyncInsert`` — thin wrapper over the generic one.

Most of :class:`AsyncInsert`'s shape (metadata, serialisation, merge,
SQL building, auto statement generator) lives in
:mod:`yggdrasil.data.async_insert`. This module adds only the bits that
require a live Databricks workspace:

- the staging entry point :func:`stage_async_insert` (writes a Parquet
  payload + sibling JSON metadata under the table's
  ``stg_<table>/.sql/async/insert`` folder),
- the path-resolution hook (routes through :class:`DatabricksPath` so
  the lifecycle hook can unlink the staged files), and
- the cleanup-paths hook (binds resolved paths to the
  :class:`WarehousePreparedStatement` via ``external_volume_paths``).

The generic class already covers the rest — ``target`` (a
:class:`URL`) and ``schema`` (a :class:`Schema`) live on the base
:class:`StatementBatch` surface, the legacy ``target_*_name`` strings
are derived properties, and :meth:`AsyncInsert.to_statements` /
:meth:`AsyncInsert.execute` work across backends.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import time
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from yggdrasil.data.async_insert import (
    ASYNC_INSERT_DATA_SUBDIR,
    ASYNC_INSERT_LOGS_SUBDIR,
    ASYNC_INSERT_ROOT,
    METADATA_VERSION,
    AsyncInsert as _GenericAsyncInsert,
    AsyncWrite as _GenericAsyncWrite,
    iter_records,
    path_for_sql as _path_for_sql,
)
from yggdrasil.data.enums import Mode
from yggdrasil.data.options import CastOptions
from yggdrasil.data.statement import PreparedStatement
from yggdrasil.databricks.warehouse.statement import (
    WarehousePreparedStatement,
    WarehouseStatementBatch,
)
from yggdrasil.io.tabular.execution.expr import Predicate

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.fs import VolumePath
    from yggdrasil.databricks.sql.engine import SQLEngine
    from yggdrasil.databricks.warehouse import SQLWarehouse
    from .table import Table


__all__ = [
    "AsyncInsert",
    "AsyncWrite",
    "ASYNC_INSERT_ROOT",
    "ASYNC_INSERT_DATA_SUBDIR",
    "ASYNC_INSERT_LOGS_SUBDIR",
    "METADATA_VERSION",
    "stage_async_insert",
    "_iter_records",
    "_path_for_sql",
]


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AsyncInsert — Databricks-specific override of the generic AsyncInsert
# ---------------------------------------------------------------------------


class AsyncInsert(_GenericAsyncInsert, WarehouseStatementBatch):
    """Databricks-flavoured :class:`AsyncInsert`.

    Inherits the metadata, serialisation, merge, and SQL-building
    logic from :class:`yggdrasil.data.async_insert.AsyncInsert`, and
    plugs into :class:`WarehouseStatementBatch` so the same instance
    flows from "staged-on-disk metadata" to "live in-flight Databricks
    SQL batch" without an intermediate factory.

    Three hooks differ from the generic shape:

    - :attr:`_PREPARED_CLASS` is :class:`WarehousePreparedStatement` so
      :meth:`to_statements` produces warehouse-ready statements.
    - :meth:`_resolve_path` routes path strings through
      :meth:`DatabricksPath.from_` so the resulting handle is the
      singleton-cached one the lifecycle hook can ``unlink`` on
      success.
    - :meth:`_attach_cleanup_paths` binds the resolved
      :class:`VolumePath` handles to
      :attr:`WarehousePreparedStatement.external_volume_paths`, with
      ``temporary=True`` when ``cleanup`` is set so
      :meth:`WarehousePreparedStatement.clear_temporary_resources`
      unlinks every attached file when the statement lands.
    """

    _PREPARED_CLASS: ClassVar[type[PreparedStatement]] = WarehousePreparedStatement

    # ---- backend extension points ------------------------------------ #

    @classmethod
    def _resolve_executor(cls, target: Any) -> Any:
        """Coerce *target* into a :class:`SQLWarehouse`.

        Accepts either a live :class:`SQLWarehouse` (returned as-is)
        or an :class:`SQLEngine` whose :meth:`SQLEngine.warehouse`
        yields one. Anything else is returned unchanged — the caller
        owns the choice, the executor lifecycle handles the contract
        violations downstream.
        """
        warehouse = getattr(target, "warehouse", None)
        if callable(warehouse):
            return warehouse()
        return target

    def _resolve_path(self, path: Any, *, client: Any = None) -> Any:
        # Local import — keeps the module importable when the fs/path
        # graph isn't ready yet (circular at module load).
        from yggdrasil.databricks.path import DatabricksPath

        return DatabricksPath.from_(path, client=client)

    def _attach_cleanup_paths(
        self,
        prepared: PreparedStatement,
        *,
        parquet_paths: Sequence[Any],
        metadata_paths: Sequence[Any],
        cleanup: bool,
    ) -> None:
        """Bind staged paths to the :class:`WarehousePreparedStatement`.

        Each parquet payload rides under ``__pN__`` (also referenced
        in the SQL text) and each metadata file under ``__mN__`` (not
        referenced — it's there purely so the lifecycle hook removes
        the JSON sibling alongside the Parquet payload). With
        ``cleanup=True`` (the default) the path's ``temporary`` flag
        is flipped on so
        :meth:`WarehousePreparedStatement.clear_temporary_resources`
        unlinks it when the statement lands.
        """
        ext_paths: dict[str, Any] = dict(prepared.external_volume_paths or {})
        for i, path in enumerate(parquet_paths):
            if cleanup:
                # Mutate the (singleton-cached) path so the statement
                # batch's ``clear_temporary_resources`` walks it on
                # success and unlinks the file. The path is about to
                # be deleted; no other code should be holding it.
                path.temporary = True
            ext_paths[f"__p{i}__"] = path
        for i, path in enumerate(metadata_paths):
            if cleanup:
                path.temporary = True
            ext_paths[f"__m{i}__"] = path
        prepared.external_volume_paths = ext_paths or None

    def _build_prepared_statement(
        self,
        sql: str,
        *,
        retry: Any = None,
        client: Any = None,
    ) -> PreparedStatement:
        """Route through :meth:`WarehousePreparedStatement.prepare`.

        Threads the Databricks-specific routing kwargs (catalog /
        schema name) so the statement binds against the right
        namespace before submission. ``client`` is forwarded too so
        path coercion inside ``prepare`` reuses the bound workspace.
        """
        return WarehousePreparedStatement.prepare(
            sql,
            client=client,
            catalog_name=self.target_catalog_name,
            schema_name=self.target_schema_name,
            retry=retry,
        )

    # ---- file loading ------------------------------------------------ #

    @classmethod
    def from_file(
        cls,
        path: Any,
        *,
        client: "DatabricksClient | None" = None,
    ) -> "AsyncInsert":
        """Read a metadata JSON file and rebuild the :class:`AsyncInsert`."""
        from yggdrasil.databricks.path import DatabricksPath

        if not hasattr(path, "read_bytes"):
            path = DatabricksPath.from_(path, client=client)
        return cls.from_json_bytes(path.read_bytes())



# ---------------------------------------------------------------------------
# AsyncWrite — pinned to the Databricks AsyncInsert / WarehouseStatementBatch
# ---------------------------------------------------------------------------


class AsyncWrite(_GenericAsyncWrite):
    """Databricks-flavoured :class:`AsyncWrite`.

    Pins :attr:`_BATCH_CLASS` to :class:`WarehouseStatementBatch` and
    :attr:`_RECORD_CLASS` to the Databricks :class:`AsyncInsert`, so
    factory calls (``from_records`` / ``from_source``) submit through
    the warehouse path.
    """

    _BATCH_CLASS = WarehouseStatementBatch
    _RECORD_CLASS = AsyncInsert


# ---------------------------------------------------------------------------
# Databricks-pinned iter_records — pins the record class so :meth:`from_file`
# routes path strings through ``DatabricksPath.from_``.
# ---------------------------------------------------------------------------


def _iter_records(source: Any, *, client: Any = None):
    """Iterate :class:`AsyncInsert` records under *source* — Databricks variant.

    Pins the record class to the Databricks :class:`AsyncInsert` so
    :meth:`AsyncInsert.from_file` routes path strings through
    :meth:`DatabricksPath.from_` for the read.
    """
    return iter_records(source, cls=AsyncInsert, client=client)


# ---------------------------------------------------------------------------
# Stage helper (top-level API)
# ---------------------------------------------------------------------------


def _make_operation_id() -> str:
    """Unique per-operation id, monotonic-ish on epoch ms + random seed."""
    return f"async-{int(time.time() * 1000)}-{os.urandom(4).hex()}"


def _predicate_to_sql(value: Any) -> Optional[str]:
    """Best-effort SQL rendering of a :class:`Predicate` or pass-through string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Predicate):
        try:
            from yggdrasil.io.tabular.execution.expr.backends.sql import (
                Dialect,
                to_sql as expr_to_sql,
            )
            return expr_to_sql(value, dialect=Dialect.SPARK)
        except Exception:
            LOGGER.debug(
                "Could not render predicate %r to SQL for async insert metadata; "
                "falling back to repr.",
                value, exc_info=True,
            )
            return repr(value)
    return str(value)


def _enum_to_value(value: Any) -> Any:
    """Return ``value.value`` for enums, ``str(value)`` for unknown shapes,
    or pass primitives through unchanged."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value
    return str(value)


def _normalize_prune_by(prune_by: Any) -> Optional[Tuple[str, ...]]:
    if prune_by is None:
        return None
    if isinstance(prune_by, str):
        return (prune_by,)
    return tuple(prune_by)


def _normalize_prune_values(
    prune_values: Optional[Mapping[str, Any]],
) -> Optional[dict]:
    if not prune_values:
        return None
    out: dict[str, tuple] = {}
    for key, values in prune_values.items():
        if values is None:
            continue
        out[str(key)] = tuple(_enum_to_value(v) for v in values)
    return out or None


def stage_async_insert(
    table: "Table",
    data: Any,
    *,
    mode: Any = None,
    schema_mode: Any = None,
    cast_options: Optional[CastOptions] = None,
    overwrite_schema: bool | None = None,
    match_by: Optional[Sequence[str]] = None,
    update_column_names: Optional[Sequence[str]] = None,
    zorder_by: Optional[Sequence[str]] = None,
    optimize_after_merge: bool = False,
    vacuum_hours: int | None = None,
    where: Any = None,
    prune_by: Any = None,
    prune_values: Optional[Mapping[str, Any]] = None,
    safe_merge: bool = False,
    operation_id: str | None = None,
    lazy: bool = False,
) -> "VolumePath | AsyncInsert":
    """Stage *data* and a sibling :class:`AsyncInsert` metadata file.

    The Parquet payload is cast to the target table's existing schema
    when one can be resolved (the usual case — the target exists).
    When the target can't be inspected (e.g. it doesn't exist yet),
    the source rows are written as-is and the metadata records the
    fact so the applier can decide whether to ``CREATE TABLE`` first.

    Returns the :class:`VolumePath` to the staged Parquet file by
    default. With ``lazy=True``, returns the constructed
    :class:`AsyncInsert` record instead so the caller can ``execute``,
    ``merge_with``, or schedule it directly without re-reading the
    sibling metadata file.
    """
    op_id = operation_id or _make_operation_id()
    folder = table.staging_folder(temporary=False, async_write=True)

    # Data and logs live in sibling subfolders so the applier can walk
    # one without filtering past the other — see the module docstring.
    parquet_path = folder.joinpath(ASYNC_INSERT_DATA_SUBDIR).joinpath(
        f"{op_id}.parquet"
    )
    meta_path = folder.joinpath(ASYNC_INSERT_LOGS_SUBDIR).joinpath(
        f"{op_id}.json"
    )

    # Best-effort target schema resolution. ``collect_schema`` raises
    # when the table doesn't exist yet — that's fine, the applier
    # handles the cold-start case via the recorded ``schema_mode``
    # plus the source schema embedded in the Parquet.
    existing_schema = None
    try:
        existing_schema = table.collect_schema()
    except Exception:
        LOGGER.debug(
            "Target table %r has no resolvable schema; writing rows as-is.",
            table, exc_info=True,
        )

    opts = CastOptions.check(options=cast_options)
    if existing_schema is not None:
        opts = opts.with_target(existing_schema)

    parquet_path.write_table(data, opts, mode=Mode.OVERWRITE)

    # Carry the live :class:`VolumePath` objects on the record so
    # downstream steps (``to_statements`` / ``cleanup`` / inspection in
    # tests) reuse the same singleton-cached instance instead of
    # re-coercing the string back into a path. ``to_dict`` /
    # ``to_json_bytes`` / pickle emit URL strings (see ``to_dict``
    # / ``__getstate__`` on the base class), so the staged JSON
    # metadata format is unchanged.
    record = AsyncInsert(
        target=table.url,
        schema=existing_schema,
        parquet_paths=(parquet_path,),
        metadata_paths=(meta_path,),
        operation_ids=(op_id,),
        created_at=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        mode=_enum_to_value(mode),
        schema_mode=_enum_to_value(schema_mode),
        overwrite_schema=overwrite_schema,
        match_by=tuple(match_by) if match_by else None,
        update_column_names=(
            tuple(update_column_names) if update_column_names else None
        ),
        zorder_by=tuple(zorder_by) if zorder_by else None,
        optimize_after_merge=bool(optimize_after_merge),
        vacuum_hours=vacuum_hours,
        where=_predicate_to_sql(where),
        prune_by=_normalize_prune_by(prune_by),
        prune_values=_normalize_prune_values(prune_values),
        safe_merge=bool(safe_merge),
    )

    meta_path.write_bytes(record.to_json_bytes())

    LOGGER.info(
        "Staged async insert %s for %r at %r",
        op_id, table, parquet_path,
    )
    return record if lazy else parquet_path


def _resolve_current_client() -> "DatabricksClient":
    """Lazy import + ``current()`` shortcut so the applier helpers
    don't drag :class:`DatabricksClient` into every call signature."""
    from yggdrasil.databricks.client import DatabricksClient
    return DatabricksClient.current()
