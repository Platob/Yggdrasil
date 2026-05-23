"""Stage table inserts as Parquet + metadata for asynchronous execution.

When :meth:`Table.insert` is called with ``lazy=True``, the caller is
not waiting on a warehouse round trip — the rows are cast to the
target schema, written as Parquet under the table's
``<table>/.sql/async/insert`` staging folder, and a sibling JSON
file carries an :class:`AsyncInsert` record so a downstream applier
(typically a job-driven loop) can replay the operation against the
target table when it's convenient. The :class:`AsyncInsert` itself is
returned: it's a :class:`WarehouseStatementBatch` subclass, so the
caller can ``.execute(engine)`` straight away (or
``.merge_with(other)`` peers, or schedule a Databricks applier job
via :class:`AsyncInsertJob` in :mod:`.async_job`).

File layout under the table's async staging folder::

    .sql/async/insert/
        data/async-<epoch_ms>-<seed>.parquet   # rows, cast to target schema
        logs/async-<epoch_ms>-<seed>.json      # operation metadata (orjson)

Data files and metadata logs live in sibling folders so the applier
can drain one without listing the other — bulk-deleting ``data/``
after a successful apply doesn't have to skip past JSON entries, and
walking ``logs/`` for merge candidates doesn't have to filter out
Parquet payloads.

This module lives outside ``table.py`` so the latter doesn't pick up
async-specific helpers; :meth:`Table.insert` delegates to
:func:`stage_async_insert` only when ``lazy=True``.

Multiple staged operations against the same table can be folded into
one logical insert via :meth:`AsyncInsert.merge` — pure appends fold
together, an overwrite drops every earlier op for that target, and
the resulting record renders to a single ``INSERT INTO`` / ``INSERT
OVERWRITE`` statement via :meth:`AsyncInsert.to_sql`. The unified
apply path is :class:`AsyncWrite`, a :class:`WarehouseStatementBatch`
that submits every merged target as one batch of prepared statements
and unlinks the staged files via the batch's wait-hook on success.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import time
from collections import OrderedDict
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from yggdrasil.data.enums import Mode
from yggdrasil.data.options import CastOptions
from yggdrasil.databricks.warehouse.statement import WarehouseStatementBatch
from yggdrasil.execution.expr import Predicate
from yggdrasil.pickle import json as ygg_json

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.fs import VolumePath
    from yggdrasil.databricks.sql.engine import SQLEngine
    from yggdrasil.databricks.warehouse import SQLWarehouse
    from yggdrasil.io.path import Path
    from .table import Table


__all__ = [
    "AsyncInsert",
    "AsyncWrite",
    "ASYNC_INSERT_ROOT",
    "ASYNC_INSERT_DATA_SUBDIR",
    "ASYNC_INSERT_LOGS_SUBDIR",
    "METADATA_VERSION",
    "stage_async_insert",
]


LOGGER = logging.getLogger(__name__)

# Wire-format version. Bump when the JSON schema changes in a way that
# breaks existing appliers so the consumer can fail loudly instead of
# mis-applying an operation.
METADATA_VERSION = 1

# Staging folder layout — Parquet payloads go under ``data/`` and the
# JSON metadata logs go under ``logs/``. Sibling folders keep the two
# concerns separable: appliers walk ``logs/`` for merge candidates and
# walk ``data/`` for cleanup without filtering past the other.
ASYNC_INSERT_ROOT: str = ".sql/async/insert"
ASYNC_INSERT_DATA_SUBDIR: str = "data"
ASYNC_INSERT_LOGS_SUBDIR: str = "logs"

# Mode tokens, normalized to lowercase. Anything else is treated as an
# append unless explicitly listed.
_APPEND_TOKENS: frozenset[str] = frozenset({"append", "", "auto", "insert"})
_OVERWRITE_TOKENS: frozenset[str] = frozenset({"overwrite"})


# ---------------------------------------------------------------------------
# AsyncInsert — a deferred table insert that's also an executable
# WarehouseStatementBatch.
# ---------------------------------------------------------------------------


# Ordered list of metadata field names + defaults — drives :meth:`__init__`,
# :meth:`_replace`, :meth:`to_dict`, :meth:`from_dict`, equality, and the
# tuple-coercion set in :meth:`from_dict`. Kept module-level so the hot
# per-record walks don't rebuild it on every call.
_METADATA_FIELDS: Tuple[Tuple[str, Any], ...] = (
    ("target_full_name", ""),
    ("parquet_paths", ()),
    ("metadata_paths", ()),
    ("operation_ids", ()),
    ("created_at", ""),
    ("target_catalog_name", None),
    ("target_schema_name", None),
    ("target_table_name", None),
    ("target_field_names", None),
    ("mode", None),
    ("schema_mode", None),
    ("overwrite_schema", None),
    ("match_by", None),
    ("update_column_names", None),
    ("zorder_by", None),
    ("optimize_after_merge", False),
    ("vacuum_hours", None),
    ("where", None),
    ("prune_by", None),
    ("prune_values", None),
    ("safe_merge", False),
    ("version", METADATA_VERSION),
)


class AsyncInsert(WarehouseStatementBatch):
    """JSON-serialisable description of a deferred table insert.

    Records a list of Parquet payloads (one per merged operation, all
    rooted under the same target's async staging folder) plus the
    insert-time parameters needed to replay the operation against the
    target table.

    Also a :class:`WarehouseStatementBatch` — :meth:`execute` plugs in
    an executor and submits the rendered prepared statements through
    it, so the same instance flows from "staged-on-disk metadata" to
    "live in-flight batch" without an intermediate factory.

    Constructed without an executor it sits in metadata-only mode
    (``self.executor is None``, ``self.results`` empty), suitable for
    JSON round-tripping, merging, and :meth:`to_sql` rendering. Pass
    ``executor=`` to submit straight away.

    ``parquet_paths`` / ``metadata_paths`` accept either :class:`str`
    URLs or live :class:`DatabricksPath` instances; in-memory the
    record carries whatever it was handed (so :func:`stage_async_insert`
    can stash the freshly-built :class:`VolumePath` objects without
    forcing a round-trip through string coercion). On any serialisation
    path — :meth:`to_dict`, :meth:`to_json_bytes`, pickle via
    :meth:`__reduce__` — path entries are dumped as their URL strings
    and the live :attr:`executor` / :attr:`results` are stripped so the
    output is a clean metadata snapshot.
    """

    _METADATA_FIELDS: ClassVar[Tuple[Tuple[str, Any], ...]] = _METADATA_FIELDS

    # Live handles + in-flight state that don't survive pickling — the
    # bound warehouse, the queue of submitted prepared statements, and
    # the schema cache rebuild on the receiver. Matches the
    # ``yggdrasil.io.session.Session`` / ``yggdrasil.aws.AWSClient``
    # pickle pattern (see ``AGENTS.md`` → "Make objects picklable").
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({
        "executor",
        "results",
        "external_volume_paths",
        "_cached_schema",
        "start_timestamp",
    })

    def __init__(
        self,
        target_full_name: str = "",
        *,
        parquet_paths: Tuple["Path | str", ...] = (),
        metadata_paths: Tuple["Path | str", ...] = (),
        operation_ids: Tuple[str, ...] = (),
        created_at: str = "",
        target_catalog_name: Optional[str] = None,
        target_schema_name: Optional[str] = None,
        target_table_name: Optional[str] = None,
        target_field_names: Optional[Tuple[str, ...]] = None,
        mode: Optional[str] = None,
        schema_mode: Optional[str] = None,
        overwrite_schema: Optional[bool] = None,
        match_by: Optional[Tuple[str, ...]] = None,
        update_column_names: Optional[Tuple[str, ...]] = None,
        zorder_by: Optional[Tuple[str, ...]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: Optional[int] = None,
        where: Optional[str] = None,
        prune_by: Optional[Tuple[str, ...]] = None,
        prune_values: Optional[Mapping[str, Tuple[Any, ...]]] = None,
        safe_merge: bool = False,
        version: int = METADATA_VERSION,
        executor: "SQLWarehouse | None" = None,
        parallel: int = 1,
    ):
        self.target_full_name = target_full_name
        self.parquet_paths = parquet_paths
        self.metadata_paths = metadata_paths
        self.operation_ids = operation_ids
        self.created_at = created_at
        self.target_catalog_name = target_catalog_name
        self.target_schema_name = target_schema_name
        self.target_table_name = target_table_name
        self.target_field_names = target_field_names
        self.mode = mode
        self.schema_mode = schema_mode
        self.overwrite_schema = overwrite_schema
        self.match_by = match_by
        self.update_column_names = update_column_names
        self.zorder_by = zorder_by
        self.optimize_after_merge = optimize_after_merge
        self.vacuum_hours = vacuum_hours
        self.where = where
        self.prune_by = prune_by
        self.prune_values = prune_values
        self.safe_merge = safe_merge
        self.version = version

        # WarehouseStatementBatch handles results / external_volume_paths.
        # No statements are eagerly submitted: rendering only happens once
        # an executor is bound (via ``execute``) so the metadata-only mode
        # never touches a warehouse.
        super().__init__(executor=executor, statements=None, parallel=parallel)

    # ------------------------------------------------------------------ #
    # Replace / equality / hash / repr
    # ------------------------------------------------------------------ #
    def _replace(self, **changes: Any) -> "AsyncInsert":
        """Return a copy of this record with *changes* applied.

        Stands in for :func:`dataclasses.replace` now that AsyncInsert
        is a regular class; carries only the metadata fields so the
        result is in metadata-only mode regardless of *self*'s state.
        """
        kwargs = {name: getattr(self, name) for name, _ in _METADATA_FIELDS}
        kwargs.update(changes)
        return type(self)(**kwargs)

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, AsyncInsert):
            return NotImplemented
        for name, _ in _METADATA_FIELDS:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self) -> int:
        # Identity-only hash keeps records usable as dict keys without
        # walking every field. ``operation_ids`` + target uniquely
        # identifies a (merged) record.
        return hash((self.target_full_name, self.operation_ids))

    def __repr__(self) -> str:
        return (
            f"AsyncInsert(target={self.target_full_name!r}, "
            f"ops={len(self.operation_ids)}, mode={self.mode!r})"
        )

    def __getstate__(self) -> dict[str, Any]:
        """Pickle the metadata snapshot — drop the live executor.

        Follows the project's transient-state pattern (see
        :class:`yggdrasil.io.session.Session`,
        :class:`yggdrasil.aws.AWSClient`): attributes named in
        :attr:`_TRANSIENT_STATE_ATTRS` (executor, results, schema cache,
        …) are stripped from ``__dict__`` so the live warehouse handle
        and in-flight result map never cross the wire. Path entries on
        :attr:`parquet_paths` / :attr:`metadata_paths` are dumped as
        URL strings via :func:`_path_for_sql` — the receiver gets
        plain strings, ready to coerce lazily through
        :meth:`DatabricksPath.from_` when it needs a live handle again.
        """
        state = {
            k: v for k, v in self.__dict__.items()
            if k not in self._TRANSIENT_STATE_ATTRS
        }
        state["parquet_paths"] = tuple(_path_for_sql(p) for p in self.parquet_paths)
        state["metadata_paths"] = tuple(_path_for_sql(p) for p in self.metadata_paths)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore *state* and re-init the dropped transient handles."""
        self.__dict__.update(state)
        # Reset transients to the same defaults ``WarehouseStatementBatch``
        # / ``StatementBatch`` set up in ``__init__``.
        self.executor = None  # type: ignore[assignment]
        self.results = OrderedDict()
        self.external_volume_paths = {}
        self._cached_schema = None
        self.start_timestamp = None

    # ---- derived ---------------------------------------------------------
    @property
    def operation_id(self) -> str:
        """First operation id (the primary one before any merge)."""
        return self.operation_ids[0] if self.operation_ids else ""

    @property
    def is_overwrite(self) -> bool:
        return (self.mode or "").lower() in _OVERWRITE_TOKENS

    @property
    def is_append(self) -> bool:
        mode = (self.mode or "").lower()
        return mode in _APPEND_TOKENS or not self.is_overwrite

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a plain-dict view suitable for JSON serialisation.

        Tuples are emitted as lists so a downstream reader doesn't need
        to import this module to read the file. The ``prune_values``
        mapping is shallow-copied with list-of-values entries.

        Manually inlined (vs. :func:`dataclasses.asdict`) so JSON
        serialisation in the hot per-stage / per-apply path doesn't
        pay the deep-walk + copy cost asdict adds for every flat
        scalar field.
        """
        pv = self.prune_values
        tfn = self.target_field_names
        mb = self.match_by
        ucn = self.update_column_names
        zb = self.zorder_by
        pb = self.prune_by
        # Path entries may be live :class:`DatabricksPath` objects (set
        # by :func:`stage_async_insert`) — coerce each to its URL string
        # so the dict survives orjson / pickle. ``_path_for_sql`` is a
        # no-op for plain strings.
        return {
            "target_full_name": self.target_full_name,
            "parquet_paths": [_path_for_sql(p) for p in self.parquet_paths],
            "metadata_paths": [_path_for_sql(p) for p in self.metadata_paths],
            "operation_ids": list(self.operation_ids),
            "created_at": self.created_at,
            "target_catalog_name": self.target_catalog_name,
            "target_schema_name": self.target_schema_name,
            "target_table_name": self.target_table_name,
            "target_field_names": list(tfn) if tfn else None,
            "mode": self.mode,
            "schema_mode": self.schema_mode,
            "overwrite_schema": self.overwrite_schema,
            "match_by": list(mb) if mb else None,
            "update_column_names": list(ucn) if ucn else None,
            "zorder_by": list(zb) if zb else None,
            "optimize_after_merge": self.optimize_after_merge,
            "vacuum_hours": self.vacuum_hours,
            "where": self.where,
            "prune_by": list(pb) if pb else None,
            "prune_values": (
                {k: list(v) for k, v in pv.items()} if pv else None
            ),
            "safe_merge": self.safe_merge,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AsyncInsert":
        """Rebuild an :class:`AsyncInsert` from a JSON-loaded dict."""
        # ``_FIELD_NAMES`` and ``_TUPLE_FIELD_NAMES`` are module-level
        # frozensets baked at class-definition time so the hot per-apply
        # walk doesn't re-derive them via ``fields(cls)`` on every call.
        kwargs: dict[str, Any] = {}
        tuple_fields = _TUPLE_FIELD_NAMES
        for key, value in data.items():
            if key not in _FIELD_NAMES:
                continue
            if key in tuple_fields and isinstance(value, list):
                value = tuple(value)
            kwargs[key] = value

        prune_values = kwargs.get("prune_values")
        if isinstance(prune_values, Mapping):
            kwargs["prune_values"] = {
                str(k): tuple(v) if isinstance(v, list) else v
                for k, v in prune_values.items()
            }

        return cls(**kwargs)

    def to_json_bytes(self) -> bytes:
        return ygg_json.dumps(self.to_dict())

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "AsyncInsert":
        return cls.from_dict(ygg_json.loads(data))

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

    # ------------------------------------------------------------------ #
    # Merge
    # ------------------------------------------------------------------ #
    def merge_with(self, other: "AsyncInsert") -> "AsyncInsert":
        """Combine two records for the same target into one.

        Both records must share ``target_full_name``. Semantics:

        - Both append → parquet paths are concatenated, the rest of the
          spec inherits the older record with newer non-``None`` values
          overriding (``match_by``, ``zorder_by``, …).
        - One overwrite (and it's the newer one) → drops the older
          record entirely; the newer overwrite stands alone.
        - One overwrite (and it's the older one) → the newer append's
          rows are folded into the overwrite scope so a single
          ``INSERT OVERWRITE`` carries every staged row.
        - Both overwrite → take the newer overwrite.
        """
        if self.target_full_name != other.target_full_name:
            raise ValueError(
                f"Cannot merge AsyncInsert records for different targets "
                f"({self.target_full_name!r} vs {other.target_full_name!r}); "
                "merge groups records by target_full_name before pairwise combine."
            )

        # Pairwise: one comparison beats ``sorted([self, other], key=…)``
        # by ~30% on the per-record hot path :meth:`merge` drives.
        if self.created_at <= other.created_at:
            older, newer = self, other
        else:
            older, newer = other, self

        # Newer overwrite wins outright: the older record's Parquet
        # data is dropped from the SQL projection (parquet_paths stays
        # as the newer's set), but the older record's Parquet file +
        # metadata still need to be cleaned up after execute. Pack
        # them into ``_extra_cleanup_paths`` via the metadata_paths
        # list — they're listed in the cleanup walk anyway.
        if newer.is_overwrite:
            return newer._replace(
                # ``parquet_paths`` stays the newer's set — older's
                # Parquet is dropped from the SQL projection. We still
                # need to clean it up; record it on metadata_paths so
                # ``cleanup()`` removes it during the same sweep.
                metadata_paths=(
                    older.parquet_paths + older.metadata_paths
                    + newer.metadata_paths
                ),
                operation_ids=older.operation_ids + newer.operation_ids,
            )

        # All other cases (both append, or older overwrite + newer
        # append): combine parquets into one record. When the older is
        # an overwrite, the merged record stays an overwrite (the newer
        # append rows are pulled into the overwrite scope).
        merged_mode = older.mode if older.is_overwrite else newer.mode or older.mode

        return older._replace(
            parquet_paths=older.parquet_paths + newer.parquet_paths,
            metadata_paths=older.metadata_paths + newer.metadata_paths,
            operation_ids=older.operation_ids + newer.operation_ids,
            created_at=newer.created_at,
            mode=merged_mode,
            schema_mode=newer.schema_mode or older.schema_mode,
            overwrite_schema=(
                older.overwrite_schema
                if newer.overwrite_schema is None
                else newer.overwrite_schema
            ),
            match_by=newer.match_by or older.match_by,
            update_column_names=newer.update_column_names or older.update_column_names,
            zorder_by=newer.zorder_by or older.zorder_by,
            optimize_after_merge=older.optimize_after_merge or newer.optimize_after_merge,
            vacuum_hours=newer.vacuum_hours or older.vacuum_hours,
            where=newer.where or older.where,
            prune_by=newer.prune_by or older.prune_by,
            prune_values=newer.prune_values or older.prune_values,
            safe_merge=older.safe_merge or newer.safe_merge,
            target_field_names=newer.target_field_names or older.target_field_names,
        )

    @classmethod
    def merge(
        cls,
        source: Any,
        *,
        client: "DatabricksClient | None" = None,
    ) -> List["AsyncInsert"]:
        """Collapse multiple staged operations into one per target.

        ``source`` may be:

        - A folder-like :class:`VolumePath` (or path string) — every
          ``*.json`` entry under it is read and merged.
        - An iterable of metadata file paths (or path strings).
        - An iterable of already-loaded :class:`AsyncInsert` records.

        Returns one merged record per unique target. Within each
        target, an overwrite drops every earlier operation; the
        remaining records (appends and at most one trailing overwrite)
        fold into a single record via :meth:`merge_with`.
        """
        records = list(_iter_records(source, client=client))

        groups: dict[str, list[AsyncInsert]] = {}
        for record in records:
            groups.setdefault(record.target_full_name, []).append(record)

        merged: list[AsyncInsert] = []
        for target, recs in groups.items():
            recs.sort(key=lambda r: r.created_at)
            # The latest overwrite wipes everything before it.
            last_overwrite = max(
                (i for i, r in enumerate(recs) if r.is_overwrite),
                default=-1,
            )
            kept = recs[last_overwrite:] if last_overwrite >= 0 else recs

            head = kept[0]
            for r in kept[1:]:
                head = head.merge_with(r)

            # When earlier records were dropped by an overwrite, their
            # data is NOT part of the merged SQL projection — but their
            # staged files still have to be removed. Pile both the
            # dropped parquet and metadata paths onto ``metadata_paths``
            # so :meth:`cleanup` removes them during the same sweep.
            if last_overwrite > 0:
                dropped = recs[:last_overwrite]
                dropped_cleanup = (
                    tuple(p for r in dropped for p in r.parquet_paths)
                    + tuple(p for r in dropped for p in r.metadata_paths)
                )
                head = head._replace(
                    metadata_paths=dropped_cleanup + head.metadata_paths,
                    operation_ids=(
                        tuple(o for r in dropped for o in r.operation_ids)
                        + head.operation_ids
                    ),
                )
            merged.append(head)

        return merged


    # ------------------------------------------------------------------ #
    # SQL rendering
    # ------------------------------------------------------------------ #
    def _build_sql(self, parquet_refs: Sequence[str]) -> Optional[str]:
        r"""Common SQL shape used by both raw-text and prepared paths.

        Returns ``None`` when there is nothing to apply (no Parquet
        payloads or no target). Each entry in ``parquet_refs`` is a
        full table expression spliced into ``SELECT * FROM <ref>`` —
        either ``parquet.\`<full_path>\``` for raw SQL or a bare
        ``{alias}`` placeholder whose substituted ``text_value``
        already carries the full ``parquet.\`...\``` form. Wrapping
        is the caller's job so the prepared-statement path doesn't
        end up double-wrapping (the alias substitution would yield
        ``parquet.\`parquet.\`<path>\`\``` otherwise).
        """
        if not parquet_refs or not self.target_full_name:
            return None

        selects = [f"SELECT * FROM {ref}" for ref in parquet_refs]
        source = " UNION ALL ".join(selects)

        target = self.target_full_name
        if self.target_field_names:
            cols = ", ".join(f"`{c}`" for c in self.target_field_names)
            target = f"{target} ({cols})"

        prefix = (
            f"INSERT OVERWRITE {target}"
            if self.is_overwrite
            else f"INSERT INTO {target}"
        )
        where = f" WHERE {self.where}" if self.where else ""
        return f"{prefix} {source}{where}"

    def to_sql(self) -> List[str]:
        r"""Render the operation as one or more raw SQL strings.

        Emits a single statement: ``INSERT INTO`` (append) or
        ``INSERT OVERWRITE`` (overwrite) whose source is the staged
        Parquet payloads read via ``parquet.\`<path>\``. Multiple
        paths are unioned with ``UNION ALL``. Returns an empty list
        when no Parquet payloads are recorded.

        Useful for inspection / tests. The execution path
        (:meth:`to_statements`, :meth:`execute`, :meth:`concat`,
        :class:`AsyncWrite`) prefers prepared statements so the
        staged files are cleaned up automatically via the statement
        lifecycle.
        """
        refs = tuple(f"parquet.`{p}`" for p in self.parquet_paths)
        sql = self._build_sql(refs)
        return [sql] if sql is not None else []

    def to_statements(
        self,
        *,
        client: "DatabricksClient | None" = None,
        retry: Any = None,
        cleanup: bool = True,
    ) -> List[Any]:
        """Render the operation as a list of :class:`WarehousePreparedStatement`.

        Each Parquet payload + every contributing metadata file are
        attached to the statement as ``external_volume_paths`` (with
        ``temporary=True`` when ``cleanup=True``), so the statement
        lifecycle owns cleanup: on success
        :meth:`WarehousePreparedStatement.clear_temporary_resources`
        — auto-fired by :meth:`StatementResult.wait` and
        :meth:`StatementBatch.wait` — unlinks every attached
        temporary path. On failure the same hook runs from
        :meth:`StatementResult.raise_for_status`.

        Aliases follow ``__p0__ / __p1__ / …`` for the Parquet
        payloads referenced by SQL and ``__m0__ / __m1__ / …`` for
        the metadata files that are attached purely for cleanup
        (they never appear in the substituted text).

        Pass ``cleanup=False`` to attach the same paths without
        marking them temporary — useful when debugging an applier
        and the staged files should survive the run.
        """
        from yggdrasil.databricks.path import DatabricksPath
        from yggdrasil.databricks.warehouse.statement import (
            WarehousePreparedStatement,
        )

        parquet_aliases = [f"__p{i}__" for i in range(len(self.parquet_paths))]
        # Build SQL against ``{alias}`` placeholders. Empty payload
        # returns no statements.
        sql = self._build_sql(
            tuple(f"{{{alias}}}" for alias in parquet_aliases),
        )
        if sql is None:
            return []

        ext_paths: dict[str, Any] = {}
        for alias, full_path in zip(parquet_aliases, self.parquet_paths):
            path = DatabricksPath.from_(full_path, client=client)
            if cleanup:
                # Mutate the (singleton-cached) path so the statement
                # batch's ``clear_temporary_resources`` walks it on
                # success and unlinks the file. The path is about to
                # be deleted; no other code should be holding it.
                path.temporary = True
            ext_paths[alias] = path

        for i, full_path in enumerate(self.metadata_paths):
            # ``__m{i}__`` aliases don't appear in SQL — they ride
            # the statement purely so the lifecycle hook removes the
            # metadata files alongside the Parquet payloads.
            path = DatabricksPath.from_(full_path, client=client)
            if cleanup:
                path.temporary = True
            ext_paths[f"__m{i}__"] = path

        stmt = WarehousePreparedStatement.prepare(
            sql,
            client=client,
            external_volume_paths=ext_paths,
            catalog_name=self.target_catalog_name,
            schema_name=self.target_schema_name,
            retry=retry,
        )
        return [stmt]

    # ------------------------------------------------------------------ #
    # Concat — render a batch-execution suite across many records
    # ------------------------------------------------------------------ #
    @classmethod
    def concat(
        cls,
        source: Any,
        *,
        engine: "SQLEngine | None" = None,
        client: "DatabricksClient | None" = None,
        wait: Any = True,
        raise_error: bool = True,
        cleanup: bool = True,
    ) -> Any:
        """Concatenate staged operations into the SQL suite that applies them.

        Pipes everything :meth:`merge` accepts (a folder
        :class:`VolumePath`, an iterable of metadata file paths, an
        iterable of in-memory records, or a single
        :class:`AsyncInsert`) through the per-target merge.

        With *engine* supplied: every merged record is wrapped in
        one unified :class:`AsyncWrite` batch — the staged Parquet
        + metadata files attach to each prepared statement as
        ``external_volume_paths`` (marked ``temporary=True`` unless
        ``cleanup=False``). The batch's ``wait`` hook auto-fires
        ``clear_temporary_resources`` so every attached file is
        unlinked when the run lands successfully. Returns the
        resulting :class:`AsyncWrite`.

        Without *engine*: returns the bare list of SQL strings (via
        :meth:`to_sql`) for callers that want to inspect the
        rendered text or run it through a different path. No
        prepared statements are built and no cleanup is wired —
        reach for :class:`AsyncWrite` for the wire-up.

        Returns
        -------
        list[str]
            When *engine* is ``None``.
        AsyncWrite
            When *engine* is supplied. Empty input returns ``[]`` /
            ``None`` without touching the engine.
        """
        merged = cls.merge(source, client=client)
        if not merged:
            return [] if engine is None else None

        if engine is None:
            statements: list[str] = []
            for record in merged:
                statements.extend(record.to_sql())
            return statements

        return AsyncWrite.from_records(
            merged,
            executor=engine.warehouse(),
            client=client,
            cleanup=cleanup,
            wait=wait,
            raise_error=raise_error,
        )

    def __call__(
        self,
        engine: "SQLEngine | None" = None,
        *others: "AsyncInsert",
        client: "DatabricksClient | None" = None,
        wait: Any = True,
        raise_error: bool = True,
        cleanup: bool = True,
    ) -> Any:
        """Shorthand for :meth:`concat` keyed off this record.

        ``record(engine, *others)`` is equivalent to
        ``AsyncInsert.concat([record, *others], engine=engine, …)``,
        so a caller that just staged an insert can apply it (plus
        any peers) without re-typing the class name.
        """
        return type(self).concat(
            [self, *others],
            engine=engine,
            client=client,
            wait=wait,
            raise_error=raise_error,
            cleanup=cleanup,
        )

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def execute(
        self,
        engine: "SQLEngine",
        *,
        wait: Any = True,
        raise_error: bool = True,
        cleanup: bool = True,
        client: "DatabricksClient | None" = None,
    ) -> "AsyncInsert | None":
        """Submit this record's prepared statements through *engine*.

        Binds *engine*'s warehouse as :attr:`executor`, renders the
        record via :meth:`to_statements`, and extends self with the
        resulting prepared statements (which submits them). Every
        staged Parquet + metadata file rides as an
        ``external_volume_paths`` entry (marked temporary unless
        ``cleanup=False``); on success :class:`WarehouseStatementBatch`'s
        ``wait`` hook auto-fires ``clear_temporary_resources`` and
        unlinks every attached path — no explicit :meth:`cleanup` call
        is needed.

        Empty operations (no Parquet paths) return ``None`` without
        touching *engine*. Set ``cleanup=False`` to attach the files
        without marking them temporary (useful for debugging an
        applier when the staged files should survive the run).
        """
        if not self.parquet_paths or not self.target_full_name:
            return None

        statements = self.to_statements(client=client, cleanup=cleanup)
        if not statements:
            return None

        self.executor = engine.warehouse()
        self.extend(statements)
        self.wait(wait=wait, raise_error=raise_error)
        return self

    def cleanup(self, *, client: "DatabricksClient | None" = None) -> None:
        """Force-remove every staged Parquet + metadata file recorded on this op.

        Best-effort: missing files are tolerated, individual delete
        failures are logged and swallowed so a partial cleanup
        doesn't mask the (already-successful) execute.

        The normal apply path (:meth:`execute` / :meth:`concat` with
        an engine) no longer calls this — staged files ride the
        :class:`WarehousePreparedStatement` lifecycle and get
        unlinked automatically when the statement batch lands
        successfully. Keep this method for abandoned-record cleanup
        (e.g. an applier sweep that wants to drop stale metadata
        files without going through SQL).
        """
        from yggdrasil.databricks.path import DatabricksPath

        for full_path in tuple(self.parquet_paths) + tuple(self.metadata_paths):
            if not full_path:
                continue
            try:
                DatabricksPath.from_(full_path, client=client).remove(
                    missing_ok=True, wait=False, recursive=False,
                )
            except Exception:
                LOGGER.exception(
                    "Failed to clean up staged async-insert artifact %r; "
                    "continuing.",
                    full_path,
                )


# ---------------------------------------------------------------------------
# AsyncWrite — unified WarehouseStatementBatch over staged AsyncInsert records
# ---------------------------------------------------------------------------


class AsyncWrite:
    """Unified :class:`WarehouseStatementBatch` over staged async inserts.

    The single apply path for :class:`AsyncInsert` records. Submits
    one prepared statement per merged target as one batch, with the
    contributing Parquet payloads and metadata logs attached as
    ``temporary`` external volume paths so the batch's wait-hook
    unlinks them on success.

    Construct via :meth:`from_records` (in-memory records) or
    :meth:`from_source` (folder :class:`VolumePath` / iterable of
    metadata files / iterable of records — anything
    :meth:`AsyncInsert.merge` accepts). Each classmethod returns a
    submitted :class:`WarehouseStatementBatch` whose ``wait``-hook
    (auto-fired on success) cleans up the staged files.

    This is a thin factory wrapping :class:`WarehouseStatementBatch`
    — not a subclass — because :class:`WarehouseStatementBatch`
    submits statements eagerly in ``__init__``, and the records →
    statements rendering wants to run *before* construction so empty
    inputs short-circuit without touching the executor.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "WarehouseStatementBatch":
        """Constructing ``AsyncWrite(...)`` returns a submitted batch.

        Delegates to :meth:`from_records` when *records* / *source*
        is supplied; otherwise to :meth:`from_source`. Useful as
        ``AsyncWrite(records, executor=...)`` shorthand.
        """
        if "source" in kwargs:
            return cls.from_source(*args, **kwargs)
        return cls.from_records(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_records(
        cls,
        records: Iterable[AsyncInsert],
        *,
        executor: Any,
        client: "DatabricksClient | None" = None,
        cleanup: bool = True,
        retry: Any = None,
        parallel: int = 1,
        wait: Any = True,
        raise_error: bool = True,
    ) -> Any:
        """Build, submit, and (by default) wait on the batch for *records*.

        Each record is rendered via :meth:`AsyncInsert.to_statements`;
        the resulting prepared statements are submitted as a single
        :class:`WarehouseStatementBatch`. Empty inputs return
        ``None`` without touching the executor.

        ``wait=False`` returns the live batch without blocking; the
        caller owns ``batch.wait(...)``.
        """
        # Local import — keeps async_write importable when the
        # warehouse module isn't ready yet (circular at module load).
        from yggdrasil.databricks.warehouse.statement import (
            WarehouseStatementBatch,
        )

        statements: list[Any] = []
        record_count = 0
        for record in records:
            record_count += 1
            statements.extend(
                record.to_statements(
                    client=client,
                    cleanup=cleanup,
                    retry=retry,
                )
            )

        if not statements:
            return None

        LOGGER.info(
            "Applying %d-statement async-insert batch across %d target(s)",
            len(statements), record_count,
        )

        batch = WarehouseStatementBatch(
            executor=executor,
            statements=statements,
            parallel=parallel,
        )
        batch.wait(wait=wait, raise_error=raise_error)
        return batch

    @classmethod
    def from_source(
        cls,
        source: Any,
        *,
        engine: "SQLEngine | None" = None,
        executor: Any = None,
        client: "DatabricksClient | None" = None,
        cleanup: bool = True,
        retry: Any = None,
        parallel: int = 1,
        wait: Any = True,
        raise_error: bool = True,
    ) -> Any:
        """Merge *source* through :meth:`AsyncInsert.merge`, then apply.

        ``source`` accepts the same shapes :meth:`AsyncInsert.merge`
        does (a folder :class:`VolumePath`, an iterable of metadata
        file paths, an iterable of in-memory records, or a single
        record). The merged records are submitted via
        :meth:`from_records`.

        One of *engine* or *executor* is required — pass *engine*
        to derive the warehouse via :meth:`SQLEngine.warehouse`, or
        pass an explicit *executor* (a :class:`SQLWarehouse`).
        """
        if executor is None:
            if engine is None:
                raise ValueError(
                    "AsyncWrite.from_source needs either ``engine`` or "
                    "``executor`` — got neither."
                )
            executor = engine.warehouse()

        merged = AsyncInsert.merge(source, client=client)
        if not merged:
            return None

        return cls.from_records(
            merged,
            executor=executor,
            client=client,
            cleanup=cleanup,
            retry=retry,
            parallel=parallel,
            wait=wait,
            raise_error=raise_error,
        )


# ---------------------------------------------------------------------------
# Field-name caches (bake field set at import time so :meth:`from_dict`
# doesn't re-walk ``_METADATA_FIELDS`` on every call).
# ---------------------------------------------------------------------------


_FIELD_NAMES: frozenset[str] = frozenset(name for name, _ in _METADATA_FIELDS)
_TUPLE_FIELD_NAMES: frozenset[str] = frozenset({
    "parquet_paths",
    "metadata_paths",
    "operation_ids",
    "target_field_names",
    "match_by",
    "update_column_names",
    "zorder_by",
    "prune_by",
})


# ---------------------------------------------------------------------------
# Iter records helper
# ---------------------------------------------------------------------------


def _iter_records(
    source: Any,
    *,
    client: "DatabricksClient | None" = None,
) -> Iterable[AsyncInsert]:
    """Yield :class:`AsyncInsert` records from a folder / iterable / records.

    Centralised so :meth:`AsyncInsert.merge` accepts the same shapes
    callers naturally have on hand (a folder path, a list of metadata
    files, a list of already-loaded records).

    Folder-like sources are walked for ``*.json`` entries; a
    ``logs/`` subdirectory is descended into automatically so
    callers can hand in either the staging root
    (``.sql/async/insert/``) or the explicit logs folder
    (``.sql/async/insert/logs/``) and get the same records back.
    """
    if isinstance(source, AsyncInsert):
        yield source
        return

    # Folder-like: walk for ``*.json`` files. The staging root only ever
    # contains the ``data/`` + ``logs/`` siblings (see
    # :func:`stage_async_insert`), and every metadata file lives under
    # ``logs/`` — so when the caller hands the root, skip the parent
    # listing entirely and descend directly into ``logs/``. Saves one
    # remote ``ls`` round trip per merged target (the volume listing in
    # the log shows ~700ms each on a live workspace).
    if hasattr(source, "ls"):
        source_name = getattr(source, "name", "") or ""
        if (
            source_name == "insert"
            and hasattr(source, "joinpath")
        ):
            try:
                logs_folder = source.joinpath(ASYNC_INSERT_LOGS_SUBDIR)
                entries = logs_folder.ls(recursive=False)
            except FileNotFoundError:
                return
            for entry in entries:
                if (getattr(entry, "name", "") or "").endswith(".json"):
                    yield AsyncInsert.from_file(entry, client=client)
            return

        for entry in source.ls(recursive=False):
            name = getattr(entry, "name", "") or ""
            if name == ASYNC_INSERT_LOGS_SUBDIR and hasattr(entry, "ls"):
                yield from _iter_records(entry, client=client)
                continue
            if name.endswith(".json"):
                yield AsyncInsert.from_file(entry, client=client)
        return

    # String / Path: treat as a folder if it has no extension, otherwise
    # a single file.
    if isinstance(source, str):
        if source.endswith(".json"):
            yield AsyncInsert.from_file(source, client=client)
            return
        # Fall through to DatabricksPath.from_ which dispatches to the
        # right subclass; then walk it as a folder.
        from yggdrasil.databricks.path import DatabricksPath

        path = DatabricksPath.from_(source, client=client)
        yield from _iter_records(path, client=client)
        return

    # Iterable: dispatch each item recursively.
    try:
        items = iter(source)
    except TypeError as exc:
        raise TypeError(
            f"AsyncInsert.merge cannot iterate {source!r} "
            f"(expected folder VolumePath, path string, or iterable of "
            f"metadata files / AsyncInsert records)."
        ) from exc

    for item in items:
        yield from _iter_records(item, client=client)


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
            from yggdrasil.execution.expr.backends.sql import (
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
    # ``to_json_bytes`` / pickle still emit URL strings (see ``to_dict``
    # / ``__reduce__``), so the staged JSON metadata format is unchanged.
    record = AsyncInsert(
        target_full_name=table.full_name(safe=True),
        parquet_paths=(parquet_path,),
        metadata_paths=(meta_path,),
        operation_ids=(op_id,),
        created_at=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        target_catalog_name=table.catalog_name,
        target_schema_name=table.schema_name,
        target_table_name=table.table_name,
        target_field_names=(
            tuple(existing_schema.field_names())
            if existing_schema is not None else None
        ),
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


def _path_for_sql(path: Any) -> str:
    r"""Return the path string used inside SQL / metadata.

    Prefers the Unity-style ``/Volumes/...`` shape from
    :meth:`VolumePath.full_path` when available — that's what
    ``parquet.\`<path>\`` expects in Databricks SQL.
    """
    full_path = getattr(path, "full_path", None)
    if callable(full_path):
        try:
            return full_path()
        except Exception:
            pass
    url = getattr(path, "url", None)
    if url is not None:
        return str(url)
    return str(path)
