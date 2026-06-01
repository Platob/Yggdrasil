"""Databricks table insert — the one place insert DML is generated.

This module centralizes *everything* about loading rows into a Unity Catalog
table, for both the synchronous warehouse / Spark / SQL paths and the async,
file-arrival drop pipeline:

- :class:`DatabricksTableInsert` is the full description of one insert
  operation — target, mode, the staged source, plus the keyed-write surface
  (``match_by``, ``update_column_names``, ``predicate``, ``zorder_by``,
  maintenance flags, ``safe_merge``). It is also the typed content of an
  async op-log: :meth:`~DatabricksTableInsert.from_log` parses a dropped log
  and :meth:`~DatabricksTableInsert.to_json` serializes one.
- :class:`DatabricksInsertBatch` aggregates a target's ops into one load. For
  ``OVERWRITE`` an op supersedes everything staged before it; for a keyed
  ``MERGE`` / ``UPSERT`` the unioned source is deduplicated by the match keys
  so Delta's "multiple source rows matched a target row" error can't fire.
- :func:`make_sql_select` renders the per-op ``SELECT`` over its staged source.
- :func:`make_sql_insert` renders the full INSERT / MERGE / DELETE+INSERT /
  TRUNCATE / OPTIMIZE / VACUUM statement list — it accepts either a single op
  (the atomic sync path) or a batch (the async aggregated load).

The async drop pipeline on top of this:

- A table's ``insert(..., wait=False)`` calls :func:`stage_async_insert`: the
  staged Parquet is written to the table's default tmp staging path and a
  small JSON op-log is dropped at ``.sql/async/logs/`` recording **where the
  data was written** (its uniform URL) — no warehouse statement runs.
- A **file-arrival trigger** on ``logs/`` wakes a serverless job
  (:func:`ensure_async_job`) whose entry point runs
  ``ygg databricks table execute_async_insert --logs <dir>`` → :func:`load_async`:
  read every pending log, group by **target table** into a
  :class:`DatabricksInsertBatch`, run one aggregated load per target, then
  clear the consumed logs + data.

Reach :func:`ensure_async_job` for a table via
:meth:`yggdrasil.databricks.table.table.Table.async_job`.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Optional

from yggdrasil.data import Field
from yggdrasil.databricks.sql.sql_utils import quote_ident
from yggdrasil.enums.mode import Mode
from yggdrasil.execution.expr.backends.sql import Dialect, to_sql as expr_to_sql
from yggdrasil.path import Path

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.fs.volume_path import VolumePath
    from yggdrasil.databricks.table.table import Table
    from yggdrasil.execution.expr import Predicate

__all__ = [
    "ASYNC_ROOT",
    "LOGS_SUBDIR",
    "ASYNC_MODES",
    "BUFFER_SECONDS",
    "ALIAS_TMPSRC",
    "DatabricksTableInsert",
    "DatabricksInsertBatch",
    "make_sql_select",
    "make_sql_insert",
    "stage_async_insert",
    "load_async",
    "dispatch_async",
    "job_name",
    "logs_path",
    "ensure_async_job",
]

logger = logging.getLogger(__name__)

#: Root (under a table's staging volume) for the async drop pipeline. Only the
#: ``logs/`` directory is fixed — the file-arrival trigger watches it. The
#: staged Parquet lives wherever the producer wrote it (the table's default tmp
#: staging path); each operation log records that location.
ASYNC_ROOT = ".sql/async"
LOGS_SUBDIR = f"{ASYNC_ROOT}/logs"

#: Modes the async *drop* path accepts — keyed merges have no aggregation story
#: at drop time. (The synchronous path supports every mode.)
ASYNC_MODES = (Mode.OVERWRITE, Mode.APPEND)

#: File-arrival buffering window (seconds): wait this long after the last
#: dropped log before firing, and fire at most this often — so a burst of
#: ``async_insert`` drops batches into one aggregated load. Default 2 min.
BUFFER_SECONDS = 120

#: Job-name prefix so the file-arrival loader jobs are easy to spot.
_NAME_PREFIX = "[YGG][ASYNC]"

#: Placeholder a staged Parquet source is referenced by on the warehouse path —
#: substituted for the concrete external-data ``VolumePath`` at prepare time.
ALIAS_TMPSRC = "__tmpsrc__"


def _new_op_id() -> str:
    return f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


@dataclass
class DatabricksTableInsert:
    """One insert operation — the full ``arrow_insert`` surface in one object.

    Carries the ``target`` table, the save ``mode``, the staged ``data``
    location (a uniform URL on disk, a :class:`Path` once read back), and the
    keyed-write surface (``schema``, ``predicate``, ``match_by``,
    ``update_column_names``, ``schema_mode``, ``zorder_by``,
    ``optimize_after_merge``, ``vacuum_hours``, ``safe_merge``).

    It is also the typed content of an async op-log: :meth:`from_log` parses a
    dropped log and :meth:`to_json` serializes one. Only the JSON-friendly
    fields round-trip through the log — runtime-only carriers (``client`` and
    the live ``predicate`` tree) stay local to the
    producing process; ``predicate`` is also recorded as its Databricks SQL
    string for human inspection but isn't reconstructed (there's no SQL→AST
    parser, and the async drop path never carries one anyway).

    ``target`` may be a :class:`Table` or its full name; ``data`` may be a
    :class:`Path` or a uniform-URL string — both are normalized lazily through
    the bound :attr:`client` when a concrete object is needed.
    """

    target: "Table | str"
    mode: Mode
    data: "Path | str"
    client: "DatabricksClient | None" = None
    op_id: str = field(default_factory=_new_op_id)
    ts: float = field(default_factory=time.time)
    #: the on-disk op-log (set when read back) — cleaned up after a load.
    log_file: "Path | None" = None

    #: the data's schema (the same :class:`Field` / :class:`Schema` type
    #: ``Table.collect_schema()`` returns); drives the SELECT projection.
    schema: "Field | None" = None
    #: target-row prune / source-row filter, folded in from ``cast_options``.
    predicate: "Predicate | None" = None
    match_by: "list[str] | None" = None
    update_column_names: "list[str] | None" = None
    schema_mode: "Mode | str | None" = None
    zorder_by: "list[str] | None" = None
    optimize_after_merge: bool = False
    vacuum_hours: "int | None" = None
    safe_merge: bool = False

    def __post_init__(self) -> None:
        self.mode = Mode.from_(self.mode, default=Mode.APPEND)
        # The drop pipeline only knows how to aggregate OVERWRITE / APPEND;
        # the keyed modes have no UNION-ALL story. The synchronous path
        # (``Table.arrow_insert``) builds a richer op directly — carrying a
        # ``schema`` / ``match_by`` — and supports every mode, so the
        # OVERWRITE-only guard only fires for the bare op shape that is the
        # content of an async op-log.
        if (
            self.mode not in ASYNC_MODES
            and self.schema is None
            and not self.match_by
        ):
            raise ValueError(
                f"async insert supports only OVERWRITE / APPEND, got {self.mode.name}"
            )

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_log(cls, log_file: Any, *, client: Any = None) -> "DatabricksTableInsert":
        """Parse an op-log file into an op (keeps *log_file* for cleanup)."""
        mapping = json.loads(bytes(log_file.read_bytes()))
        op = cls.from_json(mapping, client=client)
        op.log_file = log_file
        return op

    @classmethod
    def from_json(cls, mapping: dict, *, client: Any = None) -> "DatabricksTableInsert":
        """Rebuild an op from its serialized op-log payload."""
        raw_schema = mapping.get("schema")
        return cls(
            target=mapping["target"],
            mode=Mode.from_(mapping["mode"]),
            data=mapping["data"],
            client=client,
            op_id=mapping.get("op_id") or _new_op_id(),
            ts=mapping.get("ts") or time.time(),
            schema=Field.from_json(raw_schema) if raw_schema else None,
            match_by=mapping.get("match_by"),
            update_column_names=mapping.get("update_column_names"),
            zorder_by=mapping.get("zorder_by"),
            optimize_after_merge=bool(mapping.get("optimize_after_merge", False)),
            vacuum_hours=mapping.get("vacuum_hours"),
            safe_merge=bool(mapping.get("safe_merge", False)),
        )

    def to_json(self) -> bytes:
        """Serialize to the JSON op-log payload.

        ``target`` / ``mode`` / ``data`` keep the historical shape the loader
        reads back; the keyed-write fields are persisted when set so a richer
        async op can round-trip. ``predicate`` is recorded as its Databricks
        SQL string for inspection only (one-way — see the class docstring).
        """
        payload: dict[str, Any] = {
            "op_id": self.op_id,
            "target": self.target_name,
            "mode": self.mode.name.lower(),
            "data": self.data_url,
            "ts": self.ts,
        }
        if self.schema is not None:
            payload["schema"] = json.loads(self.schema.to_json())
        if self.predicate is not None:
            payload["predicate"] = expr_to_sql(self.predicate, dialect=Dialect.DATABRICKS)
        if self.match_by:
            payload["match_by"] = list(self.match_by)
        if self.update_column_names:
            payload["update_column_names"] = list(self.update_column_names)
        if self.zorder_by:
            payload["zorder_by"] = list(self.zorder_by)
        if self.optimize_after_merge:
            payload["optimize_after_merge"] = True
        if self.vacuum_hours is not None:
            payload["vacuum_hours"] = int(self.vacuum_hours)
        if self.safe_merge:
            payload["safe_merge"] = True
        return json.dumps(payload).encode()

    # -- normalized views -----------------------------------------------------

    @property
    def target_name(self) -> str:
        """``catalog.schema.table`` — the grouping key and log ``target``."""
        full_name = getattr(self.target, "full_name", None)
        return full_name() if callable(full_name) else str(self.target)

    @property
    def data_url(self) -> str:
        """The staged data's uniform URL (round-trippable, location-agnostic)."""
        to_url = getattr(self.data, "to_url", None)
        return to_url().to_string() if callable(to_url) else str(self.data)

    @property
    def group_key(self) -> str:
        """The target a batch aggregates over — one load per target."""
        return self.target_name

    def data_path(self, client: Any = None) -> "Path":
        """Resolve the staged data to a concrete :class:`Path`.

        Already a :class:`Path` → returned as-is; a uniform-URL string →
        reconstructed through the bound (or supplied) client so the loader can
        read / clean it up wherever it landed.
        """
        if isinstance(self.data, Path):
            return self.data
        from yggdrasil.databricks.path import DatabricksPath

        return DatabricksPath.from_(self.data, client=client or self.client)

    def select_sql(self, client: Any = None) -> str:
        """Back-compat alias for :func:`make_sql_select` over this op."""
        return make_sql_select(self, client=client)


@dataclass
class DatabricksInsertBatch:
    """All ops staged for one ``target`` — aggregated into a single load.

    Groups a target's :class:`DatabricksTableInsert` ops and centralizes the
    statement generation via :func:`make_sql_insert`. :attr:`mode` is
    ``OVERWRITE`` when any retained op overwrites (an OVERWRITE supersedes
    everything staged before it), else the first active op's mode. Every
    consumed op is retained on :attr:`logs` so the loader can clean up
    superseded data too.
    """

    target: "Table"
    logs: "list[DatabricksTableInsert]" = field(default_factory=list)

    def append(self, op: "DatabricksTableInsert") -> "DatabricksInsertBatch":
        self.logs.append(op)
        return self

    @property
    def mode(self) -> Mode:
        """``OVERWRITE`` if any op overwrites, else the first active op's mode."""
        if any(op.mode is Mode.OVERWRITE for op in self.logs):
            return Mode.OVERWRITE
        active = self.active
        return active[0].mode if active else Mode.APPEND

    @property
    def active(self) -> "list[DatabricksTableInsert]":
        """The ops that actually feed the load, in timestamp order.

        Everything from the latest OVERWRITE onward — earlier ops are
        superseded by it (but still tracked on :attr:`logs` for cleanup).
        """
        ordered = sorted(self.logs, key=lambda op: op.ts)
        cut = 0
        for i, op in enumerate(ordered):
            if op.mode is Mode.OVERWRITE:
                cut = i
        return ordered[cut:]

    @property
    def match_by(self) -> "list[str] | None":
        """The keyed-write columns — taken from the first active op."""
        active = self.active
        return active[0].match_by if active else None

    def make_sql(self, client: Any = None) -> str:
        """The aggregated source body: one ``SELECT`` per active op, UNION'd.

        For a keyed batch the union is deduplicated by the match keys (see
        :func:`_batch_source_sql` for the dedup CTE).
        """
        return _batch_source_sql(self, client=client)

    @classmethod
    def group(
        cls, ops: "Iterable[DatabricksTableInsert]",
    ) -> "list[DatabricksInsertBatch]":
        """Group *ops* by target into one batch each."""
        grouped: "dict[str, DatabricksInsertBatch]" = {}
        for op in ops:
            grouped.setdefault(
                op.group_key, cls(target=op.target),
            ).append(op)
        return list(grouped.values())


# ===========================================================================
# DML generation — the single source of truth for every insert path.
#
# ``make_sql_insert`` dispatches on atomic op vs batch and reproduces the
# arrow / spark / sql sync-path DML exactly:
#
#   - ``overwrite`` (no keys)  → plain ``INSERT OVERWRITE``
#   - ``truncate`` (no keys)   → ``TRUNCATE TABLE`` + ``INSERT``
#   - ``truncate``/``overwrite`` with keys → keyed ``DELETE`` + ``INSERT``
#   - keyed, ``safe_merge=False`` → single ``MERGE INTO`` (insert-only for
#     APPEND/AUTO, full update+insert for UPSERT/MERGE)
#   - keyed UPSERT/MERGE, ``safe_merge=True`` → keyed ``DELETE`` + ``INSERT``
#   - keyed APPEND/AUTO, ``safe_merge=True`` → ``INSERT … WHERE NOT EXISTS``
#   - no keys                  → plain ``INSERT INTO``
#   - maintenance tail         → ``OPTIMIZE [ZORDER]`` / ``VACUUM``
#
# Merge ``ON`` is built null-safe (``<=>``) so NULL matches NULL.
# ===========================================================================


def make_sql_select(
    op: "DatabricksTableInsert",
    *,
    client: Any = None,
    source: "str | None" = None,
) -> str:
    """The atomic per-op ``SELECT`` over the op's staged source.

    Two source shapes, both supported because the sync and async paths
    reference the staged data differently:

    * **default** — render ``SELECT * FROM parquet.`<warehouse path>``` over
      the op's staged Parquet (resolved from its uniform URL). This is the
      async loader's per-file SELECT.
    * **explicit *source*** — when the caller already has a source reference
      (the sync ``arrow_insert`` passes the ``{__tmpsrc__}`` placeholder, which
      is substituted for the external-data ``VolumePath`` at prepare time),
      project the op's schema columns from it: ``SELECT <projection> FROM
      <source>``.
    """
    if source is not None:
        if op.schema is not None and op.schema.fields:
            projection = _build_column_projection(op.schema.fields)
        else:
            projection = "*"
        return f"SELECT {projection} FROM {source}"

    path = op.data_path(client)
    full_path = getattr(path, "full_path", None)
    ref = full_path() if callable(full_path) else str(op.data)
    return f"SELECT * FROM parquet.`{ref}`"


def make_sql_insert(
    target: "DatabricksTableInsert | DatabricksInsertBatch",
    *,
    target_location: "str | None" = None,
    source_sql: "str | None" = None,
    columns: "list[str] | None" = None,
    client: Any = None,
) -> list[str]:
    """Render the full statement list for one insert — atomic op or batch.

    Dispatches on the argument: a :class:`DatabricksTableInsert` yields the
    atomic INSERT / MERGE / … list; a :class:`DatabricksInsertBatch` yields the
    aggregated load over the unioned (and, when keyed, deduplicated) source.

    ``target_location`` / ``source_sql`` / ``columns`` let the synchronous
    paths supply their own source reference (the ``{__tmpsrc__}`` placeholder,
    a Spark temp-view name, or a wrapped user query) and pre-resolved target
    location; when omitted they're derived from the op/batch's ``target`` and
    staged data.
    """
    if isinstance(target, DatabricksInsertBatch):
        return _make_batch_insert(
            target, target_location=target_location, client=client,
        )
    return _make_op_insert(
        target,
        target_location=target_location,
        source_sql=source_sql,
        columns=columns,
        client=client,
    )


def _resolve_target_location(target: Any, fallback: "str | None") -> str:
    if fallback is not None:
        return fallback
    full_name = getattr(target, "full_name", None)
    if callable(full_name):
        return full_name(safe=True)
    return str(target)


def _make_op_insert(
    op: "DatabricksTableInsert",
    *,
    target_location: "str | None",
    source_sql: "str | None",
    columns: "list[str] | None",
    client: Any,
) -> list[str]:
    location = _resolve_target_location(op.target, target_location)
    if columns is None:
        columns = (
            [f.name for f in op.schema.fields]
            if op.schema is not None and op.schema.fields
            else []
        )
    if source_sql is None:
        source_sql = make_sql_select(op, client=client)
    prune_predicates = _build_where_predicates(op.predicate, target_alias="T")
    return _build_dml_statements(
        target_location=location,
        source_sql=source_sql,
        columns=columns,
        mode=op.mode,
        match_by=op.match_by,
        update_column_names=op.update_column_names,
        prune_predicates=prune_predicates,
        zorder_by=op.zorder_by,
        optimize_after_merge=op.optimize_after_merge,
        vacuum_hours=op.vacuum_hours,
        safe_merge=op.safe_merge,
    )


def _batch_source_sql(batch: "DatabricksInsertBatch", *, client: Any) -> str:
    """The unioned source body for a batch, deduplicated when keyed.

    A plain ``SELECT … UNION ALL …`` over each active op. When the batch runs
    a keyed ``MERGE`` / ``UPSERT``, the staged files can carry duplicate keys
    across drops, and Delta's MERGE errors when several source rows match the
    same target row — so the union is wrapped in a ``ROW_NUMBER() … = 1`` CTE
    that keeps the **last** row per key (incoming-wins, matching the
    DELETE+INSERT semantics). ``OVERWRITE`` / ``APPEND`` need no dedup.
    """
    union = " UNION ALL ".join(
        op.select_sql(client=client) for op in batch.active
    )
    match_by = batch.match_by
    if not match_by or batch.mode not in (Mode.UPSERT, Mode.MERGE):
        return union

    partition = ", ".join(quote_ident(k) for k in match_by)
    # The staged files have no natural ordering column to break ties on, so
    # ROW_NUMBER orders by the partition keys themselves — within a key every
    # row is interchangeable except that the window keeps exactly one,
    # collapsing the cross-file duplicates so the downstream MERGE sees at most
    # one source row per target key.
    return (
        f"SELECT * EXCEPT (__ygg_rn__) FROM (\n"
        f"  SELECT *, ROW_NUMBER() OVER (\n"
        f"    PARTITION BY {partition} ORDER BY {partition}\n"
        f"  ) AS __ygg_rn__ FROM (\n{union}\n  ) AS __ygg_union__\n"
        f") AS __ygg_dedup__\nWHERE __ygg_rn__ = 1"
    )


def _make_batch_insert(
    batch: "DatabricksInsertBatch",
    *,
    target_location: "str | None",
    client: Any,
) -> list[str]:
    location = _resolve_target_location(batch.target, target_location)
    source_sql = _batch_source_sql(batch, client=client)
    head = batch.active[0] if batch.active else None
    columns = (
        [f.name for f in head.schema.fields]
        if head is not None and head.schema is not None and head.schema.fields
        else []
    )
    prune_predicates = _build_where_predicates(
        head.predicate if head is not None else None, target_alias="T",
    )
    return _build_dml_statements(
        target_location=location,
        source_sql=source_sql,
        columns=columns,
        mode=batch.mode,
        match_by=batch.match_by,
        update_column_names=head.update_column_names if head is not None else None,
        prune_predicates=prune_predicates,
        zorder_by=head.zorder_by if head is not None else None,
        optimize_after_merge=bool(head.optimize_after_merge) if head is not None else False,
        vacuum_hours=head.vacuum_hours if head is not None else None,
        safe_merge=bool(head.safe_merge) if head is not None else False,
    )


# ---------------------------------------------------------------------------
# DML builders — moved here verbatim from table.py so every insert path shares
# one generator. The synchronous paths import these back through table.py.
# ---------------------------------------------------------------------------


def _build_match_condition(
    match_by: list[str],
    *,
    left_alias: str,
    right_alias: str,
    null_safe: bool = True,
    extra_predicates: Optional[Iterable[str]] = None,
) -> str:
    """Build a merge ``ON`` expression from key columns and optional extras."""
    op = "<=>" if null_safe else "="
    clauses = [
        f"{left_alias}.{quote_ident(k)} {op} {right_alias}.{quote_ident(k)}"
        for k in match_by
    ]
    if extra_predicates:
        clauses.extend(p for p in extra_predicates if p)
    return " AND ".join(clauses)


def _build_where_predicates(
    where: "Predicate | None",
    *,
    target_alias: str,
) -> list[str]:
    """Render *where* as a target-aliased SQL clause for MERGE / DELETE.

    Returns a ``list[str]`` (0 or 1 elements) so the downstream DML
    builders can splice it into an AND chain.
    """
    if where is None:
        return []
    aliased = _alias_columns(where, target_alias)
    sql = expr_to_sql(aliased, dialect=Dialect.DATABRICKS)
    return [sql]


def _alias_columns(expr, alias: str):
    """Walk *expr* and stamp every :class:`Column` with ``alias``.

    Lifts a user predicate onto the target-side of a MERGE so ``foo`` becomes
    ``T.foo``. Returns a new tree — the AST is immutable so we never mutate
    the caller's predicate.
    """
    from yggdrasil.execution.expr.nodes import (
        Arithmetic,
        Between,
        Cast,
        Column,
        Comparison,
        InList,
        IsNull,
        Like,
        Logical,
        Not,
    )

    if isinstance(expr, Column):
        return type(expr)(
            name=expr.name,
            field=expr.field,
            alias=expr.alias,
            qualifier=alias,
        )
    if isinstance(expr, Comparison):
        return Comparison(
            _alias_columns(expr.left, alias),
            expr.op,
            _alias_columns(expr.right, alias),
        )
    if isinstance(expr, Logical):
        return Logical(
            expr.op,
            tuple(_alias_columns(o, alias) for o in expr.operands),
        )
    if isinstance(expr, Not):
        return Not(_alias_columns(expr.operand, alias))
    if isinstance(expr, Between):
        return Between(
            _alias_columns(expr.target, alias),
            _alias_columns(expr.low, alias),
            _alias_columns(expr.high, alias),
            negated=expr.negated,
        )
    if isinstance(expr, InList):
        return InList(
            target=_alias_columns(expr.target, alias),
            values=expr.values,
            negated=expr.negated,
            includes_null=expr.includes_null,
        )
    if isinstance(expr, IsNull):
        return IsNull(_alias_columns(expr.target, alias), negated=expr.negated)
    if isinstance(expr, Like):
        return Like(
            target=_alias_columns(expr.target, alias),
            pattern=expr.pattern,
            case_insensitive=expr.case_insensitive,
            negated=expr.negated,
        )
    if isinstance(expr, Cast):
        return Cast(_alias_columns(expr.operand, alias), expr.dtype)
    if isinstance(expr, Arithmetic):
        return Arithmetic(
            expr.op,
            _alias_columns(expr.left, alias),
            _alias_columns(expr.right, alias),
        )
    return expr


def _build_column_projection(
    fields: "Iterable[Field]",
    *,
    source_alias: "str | None" = None,
) -> str:
    """Build a plain column-reference projection list for INSERT/MERGE.

    Each :class:`Field` contributes one bare (or alias-qualified) quoted
    column reference — no per-column ``CAST(... AS <ddl>)`` wrapper. The data
    has already been aligned to the target schema upstream (the arrow cast
    pipeline / Spark dataframe coercion / the warehouse's implicit
    column-boundary cast), so the engine accepts the rows as-is. Skipping the
    explicit CAST keeps the SQL short — important for wide / deeply nested
    schemas where the spelled-out DDL bloated statements past the warehouse
    text limits.
    """
    parts: list[str] = []
    alias = quote_ident(source_alias) if source_alias else None
    for f in fields:
        col = quote_ident(f.name)
        parts.append(f"{alias}.{col}" if alias else col)
    return ", ".join(parts)


def _build_cast_column_projection(
    target_fields: "Iterable[Field]",
    *,
    source: "Field | None" = None,
    source_alias: str,
) -> str:
    """Build a SELECT projection that CASTs source columns to target Spark types.

    For each target field:

    * **present in source, same type** — bare ``alias.`col``` (no CAST)
    * **present in source, different type** —
      ``CAST(alias.`col` AS <spark_type>)``
    * **missing from source** — ``CAST(NULL AS <spark_type>) AS `col```

    *source* is the :class:`Field` describing the source schema. Child lookup
    uses :meth:`Field.get` — no intermediate dict. When *source* is ``None``
    every target column is assumed present with an unknown type (always CAST).
    """
    alias = quote_ident(source_alias)
    parts: list[str] = []
    for f in target_fields:
        col = quote_ident(f.name)
        target_spark = f.to_spark_name(
            with_name=False, with_nullable=False, with_comment=False,
        )
        src = source.get(f.name) if source is not None else ...

        if src is None:
            parts.append(f"CAST(NULL AS {target_spark}) AS {col}")
        elif src is ...:
            parts.append(f"CAST({alias}.{col} AS {target_spark})")
        else:
            source_spark = src.to_spark_name(
                with_name=False, with_nullable=False, with_comment=False,
            )
            if source_spark == target_spark:
                parts.append(f"{alias}.{col}")
            else:
                parts.append(f"CAST({alias}.{col} AS {target_spark})")
    return ", ".join(parts)


def _build_delete_insert_statements(
    *,
    target_location: str,
    source_sql: str,
    columns: list[str],
    match_by: list[str],
    prune_predicates: list[str],
) -> list[str]:
    """Build the keyed-DELETE + INSERT pair used by upsert and the merge fallback.

    Always emits exactly two statements: a key-scoped ``DELETE`` against
    target rows whose match keys appear in ``source_sql``, followed by a plain
    ``INSERT INTO ... SELECT``. Databricks/Spark SQL doesn't accept ``DELETE
    FROM target USING source``; the keyed delete is rendered as ``DELETE FROM
    target T WHERE EXISTS (...)`` so it parses on Delta. ``prune_predicates``
    lift onto the outer ``WHERE`` so the target scan is bounded before the
    EXISTS subquery runs.
    """
    cols_quoted = ", ".join(quote_ident(c) for c in columns)
    key_cols = ", ".join(quote_ident(k) for k in match_by)
    key_match = " AND ".join(
        f"T.{quote_ident(k)} <=> S.{quote_ident(k)}"
        for k in match_by
    )
    exists_clause = (
        f"EXISTS (\n"
        f"  SELECT 1 FROM (\n"
        f"    SELECT DISTINCT {key_cols} FROM ({source_sql}) AS src\n"
        f"  ) AS S\n"
        f"  WHERE {key_match}\n"
        f")"
    )
    where_clauses = [*prune_predicates, exists_clause]
    where_sql = "\n  AND ".join(where_clauses)
    return [
        f"DELETE FROM {target_location} AS T\nWHERE {where_sql}",
        f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}",
    ]


def _build_merge_statement(
    *,
    target_location: str,
    source_sql: str,
    columns: list[str],
    match_by: list[str],
    update_column_names: Optional[list[str]],
    prune_predicates: list[str],
    insert_only: bool = False,
) -> str:
    """Render a single ``MERGE INTO ...`` statement.

    Used by :func:`_build_dml_statements` when ``safe_merge=False`` (the
    default) — Databricks / Delta supports MERGE natively and plans the keyed
    dedup once instead of twice (one delete + one insert) the way the safe path
    does.

    ``insert_only=True`` emits a MERGE with only the ``WHEN NOT MATCHED THEN
    INSERT`` clause — the keyed-APPEND shape. Without it, the full
    update-and-insert MERGE runs.
    """
    cols_quoted = ", ".join(quote_ident(c) for c in columns)
    on_condition = _build_match_condition(
        match_by, left_alias="T", right_alias="S",
        null_safe=True, extra_predicates=prune_predicates,
    )
    insert_clause = (
        f"WHEN NOT MATCHED THEN INSERT ({cols_quoted}) "
        f"VALUES ({', '.join(f'S.{quote_ident(c)}' for c in columns)})"
    )
    if insert_only:
        return (
            f"MERGE INTO {target_location} AS T\n"
            f"USING (\n{source_sql}\n) AS S\n"
            f"ON {on_condition}\n"
            f"{insert_clause}"
        )

    update_column_names_effective = (
        update_column_names
        if update_column_names is not None
        else [c for c in columns if c not in match_by]
    )
    update_clause = ""
    if update_column_names_effective:
        update_set = ", ".join(
            f"T.{quote_ident(c)} = S.{quote_ident(c)}"
            for c in update_column_names_effective
        )
        update_clause = f"WHEN MATCHED THEN UPDATE SET {update_set}\n"

    return (
        f"MERGE INTO {target_location} AS T\n"
        f"USING (\n{source_sql}\n) AS S\n"
        f"ON {on_condition}\n"
        f"{update_clause}"
        f"{insert_clause}"
    )


def _build_anti_join_insert(
    *,
    target_location: str,
    source_sql: str,
    columns: list[str],
    match_by: list[str],
    prune_predicates: list[str],
) -> list[str]:
    """Build a keyed APPEND that filters out rows already in the target.

    Renders one statement of the shape::

        INSERT INTO target (cols)
        SELECT cols FROM (source) AS S
        WHERE NOT EXISTS (
          SELECT 1 FROM target AS T
          WHERE T.k1 <=> S.k1 AND ... [AND prune_predicates]
        )

    No ``MERGE``, no fallback, no retry. The ``<=>`` null-safe comparison
    matches the MERGE behavior so rows with NULL key columns line up.
    ``prune_predicates`` (target-aliased) narrow the EXISTS scan to the
    matching partitions, so the keyed dedup doesn't read the whole target.
    """
    cols_quoted = ", ".join(quote_ident(c) for c in columns)
    key_match = " AND ".join(
        f"T.{quote_ident(k)} <=> S.{quote_ident(k)}"
        for k in match_by
    )
    exists_where = "\n    AND ".join([key_match, *prune_predicates])
    exists_clause = (
        f"NOT EXISTS (\n"
        f"  SELECT 1 FROM {target_location} AS T\n"
        f"  WHERE {exists_where}\n"
        f")"
    )
    return [
        f"INSERT INTO {target_location} ({cols_quoted})\n"
        f"SELECT {cols_quoted} FROM (\n{source_sql}\n) AS S\n"
        f"WHERE {exists_clause}"
    ]


def _append_maintenance_statements(
    statements: list[str],
    *,
    target_location: str,
    zorder_by: Optional[list[str]],
    optimize_after_merge: bool,
    keyed: bool,
    vacuum_hours: Optional[int],
) -> None:
    """Tack on OPTIMIZE / VACUUM statements when configured."""
    if zorder_by:
        zorder_cols = ", ".join(quote_ident(c) for c in zorder_by)
        statements.append(f"OPTIMIZE {target_location} ZORDER BY ({zorder_cols})")
    if optimize_after_merge and keyed:
        statements.append(f"OPTIMIZE {target_location}")
    if vacuum_hours is not None:
        statements.append(f"VACUUM {target_location} RETAIN {int(vacuum_hours)} HOURS")


def _build_dml_statements(
    *,
    target_location: str,
    source_sql: str,
    columns: list[str],
    mode: Mode,
    match_by: Optional[list[str]],
    update_column_names: Optional[list[str]],
    prune_predicates: list[str],
    zorder_by: Optional[list[str]] = None,
    optimize_after_merge: bool = False,
    vacuum_hours: Optional[int] = None,
    safe_merge: bool = False,
) -> list[str]:
    """Generate INSERT / MERGE / DELETE / OPTIMIZE / VACUUM SQL.

    Mode dispatch when ``match_by`` is set:

    * **safe_merge=False (default)** — emit a single ``MERGE INTO`` statement.
      :attr:`Mode.UPSERT` / :attr:`Mode.MERGE` get the full update-and-insert
      MERGE, :attr:`Mode.APPEND` / :attr:`Mode.AUTO` get the insert-only
      variant.
    * **safe_merge=True** — sidestep MERGE entirely.
      :attr:`Mode.UPSERT` / :attr:`Mode.MERGE` run a keyed ``DELETE`` followed
      by ``INSERT`` (incoming wins on overlap). :attr:`Mode.APPEND` /
      :attr:`Mode.AUTO` run ``INSERT ... WHERE NOT EXISTS (...)`` so existing
      rows are filtered at INSERT time.

    Mode without keys:

    * :attr:`Mode.TRUNCATE` with ``match_by`` → DELETE + INSERT.
    * :attr:`Mode.TRUNCATE` no keys → ``TRUNCATE TABLE`` + INSERT.
    * :attr:`Mode.OVERWRITE` with ``match_by`` → keyed DELETE + INSERT.
    * :attr:`Mode.OVERWRITE` no keys → ``INSERT OVERWRITE`` (atomic full
      replace; the live table + schema are preserved, no drop up front).
    """
    cols_quoted = ", ".join(quote_ident(c) for c in columns)
    statements: list[str] = []

    if mode in (Mode.TRUNCATE, Mode.OVERWRITE):
        if mode == Mode.TRUNCATE and match_by:
            statements.extend(_build_delete_insert_statements(
                target_location=target_location,
                source_sql=source_sql,
                columns=columns,
                match_by=match_by,
                prune_predicates=prune_predicates,
            ))
        elif mode == Mode.TRUNCATE:
            statements.extend([
                f"TRUNCATE TABLE {target_location}",
                f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}",
            ])
        elif match_by:
            statements.extend(_build_delete_insert_statements(
                target_location=target_location,
                source_sql=source_sql,
                columns=columns,
                match_by=match_by,
                prune_predicates=prune_predicates,
            ))
        else:
            statements.append(
                f"INSERT OVERWRITE {target_location} ({cols_quoted})\n{source_sql}"
            )

    elif match_by and not safe_merge:
        # Native MERGE INTO — Databricks / Delta plans the dedup once.
        # Insert-only for APPEND/AUTO; full update + insert for UPSERT/MERGE.
        statements.append(_build_merge_statement(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            match_by=match_by,
            update_column_names=update_column_names,
            prune_predicates=prune_predicates,
            insert_only=mode in (Mode.APPEND, Mode.AUTO),
        ))

    elif match_by and mode in (Mode.UPSERT, Mode.MERGE):
        # safe_merge=True + UPSERT — keyed DELETE then INSERT.
        statements.extend(_build_delete_insert_statements(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            match_by=match_by,
            prune_predicates=prune_predicates,
        ))

    elif match_by:
        # safe_merge=True + AUTO/APPEND — INSERT NOT EXISTS so existing rows
        # are filtered at INSERT time.
        statements.extend(_build_anti_join_insert(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            match_by=match_by,
            prune_predicates=prune_predicates,
        ))

    else:
        statements.append(
            f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}"
        )

    _append_maintenance_statements(
        statements,
        target_location=target_location,
        zorder_by=zorder_by,
        optimize_after_merge=optimize_after_merge,
        keyed=bool(match_by),
        vacuum_hours=vacuum_hours,
    )
    return statements


# ===========================================================================
# Async drop pipeline — stage a drop, deploy the trigger job, run the loader.
# ===========================================================================


def job_name(table: "Table") -> str:
    """The deployed loader job's name for *table*."""
    return f"{_NAME_PREFIX} {table.catalog_name}.{table.schema_name}.{table.table_name}"


def logs_path(table: "Table") -> "VolumePath":
    """``<staging_volume>/.sql/async/logs`` — the trigger's watch dir and where
    :func:`stage_async_insert` drops its operation logs."""
    return table.staging_volume.path(LOGS_SUBDIR)


def _trigger_url(table: "Table") -> str:
    # Databricks requires the file-arrival URL to end with '/'.
    url = logs_path(table).full_path()
    return url if url.endswith("/") else url + "/"


def stage_async_insert(
    table: "Table",
    data: Any,
    *,
    mode: Any = None,
    match_by: "list[str] | None" = None,
    cast_options: Any = None,
) -> "VolumePath":
    """Stage *data* as Parquet + drop a :class:`DatabricksTableInsert` op-log.

    The producer behind ``Table.insert(..., wait=False)``: write the rows to
    the table's default tmp staging path and drop a JSON op-log under
    :func:`logs_path` recording the staged data's uniform URL (so it can live
    anywhere). A path/URL *string* source is read into Arrow first. Returns the
    op-log path.

    Supports ``APPEND`` / ``OVERWRITE`` (no keys) and ``MERGE`` / ``UPSERT``
    (which require ``match_by`` key columns) — the loader aggregates the staged
    ops into one ``INSERT`` / ``INSERT OVERWRITE`` / ``MERGE INTO`` per target.
    """
    mode_enum = Mode.from_(mode, default=Mode.APPEND)
    keyed = mode_enum in (Mode.MERGE, Mode.UPSERT)
    if keyed and not match_by:
        raise ValueError(
            f"async {mode_enum.name.lower()} requires match_by key columns"
        )
    if not keyed and mode_enum not in ASYNC_MODES:
        raise ValueError(
            f"async insert supports OVERWRITE / APPEND / MERGE / UPSERT, "
            f"got {mode_enum.name}"
        )
    if isinstance(match_by, str):
        raise ValueError(
            "async match_by must be an explicit list of key columns, not a string"
        )

    if isinstance(data, str):
        from yggdrasil.io.holder import IO
        data = IO.from_(data).read_arrow_table()

    # Data goes to the default tmp staging path (kept until consumed); the log
    # records its uniform URL so the loader reads it wherever it landed.
    data_file = table.insert_volume_path(table, temporary=False)
    data_file.write_table(data, cast_options, mode=Mode.OVERWRITE)

    op = DatabricksTableInsert(
        target=table,
        mode=mode_enum,
        data=data_file,
        client=table.client,
        match_by=list(match_by) if match_by else None,
    )
    log_file = logs_path(table) / f"{op.op_id}.json"
    log_file.write_bytes(op.to_json())
    return log_file


def load_async(
    tables: Any,
    logs: Any = None,
    *,
    log_files: "Iterable[Any] | None" = None,
    wait: Any = True,
    limit: "int | None" = None,
) -> int:
    """Run the async loader over pending op-logs — group, aggregate, load, clean.

    Pass **either** *logs* — a path to a JSON op-log file or a directory of
    them (a :class:`Path` or a path string), scanned for ``*.json`` — **or**
    *log_files*, an explicit, pre-gathered iterable of log files (paths or
    strings) to consume directly, skipping the scan.

    Each log carries the full metadata, so the loader parses each into a
    :class:`DatabricksTableInsert`, groups by **target table** into a
    :class:`DatabricksInsertBatch`, runs one aggregated load per target via
    the target table's ``insert_into`` (the batch renders one ``UNION ALL``
    body — deduplicated when keyed), then clears the consumed logs + data.
    Returns the number of operations processed.

    The loader behind the file-arrival job and the ``ygg databricks table
    execute_async_insert`` CLI. *tables* is the :class:`Tables` service.
    """
    from yggdrasil.databricks.path import DatabricksPath

    client = tables.client
    if log_files is not None:
        files = [
            DatabricksPath.from_(f, client=client) if isinstance(f, str) else f
            for f in log_files
        ]
    else:
        logs_dir = (
            DatabricksPath.from_(logs, client=client)
            if isinstance(logs, str) else logs
        )
        if logs_dir is None or not logs_dir.exists():
            logger.info("async loader: %s does not exist — nothing to do", logs_dir)
            return 0
        files = (
            [f for f in logs_dir.iterdir() if str(f.name).endswith(".json")]
            if logs_dir.is_dir() else [logs_dir]
        )

    ops: list[DatabricksTableInsert] = []
    for log_file in files:
        try:
            ops.append(DatabricksTableInsert.from_log(log_file, client=client))
        except Exception:
            logger.warning("skipping unreadable async log %s", log_file)
            continue
        if limit is not None and len(ops) >= limit:
            break

    return dispatch_async(tables, ops, wait=wait)


def dispatch_async(tables: Any, ops: "Iterable[Any]", *, wait: Any = True) -> int:
    """Group parsed ops by target into a :class:`DatabricksInsertBatch` and load
    each through the target's ``insert_into`` (one aggregated body per target),
    clearing the consumed logs + data afterward. Returns the op count."""
    client = tables.client
    batches = DatabricksInsertBatch.group(ops)
    if not batches:
        logger.info("async loader: no pending operation logs")
        return 0
    logger.info("async loader: %d target group(s)", len(batches))

    processed = 0
    for batch in batches:
        target_name = batch.logs[0].target_name
        target = tables[target_name]
        mode = batch.mode
        logger.info(
            "loading %d file(s) into %s (%s)",
            len(batch.active), target_name, mode.name.lower(),
        )
        # The batch is the single place the load source is generated.
        target.insert_into(
            batch.make_sql(client), mode=mode.name.lower(),
            match_by=batch.match_by, wait=wait,
        )
        # Clear consumed logs + data (incl. superseded ops) after a load.
        for op in batch.logs:
            _best_effort_unlink(op.log_file)
            _best_effort_unlink(op.data_path(client))
        processed += len(batch.logs)
    return processed


def _best_effort_unlink(path: Any) -> None:
    """Remove *path* if present; cleanup failures are logged, never raised."""
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - cleanup is best-effort
        logger.debug("async cleanup: failed to remove %s", path, exc_info=True)


def ensure_async_job(
    table: "Table", *, client: Any = None, rebuild: bool = False,
) -> Any:
    """Get-or-create the file-arrival loader job for *table*, return the Job.

    **Get** — when a job with the same name already exists it is returned
    untouched: no wheel build, no upsert, no prune. A steady-state call is
    therefore cheap. Pass ``rebuild=True`` to force the create path and refresh
    the deployment (re-resolve the wheel, re-upsert the job).

    **Create** — provisions the watched ``logs/`` dir, resolves the full ygg
    wheel bundle for the current version
    (:func:`~yggdrasil.databricks.job.wheel.ensure_ygg_wheel` — reusing an
    already-deployed bundle when present, else building + uploading it to the
    shared workspace wheel path), and upserts a serverless job whose single
    python-wheel task runs ``ygg databricks table execute_async_insert --logs
    <dir>`` when a log lands. Any stale job watching the same logs dir is pruned
    so a single job owns the trigger.
    """
    from databricks.sdk.service.jobs import (
        FileArrivalTriggerConfiguration,
        PythonWheelTask,
        Task as DBTask,
        TriggerSettings,
    )

    from yggdrasil.databricks.job.skeleton import ensure_console_logging

    client = client or table.client
    name = job_name(table)

    if not rebuild:
        existing = client.jobs.get(name=name, default=None)
        if existing is not None:
            logger.info(
                "async job %r already deployed (id=%s) — reusing",
                name, getattr(existing, "job_id", None),
            )
            return existing

    ensure_console_logging()  # surface the deploy CRUD interactively

    # The trigger watches the logs dir — create it first so Databricks accepts
    # the URL (and the first drop lands cleanly).
    logs = logs_path(table)
    logger.info("async job: ensuring logs dir %s", logs.full_path())
    logs.mkdir(parents=True, exist_ok=True)

    # The pinned, versioned ygg image (latest serverless v5 + ygg CLI +
    # databricks-sdk + deps, installed by path) — get-or-created on the client.
    environment = client.ygg_environment(environment_key="default", rebuild=rebuild)

    logger.info("create-or-update async job %r", name)
    job = client.jobs.create_or_update(
        name=name,
        tasks=[
            DBTask(
                task_key="async-load",
                environment_key="default",
                python_wheel_task=PythonWheelTask(
                    # Run the ygg CLI on the cluster:
                    #   ygg databricks table execute_async_insert --logs <dir>
                    package_name="ygg",
                    entry_point="ygg",
                    parameters=[
                        "databricks", "table", "execute_async_insert",
                        "--logs", logs.full_path(),
                    ],
                ),
            )
        ],
        environments=[environment],
        trigger=TriggerSettings(
            file_arrival=FileArrivalTriggerConfiguration(
                url=_trigger_url(table),
                wait_after_last_change_seconds=BUFFER_SECONDS,
                min_time_between_triggers_seconds=BUFFER_SECONDS,
            ),
        ),
    )
    logger.info("deployed async job %r (id=%s)", name, getattr(job, "job_id", None))
    _prune_duplicates(client, _trigger_url(table), keep=getattr(job, "job_id", None))
    return job


def _prune_duplicates(client: Any, url: str, *, keep: Any) -> None:
    """Delete any *other* job whose file-arrival trigger watches the same logs
    dir — orphans left by an earlier naming scheme keep firing on the shared
    trigger (and fail), so the deploy collapses to a single job."""
    try:
        for other in client.jobs.list():
            if other.job_id == keep:
                continue
            trigger = getattr(other.settings, "trigger", None)
            file_arrival = getattr(trigger, "file_arrival", None)
            if file_arrival is not None and file_arrival.url == url:
                try:
                    other.delete()
                    logger.info("removed stale async job %s (%s)", other.job_id, url)
                except Exception:
                    logger.warning("could not delete stale async job %s", other.job_id)
    except Exception:
        logger.debug("stale-async-job prune skipped", exc_info=True)
