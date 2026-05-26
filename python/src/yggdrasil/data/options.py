"""`CastOptions` — the one options object every :class:`DataIO` takes.

What it carries
---------------

Two :class:`Field` s (``source``, ``target``) — the inferred /
desired schema at each end of a cast. Because :class:`Schema` is a
subclass of :class:`StructField` which is a :class:`Field`, the same
slot covers "I have a single column" and "I have a full schema" —
the type just promotes through :meth:`Field.from_` on construction
(``pa.Schema`` lands as :class:`StructField`, ``pa.DataType`` lands
as a leaf :class:`Field`). Callers that want a Schema-shaped view
specifically can call ``.target.to_schema()`` / ``.source.to_schema()``.

A ``safe`` flag for strict-vs-permissive semantics (overflow,
truncation, nulls-in-non-nullable). Sizing knobs ``row_size`` /
``byte_size`` that batch-oriented readers/writers honour. A
:class:`Mode` pair for write semantics: ``mode`` controls the data
write (overwrite / append / error-if-exists), ``schema_mode``
controls how schema drift is handled. An optional
``arrow_memory_pool`` for callers routing allocations through a
bounded pool.

The canonical entry point
-------------------------

:meth:`CastOptions.check` is what every :class:`DataIO` public method
funnels through. It accepts any of: an existing CastOptions to reuse,
a dict of overrides, a :class:`pa.DataType` / :class:`pa.Field` /
:class:`pa.Schema` to promote to a target hint, or ``None`` for
defaults. It merges everything into a single :class:`CastOptions`
instance — immutable from there, so no per-call mutation hazards.

``...`` sentinel
------------------

:data:`...` distinguishes "caller didn't pass this" from "caller
passed ``None``". The latter is a real value (e.g., ``row_size=None``
means "no row cap"); the former should inherit whatever the base
options had. :meth:`strip_...` drops ...-valued keys from a
mapping so ``.check()`` doesn't overwrite existing values with "I
didn't say anything."

Field normalization
-------------------

``__post_init__`` runs every field-shaped input through
:meth:`Field.from_`, so callers can pass a :class:`pa.Schema`, a
:class:`pa.Field`, a :class:`pa.DataType`, a yggdrasil :class:`Schema`
or :class:`Field`, or a dict spec — all land as a uniform
:class:`Field`. Stops ``CastOptions(target=pa_schema)`` from
propagating an un-wrapped pyarrow object into the casting engines.

Engine dispatch
---------------

:meth:`cast` and the per-engine / per-shape variants all delegate to
the matching :class:`Field` methods — :class:`Field` owns the
dispatch table (engine detection via :meth:`ObjectSerde.module_and_name`,
shape detection via isinstance under the engine's lazy-imported
module). :class:`CastOptions` adds exactly two things on top:

1. The ``target is None`` short-circuit — valid on options but not
   on a Field, which always carries a dtype. Lets
   :meth:`CastOptions.cast` pass *obj* through unchanged when no
   target is bound.
2. ``options=self`` plumbing — callers threading options through a
   long pipeline don't have to re-specify at each cast site.

Single source of truth for the dispatch table: adding a new engine
is a one-site edit in :class:`Field`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import functools
from typing import Any, Iterable, Iterator, Mapping, TypeVar, Union, TYPE_CHECKING

import pyarrow as pa

from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.data.enums import Mode
from yggdrasil.environ import PyEnv
from yggdrasil.lazy_imports import field_class, schema_class

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from yggdrasil.data.data_field import Field
    from yggdrasil.data.schema import Schema
    from yggdrasil.execution.expr import Predicate

# ``Field`` / ``Schema`` are imported lazily — top-level imports here
# would form a cycle through ``yggdrasil.data.schema`` ↔
# ``yggdrasil.data.data_field`` during ``yggdrasil.data`` bootstrap
# (this module is reached transitively from ``data_field`` via the
# ``io`` chain). The deferred ``field_class()`` / ``schema_class()``
# helpers from :mod:`yggdrasil.lazy_imports` keep the runtime contract
# without anchoring the cycle.


__all__ = [
    "CastOptions",
    "CastOptionsArg",
]


# ``...`` (Ellipsis) is the project-wide "unset" sentinel — see the
# convention note in ``AGENTS.md`` / ``CLAUDE.md``. ``None`` is a
# valid computed merge result, so we can't use it to mean
# "not yet computed".


# Slots whose values pass through ``__post_init__`` normalization
# (Field coercion, Mode coercion, match_by list rebuild). The
# fast-clone path in :meth:`CastOptions._fast_clone` uses this to
# decide whether the new instance needs a normalization re-run after
# applying overrides — copies that only touch primitive flags
# (``safe``, ``row_size``, ``mode`` *already as* a Mode enum, …) can
# skip ``__post_init__`` entirely.
_NORMALIZED_KEYS = frozenset({
    "source", "target", "match_by", "unique_by", "time_sample_by",
    "mode", "schema_mode",
})


# Field metadata key (non-prefixed — *not* a yggdrasil schema-level
# tag) carrying the ISO-8601 sampling interval for entries in
# :attr:`CastOptions.time_sample_by`. Kept off the ``t:`` prefix
# registry so the value doesn't get auto-propagated through
# :meth:`Field.autotag` / arrow round-trips — it's a per-call option,
# not a schema-level contract.
_TIME_SAMPLING_METADATA_KEY: bytes = b"time_sampling"


def _field_time_sampling_seconds(field: Any) -> int:
    """Decode the ISO-8601 sampling stored on a :class:`Field`.

    Returns ``0`` when the field carries no ``b"time_sampling"`` key
    or the stored value fails to parse — callers treat ``0`` as the
    "no sampling requested" sentinel. Local import of
    :func:`yggdrasil.data.types.primitive.temporal._parse_iso_duration`
    keeps the options module free of the temporal import chain on
    the common construction path.
    """
    md = getattr(field, "metadata", None)
    if not md:
        return 0
    raw = md.get(_TIME_SAMPLING_METADATA_KEY)
    if raw is None:
        return 0
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    from yggdrasil.data.types.primitive.temporal import _parse_iso_duration
    td = _parse_iso_duration(text)
    if td is None:
        return 0
    return max(0, int(td.total_seconds()))


def timedelta_to_iso_duration(td: dt.timedelta) -> str:
    """Render a :class:`datetime.timedelta` as an ISO-8601 duration string.

    Inverse of
    :func:`yggdrasil.data.types.primitive.temporal._parse_iso_duration`,
    minus the calendar units (Y/M/W) that the parser collapses on
    the way in. Useful for stamping ``b"time_sampling"`` metadata on
    a :class:`Field` before handing it to
    :attr:`CastOptions.time_sample_by`::

        from yggdrasil.data import field
        from yggdrasil.data.options import timedelta_to_iso_duration
        import datetime as dt, pyarrow as pa

        f = field("ts", pa.timestamp("us", "UTC"),
                  metadata={b"time_sampling":
                            timedelta_to_iso_duration(dt.timedelta(hours=1)).encode()})
    """
    total = td.total_seconds()
    sign = "-" if total < 0 else ""
    total = abs(total)
    days, rem = divmod(int(total), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    parts = ["P"]
    if days:
        parts.append(f"{days}D")
    time_parts: list[str] = []
    if hours:
        time_parts.append(f"{hours}H")
    if minutes:
        time_parts.append(f"{minutes}M")
    if seconds or (not days and not hours and not minutes):
        time_parts.append(f"{seconds}S")
    if time_parts:
        parts.append("T" + "".join(time_parts))
    return sign + "".join(parts)


def _struct_of_objects(columns: "Iterable[str]") -> "Schema":
    """Build a :class:`Schema` from *columns* with :class:`ObjectType` fields.

    Used by :meth:`CastOptions.check` when the caller passed a bare
    ``columns=`` list and no target schema bound. The children default
    to :class:`ObjectType` so casts pass through untouched.
    """
    from yggdrasil.data.data_field import Field as _Field
    from yggdrasil.data.schema import Schema as _Schema
    from yggdrasil.data.types.primitive.object import ObjectType

    children = [_Field(name=str(c), dtype=ObjectType()) for c in columns]
    return _Schema(children)



# ---------------------------------------------------------------------------
# Type alias for "anything that can be coerced into a CastOptions"
# ---------------------------------------------------------------------------

# Accept at the public boundary: an existing CastOptions, a dict of
# field overrides, a pa.DataType/Field/Schema that we promote to
# target, or None for defaults. Kept intentionally broad —
# .check() handles the routing.
CastOptionsArg = Union[
    "CastOptions",
    Mapping[str, Any],
    pa.DataType,
    pa.Field,
    pa.Schema,
    None,
]


# Cached tuple for the schema-promote isinstance check on the
# :meth:`CastOptions.check` hot path. The tuple is rebuilt the
# first time a non-CastOptions / non-None / non-Mapping value
# lands in ``check``; ``field_class()`` / ``schema_class()`` are
# themselves lru-cached, but resolving + tuple-allocating on every
# call shows up as ~200ns of pure overhead in the bench. Stored as
# a module-level slot to avoid a second function-call hop.
_PROMOTE_TYPES: "tuple[type, ...] | None" = None


def _promote_types() -> "tuple[type, ...]":
    global _PROMOTE_TYPES
    cached = _PROMOTE_TYPES
    if cached is None:
        cached = (
            pa.DataType, pa.Field, pa.Schema,
            field_class(), schema_class(),
        )
        _PROMOTE_TYPES = cached
    return cached


# ---------------------------------------------------------------------------
# CastOptions
# ---------------------------------------------------------------------------


T = TypeVar("T", bound="CastOptions")


@dataclasses.dataclass(frozen=True, slots=True)
class CastOptions:
    """Options carried through every :class:`DataIO` read and write.

    Frozen so a single instance can safely be shared across threads /
    tasks; mutation requires :meth:`copy`. Slotted for cheap
    construction on the hot path (hundreds of options objects get
    built per batched write in a folder-of-folders persist).

    All fields default to safe no-ops:

    * ``source`` / ``target`` = ``None`` → no cast coercion.
      :meth:`cast` returns inputs unchanged.
    * ``safe`` = ``False`` → permissive cast: bad rows / overflow
      become null. Strict semantics are opt-in via ``safe=True``.
    * ``mode`` / ``schema_mode`` = :attr:`Mode.AUTO` → writer
      picks the appropriate behaviour from context.
    * ``row_size`` / ``byte_size`` = ``None`` → no batch caps; readers
      stream whatever size is natural for the format.
    * ``arrow_memory_pool`` = ``None`` → use pyarrow's default pool.
    """

    source: "Field | None" = None
    target: "Field | None" = None
    #: ``False`` by default: bad rows / overflow / out-of-range values
    #: become null instead of raising. The wider data plane (CSV-from-web,
    #: JSON-from-API, Spark joins on partial frames) is messy enough that
    #: a strict cast is almost never what the caller actually wants — they
    #: want to land the rows they can and surface the ones they couldn't.
    #: Flip to ``True`` at the call site to opt into strict semantics
    #: (overflow / parse / type-mismatch → raise).
    safe: bool = False
    #: ``False`` (default): each batch handed to ``write_arrow_batches``
    #: is run through :meth:`CastOptions.check_source` /
    #: :meth:`check_target` to align its schema against the target.
    #: That covers heterogenous inputs (an Arrow stream from one feed,
    #: a polars frame from another, a row dict from a callsite) where
    #: the writer can't trust the batch shape matches what the leaf
    #: holds on disk.
    #:
    #: ``True``: the caller guarantees every batch *already* matches
    #: the target schema (came from a :class:`pa.Table`, a
    #: :class:`pa.RecordBatchReader`, a polars / pandas frame whose
    #: shape was resolved upstream, or another writer that just
    #: emitted the same schema). Inner ``write_arrow_batches`` skips
    #: the per-batch source rebuild + cast pass entirely — the bytes
    #: go straight to the format's writer. Use only when you control
    #: the source.
    checked_cast: bool = False
    mode: Mode = Mode.AUTO
    schema_mode: Mode = Mode.IGNORE
    row_size: int | None = None
    byte_size: int | None = None
    row_limit: int | None = None
    #: Allow format readers / writers to use a thread pool when the
    #: backend supports it. Universally honored across CSV, Parquet,
    #: Arrow IPC, and NDJSON; format-specific options can override
    #: the default by re-declaring the field.
    use_threads: bool = True
    recursive: bool = False
    #: Field-typed column references to dedup on. Each entry's
    #: :attr:`Field.name` resolves to the target column. Bare strings
    #: passed in ``__init__`` are coerced to a default-typed
    #: :class:`Field` in :meth:`__post_init__` so callers can still
    #: pass plain key names — the Field-typed shape is the canonical
    #: surface.
    match_by: list["Field"] | None = None
    #: Field-typed column references to dedup on at read time. Same
    #: shape as :attr:`match_by` — each entry's :attr:`Field.name`
    #: names the column whose values must be distinct in the read
    #: output. :meth:`dedup_arrow_batches` collapses duplicates on
    #: these columns via :func:`yggdrasil.arrow.ops.dedup_arrow_table`
    #: (first occurrence wins). Unset → no dedup pass; the read
    #: pipeline yields rows unchanged.
    unique_by: list["Field"] | None = None
    #: Field-typed timestamp references to resample on at read time.
    #: Each entry's :attr:`Field.metadata` carries a non-tag
    #: ``b"time_sampling"`` key whose value is the ISO-8601 duration
    #: (``"PT1H"``, ``"P1D"``, etc.) that
    #: :func:`yggdrasil.arrow.ops.resample_arrow_table` snaps the
    #: column to. Unset → no resample pass; the read pipeline yields
    #: rows unchanged. Plain ``list[Field]`` keeps the API shape
    #: parallel to :attr:`match_by` / :attr:`unique_by` — the
    #: sampling rides on the Field itself rather than a separate
    #: ``dict`` so the bundle is a single round-trippable Field
    #: collection.
    time_sample_by: list["Field"] | None = None
    #: How to fill nulls left by the resample pass. ``"ffill"``
    #: (default) propagates the last non-null value forward into
    #: subsequent nulls within the same partition; ``"bfill"``
    #: propagates the next non-null backward; ``"none"`` / ``None``
    #: disables the pass so resample emits whatever the ``"first"``
    #: aggregate picked. Same vocabulary on every engine — the
    #: arrow / spark ops in :mod:`yggdrasil.arrow.ops` and
    #: :mod:`yggdrasil.spark.ops` consume this verbatim.
    fill_strategy: str = "ffill"
    read_seek: int | None = None
    write_seek: int | None = None
    #: Row-level predicate. Evaluated by every IO that reads tabular
    #: rows: applied to each Arrow batch before it leaves the read
    #: pipeline so callers don't have to wrap the result by hand.
    #:
    #: When the predicate references a column the *source* doesn't
    #: have (different schema, optional column not present in this
    #: file), the predicate degrades to *accept everything* —
    #: missing inputs can't yield a coherent boolean, and the
    #: alternative ("drop everything") is almost always wrong for
    #: heterogeneous-source folders. Backends that can push the
    #: predicate down (Delta, warehouse SQL) skip the per-batch
    #: filter once they've translated it.
    predicate: Predicate | None = None
    #: Predicate evaluated against a discovered child (``name``,
    #: ``path``, ``is_dir``, ``is_private``) for IOs that aggregate
    #: sub-IOs (folders, zips, partitioned tables). Replaces the
    #: legacy ``include_patterns`` / ``exclude_patterns`` /
    #: ``exclude_private`` glob knobs with one composable predicate
    #: backed by :mod:`yggdrasil.data.expr` — for example
    #: ``~col("name").like(".%") & col("name").like("%.parquet")``.
    children_predicate: Predicate | None = None
    read_write_upsert: bool = False
    wait: WaitingConfig = WaitingConfig.default()
    spark_session: "SparkSession | None" = None
    arrow_memory_pool: pa.MemoryPool | None = None

    # --- Upsert / merge shape -------------------------------------------
    update_column_names: list[str] | None = None

    # --- Partition pruning ----------------------------------------------
    # ``prune_by`` accepts the literal string ``"auto"`` to mean
    # "use the partition columns from the target schema" — Table.*_insert
    # resolves that into a real column list.
    prune_by: "list[str] | str | None" = None
    prune_values: Mapping[str, tuple[Any, ...]] | None = None

    # --- Trailing maintenance -------------------------------------------
    zorder_by: list[str] | None = None
    optimize_after_merge: bool = False
    vacuum_hours: int | None = None

    # --- Engine knobs ----------------------------------------------------
    overwrite_schema: bool | None = None
    spark_options: dict[str, Any] | None = None

    # --- Statement-level retry ------------------------------------------
    # Threaded onto each DML WarehousePreparedStatement on the warehouse
    # path; ignored on Spark (driver-side retry handles it there).
    retry: WaitingConfigArg | None = None

    # --- Capture inserted rows as a return value -----------------------
    # When True, mutating tabular operations (table inserts, MERGE-style
    # writes, …) hand back the rows they actually wrote as a
    # :class:`Tabular` — typically a :class:`ArrowTabular` or a
    # :class:`Dataset` depending on the engine that ran the write.
    # Default ``False`` keeps the historical "fire-and-forget" return
    # contract; flip it on when downstream code wants to chain on the
    # newly-appended payload (logging, follow-up tasks, downstream
    # writes) without re-querying the target.
    return_data: bool = False

    # --- Keyed-write strategy toggle ------------------------------------
    # When False (the default), keyed inserts use the engine's native
    # ``MERGE INTO`` statement — Databricks / Delta plans the dedup
    # once. When True, the table layer sidesteps MERGE entirely:
    # ``Mode.APPEND`` becomes ``INSERT ... WHERE NOT EXISTS`` (or a
    # Spark DataFrame anti-join, when a session is reachable);
    # ``Mode.UPSERT`` / ``Mode.MERGE`` become a keyed ``DELETE``
    # followed by ``INSERT``. Useful for backends without native
    # MERGE, for callers that want explicit dedup semantics, or for
    # the Spark fast path where pre-filtering the DataFrame is much
    # cheaper than the SQL ``NOT EXISTS`` plan.
    safe_merge: bool = False

    # --- Metadata sync after writes -------------------------------------
    # When True (the default), a writer commits the holder-side IO
    # metadata (size, mtime, media_type) once the operation finishes.
    # Bulk writers (``_write_arrow_table``, ``_write_polars_frame``,
    # ``_write_records``) flip this to False on the per-batch sub-call
    # and run a single ``_commit_metadata`` at the end so multi-batch
    # writes don't pay the stat-refresh cost on every batch. Set to
    # False at the call site to suppress the final commit too — for
    # callers that intend to chain another write before any reader sees
    # the result.
    sync_metadata: bool = True

    # --- Memoization slot ----------------------------------------------
    # ``merged`` is read repeatedly by every cast / fill / alias
    # entry point on this class — once per dispatch call, often dozens
    # of times across a single batch pipeline. The underlying
    # ``Field.merge_with`` walks the full struct tree on each call.
    # ``CastOptions`` is frozen, so the inputs to that merge
    # (``source``, ``target``, ``schema_mode``) cannot change for a
    # given instance — the result is safe to cache for the lifetime
    # of the options object.
    #
    # ``merged_schema`` returns the same object: now that
    # :class:`StructField` *is* a :class:`Field`, the merged result
    # for struct sides already satisfies the Schema interface — one
    # cache slot covers both views.  The same logic collapses
    # ``source`` / ``target`` themselves: a single :class:`Field`
    # slot serves both "I want a column-ish Field" and "I want a
    # schema-shaped Field" callers — no parallel ``*_schema`` cache
    # is necessary.
    #
    # ``init=False`` keeps the slot out of the constructor signature;
    # ``compare=False`` keeps two functionally identical CastOptions
    # equal regardless of which one happened to have its cache warmed;
    # ``repr=False`` keeps the cache out of the dataclass-generated
    # repr (the custom repr below also skips non-repr fields).
    _merged_cache: Any = dataclasses.field(
        default=..., init=False, repr=False, compare=False
    )

    # ==================================================================
    # Normalization — runs after dataclass init
    # ==================================================================

    def __post_init__(self) -> None:
        """Coerce field-shaped inputs to :class:`Field`, mode inputs to
        :class:`Mode`.

        Frozen dataclasses can't use normal attribute assignment in
        ``__post_init__`` — we go through :func:`object.__setattr__``
        directly, which is the standard escape hatch for this exact
        case. The body is structured around "skip every branch whose
        input is already normalized" — the default ``CastOptions()``
        path (no source/target, ``Mode`` enums already in place, no
        ``match_by``) is the hottest construction shape and pays no
        normalization cost beyond the isinstance gates.
        """
        setattr_ = object.__setattr__

        # Field normalization — accept pa.Schema/Field/DataType, dict,
        # yggdrasil Field/Schema, or None. We defer the ``field_class()``
        # lookup until we actually need the Field type, so the common
        # "no source, no target, no match_by" path skips the import
        # cache hit entirely.
        src = self.source
        tgt = self.target
        match_by = self.match_by
        unique_by = self.unique_by
        time_sample_by = self.time_sample_by
        if (
            src is not None or tgt is not None
            or match_by or unique_by or time_sample_by
        ):
            Field = field_class()
            if src is not None and not isinstance(src, Field):
                setattr_(self, "source", Field.from_(src))
            if tgt is not None and not isinstance(tgt, Field):
                setattr_(self, "target", Field.from_(tgt))
            # match_by / unique_by / time_sample_by normalization —
            # each entry accepts the same shapes (Field | str |
            # dict | pa.Field). Plain strings become a default-typed
            # Field so the selector machinery (alias / position
            # lookup) has a Field to drive.
            def _to_fields(items):
                return [
                    item if isinstance(item, Field)
                    else Field.default(name=item) if isinstance(item, str)
                    else Field.from_(item)
                    for item in items
                ]
            if match_by:
                setattr_(self, "match_by", _to_fields(match_by))
            if unique_by:
                setattr_(self, "unique_by", _to_fields(unique_by))
            if time_sample_by:
                setattr_(self, "time_sample_by", _to_fields(time_sample_by))
        # Empty match_by / unique_by / time_sample_by collapses to
        # ``None`` so consumers can branch on truthiness. Lives
        # outside the field-lookup gate since it doesn't need the
        # Field type.
        if match_by is not None and not match_by:
            setattr_(self, "match_by", None)
        if unique_by is not None and not unique_by:
            setattr_(self, "unique_by", None)
        if time_sample_by is not None and not time_sample_by:
            setattr_(self, "time_sample_by", None)

        # Mode normalization — accept Mode, string ("overwrite"),
        # or None (→ AUTO). Already a Mode instance is the overwhelmingly
        # common case (defaults plus enum-literal callers), so gate the
        # ``Mode.from_`` call behind an isinstance check.
        mode = self.mode
        if not isinstance(mode, Mode):
            setattr_(self, "mode", Mode.from_(mode, default=Mode.AUTO))
        schema_mode = self.schema_mode
        if not isinstance(schema_mode, Mode):
            setattr_(self, "schema_mode", Mode.from_(schema_mode, default=Mode.IGNORE))

    # ==================================================================
    # Derived properties
    # ==================================================================

    @property
    def merged(self) -> Field | None:
        cached = self._merged_cache
        if cached is not ...:
            return cached
        if self.source and self.target:
            result = self.target.merge_with(
                self.source, mode=self.schema_mode
            )
        else:
            result = self.target or self.source
        object.__setattr__(self, "_merged_cache", result)
        return result

    def select_source_column_names(self) -> list[str] | None:
        """The source field's column names, if a source field is bound."""
        if self.source is None:
            return None

        source_names = list(self.source.names)

        if self.target is None:
            return source_names

        selected: list[str] = []

        for target_name in self.target.names:
            found = self.source.field(name=target_name, raise_error=False)
            if found is not None:
                selected.append(found.name)

        return selected or None

    @property
    def column_names(self) -> list[str] | None:
        """The target field's column names, if a target field is bound."""
        merged = self.merged

        if merged is None:
            return None

        return merged.names

    @property
    def match_by_keys(self) -> list[str] | None:
        """Resolved key column names to dedup on.

        Pulls the :attr:`Field.name` of each entry in
        :attr:`match_by`. Returns ``None`` when no keys are set so
        callers can branch on "keys vs no-keys" with a single
        truthiness check.
        """
        if not self.match_by:
            return None
        return [f.name for f in self.match_by]

    # ==================================================================
    # Construction / merge entry point
    # ==================================================================

    @classmethod
    def check(
        cls: type[T],
        options: CastOptionsArg = None,
        /,
        **overrides: Any,
    ) -> T:
        """Canonical entry point — coerce anything into a :class:`CastOptions`.

        Dispatch by what *options* is:

        * ``None`` — construct fresh :class:`CastOptions(**overrides)`.
        * :class:`CastOptions` — if no overrides given, return it
          unchanged; if overrides given, ``.copy(**overrides)``.
        * :class:`Mapping` (including ``dict``) — merge into overrides
          and construct fresh (override args win on key collision).
        * :class:`pa.DataType` / :class:`pa.Field` / :class:`pa.Schema`
          / :class:`Field` / :class:`Schema` — treat as a target hint.
          Equivalent to ``check(target=options, **overrides)``.

        ``source=`` / ``target=`` go straight into the dataclass slots
        (after :meth:`Field.from_` normalization in ``__post_init__``).
        Callers that want the peek-and-bind "only set if not already
        bound" semantic should chain :meth:`check_source` /
        :meth:`check_target` after the call.

        ``columns=`` shortcut: a sequence of column names. When the
        caller didn't bind a source by any other means (no ``source=``
        override, the wrapped options didn't carry one either), the
        names are promoted to a struct-shaped source field whose
        children default to :class:`ObjectType` — a "I have these
        columns, no idea what's in them yet" placeholder that
        downstream casts treat as passthroughs. Ignored when the source
        is already bound, so callers can pass it defensively without
        clobbering richer schemas.

        :raises TypeError: if *options* is a type the dispatch table
            doesn't cover.
        """
        # Hot fast-path: ``check(opts)`` / ``check(None)`` with no
        # overrides. Every cast wrapper in arrow/polars/pandas/spark
        # ``cast.py`` funnels through that exact call shape, and the
        # passthrough is by far the most common dispatch. Resolving
        # it before the ``columns`` pop, the ``Mapping`` ABC check,
        # and the promotable-types tuple build saves ~250ns on each
        # passthrough.
        if not overrides:
            if options is None or options is ...:
                return cls()
            if isinstance(options, cls):
                return options
            # Other shapes (Mapping, schema-promote, subclass re-home)
            # still need full processing — fall through.

        # ``columns`` is not a CastOptions field — pop it from both
        # ``overrides`` and any Mapping-shaped ``options`` before the
        # field-name filter in :meth:`_build` discards it.
        columns = overrides.pop("columns", None) if overrides else None
        if columns is None and isinstance(options, Mapping):
            # Only copy the mapping when the caller actually stuffed
            # ``columns`` into it — the common dict shape (target +
            # safe + row_size, no columns) skips the copy entirely.
            cols = options.get("columns")
            if cols is not None:
                options = dict(options)
                columns = options.pop("columns")

        # 1. None / ... → fresh construction. Fast path when nothing
        # else is bound either — every DataIO public method that's
        # called without an explicit options hits this branch, and
        # going through ``_build`` (with its dict comprehension over
        # ``field_names()``) is pure overhead when there's nothing to
        # merge.
        if options is None or options is ...:
            if not overrides and not columns:
                instance = cls()
            else:
                instance = cls._build(overrides)

        # 2. Already a CastOptions — reuse via copy if overrides
        # given, otherwise passthrough. Typed check with ``cls`` so a
        # subclass CastOptions stays on its subclass.
        elif isinstance(options, cls):
            if not overrides and not columns:
                instance = options
            else:
                instance = options.copy(**overrides)
        elif isinstance(options, CastOptions):
            # Different CastOptions subclass → re-home onto cls. Only
            # carry over fields shared with cls (base CastOptions
            # always; format-specific fields stay on their owner). A
            # name like ``compression`` lives on both ZipOptions and
            # ArrowIPCOptions but means different things; keeping the
            # source value would corrupt the target's writer config.
            target_fields = cls.field_names()
            base_fields = CastOptions.field_names()
            carry = base_fields & target_fields
            merged = {
                f.name: getattr(options, f.name)
                for f in dataclasses.fields(options)
                if f.name in carry
            }
            if overrides:
                merged.update(overrides)
            instance = cls._build(merged)

        # 3. Mapping → merge into overrides (explicit kwargs win).
        elif isinstance(options, Mapping):
            # ``_build`` mutates the passed dict (pops ``"options"``),
            # so the caller's mapping is never the one we hand on.
            # Skip the ``{**options, **overrides}`` double-spread when
            # the caller passed a dict alone — a single ``dict(options)``
            # copy is enough and meaningfully cheaper on small dicts.
            if overrides:
                merged = {**options, **overrides}
            else:
                merged = dict(options)
            instance = cls._build(merged)

        # 4. Schema-shaped → promote to a target hint.
        elif isinstance(options, _promote_types()):
            overrides.setdefault("target", options)
            instance = cls._build(overrides)

        else:
            raise TypeError(
                f"CastOptions.check cannot coerce {type(options).__name__}. "
                "Expected CastOptions, dict, pa.DataType/Field/Schema, "
                "yggdrasil Field/Schema, or None."
            )

        if columns:
            if instance.target is not None:
                instance = instance.copy(target=instance.target.select(columns))
            elif instance.source is None:
                instance = instance.copy(source=_struct_of_objects(columns))
        return instance

    @classmethod
    @functools.cache
    def _init_field_names_tuple(cls) -> tuple[str, ...]:
        """Ordered tuple of init-eligible field names — cached per-class.

        Sibling of :meth:`field_names` but tuple-shaped: the
        fast-clone loop in :meth:`_fast_clone` iterates this list
        once per copy and tuple iteration is measurably tighter than
        frozenset iteration (no hash probes; predictable cache locality).
        Cached on the class via :func:`functools.cache` so subclasses
        get their own expanded tuple on first access.
        """
        return tuple(f.name for f in dataclasses.fields(cls) if f.init)

    @classmethod
    @functools.cache
    def field_names(cls) -> frozenset[str]:
        """Frozenset of this class's constructor-accepting field names.

        Used by :meth:`_build` to filter ``**overrides`` down to keys
        the constructor will accept — callers funnel mixed kwargs
        through ``.check()`` (DataIO public methods often pass user
        kwargs straight through), and we don't want a stray
        ``filter=`` or ``columns=`` to crash construction.

        Excludes ``init=False`` fields (the private memoization slots
        for ``merged`` / ``merged_schema``); those are not valid
        ``__init__`` keywords and a copy via ``dataclasses.replace``
        would crash if it tried to forward them.

        Cached per-class via :func:`functools.cache` so subclasses
        with extra fields get their own expanded set on first access.
        """
        return frozenset(f.name for f in dataclasses.fields(cls) if f.init)

    @classmethod
    def _build(
        cls: type[T],
        overrides: Mapping[str, Any],
    ) -> T:
        # Single filter pass over the override mapping: drop ``...``
        # sentinels (caller didn't say) and foreign keys (DataIO
        # public methods pass mixed kwargs through ``.check()`` and
        # only CastOptions fields are valid for construction). The
        # earlier two-pass shape called ``field_names()`` twice and
        # walked the dict twice for the same final set; one pass is
        # functionally identical and noticeably faster on hot caller
        # sites.
        overrides = overrides or {}
        options = overrides.pop("options", None)
        allowed = cls.field_names()
        clean = {
            k: v
            for k, v in overrides.items()
            if v is not ... and k in allowed
        }
        if options is None:
            return cls(**clean)
        return options.copy(**clean)

    # ==================================================================
    # Copy / mutation helpers
    # ==================================================================

    def copy(
        self: T,
        /,
        **overrides: Any,
    ) -> T:
        """Return a copy with *overrides* applied.

        ... values in *overrides* are ignored (keep existing). Pass
        ``source=``/``target=`` to swap either slot — :class:`Field`
        normalization runs in ``__post_init__`` so any
        :class:`Field`-shaped input (``pa.Schema``, ``pa.DataType``,
        dict, …) is accepted.

        Implementation note: bypasses :func:`dataclasses.replace`, which
        rebuilds via ``cls(**all_fields)`` and pays a full ``__init__``
        + ``__post_init__`` traversal even when the caller only tweaked
        a single bool. Cast pipelines call ``copy`` repeatedly per
        batched write (``with_source`` / ``check_source`` /
        ``with_target`` all funnel through here), so the fast clone
        below — :func:`object.__new__` + slot copy + targeted
        ``__post_init__`` normalization for the overridden keys — is
        meaningfully cheaper.
        """
        cls = type(self)
        allowed = cls.field_names()
        clean = {
            k: v
            for k, v in overrides.items()
            if v is not ... and k in allowed
        }
        return self._fast_clone(clean)

    def _fast_clone(self: T, overrides: Mapping[str, Any]) -> T:
        """Build a fresh instance by copying slots, applying *overrides*.

        Mirrors what :func:`dataclasses.replace` would do for a frozen
        slotted dataclass, minus the kwargs-construction overhead.

        Semantics preserved:

        - Every init-eligible field on *self* is copied to the new
          instance, then *overrides* take precedence.
        - ``__post_init__`` runs to normalize Field/Mode-shaped inputs
          *if* the override set touches any of those slots. When the
          overrides are all already-normalized primitives (the typical
          ``copy(safe=True)`` / ``copy(row_size=1024)`` shape), the
          normalization pass is skipped — the self-copied values are
          already normalized from *self*'s own ``__post_init__``.
        - Memoization slots (``_merged_cache`` etc.) are reset
          on the new instance — :func:`dataclasses.replace` already
          did this implicitly via the ``init=False`` default, and we
          keep the same invariant.
        """
        cls = type(self)
        new = object.__new__(cls)
        setattr_ = object.__setattr__
        names = cls._init_field_names_tuple()
        # Specialize on whether there are overrides at all — the empty
        # case is the hottest (every ``with_source(copy=True)`` /
        # ``check_source(copy=True)`` path lands here) and the dict-
        # membership check per field is pure waste when the dict is
        # empty.
        if overrides:
            for fname in names:
                if fname in overrides:
                    setattr_(new, fname, overrides[fname])
                else:
                    setattr_(new, fname, getattr(self, fname))
        else:
            for fname in names:
                setattr_(new, fname, getattr(self, fname))
        # Init=False memoization slot: always start cleared on the
        # clone — overridden source / target / schema_mode would
        # otherwise read a stale cached merge.
        setattr_(new, "_merged_cache", ...)
        # Only re-run normalization when an override could plausibly
        # need it. Slots not touched here were already normalized on
        # *self* and just propagated via getattr.
        if overrides and _NORMALIZED_KEYS.intersection(overrides):
            new.__post_init__()
        return new

    def check_source(
        self: T,
        obj: Any = None,
        *,
        copy: bool = True,
    ) -> T:
        """Bind a :attr:`source` if one isn't already set.

        Two ways to supply one:

        1. ``source=`` on :meth:`check` / :meth:`copy` — explicit
           Field / Schema / pa type. Wins even if ``self.source`` is
           already set (explicit override).
        2. ``obj=`` here — a peekable object. Only runs the peek
           when ``self.source`` is currently ``None`` — an already-
           bound field is never clobbered by a peek.

        Returns self unchanged when neither is given. Used from
        :class:`DataIO` methods (``collect_schema``, ``read_arrow_dataset``)
        that want to pin a source schema before running a batch walk.

        ``checked_cast=True`` short-circuits — the caller guarantees
        the batch shape matches the target, so the peek (which would
        rebuild a yggdrasil :class:`Field` from the batch's
        :class:`pa.Schema`) is wasted work. Combined with the
        :meth:`cast_arrow_tabular` short-circuit, this collapses every
        per-batch cast pass to a single attribute read on the leaf
        write path — ~150 us / batch saved on a RESPONSE_SCHEMA-shaped
        write.
        """
        if self.checked_cast:
            return self
        if self.source is None and obj is not None:
            try:
                peeked = obj() if callable(obj) else field_class().from_(obj)
            except (TypeError, ValueError):
                # Some inputs can't be peeked into a Field — e.g. an
                # unbound pyspark ``Column`` carries no usable dtype
                # (``df["x"]`` resolves through the DataFrame schema,
                # not the Column object). Treat those as "no source"
                # rather than crashing the cast pipeline.
                return self
            return self.with_source(peeked, copy=copy)
        return self

    def check_target(
        self: T,
        obj: Any = ...,
        *,
        copy: bool = True
    ) -> T:
        """Bind a :attr:`target` if one isn't already set.

        Symmetry partner for :meth:`check_source`. See that method
        for the argument semantics — source/target behave identically.
        """
        if self.target is None and obj is not None:
            return self.with_target(
                obj() if callable(obj) else field_class().from_(obj),
                copy=copy
            )
        return self

    def with_source(self: T, source: "Field", copy: bool = False) -> T:
        """Return a copy with *source* as the new source field.

        Accepts the same shapes :meth:`Field.from_` does (pa schema,
        yggdrasil Field, dict, etc.) — normalized in ``__post_init__``
        via :func:`dataclasses.replace`. The frozen slot is updated
        through :func:`object.__setattr__` in the post-init hook; we
        don't bypass it here because going through ``replace`` gets
        the normalization for free.
        """
        setattr_ = object.__setattr__
        if source is None:
            if copy:
                return self.copy(source=None)
            setattr_(self, "source", None)
            setattr_(self, "_merged_cache", ...)
            return self

        source = field_class().from_(source)

        if source == self.source:
            return self

        if copy:
            # ``replace`` drops init=False fields back to their
            # defaults, which clears the merged cache for free.
            return dataclasses.replace(self, source=source)
        setattr_(self, "source", source)
        # In-place edit invalidates whatever the merged
        # cache held.
        setattr_(self, "_merged_cache", ...)
        return self

    def with_target(self: T, target: "Field", copy: bool = True) -> T:
        """Return a copy with *target* as the new target field."""
        setattr_ = object.__setattr__
        if target is None:
            if copy:
                return self.copy(target=None)
            setattr_(self, "target", None)
            setattr_(self, "_merged_cache", ...)
            return self

        target = field_class().from_(target)

        if target == self.target:
            return self

        if copy:
            return dataclasses.replace(self, target=target)
        setattr_(self, "target", target)
        setattr_(self, "_merged_cache", ...)
        return self

    def with_checked_cast(self: T, value: bool = True, copy: bool = False) -> T:
        """Return a copy (or in-place) with :attr:`checked_cast` set.

        Mirror of :meth:`with_source` / :meth:`with_target` — keeps
        the per-call mutation behind a named method instead of having
        every writer-side caller reach for
        :func:`dataclasses.replace` / :func:`object.__setattr__`. Set
        when the caller knows every batch already matches the target
        (came from a :class:`pa.Table`, a :class:`pa.RecordBatchReader`,
        a polars / pandas frame, or another writer that just emitted
        the same schema); the leaf's :meth:`check_source` /
        :meth:`cast_arrow_tabular` then short-circuit straight to the
        write path.
        """
        if bool(value) == self.checked_cast:
            return self
        if copy:
            return dataclasses.replace(self, checked_cast=bool(value))
        object.__setattr__(self, "checked_cast", bool(value))
        return self

    # ==================================================================
    # Cast-need inspection
    # ==================================================================

    def need_cast(
        self,
        source: Any | None = None,
        target: Any | None = None,
        check_names=False, check_dtypes=True, check_metadata=False,
        check_nullable: bool = False,
    ) -> bool:
        """Return ``True`` if source and target fields differ enough to need casting.

        When either field is unbound, returns ``False`` — there's
        nothing to compare against, so assume caller already sorted it.

        Field equality semantics are the :meth:`Field.equals` rules:
        names, dtypes, metadata — each independently gateable.
        Metadata is off by default because it's commonly decorative
        (pandas preserves indices through metadata, arrow carries codec
        hints in field metadata) and comparing on it would demand a
        cast for cosmetic differences.

        ``check_nullable`` is off by default because nullability rarely
        warrants a real value-level cast — primitives and lists pass
        through unchanged when only the flag differs. Tabular / struct
        casts pass ``check_nullable=True`` so the rebuild fires when
        child fields differ on nullability: Spark / Delta refuse to
        implicitly cast nullable→``NOT NULL`` inside a struct (even
        when the data is in fact non-null), so the cast has to emit
        the target's field types verbatim to keep MERGE happy.
        """
        # Fast path: no args means "use already-bound source / target"
        # — skip the two ``check_*`` method-call hops that otherwise
        # fire on every cast site.
        if source is None and target is None:
            src = self.source
            tgt = self.target
        else:
            clean = self.check_source(source, copy=False).check_target(target, copy=False)
            src = clean.source
            tgt = clean.target
        if src is None or tgt is None:
            return False
        if src is tgt:
            return False

        return not src.equals(
            tgt,
            check_names=check_names,
            check_dtypes=check_dtypes,
            check_metadata=check_metadata,
            check_nullable=check_nullable,
        )

    def finalize(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Finalize any object — delegates to :meth:`Field.finalize`."""
        if self.target is None:
            return obj
        return self.target.finalize(obj, default_scalar=default_scalar)

    def finalize_spark_cast(
        self,
        obj: Any,
        *,
        default_scalar: Any = None,
    ) -> Any:
        """Fill nulls and alias a Spark Column to the target name.

        Direct parallel of :meth:`finalize_polars_cast` — Spark
        Columns, like polars Series/Expr, carry a name that can
        diverge from the target after a cast chain, so the alias
        step belongs in finalize rather than in each cast site.
        """
        if self.target is None:
            return obj
        filled = self.fill_spark_nulls(obj, default_scalar=default_scalar)
        return self.spark_alias(filled)

    def finalize_arrow_cast(
        self,
        obj: Any,
        *,
        default_scalar: Any = None,
    ) -> Any:
        """Fill nulls on a pyarrow object to finish a cast chain.

        No alias step: :class:`pa.Array` / :class:`pa.ChunkedArray`
        don't carry a name, and tabular rename (Table/RecordBatch) is
        a schema-level rebuild that :meth:`cast_arrow_tabular`
        already handles inline via the target schema. Finalize here
        just means "apply the default-scalar null fill."
        """
        if self.target is None:
            return obj
        return self.fill_arrow_nulls(obj, default_scalar=default_scalar)

    def finalize_pandas_cast(
        self,
        obj: Any,
        *,
        default_scalar: Any = None,
    ) -> Any:
        """Fill nulls on a pandas object to finish a cast chain.

        No alias step exposed on :class:`CastOptions` for pandas —
        Series ``.name`` and DataFrame column labels get set by the
        cast methods directly. Finalize is fill-only, matching
        :meth:`finalize_arrow_cast`.
        """
        if self.target is None:
            return obj
        return self.fill_pandas_nulls(obj, default_scalar=default_scalar)

    # ==================================================================
    # Engine dispatch — delegate to Field, which owns the dispatch table
    # ==================================================================
    #
    # :class:`Field` now carries the full engine + shape dispatch tree
    # (``Field.cast`` → ``cast_arrow``/``cast_polars``/... → narrow
    # ``cast_*_tabular``/``cast_*_series``/...). :class:`CastOptions`
    # wraps those methods with:
    #
    # 1. The ``target is None`` short-circuit — valid on options
    #    but not on a Field (which always has a dtype).
    # 2. Ergonomic ``options=self`` plumbing — so callers of the wider
    #    DataIO surface don't have to thread options through every
    #    cast site.
    #
    # Single source of truth for engine dispatch = one place to update
    # when a new engine lands. When ``target`` is bound, every
    # method in this section is one-line-delegate.

    # ---- Top-level dispatch ------------------------------------------

    def cast(self, obj: Any) -> Any:
        """Cast *obj* to :attr:`target` using its native engine.

        Short-circuits to *obj* unchanged when ``target is None``.
        Otherwise delegates to :meth:`Field.cast`, which handles engine
        detection (pyarrow/polars/pandas/spark + iterable fallback) and
        shape dispatch (tabular vs array/series/column/expr). Source-side
        metadata still flows through via ``options=self`` for the inner
        cast paths that need it.
        """
        if self.target is None:
            return obj
        return self.target.cast(obj, options=self)

    # ---- pyarrow -----------------------------------------------------

    def cast_pyarrow(self, obj: Any) -> Any:
        """Cast any pyarrow object — delegates to :meth:`Field.cast_arrow`."""
        if self.target is None:
            return obj
        return self.target.cast_arrow(obj, options=self)

    def cast_arrow_array(self, array: Any) -> Any:
        """Cast a :class:`pa.Array` or :class:`pa.ChunkedArray`."""
        if self.target is None:
            return array
        return self.target.cast_arrow_array(array, options=self)

    def cast_arrow_tabular(self, table: Any) -> Any:
        """Cast a :class:`pa.Table` or :class:`pa.RecordBatch`.

        ``checked_cast=True`` short-circuits to the input unchanged —
        the caller guarantees every batch already matches the target,
        so the per-batch cast pass (and the schema rebuild upstream
        in :meth:`check_source`) is wasted work. Use only when you
        control the source.
        """
        if self.target is None or self.checked_cast:
            return table
        return self.target.cast_arrow_tabular(table, options=self)

    def dedup_columns_on_read(self) -> "list[str]":
        """Return the column names that need client-side dedup at read time.

        Sourced from :attr:`unique_by` — each Field's
        :attr:`Field.name` is the column the read pass must
        deduplicate on. Returns an empty list when :attr:`unique_by`
        is unset / empty.
        """
        if not self.unique_by:
            return []
        return [f.name for f in self.unique_by]

    def dedup_arrow_batches(
        self, batches: "Iterator[pa.RecordBatch]",
    ) -> "Iterator[pa.RecordBatch]":
        """Collapse duplicate rows on the columns flagged ``unique``.

        Resolves the dedup column set via
        :meth:`dedup_columns_on_read`, then delegates to
        :func:`yggdrasil.arrow.ops.dedup_arrow_batches` for the
        pure-Arrow group-by + take pass. Identity short-circuit when
        no column needs collapsing keeps the read path zero-cost on
        the common case (no target / no unique column / source
        already unique).
        """
        cols = self.dedup_columns_on_read()
        if not cols:
            yield from batches
            return
        # Local import to avoid the top-of-module cycle (arrow.ops
        # imports back into ``yggdrasil.data`` for the cast helpers).
        from yggdrasil.arrow.ops import dedup_arrow_batches as _dedup
        yield from _dedup(batches, cols)

    def resample_on_read(self) -> "tuple[str, int, list[str], str] | None":
        """Return ``(time_column, sampling_seconds, partition_by, fill_strategy)`` to resample.

        Picks the first entry of :attr:`time_sample_by` whose
        ``time_sampling`` metadata carries a positive ISO-8601
        duration. The result drives
        :func:`yggdrasil.arrow.ops.resample_arrow_table` — a single
        (column, interval) is all that op consumes (you can only
        have one time axis per table to resample on at a time).

        Each Field's sampling lives under its
        :attr:`Field.metadata`'s non-prefixed ``b"time_sampling"``
        key as an ISO-8601 duration string (``"PT1H"`` / ``"P1D"``).
        The non-prefixed key keeps the value off the schema-level
        tag registry (it's a per-call option, not a contract that
        rides with the data on disk).

        ``partition_by`` is derived from the target schema's
        :attr:`Field.primary_key` set, *minus* the resample column
        itself if it's also primary. The rationale: on a per-entity
        time series (one symbol per row, partitioned by symbol),
        each entity's timeline should bucket independently — without
        ``partition_by`` the resample would collapse rows across
        instruments. Schemas with no primary keys (or where the
        only primary is the timestamp) fall back to a flat resample.

        Returns ``None`` when :attr:`time_sample_by` is unset /
        empty or every listed Field's metadata fails to parse.
        """
        if not self.time_sample_by:
            return None
        for child in self.time_sample_by:
            seconds = _field_time_sampling_seconds(child)
            if seconds <= 0:
                continue
            partition_by = self._resample_partition_by(child.name)
            return child.name, seconds, partition_by, self.fill_strategy
        return None

    def _resample_partition_by(self, time_column: str) -> "list[str]":
        """Default ``partition_by`` for :meth:`resample_on_read`.

        Walks the target schema's children for ``primary_key=True``
        fields (excluding ``time_column`` itself). Returns ``[]`` when
        no target / no primary keys are declared, or when the only
        primary is the resample column.
        """
        target = self.target
        if target is None:
            return []
        children = getattr(target, "children", None)
        if not children:
            return []
        out: list[str] = []
        for child in children:
            if not getattr(child, "primary_key", False):
                continue
            if child.name == time_column:
                continue
            out.append(child.name)
        return out

    def resample_arrow_batches(
        self, batches: "Iterator[pa.RecordBatch]",
    ) -> "Iterator[pa.RecordBatch]":
        """Snap rows to the target's ``time_sampling`` grid.

        Resolves the resample column / interval / partition keys via
        :meth:`resample_on_read`, then delegates to
        :func:`yggdrasil.arrow.ops.resample_arrow_batches`. Identity
        short-circuit when no field is flagged keeps the read path
        zero-cost on the common case.
        """
        target = self.resample_on_read()
        if target is None:
            yield from batches
            return
        from yggdrasil.arrow.ops import resample_arrow_batches as _resample
        time_column, sampling_seconds, partition_by, fill_strategy = target
        yield from _resample(
            batches,
            time_column=time_column,
            sampling_seconds=sampling_seconds,
            partition_by=partition_by or None,
            fill_strategy=fill_strategy,
        )

    # ------------------------------------------------------------------
    # Post-read passes on a *materialised* table
    #
    # The streaming wraps above are the right shape for iterator-driven
    # read paths (``read_arrow_batches`` / ``read_arrow_batch_reader``).
    # When the caller has already built (or will immediately build) a
    # single :class:`pa.Table` — :meth:`Tabular._read_arrow_table`,
    # :meth:`read_polars_frame`, :meth:`read_pandas_frame`, the write
    # side's ``cast_arrow_tabular`` entry point — applying the resample
    # / dedup directly on the Table skips the materialise-then-rebatch
    # dance the iterator wraps do internally (each op rebuilds a
    # ``Table.from_batches`` to run group_by, then re-batches on the
    # way out, only for the outer caller to ``Table.from_batches``
    # them right back).
    # ------------------------------------------------------------------

    def apply_post_read_table(self, table: "pa.Table") -> "pa.Table":
        """Run column projection + resample + dedup on a materialised :class:`pa.Table`.

        Same operations and same order as the streaming wraps —
        column projection first (trim I/O cost before any compute),
        resample second (its bucket collapse trims rows before the
        unique-tag walk), then dedup. Identity short-circuit when
        no pass is configured so the common case stays zero-cost.

        Pyarrow / polars / pandas read paths that already produce a
        Table funnel through this method instead of the iterator
        wraps; the result is one ``Table.from_batches`` + one
        ``Table.take`` (per pass) instead of two ``Table.from_batches``
        + a ``Table.to_batches`` rebatch sandwich.
        """
        columns = self.column_names
        if columns:
            available = table.column_names
            select = [c for c in columns if c in available]
            if select and len(select) < len(available):
                table = table.select(select)

        resample = self.resample_on_read()
        unique = self.dedup_columns_on_read()
        if resample is None and not unique:
            return table
        if resample is not None:
            from yggdrasil.arrow.ops import resample_arrow_table as _resample
            time_column, sampling_seconds, partition_by, fill_strategy = resample
            table = _resample(
                table,
                time_column=time_column,
                sampling_seconds=sampling_seconds,
                partition_by=partition_by or None,
                fill_strategy=fill_strategy,
            )
        if unique:
            from yggdrasil.arrow.ops import dedup_arrow_table as _dedup
            table = _dedup(table, unique)
        return table

    def apply_post_read_spark_frame(self, df: Any) -> Any:
        """Run resample + dedup directly on a Spark DataFrame.

        Spark-side mirror of :meth:`apply_post_read_table` — same
        op order (resample first, dedup second), same identity
        short-circuit when neither is configured. Routes through
        :mod:`yggdrasil.spark.ops` so the heavy lifting stays on
        the executors (``groupBy + applyInArrow`` for the
        partitioned resample, SQL window functions otherwise)
        instead of collecting the frame to the driver as Arrow.

        Used by :meth:`yggdrasil.spark.tabular.Dataset._read_spark_frame`
        to apply the read-time passes before handing the frame back
        — saving a full ``df.toArrow → arrow.ops → createDataFrame``
        round trip per configured op.
        """
        resample = self.resample_on_read()
        unique = self.dedup_columns_on_read()
        if resample is None and not unique:
            return df
        if resample is not None:
            from yggdrasil.spark.ops import resample_spark_dataframe as _resample
            time_column, sampling_seconds, partition_by, fill_strategy = resample
            df = _resample(
                df,
                time_column=time_column,
                sampling_seconds=sampling_seconds,
                partition_by=partition_by or None,
                fill_strategy=fill_strategy,
            )
        if unique:
            from yggdrasil.spark.ops import dedup_spark_dataframe as _dedup
            df = _dedup(df, unique)
        return df

    def cast_arrow_batch_iterator(self, batches: Any) -> Any:
        """Cast a stream of :class:`pa.RecordBatch` and rechunk by ``byte_size`` / ``row_size``.

        With a bound ``target``: per-batch tabular cast + streamed
        rechunking via :meth:`Field.cast_arrow_batch_iterator` (which
        routes through the struct-side helper).

        Without a target: rechunk-only when ``byte_size`` / ``row_size``
        is set, otherwise passthrough. Lets callers that did an
        in-engine cast upstream still pick up the optimized rechunker.
        """
        if self.target is None:
            if not self.byte_size and not self.row_size:
                return batches
            from yggdrasil.arrow.cast import rechunk_arrow_batches
            return rechunk_arrow_batches(
                batches,
                byte_size=self.byte_size,
                row_size=self.row_size,
                memory_pool=self.arrow_memory_pool,
            )
        return self.target.cast_arrow_batch_iterator(batches, options=self)

    def fill_arrow_nulls(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Engine-level null fill — delegates to :meth:`Field.fill_arrow`."""
        if self.target is None:
            return obj
        return self.target.fill_arrow(obj, default_scalar=default_scalar)

    def fill_arrow_array_nulls(self, array: Any, *, default_scalar: Any = None) -> Any:
        """Narrow null fill for a :class:`pa.Array` / :class:`pa.ChunkedArray`."""
        if self.target is None:
            return array
        return self.target.fill_arrow_array_nulls(
            array, default_scalar=default_scalar
        )

    # ---- polars ------------------------------------------------------

    def cast_polars(self, obj: Any) -> Any:
        """Cast any polars object — delegates to :meth:`Field.cast_polars`."""
        if self.target is None:
            return obj
        return self.target.cast_polars(obj, options=self)

    def cast_polars_series(self, series: Any, *, default_scalar: Any = None) -> Any:
        """Cast a :class:`pl.Series`."""
        if self.target is None:
            return series
        return self.target.cast_polars_series(
            series, options=self, default_scalar=default_scalar
        )

    def cast_polars_expr(self, expr: Any, *, default_scalar: Any = None) -> Any:
        """Cast a :class:`pl.Expr`.

        Wraps the expression tree with a cast operator — actual work
        fires when the containing LazyFrame is collected.
        """
        if self.target is None:
            return expr
        return self.target.cast_polars_expr(
            expr, options=self, default_scalar=default_scalar
        )

    def cast_polars_tabular(self, data: Any) -> Any:
        """Cast a :class:`pl.DataFrame` or :class:`pl.LazyFrame`."""
        if self.target is None:
            return data
        return self.target.cast_polars_tabular(data, options=self)

    def fill_polars_nulls(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Engine-level polars null fill — delegates to :meth:`Field.fill_polars`."""
        if self.target is None:
            return obj
        return self.target.fill_polars(obj, default_scalar=default_scalar)

    def polars_alias(self, obj: Any) -> Any:
        """Rename a polars Series/Expr to the target name — no-op if matching.

        Delegates to :meth:`Field.polars_alias`. When ``target``
        is unbound there's no name to rename to, so we pass through.
        """
        if self.target is None:
            return obj
        return self.target.polars_alias(obj)

    # ---- pandas ------------------------------------------------------

    def cast_pandas(self, obj: Any) -> Any:
        """Cast any pandas object — delegates to :meth:`Field.cast_pandas`."""
        if self.target is None:
            return obj
        return self.target.cast_pandas(obj, options=self)

    def fill_pandas_nulls(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Engine-level pandas null fill — delegates to :meth:`Field.fill_pandas`."""
        if self.target is None:
            return obj
        return self.target.fill_pandas(obj, default_scalar=default_scalar)

    # ---- spark -------------------------------------------------------

    def cast_spark(self, obj: Any) -> Any:
        """Cast any spark object — delegates to :meth:`Field.cast_spark`."""
        if self.target is None:
            return obj
        return self.target.cast_spark(obj, options=self)

    def cast_spark_tabular(self, df: Any) -> Any:
        """Filter + cast a Spark DataFrame.

        Applies :attr:`predicate` (when set) then delegates to
        :meth:`Field.cast_spark_tabular` for schema coercion.
        """
        if self.predicate is not None:
            df = self.predicate.filter_spark_frame(df)
        if self.target is None:
            return df
        return self.target.cast_spark_tabular(df, options=self)

    def cast_spark_column(self, obj: Any) -> Any:
        """Cast any spark object — delegates to :meth:`Field.cast_spark`."""
        if self.target is None:
            return obj
        return self.target.cast_spark_column(obj, options=self)

    def fill_spark_nulls(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Engine-level spark null fill — delegates to :meth:`Field.fill_spark`."""
        if self.target is None:
            return obj
        return self.target.fill_spark(obj, default_scalar=default_scalar)

    def spark_alias(self, obj: Any) -> Any:
        """Rename a Spark Column to the target name — delegates to :meth:`Field.spark_alias`."""
        if self.target is None:
            return obj
        return self.target.spark_alias(obj)

    def get_spark_session(self, create: bool = False, **kwargs):
        return PyEnv.spark_session(
            self.spark_session,
            create=create,
            **kwargs
        )

    # ==================================================================
    # Dunders
    # ==================================================================

    @classmethod
    @functools.cache
    def _repr_field_meta(cls) -> tuple[tuple[str, Any], ...]:
        """Cached ``(name, default)`` pairs for repr-eligible fields.

        Walking :func:`dataclasses.fields` on every ``repr`` call is
        pure overhead — the (name, default) signature is fixed at
        class-definition time. Cached per-class so subclasses with
        extra fields get their own tuple on first access.
        """
        return tuple(
            (f.name, f.default)
            for f in dataclasses.fields(cls)
            if f.repr
        )

    def __repr__(self) -> str:
        """Compact repr — skip fields at their default values.

        The full dataclass repr is a wall of text when most fields are
        defaults (very common in practice). Showing only set fields
        keeps exception messages and debug logs readable.
        """
        parts: list[str] = []
        cls_name = type(self).__name__
        for name, default in type(self)._repr_field_meta():
            value = getattr(self, name)
            if default is dataclasses.MISSING:
                parts.append(f"{name}={value!r}")
                continue
            # Mode normalization makes mode/schema_mode non-None
            # even when default was None → compare against AUTO too.
            if value is default:
                # Pointer-equal short-circuit: most defaults (Mode enums,
                # None, False, 0) end up identity-equal after
                # ``__post_init__``; saves a ``==`` dispatch on the hot
                # default-only repr path.
                continue
            if name in ("mode", "schema_mode") and value is Mode.AUTO:
                continue
            if value == default:
                continue
            parts.append(f"{name}={value!r}")
        return f"{cls_name}({', '.join(parts)})"