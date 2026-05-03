"""`CastOptions` — the one options object every :class:`DataIO` takes.

What it carries
---------------

Two :class:`Field` s (``source_field``, ``target_field``) — the
inferred / desired schema at each end of a cast. A ``safe`` flag for
strict-vs-permissive semantics (overflow, truncation, nulls-in-non-
nullable). Sizing knobs ``row_size`` / ``byte_size`` that batch-
oriented readers/writers honour. A :class:`Mode` pair for write
semantics: ``mode`` controls the data write (overwrite / append /
error-if-exists), ``schema_mode`` controls how schema drift is
handled. An optional ``arrow_memory_pool`` for callers routing
allocations through a bounded pool.

The canonical entry point
-------------------------

:meth:`CastOptions.check` is what every :class:`DataIO` public method
funnels through. It accepts any of: an existing CastOptions to reuse,
a dict of overrides, a :class:`pa.DataType` / :class:`pa.Field` /
:class:`pa.Schema` to promote to a target-field hint, or ``None`` for
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
:class:`Field`. Stops ``CastOptions(target_field=pa_schema)`` from
propagating an un-wrapped pyarrow object into the casting engines.

Engine dispatch
---------------

:meth:`cast` and the per-engine / per-shape variants all delegate to
the matching :class:`Field` methods — :class:`Field` owns the
dispatch table (engine detection via :meth:`ObjectSerde.module_and_name`,
shape detection via isinstance under the engine's lazy-imported
module). :class:`CastOptions` adds exactly two things on top:

1. The ``target_field is None`` short-circuit — valid on options but
   not on a Field, which always carries a dtype. Lets
   :meth:`CastOptions.cast` pass *obj* through unchanged when no
   target is bound.
2. ``options=self`` plumbing — callers threading options through a
   long pipeline don't have to re-specify at each cast site.

Single source of truth for the dispatch table: adding a new engine
is a one-site edit in :class:`Field`.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Mapping, TypeVar, Union, TYPE_CHECKING

import pyarrow as pa
from yggdrasil.data.expr import Predicate

from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io.enums import Mode
from yggdrasil.lazy_imports import field_class, schema_class

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from yggdrasil.data.data_field import Field
    from yggdrasil.data.schema import Schema

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


# ---------------------------------------------------------------------------
# Type alias for "anything that can be coerced into a CastOptions"
# ---------------------------------------------------------------------------

# Accept at the public boundary: an existing CastOptions, a dict of
# field overrides, a pa.DataType/Field/Schema that we promote to
# target_field, or None for defaults. Kept intentionally broad —
# .check() handles the routing.
CastOptionsArg = Union[
    "CastOptions",
    Mapping[str, Any],
    pa.DataType,
    pa.Field,
    pa.Schema,
    None,
]


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

    * ``source_field`` / ``target_field`` = ``None`` → no cast
      coercion. :meth:`cast` returns inputs unchanged.
    * ``safe`` = ``True`` → strict overflow/truncation/null semantics.
    * ``mode`` / ``schema_mode`` = :attr:`Mode.AUTO` → writer
      picks the appropriate behaviour from context.
    * ``row_size`` / ``byte_size`` = ``None`` → no batch caps; readers
      stream whatever size is natural for the format.
    * ``arrow_memory_pool`` = ``None`` → use pyarrow's default pool.
    """

    source_field: "Field | None" = None
    target_field: "Field | None" = None
    safe: bool = True
    mode: Mode = Mode.AUTO
    schema_mode: Mode = Mode.IGNORE
    row_size: int | None = None
    byte_size: int | None = None
    recursive: bool = False
    match_by_names: list[str] | None = None
    with_io: bool = True
    seek_source: bool = False
    reset_seek: bool = False
    read_seek: int | None = None
    write_seek: int | None = None
    where: Predicate | None = None
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

    # ==================================================================
    # Normalization — runs after dataclass init
    # ==================================================================

    def __post_init__(self) -> None:
        """Coerce field-shaped inputs to :class:`Field`, mode inputs to
        :class:`Mode`.

        Frozen dataclasses can't use normal attribute assignment in
        ``__post_init__`` — we go through :func:`object.__setattr__`
        directly, which is the standard escape hatch for this exact
        case.
        """
        # Field normalization — accept pa.Schema/Field/DataType, dict,
        # yggdrasil Field/Schema, or None. Field.from_(None) returns
        # None so the no-cast case passes through cleanly.
        Field = field_class()
        if self.source_field is not None and not isinstance(self.source_field, Field):
            object.__setattr__(self, "source_field", Field.from_(self.source_field))
        if self.target_field is not None and not isinstance(self.target_field, Field):
            object.__setattr__(self, "target_field", Field.from_(self.target_field))

        # Mode normalization — accept Mode, string ("overwrite"),
        # or None (→ AUTO). Keeping None on the way in as a distinct
        # state from AUTO would invite "did I forget to set this?"
        # confusion downstream.
        object.__setattr__(self, "mode", Mode.from_(self.mode, default=Mode.AUTO))
        object.__setattr__(self, "schema_mode", Mode.from_(self.schema_mode, default=Mode.IGNORE))

    # ==================================================================
    # Derived properties
    # ==================================================================

    @property
    def source_schema(self) -> Schema | None:
        """The source field's :class:`Schema`, if a source field is bound.

        Returns ``None`` when no source is bound — callers that need a
        non-None schema should use :meth:`check_source` first to bind
        one from a peekable object.
        """
        return self.source_field.to_schema() if self.source_field is not None else None

    @property
    def target_schema(self) -> Schema | None:
        """The target field's :class:`Schema`, if a target field is bound."""
        return self.target_field.to_schema() if self.target_field is not None else None

    @property
    def merged_field(self) -> Field | None:
        if self.source_field and self.target_field:
            return self.target_field.merge_with(self.source_field, mode=self.schema_mode)
        return self.target_field or self.source_field

    @property
    def merged_schema(self) -> Schema | None:
        if self.source_field and self.target_field:
            return self.target_schema.merge_with(self.source_schema, mode=self.schema_mode)
        return self.target_schema or self.source_schema

    def select_source_column_names(self) -> list[str] | None:
        """The source field's column names, if a source field is bound."""
        if self.source_field is None:
            return None

        source_names = list(self.source_field.names)

        if self.target_field is None:
            return source_names

        selected: list[str] = []

        for target_name in self.target_field.names:
            found = self.source_field.field(name=target_name, raise_error=False)
            if found is not None:
                selected.append(found.name)

        return selected or None

    @property
    def column_names(self) -> list[str] | None:
        """The target field's column names, if a target field is bound."""
        merged = self.merged_field

        if merged is None:
            return None

        return merged.names

    # ==================================================================
    # Construction / merge entry point
    # ==================================================================

    @classmethod
    def check(
        cls: type[T],
        options: CastOptionsArg = None,
        /,
        *,
        source: Any = ...,
        target: Any = ...,
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
          / :class:`Field` / :class:`Schema` — treat as a target-field
          hint. Equivalent to ``check(target_field=options, **overrides)``.

        The ``source=`` / ``target=`` keyword-only shortcuts are for
        peek-and-bind use: ``check(target=some_table)`` runs
        :meth:`Field.peek_from` to infer a target field from the
        table's shape. ``source_field=`` / ``target_field=`` (via
        ``**overrides``) remain the explicit form.

        :raises TypeError: if *options* is a type the dispatch table
            doesn't cover.
        """
        # 1. None / ... → fresh construction.
        if options is None or options is ...:
            return cls._build(overrides, source=source, target=target)

        # 2. Already a CastOptions — reuse via copy if overrides
        # given, otherwise passthrough. Typed check with ``cls`` so a
        # subclass CastOptions stays on its subclass.
        if isinstance(options, cls):
            if not overrides and source is ... and target is ... and not overrides:
                return options
            return options.copy(source=source, target=target, **overrides)
        if isinstance(options, CastOptions):
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
            merged.update(overrides)
            return cls._build(merged, source=source, target=target)

        # 3. Mapping → merge into overrides (explicit kwargs win).
        if isinstance(options, Mapping):
            merged = {**options, **overrides}
            return cls._build(merged, source=source, target=target)

        # 4. Schema-shaped → promote to a target_field hint.
        if isinstance(options, (pa.DataType, pa.Field, pa.Schema, field_class(), schema_class())):
            overrides.setdefault("target_field", options)
            return cls._build(overrides, source=source, target=target)

        raise TypeError(
            f"CastOptions.check cannot coerce {type(options).__name__}. "
            "Expected CastOptions, dict, pa.DataType/Field/Schema, "
            "yggdrasil Field/Schema, or None."
        )

    @classmethod
    @functools.cache
    def field_names(cls) -> frozenset[str]:
        """Frozenset of this class's dataclass field names.

        Used by :meth:`_build` to filter ``**overrides`` down to keys
        the constructor will accept — callers funnel mixed kwargs
        through ``.check()`` (DataIO public methods often pass user
        kwargs straight through), and we don't want a stray
        ``filter=`` or ``columns=`` to crash construction.

        Cached per-class via :func:`functools.cache` so subclasses
        with extra fields get their own expanded set on first access.
        """
        return frozenset(f.name for f in dataclasses.fields(cls))

    @classmethod
    def _build(
        cls: type[T],
        overrides: Mapping[str, Any],
        *,
        source: Any = ...,
        target: Any = ...,
    ) -> T:
        overrides = overrides or {}
        options = overrides.pop("options", None)
        clean = {
            k: v
            for k, v in overrides.items()
            if v is not ... and k in cls.field_names()
        }
        # Drop foreign keys — DataIO public methods pass mixed kwargs
        # through .check(), and only CastOptions fields are valid for
        # construction. Foreign keys are the caller's concern (filter
        # predicates, format-specific knobs) and should be ignored
        # here, not raised on.
        allowed = cls.field_names()
        clean = {k: v for k, v in clean.items() if k in allowed}
        if options is None:
            instance = cls(**clean)
        else:
            instance = options.copy(**clean)
        if not source is ...:
            instance = instance.check_source(obj=source)
        if not target is ...:
            instance = instance.check_target(obj=target)
        return instance

    # ==================================================================
    # Copy / mutation helpers
    # ==================================================================

    def copy(
        self: T,
        /,
        *,
        source: Any = ...,
        target: Any = ...,
        **overrides: Any,
    ) -> T:
        """Return a copy with *overrides* applied.

        ... values in *overrides* are ignored (keep existing).
        ``source``/``target`` run a peek-and-bind after the base copy
        so they compose with explicit ``source_field``/``target_field``
        overrides — the explicit always wins, peek only fires when
        the override left the slot unbound.
        """
        clean = {
            k: v
            for k, v in overrides.items()
            if v is not ... and k in self.field_names()
        }
        replaced = dataclasses.replace(self, **clean)
        if source is not ... and source is not None:
            replaced = replaced.with_source(source)
        if target is not ... and target is not None:
            replaced = replaced.with_target(target)
        return replaced

    def check_source(
        self: T,
        obj: Any = None,
        *,
        copy: bool = True,
    ) -> T:
        """Bind a ``source_field`` if one isn't already set.

        Two ways to supply one:

        1. ``source_field=`` — explicit Field / Schema / pa type. Wins
           even if ``self.source_field`` is already set (explicit
           override).
        2. ``obj=`` — a peekable object. Only runs :meth:`peek_source`
           when ``self.source_field`` is currently ``None`` — an
           already-bound field is never clobbered by a peek.

        Returns self unchanged when neither is given. Used from
        :class:`DataIO` methods (``collect_schema``, ``read_arrow_dataset``)
        that want to pin a source schema before running a batch walk.
        """
        if self.source_field is None and obj is not None:
            return self.with_source(
                obj() if callable(obj) else field_class().from_(obj),
                copy=copy
            )
        return self

    def check_target(
        self: T,
        obj: Any = ...,
        *,
        copy: bool = True
    ) -> T:
        """Bind a ``target_field`` if one isn't already set.

        Symmetry partner for :meth:`check_source`. See that method
        for the argument semantics — source/target behave identically.
        """
        if self.target_field is None and obj is not None:
            return self.with_target(
                obj() if callable(obj) else field_class().from_(obj),
                copy=copy
            )
        return self

    def with_source(self: T, source: "Field", copy: bool = False) -> T:
        """Return a copy with *field* as the new source field.

        Accepts the same shapes :meth:`Field.from_` does (pa schema,
        yggdrasil Field, dict, etc.) — normalized in ``__post_init__``
        via :func:`dataclasses.replace`. The frozen slot is updated
        through :func:`object.__setattr__` in the post-init hook; we
        don't bypass it here because going through ``replace`` gets
        the normalization for free.
        """
        source = field_class().from_(source)

        if source == self.source_field:
            return self

        if copy:
            return dataclasses.replace(self, source_field=source)
        object.__setattr__(self, "source_field", source)
        return self

    def with_target(self: T, target: "Field", copy: bool = True) -> T:
        """Return a copy with *field* as the new target field."""
        target = field_class().from_(target)

        if target == self.target_field:
            return self

        if copy:
            return dataclasses.replace(self, target_field=target)
        object.__setattr__(self, "target_field", target)
        return self

    # ==================================================================
    # Cast-need inspection
    # ==================================================================

    def need_cast(
        self,
        source: Any | None = None,
        target: Any | None = None,
        check_names=False, check_dtypes=True, check_metadata=False
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
        """
        clean = self.check_source(source, copy=False).check_target(target, copy=False)
        src = clean.source_field
        tgt = clean.target_field
        if src is None or tgt is None:
            return False

        return not src.equals(
            tgt,
            check_names=check_names,
            check_dtypes=check_dtypes,
            check_metadata=check_metadata,
            check_nullable=False
        )

    def finalize(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Finalize any object — delegates to :meth:`Field.finalize`."""
        if self.target_field is None:
            return obj
        return self.target_field.finalize(obj, default_scalar=default_scalar)

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
        if self.target_field is None:
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
        if self.target_field is None:
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
        if self.target_field is None:
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
    # 1. The ``target_field is None`` short-circuit — valid on options
    #    but not on a Field (which always has a dtype).
    # 2. Ergonomic ``options=self`` plumbing — so callers of the wider
    #    DataIO surface don't have to thread options through every
    #    cast site.
    #
    # Single source of truth for engine dispatch = one place to update
    # when a new engine lands. When ``target_field`` is bound, every
    # method in this section is one-line-delegate.

    # ---- Top-level dispatch ------------------------------------------

    def cast(self, obj: Any) -> Any:
        """Cast *obj* to :attr:`target_field` using its native engine.

        Short-circuits to *obj* unchanged when ``target_field is None``.
        Otherwise delegates to :meth:`Field.cast`, which handles engine
        detection (pyarrow/polars/pandas/spark + iterable fallback) and
        shape dispatch (tabular vs array/series/column/expr).
        """
        if self.target_field is None:
            return obj
        return self.merged_field.cast(obj, options=self)

    # ---- pyarrow -----------------------------------------------------

    def cast_pyarrow(self, obj: Any) -> Any:
        """Cast any pyarrow object — delegates to :meth:`Field.cast_arrow`."""
        if self.target_field is None:
            return obj
        return self.merged_field.cast_arrow(obj, options=self)

    def cast_arrow_array(self, array: Any) -> Any:
        """Cast a :class:`pa.Array` or :class:`pa.ChunkedArray`."""
        if self.target_field is None:
            return array
        return self.merged_field.cast_arrow_array(array, options=self)

    def cast_arrow_tabular(self, table: Any) -> Any:
        """Cast a :class:`pa.Table` or :class:`pa.RecordBatch`."""
        if self.target_field is None:
            return table
        return self.merged_field.cast_arrow_tabular(table, options=self)

    def cast_arrow_batch_iterator(self, batches: Any) -> Any:
        """Cast a stream of :class:`pa.RecordBatch` and rechunk by ``byte_size`` / ``row_size``.

        With a bound ``target_field``: per-batch tabular cast + streamed
        rechunking via :meth:`Field.cast_arrow_batch_iterator` (which
        routes through the struct-side helper).

        Without a target: rechunk-only when ``byte_size`` / ``row_size``
        is set, otherwise passthrough. Lets callers that did an
        in-engine cast upstream still pick up the optimized rechunker.
        """
        if self.target_field is None:
            if not self.byte_size and not self.row_size:
                return batches
            from yggdrasil.arrow.cast import (
                rechunk_arrow_batches_by_byte_size,
            )
            return rechunk_arrow_batches_by_byte_size(
                batches,
                byte_size=self.byte_size,
                row_size=self.row_size,
                memory_pool=self.arrow_memory_pool,
            )
        return self.merged_field.cast_arrow_batch_iterator(batches, options=self)

    def fill_arrow_nulls(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Engine-level null fill — delegates to :meth:`Field.fill_arrow`."""
        if self.target_field is None:
            return obj
        return self.merged_field.fill_arrow(obj, default_scalar=default_scalar)

    def fill_arrow_array_nulls(self, array: Any, *, default_scalar: Any = None) -> Any:
        """Narrow null fill for a :class:`pa.Array` / :class:`pa.ChunkedArray`."""
        if self.target_field is None:
            return array
        return self.merged_field.fill_arrow_array_nulls(
            array, default_scalar=default_scalar
        )

    # ---- polars ------------------------------------------------------

    def cast_polars(self, obj: Any) -> Any:
        """Cast any polars object — delegates to :meth:`Field.cast_polars`."""
        if self.target_field is None:
            return obj
        return self.merged_field.cast_polars(obj, options=self)

    def cast_polars_series(self, series: Any, *, default_scalar: Any = None) -> Any:
        """Cast a :class:`pl.Series`."""
        if self.target_field is None:
            return series
        return self.merged_field.cast_polars_series(
            series, options=self, default_scalar=default_scalar
        )

    def cast_polars_expr(self, expr: Any, *, default_scalar: Any = None) -> Any:
        """Cast a :class:`pl.Expr`.

        Wraps the expression tree with a cast operator — actual work
        fires when the containing LazyFrame is collected.
        """
        if self.target_field is None:
            return expr
        return self.merged_field.cast_polars_expr(
            expr, options=self, default_scalar=default_scalar
        )

    def cast_polars_tabular(self, data: Any) -> Any:
        """Cast a :class:`pl.DataFrame` or :class:`pl.LazyFrame`."""
        if self.target_field is None:
            return data
        return self.merged_field.cast_polars_tabular(data, options=self)

    def fill_polars_nulls(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Engine-level polars null fill — delegates to :meth:`Field.fill_polars`."""
        if self.target_field is None:
            return obj
        return self.merged_field.fill_polars(obj, default_scalar=default_scalar)

    def polars_alias(self, obj: Any) -> Any:
        """Rename a polars Series/Expr to the target name — no-op if matching.

        Delegates to :meth:`Field.polars_alias`. When ``target_field``
        is unbound there's no name to rename to, so we pass through.
        """
        if self.target_field is None:
            return obj
        return self.merged_field.polars_alias(obj)

    # ---- pandas ------------------------------------------------------

    def cast_pandas(self, obj: Any) -> Any:
        """Cast any pandas object — delegates to :meth:`Field.cast_pandas`."""
        if self.target_field is None:
            return obj
        return self.merged_field.cast_pandas(obj, options=self)

    def fill_pandas_nulls(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Engine-level pandas null fill — delegates to :meth:`Field.fill_pandas`."""
        if self.target_field is None:
            return obj
        return self.merged_field.fill_pandas(obj, default_scalar=default_scalar)

    # ---- spark -------------------------------------------------------

    def cast_spark(self, obj: Any) -> Any:
        """Cast any spark object — delegates to :meth:`Field.cast_spark`."""
        if self.target_field is None:
            return obj
        return self.merged_field.cast_spark(obj, options=self)

    def cast_spark_tabular(self, obj: Any) -> Any:
        """Cast any spark object — delegates to :meth:`Field.cast_spark`."""
        if self.target_field is None:
            return obj
        return self.merged_field.cast_spark_tabular(obj, options=self)

    def cast_spark_column(self, obj: Any) -> Any:
        """Cast any spark object — delegates to :meth:`Field.cast_spark`."""
        if self.target_field is None:
            return obj
        return self.merged_field.cast_spark_column(obj, options=self)

    def fill_spark_nulls(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Engine-level spark null fill — delegates to :meth:`Field.fill_spark`."""
        if self.target_field is None:
            return obj
        return self.merged_field.fill_spark(obj, default_scalar=default_scalar)

    def spark_alias(self, obj: Any) -> Any:
        """Rename a Spark Column to the target name — delegates to :meth:`Field.spark_alias`."""
        if self.target_field is None:
            return obj
        return self.merged_field.spark_alias(obj)

    # ==================================================================
    # Dunders
    # ==================================================================

    def __repr__(self) -> str:
        """Compact repr — skip fields at their default values.

        The full dataclass repr is a wall of text when most fields are
        defaults (very common in practice). Showing only set fields
        keeps exception messages and debug logs readable.
        """
        parts: list[str] = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            default = field.default
            if default is dataclasses.MISSING:
                parts.append(f"{field.name}={value!r}")
                continue
            # Mode normalization makes mode/schema_mode non-None
            # even when default was None → compare against AUTO too.
            if field.name in ("mode", "schema_mode") and value is Mode.AUTO:
                continue
            if value == default:
                continue
            parts.append(f"{field.name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"