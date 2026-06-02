"""Pure interface for tabular I/O over Apache Arrow record batches.

:class:`Tabular[O]` declares "I yield and consume Arrow record
batches" — nothing else. No state, no lifecycle, no holder, no
cursor. Two abstract hooks plus an options-class declaration; the
engine fan-out (Arrow / Polars / Pandas / Spark / Python-native) is
derived from those hooks.

Implementers compose :class:`Tabular` with whatever concrete
substrate they need:

- :class:`yggdrasil.io.holder.IO` mixes Tabular with
  :class:`Disposable` and a :class:`Holder` cursor — the default
  byte-backed implementation.
- A hypothetical :class:`SparkCatalogTabular` could mix Tabular with
  a Spark-session reference and skip byte-level concerns entirely.

The contract
------------

A leaf implements three things:

- :meth:`options_class`        — :class:`CastOptions` subtype.
- :meth:`_read_arrow_batches`  — yield :class:`pa.RecordBatch`.
- :meth:`_write_arrow_batches` — consume an iterable of batches.

Public methods route caller kwargs through :meth:`check_options` to
a single resolved options instance, then dispatch to one of the
hooks.

Why not inherit from Holder
---------------------------

Tabular is an interface, not a substrate. Most things that satisfy
"yield Arrow batches" don't have a meaningful byte representation
(a Spark catalog table, a JDBC cursor, a remote dataset service).
Forcing Holder's byte primitives into the contract would either
require those backends to fake a buffer or split them off into a
parallel hierarchy. Keeping Tabular pure lets every backend
satisfy it without lying.

Format registry
---------------

The byte-backed format registry (ParquetFile, CSVFile, ArrowIPCFile, …)
lives on :class:`Holder` — each leaf declares
:attr:`Holder.mime_type` at the class level and
:meth:`Holder.__init_subclass__` auto-registers it in
:data:`yggdrasil.io.holder._HOLDER_FORMAT_REGISTRY`. Look up via
:meth:`Holder.class_for_media_type` /
:meth:`Holder.for_holder`. Mirror of the ``scheme``-based
dispatch on :class:`URLBased`.
"""

from __future__ import annotations

import logging
import os
import time
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Iterator, TypeVar

import pyarrow as pa
from yggdrasil.data.data_field import Field as _Field
from yggdrasil.data.data_field import _pandas_index_levels
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.disposable import Disposable
from yggdrasil.enums import MediaType, MimeType, Mode, ModeLike
from yggdrasil.io.io_stats import IOStats
from yggdrasil.lazy_imports import polars_module, pyarrow_dataset_module
from yggdrasil.url.based import URLBased

if TYPE_CHECKING:
    from yggdrasil.arrow.tabular import ArrowTabular


logger = logging.getLogger(__name__)


#: Placeholder column-name prefix pyarrow uses for pandas index levels
#: that had no ``name`` on the source DataFrame (``__index_level_0__``
#: for an unnamed top level, ``__index_level_1__`` for the second level
#: of a partially-named MultiIndex, …). On read we map these back to
#: ``None`` so the round-trip matches the source.
_INDEX_PLACEHOLDER_PREFIX: str = "__index_level_"


def _is_default_range_index(frame: "pandas.DataFrame") -> bool:
    """Is this the freebie ``RangeIndex(0, len(df), 1, name=None)``?

    Default pandas indexes carry no information the read side has to
    recover — ``to_pandas()`` synthesises the same range from scratch.
    Skipping preservation for this case avoids writing a synthetic
    ``__index_level_0__`` column that would just hold ``0, 1, …, n-1``.
    Every *other* index (named, non-zero start / non-unit step, or an
    arbitrary label set like ``[5, 3, 1]``) carries information and
    MUST be persisted so it round-trips on read.
    """
    idx = frame.index
    return (
        type(idx).__name__ == "RangeIndex"
        and idx.name is None
        and getattr(idx, "start", 0) == 0
        and getattr(idx, "step", 1) == 1
        and getattr(idx, "stop", 0) == len(frame)
    )


def _collect_index_levels(schema: pa.Schema) -> "list[tuple[int, str]]":
    """Extract pandas index levels from yggdrasil tags or ``b"pandas"`` metadata.

    Returns ``(level_position, column_name)`` pairs. Prefers the
    ``index_key`` / ``index_key_level`` per-field tags written by
    :func:`_tag_index_columns`; falls back to pyarrow's ``b"pandas"``
    JSON blob's ``index_columns`` when those are absent — so files
    produced by ``pandas.to_parquet`` / ``pandas.to_feather``, Spark,
    or other tools still restore their index on read.
    """
    levels: "list[tuple[int, str]]" = []
    for f in schema:
        meta = f.metadata
        if not meta or not meta.get(_Field._TAG_KEY_INDEX_KEY):
            continue
        raw_level = meta.get(_Field._TAG_KEY_INDEX_KEY_LEVEL)
        try:
            pos = int(raw_level) if raw_level is not None else 0
        except (TypeError, ValueError):
            pos = 0
        levels.append((pos, f.name))
    if levels:
        return levels

    # No yggdrasil tags — fall back to pandas' own ``b"pandas"`` blob so
    # files written by ``pandas.to_parquet`` / other tools still restore.
    raw = (schema.metadata or {}).get(b"pandas")
    return [(pos, name) for name, pos in _pandas_index_levels(raw).items()]


def _tag_index_columns(table: pa.Table) -> pa.Table:
    """Stamp ``index_key`` + ``index_key_level`` on pandas index columns."""
    raw = (table.schema.metadata or {}).get(b"pandas")
    index_levels = _pandas_index_levels(raw)
    if not index_levels:
        return table
    tagged: list[pa.Field] = []
    for f in table.schema:
        level = index_levels.get(f.name)
        if level is not None:
            merged = dict(f.metadata or {})
            merged[_Field._TAG_KEY_INDEX_KEY] = b"true"
            merged[_Field._TAG_KEY_INDEX_KEY_LEVEL] = str(level).encode("ascii")
            tagged.append(f.with_metadata(merged))
        else:
            tagged.append(f)
    return pa.Table.from_arrays(
        list(table.itercolumns()),
        schema=pa.schema(tagged, metadata=None),
    )

if TYPE_CHECKING:
    import pandas
    import polars as pl
    import pyarrow.dataset as pds
    from pyspark.sql import DataFrame as SparkDataFrame
    from yggdrasil.execution.expr import Predicate, PredicateLike
    from yggdrasil.dataclasses.waiting import WaitingConfigArg
    from yggdrasil.io.holder import Holder
    from yggdrasil.spark.tabular import SparkDataset


__all__ = ["Tabular", "is_tabular_source"]


# ---------------------------------------------------------------------------
# Coercion helpers — :meth:`Tabular.resample` / :meth:`Tabular.unique`
# accept the same flexible argument shapes the rest of the codebase
# offers (string names, :class:`yggdrasil.data.Field` objects,
# iterables of either, :class:`int` / :class:`float` /
# :class:`datetime.timedelta` / ISO-8601 strings for durations). The
# coercion normalises everything to the canonical typed shape
# (``list[str]`` / ``int seconds``) before the typed
# :meth:`_unique` / :meth:`_resample` hook runs.
# ---------------------------------------------------------------------------


def _coerce_column_name(value: Any) -> str:
    """Resolve *value* to a column name string.

    Accepts a bare :class:`str`, or anything with a ``.name``
    attribute (:class:`yggdrasil.data.Field` and friends). Raises
    :class:`TypeError` for anything else so the caller gets a clear
    error message at the public boundary rather than a cryptic
    pyarrow / pyspark failure mid-pipeline.
    """
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    raise TypeError(
        f"Column reference must be a string name or a Field-like with "
        f"a ``.name`` attribute, got {type(value).__name__}: {value!r}."
    )


def _coerce_column_keys(value: Any) -> list[str]:
    """Resolve *value* to a flat ``list[str]`` of column names.

    Accepts a single name / Field, an iterable mixing the two, or
    ``None`` / empty (which yields ``[]``). Duplicates inside the
    iterable are preserved — the caller's order is the contract for
    multi-key dedup / partitioning.

    A :class:`yggdrasil.data.Field` is iterable (it yields its
    children), so the scalar-vs-sequence dispatch has to check for
    the scalar ``.name`` attribute *before* trying to iterate.
    Strings and bytes also take the scalar path.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (bytes, bytearray)):
        raise TypeError(
            f"Column reference must be a string / Field / iterable of those, "
            f"got {type(value).__name__}: {value!r}."
        )
    # Scalar Field-like (has a string ``.name``) → single-element list.
    # Checked before the iterable branch because Field itself is iterable.
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return [name]
    return [_coerce_column_name(item) for item in value]


def _coerce_predicate(value: Any) -> Any:
    """Normalize *value* into a :class:`Predicate` AST node.

    Accepts the same shapes :meth:`Expression.from_` does — SQL
    strings, an existing :class:`Expression` (must mark as
    :class:`Predicate`), or a pyarrow / polars / pyspark native
    expression (lifted via the matching backend importer). The
    result is always a yggdrasil :class:`Predicate` so the typed
    :meth:`Tabular._filter` hook sees a single shape regardless of
    where the predicate came from.
    """
    from yggdrasil.execution.expr import Expression, Predicate

    if isinstance(value, Predicate):
        return value
    if isinstance(value, Expression):
        raise TypeError(
            f"filter expected a boolean expression (predicate); "
            f"got non-predicate {type(value).__name__}: {value!r}. "
            "Use a comparison (col('x') > 1), is_in / between / "
            "is_null, or a SQL predicate string."
        )
    if isinstance(value, str):
        expr = Expression.from_sql(value)
        if not isinstance(expr, Predicate):
            raise TypeError(
                f"filter expected a SQL predicate string; "
                f"{value!r} parses to a non-predicate "
                f"{type(expr).__name__}."
            )
        return expr
    # Native engine expression — pyarrow / polars / pyspark. The
    # ``Expression.from_`` dispatcher sniffs the source's module so
    # the optional deps stay optional (we never import an engine
    # we haven't seen).
    expr = Expression.from_(value)
    if not isinstance(expr, Predicate):
        raise TypeError(
            f"filter expected a boolean expression (predicate); "
            f"{type(value).__name__} lifted to non-predicate "
            f"{type(expr).__name__}."
        )
    return expr


def _coerce_sampling_seconds(value: Any) -> int:
    """Resolve *value* into an integer second count for resample.

    Accepted shapes:

    * :class:`int` — used verbatim (booleans are rejected even though
      they're a subclass; ``True == 1`` would be a footgun here).
    * :class:`float` — rounded to the nearest integer.
    * :class:`datetime.timedelta` — ``td.total_seconds()`` rounded.
    * :class:`str` — ISO-8601 duration parsed via
      :func:`yggdrasil.data.types.primitive.temporal._parse_iso_duration`.

    Anything else raises :class:`TypeError`.
    """
    import datetime as dt

    if isinstance(value, bool):
        raise TypeError(
            f"sampling must be int / float / timedelta / str, got bool: {value!r}."
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, dt.timedelta):
        return int(round(value.total_seconds()))
    if isinstance(value, str):
        from yggdrasil.data.types.primitive.temporal import _parse_iso_duration

        td = _parse_iso_duration(value)
        if td is None:
            raise ValueError(
                f"sampling={value!r} is not a recognised ISO-8601 duration. "
                "Pass an integer second count, a ``datetime.timedelta``, "
                "or a string like ``'PT1H'`` / ``'P1D'`` / ``'PT15M'``."
            )
        return int(round(td.total_seconds()))
    raise TypeError(
        f"sampling must be int / float / timedelta / str (ISO-8601 "
        f"duration), got {type(value).__name__}: {value!r}."
    )


def _default_unique(self_: "Tabular", *, keys: list[str]) -> "Tabular":
    """Engine-routing dedup — Spark-native if available, else Arrow.

    Defined at module scope so :meth:`Tabular._unique` (and any
    subclass's super-call) shares one implementation regardless of
    where it's invoked from. Spark-shape detection goes through
    :meth:`Tabular._native_spark_frame` so any holder that surfaces
    a :class:`pyspark.sql.DataFrame` natively (the
    :class:`Dataset` itself, a :class:`SparkStatementResult`) keeps
    the work on the executors.
    """
    spark_frame = self_._native_spark_frame()
    if spark_frame is not None:
        from yggdrasil.spark.ops import dedup_spark_dataframe
        from yggdrasil.spark.tabular import SparkDataset

        new_frame = dedup_spark_dataframe(spark_frame, keys)
        return SparkDataset(frame=new_frame, schema=_schema_for_new_tabular(self_))

    from yggdrasil.arrow.ops import dedup_arrow_table
    from yggdrasil.arrow.tabular import ArrowTabular

    table = self_.read_arrow_table()
    out = dedup_arrow_table(table, keys)
    return ArrowTabular(out, schema=out.schema)


def _default_resample(
    self_: "Tabular",
    *,
    time_column: str,
    sampling_seconds: int,
    partition_by: list[str],
    fill_strategy: "str | None",
) -> "Tabular":
    """Engine-routing resample — Spark-native if available, else Arrow."""
    spark_frame = self_._native_spark_frame()
    if spark_frame is not None:
        from yggdrasil.spark.ops import resample_spark_dataframe
        from yggdrasil.spark.tabular import SparkDataset

        new_frame = resample_spark_dataframe(
            spark_frame,
            time_column=time_column,
            sampling_seconds=sampling_seconds,
            partition_by=partition_by or None,
            fill_strategy=fill_strategy,
        )
        return SparkDataset(frame=new_frame, schema=_schema_for_new_tabular(self_))

    from yggdrasil.arrow.ops import resample_arrow_table
    from yggdrasil.arrow.tabular import ArrowTabular

    table = self_.read_arrow_table()
    out = resample_arrow_table(
        table,
        time_column=time_column,
        sampling_seconds=sampling_seconds,
        partition_by=partition_by or None,
        fill_strategy=fill_strategy,
    )
    return ArrowTabular(out, schema=out.schema)


def _flatten_column_args(args: "tuple[Any, ...]") -> list[str]:
    """Flatten variadic select / drop column args to a ``list[str]``.

    Accepts strings, :class:`Field`-like objects (resolved via
    :attr:`.name`), and iterables mixing those (so callers can pass
    ``select("a", "b")`` *or* ``select(["a", "b"])`` interchangeably,
    matching the rest of the library's keyword conventions).
    Preserves caller-given order; duplicates are kept on purpose —
    deduping would silently mask a select-with-duplicate bug.
    """
    out: list[str] = []
    for item in args:
        if isinstance(item, str):
            out.append(item)
            continue
        if isinstance(item, (bytes, bytearray)):
            raise TypeError(
                f"Column reference must be a string / Field / iterable of "
                f"those, got {type(item).__name__}: {item!r}."
            )
        name = getattr(item, "name", None)
        if isinstance(name, str):
            out.append(name)
            continue
        try:
            iter(item)
        except TypeError:
            raise TypeError(
                f"Column reference must be a string / Field / iterable of "
                f"those, got {type(item).__name__}: {item!r}."
            )
        out.extend(_coerce_column_name(child) for child in item)
    return out


def _default_select(self_: "Tabular", *, columns: list[str]) -> "Tabular":
    """Engine-routing select — Spark-native when available, else Arrow."""
    spark_frame = self_._native_spark_frame()
    if spark_frame is not None:
        # ``DataFrame.select`` raises on a missing column, so the
        # public method's "preserve order" contract is the caller's
        # problem to keep right. We don't filter missing names here —
        # the Spark error is more informative than a silent drop.
        from yggdrasil.spark.tabular import SparkDataset

        new_frame = spark_frame.select(*columns)
        return SparkDataset(frame=new_frame, schema=None)

    from yggdrasil.arrow.tabular import ArrowTabular

    table = self_.read_arrow_table()
    # ``pa.Table.select`` raises ``KeyError`` on a missing column —
    # match the Spark side's "fail loud" behavior so a typo in a
    # select call doesn't silently return an unexpected shape.
    out = table.select(columns)
    return ArrowTabular(out, schema=out.schema)


def _default_drop(self_: "Tabular", *, columns: list[str]) -> "Tabular":
    """Engine-routing drop — Spark-native when available, else Arrow.

    Missing columns in the source are filtered out *before* the
    underlying engine call — neither :meth:`pa.Table.drop_columns`
    nor :meth:`DataFrame.drop` raise on a missing reference by
    default in current versions, but pinning the contract here
    means callers see consistent behavior even if those defaults
    drift.
    """
    spark_frame = self_._native_spark_frame()
    if spark_frame is not None:
        from yggdrasil.spark.tabular import SparkDataset

        present = [c for c in columns if c in spark_frame.columns]
        new_frame = spark_frame.drop(*present) if present else spark_frame
        return SparkDataset(frame=new_frame, schema=None)

    from yggdrasil.arrow.tabular import ArrowTabular

    table = self_.read_arrow_table()
    present = [c for c in columns if c in table.schema.names]
    if not present:
        return ArrowTabular(table, schema=table.schema)
    out = table.drop_columns(present)
    return ArrowTabular(out, schema=out.schema)


def _default_filter(self_: "Tabular", *, predicate: "Predicate") -> "Tabular":
    """Engine-routing row filter — Spark-native when available, else Arrow."""
    spark_frame = self_._native_spark_frame()
    if spark_frame is not None:
        from yggdrasil.spark.tabular import SparkDataset

        new_frame = predicate.filter_spark_frame(spark_frame)
        return SparkDataset(frame=new_frame, schema=_schema_for_new_tabular(self_))

    from yggdrasil.arrow.tabular import ArrowTabular

    table = self_.read_arrow_table()
    out = predicate.filter_arrow_table(table)
    return ArrowTabular(out, schema=out.schema)


def _schema_for_new_tabular(self_: "Tabular") -> "Schema | None":
    """Return the source's declared schema, if one is set.

    The :meth:`Tabular._schema_cache` slot holds the per-instance
    schema (``...`` sentinel when none was declared). Reading the
    raw slot avoids triggering :meth:`collect_schema`'s materialise
    side-effect on the destination Tabular — the dedup / resample
    op preserves the source's column layout, so propagating the
    declared schema is correct without re-inferring.
    """
    cached = getattr(self_, "_schema_cache", ...)
    if cached is ...:
        return None
    return cached


def is_tabular_source(obj: Any) -> bool:
    """True iff *obj* is a shape :meth:`Tabular.from_` can coerce.

    Catches the three shapes the cast layer needs to recognise before
    dispatching to engine-native constructors:

    * an existing :class:`Tabular` (incl. :class:`Holder` / :class:`Path`
      / :class:`Memory` / :class:`IO` — anything with the read/write
      Arrow batch contract);
    * an :class:`os.PathLike` (incl. :class:`pathlib.PurePath`);
    * a path-shaped ``str`` (``"data.parquet"``, ``"s3://b/k"``,
      ``"./x.csv"``) — string heuristic deliberately tight so that a
      caller passing a plain content string (``"hello"``) still falls
      through to the existing engine-native dispatch.

    File-like objects (have ``read``) aren't probed here: dispatching
    them needs an explicit ``media_type=`` and the cast layer doesn't
    carry one — :meth:`Tabular.from_` raises :class:`TypeError` for
    those if the caller routes them through anyway.
    """
    if isinstance(obj, Tabular):
        return True
    if isinstance(obj, os.PathLike):
        return True
    if isinstance(obj, str):
        if not obj or len(obj) > 4096:
            return False
        if "://" in obj or "/" in obj or "\\" in obj:
            return True
        # Bare filename with extension — ``data.parquet``, ``q.csv``.
        return "." in obj and not obj.startswith(".")
    return False


O = TypeVar("O", bound=CastOptions)


_TABULAR_FORMAT_REGISTRY: "dict[str, type[Tabular]]" = {}


class Tabular(Singleton, URLBased, Disposable, Generic[O]):
    """Pure interface — Arrow record-batch source/sink + engine fan-out.

    No state, no lifecycle, with the single exception of a
    :attr:`tabular_parent` back-pointer for tree-shaped sources
    (folders, archives, partitioned tables) where a child Tabular
    wants to walk back up to whatever yielded it. Aggregators pass
    ``tabular_parent=self`` at child construction.

    Concrete implementers add whatever substrate they need (a
    holder + cursor for byte-backed shapes, a session reference
    for catalog-backed shapes, etc.) and override the two batch
    hooks. The byte-backed format registry (ParquetFile, CSVFile,
    ArrowIPCFile, …) lives on :class:`Holder` — each leaf declares
    :attr:`Holder.mime_type` at the class level and
    :meth:`Holder.__init_subclass__` auto-registers it. Look up via
    :meth:`Holder.class_for_media_type` / :meth:`Holder.for_holder`.
    """

    mime_type: "ClassVar[MimeType | None]" = None

    #: Whether :meth:`_read_arrow_table` returns a freshly-decoded table
    #: whose buffers no one else holds — so a consumer may destroy it in
    #: place (``to_pandas(self_destruct=True)``) to halve peak memory.
    #: True for every backend that decodes a new table per read (parquet,
    #: arrow-ipc, csv, pickle, warehouse, …). In-memory holders that hand
    #: back a *cached* table shared with the live instance override this to
    #: False so the cache survives the conversion.
    _READ_TABLE_OWNED: "ClassVar[bool]" = True

    #: Lifetime of a seeded :attr:`_stat_cached` entry. ``None`` (the
    #: default) trusts a seeded snapshot until it is explicitly
    #: invalidated — right for in-memory and local-filesystem holders.
    #: Remote backends override with a finite TTL so a snapshot can't
    #: outlive the backend's own mutation window. Only gates the *stat*
    #: fields; the cached schema (``field``) is read regardless of TTL.
    STAT_CACHE_TTL: "ClassVar[float | None]" = None
    #: The cached :class:`IOStats` (stat snapshot + schema ``field``) is
    #: per-instance, backend-derived state — never pickle it. A tabular
    #: serialized on one node and rehydrated on another must re-probe
    #: rather than trust a snapshot of *this* node's view. Subclasses
    #: that filter ``__getstate__`` by this set (e.g.
    #: :class:`DatabricksPath`) rely on the slots being listed here.
    _TRANSIENT_STATE_ATTRS: "ClassVar[frozenset[str]]" = frozenset(
        {"_stat_cached", "_stat_cached_at"}
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        mt = cls.__dict__.get("mime_type")
        if mt is None:
            return
        key = mt.name if hasattr(mt, "name") else str(mt)
        _TABULAR_FORMAT_REGISTRY[key] = cls

    @classmethod
    def class_for_media_type(cls, media_type: Any, *, default: Any = None) -> "type[Tabular] | None":
        if media_type is None:
            return default
        mt = media_type if hasattr(media_type, "mime_type") else None
        if mt is not None:
            media_type = mt.mime_type
        key = media_type.name if hasattr(media_type, "name") else str(media_type)
        return _TABULAR_FORMAT_REGISTRY.get(key, default)

    def __init__(
        self,
        *,
        tabular_parent: "Tabular | None" = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the tree-parent slot.

        Subclasses chain ``super().__init__(**kwargs)`` and don't
        otherwise touch :attr:`tabular_parent` — aggregators pass
        ``tabular_parent=self`` at child construction.

        The slot is named ``tabular_parent`` (not ``parent``) so it
        doesn't collide with :attr:`yggdrasil.io.path.path.Path.parent`,
        which is the path-shaped parent directory and therefore
        property-backed and read-only.
        """
        super().__init__()
        self.tabular_parent: "Tabular | None" = tabular_parent
        # Cached :class:`IOStats` — the single home for both the stat
        # snapshot (size / mtime / kind, used by byte-backed holders)
        # and the tabular schema (``field``, used by every Tabular). The
        # ``_schema_cache`` property reads / writes the schema through
        # it. Cold by default: ``_stat_cached_at == 0`` marks "never
        # seeded" so :meth:`_stat_cached_fresh` re-probes the stat side
        # until a listing / probe fills it in.
        self._stat_cached: "IOStats | None" = None
        self._stat_cached_at: float = 0.0
        self._url: "URL | None" = None

    def to_url(self) -> "URL":
        if self._url is not None:
            return self._url
        from yggdrasil.url import URL
        self._url = URL.from_memory_address(self)
        return self._url

    @property
    def url(self) -> "URL":
        return self.to_url()

    # ==================================================================
    # Cached IOStats — shared home for the stat snapshot AND the schema.
    # The stat fields (size / mtime / kind) are gated by
    # :attr:`STAT_CACHE_TTL` via :meth:`_stat_cached_fresh`; the schema
    # (``field``) is read through the :attr:`_schema_cache` property
    # regardless of stat freshness.
    # ==================================================================

    def _stat_cached_fresh(self) -> "IOStats | None":
        """Return the cached :class:`IOStats` when its *stat* side is fresh.

        ``None`` when the slot is cold (``_stat_cached_at <= 0`` — never
        stamped as a real stat snapshot, e.g. an entry that only carries
        a cached schema ``field``) *or* the entry has expired past
        :attr:`STAT_CACHE_TTL`. ``TTL is None`` trusts a stamped snapshot
        until it is explicitly invalidated; ``TTL <= 0`` always
        re-probes. In-memory holders never stamp the slot, so this stays
        ``None`` for them and their authoritative ``_size`` / ``_mtime``
        answer instead.
        """
        cached = self._stat_cached
        if cached is None or self._stat_cached_at <= 0.0:
            return None
        ttl = self.STAT_CACHE_TTL
        if ttl is None:
            return cached
        if ttl <= 0:
            return None
        if (time.monotonic() - self._stat_cached_at) <= ttl:
            return cached
        return None

    def _persist_stat_cache(self, stats: IOStats) -> None:
        """Seed :attr:`_stat_cached` with a known :class:`IOStats` and
        stamp it fresh.

        Useful for backends that learn metadata as a side-effect of a
        listing (S3 ``ListObjectsV2`` returns size + mtime per object,
        Databricks ``Files.list_directory_contents`` returns
        ``is_directory`` + ``file_size``, ``os.scandir`` exposes
        ``is_dir`` / ``is_file`` cheaply) or a read / write where the
        response body's length IS the new size. The next filesystem
        predicate (:attr:`size` / :meth:`exists` / :meth:`is_file` /
        :meth:`is_dir` / :attr:`mtime`) collapses to a local hit.

        A schema cached on the previous entry carries over when the
        incoming *stats* doesn't bring its own — a stat refresh
        (TTL expiry, HeadObject) shouldn't drop a known schema, which
        is invalidated separately on mutation.
        """
        prior = self._stat_cached
        if stats.field is None and prior is not None and prior.field is not None:
            stats.field = prior.field
        self._stat_cached = stats
        self._stat_cached_at = time.monotonic()

    @property
    def _schema_cache(self) -> "Schema | Any":
        """The cached schema, stored in :attr:`_stat_cached`'s ``field``.

        Returns the ``...`` sentinel when no schema is cached, matching
        the slot-based contract every caller (``collect_schema``,
        folder / spark / arrow tabulars) was written against — they keep
        working unchanged while the storage moves into the shared
        :class:`IOStats`.
        """
        stat = self._stat_cached
        if stat is not None and stat.field is not None:
            return stat.field
        return ...

    @_schema_cache.setter
    def _schema_cache(self, value: "Schema | Any") -> None:
        stat = self._stat_cached
        if value is ...:
            # Clear the schema, leave any stat snapshot intact.
            if stat is not None:
                stat.field = None
            return
        if stat is None:
            # No stat yet — mint a cold entry (``_stat_cached_at`` stays
            # 0 so :meth:`_stat_cached_fresh` won't mistake it for a real
            # stat snapshot) that carries only the schema.
            self._stat_cached = IOStats(field=value)
        else:
            stat.field = value

    # ==================================================================
    # Static values — column → constant invariants across every row
    # ==================================================================

    @property
    def static_values(self) -> "Mapping[str, Any]":
        url = getattr(self, "url", None) or getattr(self, "_url", None)
        if url is not None and hasattr(url, "static_values"):
            return url.static_values
        return {}

    def matches_static(
        self,
        predicate: "Predicate",
        *,
        free_cols: "tuple[str, ...] | None" = None,
    ) -> bool:
        """True iff *predicate* could match any row given
        :attr:`static_values`. Conservative on undecidables (column
        not in static values, predicate evaluator failure) so the
        caller still reads.

        Builds a one-row pyarrow Table from the predicate's free
        columns that we have static values for, then evaluates the
        predicate against it — generalises the partition-only
        prune so any aggregator (folder read, future warehouse
        file skip) reuses the one helper.

        ``free_cols`` lets a caller that's about to prune the same
        predicate against N children precompute the free-column
        tuple once and reuse it — :func:`free_columns` walks the
        AST every call, so on a 64-OR predicate (the cache batch
        lookup shape) the saving is N-1 full walks per
        ``iter_children`` loop. Default ``None`` keeps the call
        site short for one-off prune checks.
        """
        if predicate is None:
            return True
        if free_cols is None:
            try:
                from yggdrasil.execution.expr import free_columns
                free_cols = free_columns(predicate)
            except Exception:
                return True
        if not free_cols:
            return True
        sv = self.static_values
        relevant: "dict[str, Any]" = {}
        for name in free_cols:
            if name not in sv:
                # Predicate touches a column outside our static
                # surface — can't decide, must read.
                return True
            relevant[name] = sv[name]
        try:
            table = pa.Table.from_pydict({k: [v] for k, v in relevant.items()})
            return predicate.filter_arrow_table(table).num_rows == 1
        except Exception:
            return True

    def _should_prune_by_predicate(
        self,
        options: Any,
        *,
        free_cols: "tuple[str, ...] | None" = None,
    ) -> bool:
        """Return ``True`` iff ``options.predicate`` is provably false
        against this Tabular's :attr:`static_values` — the caller may
        skip the read entirely.

        Thin wrapper around :meth:`matches_static`: pulls the
        predicate off *options* (any object that exposes a
        ``predicate`` attribute; absence and ``None`` both mean "no
        prune") and inverts the sense so an aggregator's read loop
        reads as ``if self._should_prune_by_predicate(options):
        return``. ``free_cols`` mutualises the AST walk across a
        sibling loop — see :meth:`matches_static`.

        Conservative on undecidables: a predicate over a column we
        have no static value for returns ``False`` (no prune) and
        the caller still reads. Same contract as
        :meth:`matches_static`.
        """
        predicate = getattr(options, "predicate", None)
        if predicate is None:
            return False
        return not self.matches_static(predicate, free_cols=free_cols)

    # ==================================================================
    # Coercion entry point — delegates to the format registry on Holder
    # ==================================================================

    @classmethod
    def from_(
        cls,
        obj: Any,
        *,
        media_type: "MediaType | MimeType | str | None" = None,
        default: Any = ...,
        as_folder: bool = False,
        **kwargs: Any,
    ) -> "Tabular | None":
        """Coerce *obj* into a :class:`Tabular`.

        Routes:

        * ``None`` — returns *default* (``None`` when
          ``default=None``).
        * :class:`Tabular` — returned as-is. When *as_folder* is
          ``True`` and *obj* is a local :class:`Path`, wraps it in
          a :class:`Folder`.
        * ``str`` / :class:`os.PathLike` — coerced via
          :class:`Path.from_`. When *as_folder* is ``True``,
          wraps in :class:`Folder`.
        * File-like objects — drained into :class:`Memory`;
          *media_type* required.

        Falls back to *default* on unrecognised shapes when supplied;
        otherwise raises :class:`TypeError`.
        """
        if obj is None:
            return default if default is not ... else None

        from yggdrasil.io.holder import Holder

        if isinstance(obj, Tabular):
            if as_folder and isinstance(obj, Holder) and obj.is_local:
                from yggdrasil.path.folder import Folder
                if not isinstance(obj, Folder):
                    return Folder(path=obj)
            if media_type is not None and isinstance(obj, Holder):
                return Holder.for_holder(obj, media_type=media_type, **kwargs)
            return obj

        if isinstance(obj, (str, os.PathLike)):
            if isinstance(obj, str) and not is_tabular_source(obj):
                if default is not ...:
                    return default
                raise TypeError(
                    f"Tabular.from_ string {obj!r} is not path-shaped; "
                    "pass a URL, a filesystem path, or a name with an "
                    "extension (e.g. 'data.parquet')."
                )
            from yggdrasil.path import Path as YggPath
            path = YggPath.from_(obj)
            if as_folder:
                from yggdrasil.path.folder import Folder
                return Folder(path=path)
            if media_type is not None:
                return Holder.for_holder(path, media_type=media_type, **kwargs)
            return path

        read = getattr(obj, "read", None)
        if callable(read):
            if media_type is None:
                if default is not ...:
                    return default
                raise TypeError(
                    f"Tabular.from_ requires media_type= for file-like "
                    f"inputs ({type(obj).__name__}); no URL extension "
                    "to sniff the format."
                )
            from yggdrasil.path.memory import Memory
            data = read()
            if isinstance(data, str):
                data = data.encode()
            return Holder.for_holder(
                Memory(binary=data), media_type=media_type, **kwargs,
            )

        if default is not ...:
            return default
        raise TypeError(
            f"Cannot coerce {type(obj).__name__} to a Tabular. "
            "Pass a Tabular, a path / URL (str or os.PathLike), or "
            "a file-like object with media_type=."
        )

    # ==================================================================
    # Options
    # ==================================================================

    @classmethod
    def options_class(cls) -> "type[O]":
        """The :class:`CastOptions` subclass this implementer consumes.

        Default :class:`CastOptions`. Format-specific leaves with
        their own knobs (Parquet compression, CSV delimiter, …)
        override.
        """
        return CastOptions  # type: ignore[return-value]

    @classmethod
    def check_options(
        cls,
        options: "O | None" = None,
        overrides: "dict | None" = None,
        **kwargs: Any,
    ) -> O:
        """Validate and merge caller kwargs into a resolved options.

        Canonical pattern: a public method passes ``overrides=locals()``
        and the ``...``-defaulted entries are stripped, the rest merged.
        """
        if overrides:
            overrides = {k: v for k, v in overrides.items() if v is not ...}
            overrides.pop("self", None)
            options = overrides.pop("options", options)
            kwargs.update(overrides)
            kwargs.update(kwargs.pop("kwargs", {}))
        return cls.options_class().check(options, **kwargs)

    # ==================================================================
    # Cleanup hook — backend-specific GC of stale state
    # ==================================================================

    def cleanup(self, wait: "Any" = False) -> int:
        """Garbage-collect stale state on this backend.

        Default no-op (returns ``0``) — single-file leaves and
        warehouse-backed tables don't have a sweep concept the
        client owns. Folder-shaped subclasses override to unlink
        stale ``part-*`` files, throttled by TTL.

        ``wait`` controls sync vs async dispatch on backends that
        support it: a truthy :class:`yggdrasil.dataclasses.waiting.WaitingConfig`
        (or ``True`` / a positive timeout) blocks until the sweep
        finishes; a falsy value (the default) hands the work off to a
        background thread. Backends without an async path treat both
        the same.

        Returns the number of files / rows removed when known; ``0``
        for fire-and-forget async dispatch or a no-op backend.
        """
        return 0

    # ==================================================================
    # Compaction hook
    # ==================================================================

    def optimize(
        self,
        byte_size: "int | None" = None,
        **kwargs: Any,
    ) -> int:
        """Repartition / compact this Tabular's storage.

        Default implementation is a no-op and returns ``0`` — single-file
        leaves (parquet, csv, arrow IPC, …) don't have a compaction
        concept. Aggregator subclasses (:class:`Folder`) override
        this to walk their child leaves and bin-pack small part files
        into bundles near *byte_size*.
        Files already close to the target size are left alone so a
        repeated call is cheap.

        ``byte_size=None`` keeps the legacy "collapse every leaf with
        more than one part into a single file" behavior, which is what
        the local-cache compaction loop in :class:`Session` expects.
        Any extra keyword arguments are accepted and ignored so
        upstream callers can pass forward-compatible knobs without the
        base raising.
        """
        return 0

    def create(
        self,
        schema: Schema,
        *,
        mode: ModeLike = None,
        **kwargs
    ):
        raise NotImplementedError()

    # ==================================================================
    # Row-level delete
    # ==================================================================

    def delete(
        self,
        predicate: "PredicateLike" = None,
        *,
        wait: "WaitingConfigArg" = True,
        missing_ok: bool = False,
        delete_staging: bool = True,
        **kwargs: Any,
    ) -> "Table":
        """Delete rows matching *predicate*; return this tabular.

        *predicate* is a :class:`Predicate` from
        :mod:`yggdrasil.execution.expr` or a SQL string that parses into
        one (``"id IN (1,2,3)"``, ``"price > 100 AND region = 'EU'"``).
        ``None`` means "no filter" — every row is removed (``DELETE FROM t``
        with no ``WHERE``).

        ``wait`` / ``missing_ok`` / ``delete_staging`` are honoured by
        resource-backed subclasses (e.g.
        :class:`yggdrasil.databricks.table.table.Table`, which drops the
        table asset); the generic row-rewrite path ignores them. Any extra
        ``**kwargs`` (e.g. ``options=DeltaOptions(...)``) flow through to
        :meth:`_delete`.

        The default implementation reads every batch, drops rows the
        predicate accepts, and rewrites the leaf with the survivors.
        Aggregator subclasses (:class:`yggdrasil.path.folder.Folder`)
        override to walk children, prune subtrees whose partition bounds
        make the predicate trivially false, and only rewrite the leaves
        that actually hold matched rows.
        """
        from yggdrasil.execution.expr import Expression, Predicate

        if predicate is not None:
            if isinstance(predicate, str):
                predicate = Expression.from_sql(predicate)
            if not isinstance(predicate, Predicate):
                raise TypeError(
                    f"{type(self).__name__}.delete expected a Predicate "
                    f"(or a SQL string parseable to one); got "
                    f"{type(predicate).__name__}: {predicate!r}."
                )
        self._delete(
            predicate,
            wait=wait,
            missing_ok=missing_ok,
            delete_staging=delete_staging,
            **kwargs,
        )
        return self

    @abstractmethod
    def _delete(
        self,
        predicate: "Predicate" = None,
        *,
        wait: "WaitingConfigArg" = True,
        missing_ok: bool = False,
        delete_staging: bool = True,
        **kwargs: Any,
    ) -> int:
        """Remove rows matching *predicate* (``None`` = all); return rows deleted.

        Every concrete tabular declares its own delete strategy:

        * writable leaves (:class:`~yggdrasil.io.holder.IO`,
          :class:`~yggdrasil.arrow.tabular.ArrowTabular`,
          :class:`~yggdrasil.spark.tabular.SparkDataset`) delegate to the
          generic :meth:`_delete_rewrite` — filter every batch, rewrite
          the survivors;
        * aggregators (:class:`~yggdrasil.path.folder.Folder`,
          :class:`~yggdrasil.io.delta.delta_folder.DeltaFolder`) prune /
          rewrite per leaf;
        * read-only tabulars (:class:`LazyTabular`, :class:`ExecutionPlan`,
          :class:`StatementResult`, :class:`HTTPResponseBatch`) raise.

        ``options`` and other read/write overrides ride in via ``**kwargs``;
        ``wait`` / ``missing_ok`` / ``delete_staging`` exist for signature
        parity with resource-backed deletes and are ignored by row paths.
        """

    def _delete_rewrite(
        self,
        predicate: "Predicate" = None,
        **kwargs: Any,
    ) -> int:
        """Generic single-leaf delete: filter all batches, rewrite.

        Routes the row-level work through
        :meth:`Predicate.filter_arrow_batches`, so the streaming
        filter runs in pyarrow's C++ kernels — no Python row
        iteration. Counts rows by diffing input vs. surviving sizes
        as the stream goes by, keeping the streaming property: only
        one batch resides in memory at a time. A ``None`` *predicate*
        removes every row. Returns the number of rows deleted.

        Reusable building block for the writable ``_delete``
        implementations. ``options`` (and any other read/write override)
        ride in via ``**kwargs``.
        """
        options = self.check_options(kwargs.pop("options", None), **kwargs)
        survivors: "list[pa.RecordBatch]" = []
        kept_rows = 0
        total_rows = 0

        def _counted() -> "Iterator[pa.RecordBatch]":
            nonlocal total_rows
            for b in self._read_arrow_batches(options):
                total_rows += b.num_rows
                yield b

        if predicate is None:
            # No filter → delete every row; drain to count, keep nothing.
            for _ in _counted():
                pass
        else:
            for kept in (~predicate).filter_arrow_batches(_counted()):
                kept_rows += kept.num_rows
                survivors.append(kept)
        deleted = total_rows - kept_rows
        if deleted == 0:
            return 0
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s deleted %d / %d rows",
                type(self).__name__,
                deleted,
                total_rows,
            )
        self._write_arrow_batches(iter(survivors), options)
        return deleted

    # ==================================================================
    # Abstract batch hooks — the two things every implementer overrides
    # ==================================================================

    def _native_spark_frame(self) -> "SparkDataFrame | None":
        """Return the held :class:`pyspark.sql.DataFrame` when this
        Tabular wraps one natively (no materialise cost).

        Default ``None`` — only Spark-shaped holders override. Used by
        :meth:`_write_table` to skip the Arrow round-trip when both
        source and sink speak Spark.
        """
        return None

    @abstractmethod
    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches from the underlying source."""

    @abstractmethod
    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Consume Arrow record batches and persist them."""

    # ==================================================================
    # Schema inspection — per-instance cache
    # ==================================================================

    def collect_schema(self, options: "O | None" = None, **kwargs: Any) -> Schema:
        """Return this Tabular's :class:`Schema`, caching the first hit.

        The cache slot is :attr:`_schema_cache`; on first call this
        method stamps the resolved schema into it so subsequent
        ``collect_schema`` calls short-circuit. Writers overwrite
        the slot via :meth:`_persist_schema`; lifecycle hooks clear
        it via :meth:`_unpersist_schema`.
        """
        if self._schema_cache is not ...:
            return self._schema_cache

        options = self.check_options(options, overrides=locals())
        m = options.merged

        if m:
            return m

        schema = self._collect_schema(options)
        if schema is not None:
            self._schema_cache = schema
        return schema

    def _collect_schema(self, options: O) -> Schema:
        """First batch's schema by default; merged across batches when
        ``options.safe`` is set, so heterogeneous sources surface a
        union schema instead of "whichever batch came first"."""
        batches = self._read_arrow_batches(options)
        first = next(iter(batches), None)
        if first is None:
            return Schema.empty()
        schema = Schema.from_arrow(first.schema)
        if not getattr(options, "safe", False):
            return schema
        for batch in batches:
            schema = schema.merge_with(
                Schema.from_arrow(batch.schema), inplace=True,
            )
        return schema

    def _persist_schema(self, schema: "Schema | None") -> None:
        """Stamp *schema* into the per-instance cache (writer hook).

        Writers call this after the bytes hit the sink so the next
        :meth:`collect_schema` returns the freshly-written shape
        instead of replaying the source. ``None`` and an empty schema
        are accepted but ignored — they would shadow whatever the
        cache held without telling readers anything new.
        """
        if schema is None:
            return
        self._schema_cache = schema

    def _unpersist_schema(self) -> None:
        """Drop the per-instance schema cache.

        Called from :meth:`close` (so reopening re-collects from the
        source) and from any path that mutates the underlying bytes
        in a way the writers can't describe (truncate, clear).
        """
        self._schema_cache = ...

    # ==================================================================
    # Lifecycle hook — clears the schema cache on close
    # ==================================================================

    def close(self, force: bool = False) -> None:
        """Drop the schema cache and forward to any cooperative ``close``.

        Tabular itself has no resources to release — the schema cache
        is the only state it owns. Subclasses that mix Tabular with a
        lifecycle (``Disposable``-derived ``IO``, holders, …) inherit
        this hook through cooperative ``super().close()``; pure
        Tabular subclasses without a lifecycle peer get a harmless
        no-op forward.
        """
        self._unpersist_schema()
        sup_close = getattr(super(), "close", None)
        if callable(sup_close):
            sup_close(force=force)

    # ==================================================================
    # Static helpers
    # ==================================================================

    @staticmethod
    def _normalize_records(data: Iterable[dict]) -> "list[dict]":
        """Backfill every row to the union of keys seen across all rows.

        :meth:`pa.Table.from_pylist` infers schema from the first row
        and silently drops keys that show up only later; callers that
        build rows incrementally (e.g. JSON parsers) route through
        here first.
        """
        rows = list(data) if not isinstance(data, list) else data
        if not rows:
            return []

        all_keys: "dict[str, None]" = {}
        needs_backfill = False
        reference: "tuple[str, ...] | None" = None
        for row in rows:
            if row is None:
                needs_backfill = True
                continue
            keys = tuple(row.keys())
            if reference is None:
                reference = keys
            elif keys != reference:
                needs_backfill = True
            for k in keys:
                if k not in all_keys:
                    all_keys[k] = None

        if not needs_backfill:
            return rows
        key_tuple = tuple(all_keys)
        return [
            {k: (row.get(k) if row is not None else None) for k in key_tuple}
            for row in rows
        ]

    # ==================================================================
    # Count
    # ==================================================================

    def count(self, options: "O | None" = None, **kwargs: Any) -> int:
        """Return the number of rows in this tabular."""
        return self._count(self.check_options(options, overrides=locals()))

    def _count(self, options: O) -> int:
        return sum(b.num_rows for b in self._read_arrow_batches(options))

    # ==================================================================
    # Arrow surface
    # ==================================================================

    def read_arrow_tabular(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "ArrowTabular":
        return self._read_arrow_tabular(self.check_options(options, overrides=locals()))

    def _read_arrow_tabular(self, options: O) -> "ArrowTabular":
        from yggdrasil.arrow.tabular import ArrowTabular
        return ArrowTabular.from_arrow_batches(
            self.read_arrow_batches(options)
        )

    def read_arrow_batches(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> Iterator[pa.RecordBatch]:
        resolved = self.check_options(options, overrides=locals())
        stream = self._read_arrow_batches(resolved)
        if resolved.target is not None:
            cast = resolved.cast_arrow_batch
            stream = (cast(batch) for batch in stream)
        stream = resolved.resample_arrow_batches(stream)
        stream = resolved.dedup_arrow_batches(stream)
        row_limit = resolved.row_limit
        if row_limit is not None and row_limit > 0:
            yielded = 0
            if not logger.isEnabledFor(logging.DEBUG):
                for batch in stream:
                    remaining = row_limit - yielded
                    if remaining <= 0:
                        return
                    if batch.num_rows <= remaining:
                        yielded += batch.num_rows
                        yield batch
                    else:
                        yield batch.slice(0, remaining)
                        return
            else:
                n_batches = 0
                for batch in stream:
                    remaining = row_limit - yielded
                    if remaining <= 0:
                        return
                    if batch.num_rows > remaining:
                        batch = batch.slice(0, remaining)
                    yielded += batch.num_rows
                    n_batches += 1
                    yield batch
        elif not logger.isEnabledFor(logging.DEBUG):
            yield from stream
            return
        else:
            n_batches = 0
            n_rows = 0
            for batch in stream:
                n_batches += 1
                n_rows += batch.num_rows
                yield batch

    def read_arrow_table(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> pa.Table:
        return self._read_arrow_table(self.check_options(options, overrides=locals()))

    def _read_arrow_table(self, options: O) -> pa.Table:
        # Pull the raw batches off the underlying reader and assemble
        # them into one :class:`pa.Table`, then cast + project + resample
        # + dedup via the options pipeline.
        batches = list(self._read_arrow_batches(options))
        if not batches:
            schema = (
                getattr(options, "target_schema", None)
                or getattr(options, "source_schema", None)
                or Schema.empty()
            )
            return schema.to_arrow_schema().empty_table()
        try:
            table = pa.Table.from_batches(batches)
        except (pa.lib.ArrowInvalid, pa.lib.ArrowTypeError):
            # Part files under this tabular drifted in schema across
            # writes (e.g. a legacy ``binary`` body vs the current
            # ``large_binary``, or an ``int64`` timestamp column vs a
            # proper ``timestamp``). from_batches demands identical
            # schemas — reconcile by conforming every batch to the
            # first one's schema before combining; ``cast_arrow_table``
            # below then projects + casts to the resolved target.
            from yggdrasil.arrow.cast import conform_arrow_batch
            target = batches[0].schema
            table = pa.Table.from_batches(
                [conform_arrow_batch(b, target) for b in batches]
            )
        table = options.cast_arrow_table(table)
        table = options.apply_post_read_table(table)
        if options.row_limit is not None and table.num_rows > options.row_limit:
            table = table.slice(0, options.row_limit)
        return table

    def read_arrow_batch_reader(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pa.RecordBatchReader":
        return self._read_arrow_batch_reader(
            self.check_options(options, overrides=locals())
        )

    def _read_arrow_batch_reader(self, options: O) -> "pa.RecordBatchReader":
        schema = options.check_target(obj=self.collect_schema).merged
        stream = self._read_arrow_batches(options)
        if options.target is not None:
            cast = options.cast_arrow_batch
            stream = (cast(batch) for batch in stream)
        stream = options.resample_arrow_batches(stream)
        stream = options.dedup_arrow_batches(stream)
        # ``RecordBatchReader.from_batches`` validates every batch
        # against the declared schema as it's pulled, so a drifted part
        # file (legacy ``binary`` body, ``int64`` timestamp, …) would
        # raise mid-iteration when no target cast normalized it above.
        # Conform each batch to the declared schema — a zero-copy
        # passthrough when it already matches, so the cast path stays free.
        from yggdrasil.arrow.cast import conform_arrow_batch
        arrow_schema = schema.to_arrow_schema()
        stream = (conform_arrow_batch(b, arrow_schema) for b in stream)
        return pa.RecordBatchReader.from_batches(
            arrow_schema,
            stream,
        )

    def read_arrow_dataset(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pds.Dataset":
        return self._read_arrow_dataset(
            self.check_options(options, overrides=locals())
        )

    def _read_arrow_dataset(self, options: O) -> "pds.Dataset":
        pds = pyarrow_dataset_module()
        reader = self._read_arrow_batch_reader(options)
        return pds.dataset(reader, schema=reader.schema)

    def read_table(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "Tabular | None":
        """Read into an in-memory :class:`Tabular`.

        When ``options.spark_session`` is set, reads via
        :meth:`_read_spark_frame` and wraps in a :class:`Dataset`.
        Otherwise materializes Arrow batches into :class:`ArrowTabular`.
        Returns ``None`` when empty.
        """
        return self._read_table(self.check_options(options, overrides=locals()))

    def _read_table(self, options: O) -> "Tabular | None":
        spark = options.get_spark_session()

        if spark is not None:
            return self.read_spark_dataset(options)

        return self.read_arrow_tabular(options)

    def write_table(
        self,
        obj: Any,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Dispatch *obj* to the best ``_write_*`` hook based on its runtime type.

        Recognizes another :class:`Tabular` (drained as a pyarrow
        record-batch stream), ``pa.Table`` / ``pa.RecordBatch`` /
        ``pa.RecordBatchReader``, polars ``DataFrame`` / ``LazyFrame``,
        pandas ``DataFrame``, pyspark ``DataFrame``, ``list[dict]``,
        ``dict[str, list]``, and iterables of any of the above.
        Module-name sniffing keeps optional engine deps out of the
        import graph — we only touch a frame's API once we've
        confirmed it's an instance of one we know how to drain.
        """
        self._write_table(obj, self.check_options(options, overrides=locals()))

    def _write_table(self, obj: Any, options: O) -> None:
        if obj is None:
            return
        if isinstance(obj, Tabular):
            # Spark-native sources (``yggdrasil.spark.tabular.Dataset``)
            # expose their held :class:`pyspark.sql.DataFrame` via
            # :meth:`_native_spark_frame`. Skip the driver-side
            # ``df.toArrow()`` collect that ``_read_arrow_batches`` would
            # otherwise trigger and hand the frame straight to
            # :meth:`_write_spark_frame` — Spark-aware sinks
            # (:class:`databricks.table.Table`, another ``Dataset``) keep
            # the data on the executors; non-Spark sinks fall through the
            # base :meth:`_write_spark_frame` (``frame.toArrow()`` →
            # ``_write_arrow_table``) so the cost matches the old path.
            spark_frame = obj._native_spark_frame()
            if spark_frame is not None:
                self._write_spark_frame(spark_frame, options)
                return
            self._write_arrow_batches(obj.read_arrow_batches(), options)
            return
        if isinstance(obj, pa.Table):
            self._write_arrow_table(obj, options)
            return
        if isinstance(obj, pa.RecordBatch):
            self._write_arrow_batches([obj], options)
            return
        if isinstance(obj, pa.RecordBatchReader):
            self._write_arrow_batches(obj, options)
            return

        from yggdrasil.pickle.serde import ObjectSerde
        ns, _ = ObjectSerde.module_and_name(obj)
        if ns.startswith("polars"):
            self._write_polars_frame(obj, options)
            return
        if ns.startswith("pandas"):
            self._write_pandas_frame(obj, options)
            return
        if ns.startswith("pyspark"):
            self._write_spark_frame(obj, options)
            return

        if isinstance(obj, Mapping):
            if obj and all(isinstance(v, (list, tuple)) for v in obj.values()):
                self._write_pydict({k: list(v) for k, v in obj.items()}, options)
                return
            self._write_pylist([dict(obj)], options)
            return

        if isinstance(obj, (str, bytes, bytearray, memoryview)):
            raise TypeError(
                f"{type(self).__name__}.write_table can't infer a writer for "
                f"{type(obj).__name__}: {obj!r}. Accepted: pyarrow "
                "Table/RecordBatch/RecordBatchReader, polars DataFrame/LazyFrame, "
                "pandas DataFrame, pyspark DataFrame, list[dict], dict[str, list], "
                "or an iterable of any of those."
            )

        try:
            iterator = iter(obj)
        except TypeError as exc:
            raise TypeError(
                f"{type(self).__name__}.write_table can't infer a writer for "
                f"{ObjectSerde.full_namespace(obj)}: {obj!r}. "
                "Accepted: pyarrow Table/RecordBatch/RecordBatchReader, "
                "polars DataFrame/LazyFrame, pandas DataFrame, pyspark "
                "DataFrame, list[dict], dict[str, list], or an iterable of "
                "any of those."
            ) from exc

        try:
            first = next(iterator)
        except StopIteration:
            return
        import itertools as _it
        rest = _it.chain([first], iterator)
        if isinstance(first, pa.RecordBatch):
            self._write_arrow_batches(rest, options)
            return
        if isinstance(first, pa.Table):
            for inner in rest:
                self._write_arrow_table(inner, options)
            return
        if isinstance(first, Mapping):
            self._write_pylist(rest, options)
            return
        from yggdrasil.data.record import Record
        if isinstance(first, Record):
            self._write_records(rest, options)
            return
        raise TypeError(
            f"{type(self).__name__}.write_table can't infer a writer for "
            f"iterable of {type(first).__name__}: first item {first!r}. "
            "Accepted element types: pyarrow Table/RecordBatch, Mapping, "
            "or yggdrasil Record."
        )

    def write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        options = self.check_options(options, overrides=locals())
        if not logger.isEnabledFor(logging.DEBUG):
            self._write_arrow_batches(batches, options)
        else:
            n_batches = 0
            n_rows = 0
            def _counted() -> "Iterator[pa.RecordBatch]":
                nonlocal n_batches, n_rows
                for batch in batches:
                    n_batches += 1
                    n_rows += batch.num_rows
                    yield batch
            self._write_arrow_batches(_counted(), options)
            logger.debug(
                "%r wrote %d batches / %d rows (mode=%s)",
                self,
                n_batches,
                n_rows,
                options.mode.name,
            )

    def write_arrow_table(
        self, table: pa.Table, options: "O | None" = None, **kwargs: Any,
    ) -> None:
        self._write_arrow_table(
            table, self.check_options(options, overrides=locals()),
        )

    def _write_arrow_table(self, table: pa.Table, options: O) -> None:
        casted = options.cast_arrow_table(table)
        merged = options.merged
        inner = options.copy(
            source=merged if merged is not None else ...,
            sync_metadata=False,
            row_size=None,
            byte_size=None,
        )
        # ``to_batches()`` on a 0-row table yields nothing, dropping the schema
        # the leaf writer needs to persist a valid empty file. Pin the source
        # to the table's own schema (no-op when one is already bound) so the
        # shape survives the batch hand-off.
        inner = inner.check_source(casted.schema)
        self._write_arrow_batches(casted.to_batches(), inner)

    @staticmethod
    def _empty_overwrite_batch(options: O) -> "pa.RecordBatch | None":
        """A 0-row :class:`pa.RecordBatch` carrying the bound write schema.

        Returns ``None`` when no schema is known. The leaf
        ``_write_arrow_batches`` writers feed this through their normal
        OVERWRITE path when the incoming batch stream is empty, so an
        empty input still persists a valid, schema-bearing file (Parquet
        footer, Arrow IPC schema, CSV header, ``[]`` JSON, an empty
        pickled table) instead of a 0-byte — or, on a remote backend,
        never-uploaded — stub a downstream reader can't parse. The schema
        comes from a bound target/source, or — for data written via
        :meth:`_write_arrow_table` — the source table's own schema.
        """
        field = options.merged
        if field is None:
            return None
        return pa.RecordBatch.from_pylist([], schema=field.to_arrow_schema())

    # ==================================================================
    # Union
    # ==================================================================

    def union(self, other: "Any", *, mode: "ModeLike | None" = None) -> "Tabular":
        """Return a Tabular representing ``self UNION ALL other``.

        *mode* controls how mismatched schemas are reconciled:

        - ``Mode.IGNORE`` (default) — keep ``self``'s schema; extra
          columns in *other* are dropped, missing ones are filled null.
        - ``Mode.APPEND`` — widen to the superset schema (every field
          from both sides survives).

        Concrete subclasses override :meth:`_union` for in-place
        mutation (Arrow batch append, Spark ``unionByName``).

        Accepts :class:`Tabular`, ``pa.RecordBatch``, ``pa.Table``,
        ``list[Response]``, or a Spark DataFrame.
        ``None`` returns ``self`` unchanged.
        """
        resolved: Mode = Mode.from_(mode, default=Mode.IGNORE)
        if other is None:
            return self
        if isinstance(other, Tabular):
            return self._union(other, mode=resolved)
        if isinstance(other, (pa.RecordBatch, pa.Table)):
            from yggdrasil.arrow.tabular import ArrowTabular
            return self._union(ArrowTabular(other), mode=resolved)
        if isinstance(other, list):
            from yggdrasil.arrow.tabular import ArrowTabular
            from yggdrasil.io.response import Response as _Resp
            batches = [r.to_arrow_batch(parse=False) for r in other if isinstance(r, _Resp)]
            if not batches:
                return self
            return self._union(ArrowTabular(*batches), mode=resolved)
        if hasattr(other, "unionByName"):
            try:
                from yggdrasil.spark.tabular import SparkDataset
                return self._union(SparkDataset(frame=other), mode=resolved)
            except ImportError:
                pass
        raise TypeError(
            f"{type(self).__name__}.union does not accept "
            f"{type(other).__name__!r}"
        )

    def _union(self, other: "Tabular", *, mode: Mode = Mode.IGNORE) -> "Tabular":
        """Engine-specific union hook.  Override in subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__}._union is not implemented; "
            f"use ArrowTabular or SparkTabular which override this."
        )

    # ==================================================================
    # Polars
    # ==================================================================

    def read_polars_frame(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pl.DataFrame":
        return self._read_polars_frame(
            self.check_options(options, overrides=locals())
        )

    def _read_polars_frame(self, options: O) -> "pl.DataFrame":
        # ``rechunk=False`` keeps polars from copying every column into a
        # single contiguous chunk on ingest. The Arrow table arrives
        # already chunked (one chunk per read flush / spill part), and the
        # default ``rechunk=True`` memcpy's the whole result to coalesce
        # them — pure overhead in both time and peak memory. Skipping it
        # lets the frame share the Arrow buffers zero-copy (numeric columns)
        # or copy once during conversion (everything else); polars handles
        # a multi-chunk frame transparently, and any op that genuinely wants
        # one chunk can ``.rechunk()`` on demand.
        return polars_module().from_arrow(  # type: ignore[return-value]
            self._read_arrow_table(options), rechunk=False,
        )

    def scan_polars_frame(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pl.LazyFrame":
        return self._scan_polars_frame(
            self.check_options(options, overrides=locals())
        )

    def _scan_polars_frame(self, options: O) -> "pl.LazyFrame":
        return polars_module().scan_pyarrow_dataset(self._read_arrow_dataset(options))

    def write_polars_frame(
        self,
        frame: "pl.DataFrame | pl.LazyFrame",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_polars_frame(
            frame, self.check_options(options, overrides=locals()),
        )

    def _write_polars_frame(
        self, frame: "pl.DataFrame | pl.LazyFrame", options: O,
    ) -> None:
        pl = polars_module()

        casted = options.cast_polars_tabular(frame)

        if isinstance(casted, pl.LazyFrame):
            casted = casted.collect()

        # Empty-frame guard reads off the post-collect ``casted`` —
        # the input may have been a LazyFrame (no ``.height``), and
        # the cast can also produce an empty frame from a non-empty
        # source when projecting columns the source doesn't carry.
        if casted.height == 0:
            return

        # ``CompatLevel.newest()`` emits ``string_view`` / ``binary_view`` —
        # great for in-memory pyarrow.compute, but the pyarrow parquet
        # writer (and several downstream tabular paths) still raise
        # ``Slicing not implemented for StringView`` when fed view
        # buffers. ``oldest()`` keeps the legacy flat ``string`` /
        # ``binary`` shape that every reader/writer handles.
        self._write_arrow_table(casted.to_arrow(compat_level=pl.CompatLevel.oldest()), options)

    # ==================================================================
    # Pandas
    # ==================================================================

    def read_pandas_frame(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pandas.DataFrame":
        return self._read_pandas_frame(
            self.check_options(options, overrides=locals())
        )

    def _read_pandas_frame(self, options: O) -> "pandas.DataFrame":
        """Read the table and restore the pandas index it was written with.

        Index levels are recovered from yggdrasil's ``index_key`` /
        ``index_key_level`` per-field tags (written by
        :meth:`_write_pandas_frame`), falling back to pyarrow's
        ``b"pandas"`` ``index_columns`` blob for files produced by other
        tools — see :func:`_collect_index_levels`. Tagged columns are
        promoted into the index in level order; ``__index_level_N__``
        placeholders map back to the unnamed levels they came from.
        """
        table = self._read_arrow_table(options)
        schema = table.schema  # metadata only — safe to read post-destruct
        if self._READ_TABLE_OWNED:
            # ``self_destruct=True`` (with its required ``split_blocks=True``)
            # frees each Arrow column the instant it's converted into pandas,
            # instead of holding the whole Arrow table alongside the finished
            # DataFrame — roughly halves peak memory on a wide table. Sound
            # only because this backend's ``_read_arrow_table`` returns a
            # freshly-decoded, solely-owned table; ``table`` must not be
            # touched for data afterwards, hence the ``del``.
            df = table.to_pandas(split_blocks=True, self_destruct=True)
            del table
        else:
            # Shared/cached table (in-memory holder): destroying it would
            # corrupt later reads, so take the plain, copy-once conversion.
            df = table.to_pandas()
        levels = _collect_index_levels(schema)
        if not levels:
            return df
        levels.sort()
        present = [name for _, name in levels if name in df.columns]
        if not present:
            return df
        df = df.set_index(present)
        df.index.names = [
            None if isinstance(n, str) and n.startswith(_INDEX_PLACEHOLDER_PREFIX)
            else n
            for n in df.index.names
        ]
        return df

    def write_pandas_frame(
        self,
        frame: "pandas.DataFrame",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_pandas_frame(frame, self.check_options(options, overrides=locals()))

    def _write_pandas_frame(self, frame: "pandas.DataFrame", options: O) -> None:
        # Persist every index that carries information so it round-trips
        # on read — named, non-trivial RangeIndex, or an arbitrary label
        # set (``[5, 3, 1]``). Only the freebie ``RangeIndex(0, n, 1)``
        # is dropped: ``to_pandas`` regenerates it for free. ``any(name
        # is not None)`` used to gate this, which silently lost every
        # fully-unnamed non-default index (e.g. a sliced / reordered
        # frame) — ``preserve_index`` then materialises the levels as
        # ``__index_level_N__`` columns that :func:`_tag_index_columns`
        # marks and :meth:`_read_pandas_frame` restores.
        include_index = not _is_default_range_index(frame)

        fast_casted: "pa.Table | None" = None
        target = getattr(options, "target", None)
        if (
            target is not None
            and not include_index
            and frame.columns.is_unique
        ):
            try:
                arrow_schema = target.to_arrow_schema()
            except Exception:
                arrow_schema = None
            if arrow_schema is not None:
                target_by_name = {f.name: f for f in arrow_schema}
                try:
                    arrays: "list[pa.Array]" = []
                    names: "list[str]" = []
                    for name in frame.columns:
                        col = frame[name]
                        tgt = target_by_name.get(name)
                        if tgt is not None and col.dtype.kind == "O":
                            arrays.append(
                                pa.array(col, type=tgt.type, from_pandas=True),
                            )
                        else:
                            arrays.append(pa.array(col, from_pandas=True))
                        names.append(str(name))
                    fast_casted = pa.table(arrays, names=names)
                except (pa.ArrowException, TypeError, ValueError):
                    fast_casted = None

        if fast_casted is not None:
            self._write_arrow_table(fast_casted, options)
            return

        try:
            casted = pa.Table.from_pandas(frame, preserve_index=include_index)
            if include_index:
                casted = _tag_index_columns(casted)
            self._write_arrow_table(casted, options)
        except (pa.ArrowException, TypeError, ValueError):
            pl = polars_module()
            casted = pl.from_pandas(frame, include_index=include_index)
            self._write_polars_frame(casted, options)

    # ==================================================================
    # Spark — driver-side materialize, no streaming spill
    # ==================================================================

    def read_spark_frame(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "SparkDataFrame":
        return self._read_spark_frame(
            self.check_options(options, overrides=locals())
        )

    def _read_spark_frame(self, options: O) -> "SparkDataFrame":
        from yggdrasil.environ import PyEnv
        spark = PyEnv.spark_session(create=True)
        return options.cast_spark(
            spark.createDataFrame(self._read_arrow_table(options))
        )

    def write_spark_frame(
        self,
        frame: "SparkDataFrame",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_spark_frame(
            frame, self.check_options(options, overrides=locals()),
        )

    write_spark = write_spark_frame

    def _write_spark_frame(self, frame: "SparkDataFrame", options: O) -> None:
        to_arrow = getattr(frame, "toArrow", None)
        if to_arrow is not None:
            self._write_arrow_table(to_arrow(), options)
            return
        self._write_pandas_frame(frame.toPandas(), options)

    def read_spark_dataset(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "SparkDataset":
        """Read into a :class:`Dataset` holder.

        Mirrors :meth:`read_arrow_dataset` for the Spark engine: the
        return type is a yggdrasil holder rather than the bare engine
        frame, so callers keep the Tabular surface (chained transforms,
        ``persist`` / ``insert`` / ``schema``, …) without an extra wrap
        at the call site. :class:`Dataset` overrides
        :meth:`_read_spark_dataset` to return itself in place — no
        materialise round trip when the source already speaks Spark.
        """
        return self._read_spark_dataset(
            self.check_options(options, overrides=locals())
        )

    def _read_spark_dataset(self, options: O) -> "SparkDataset":
        from yggdrasil.spark.tabular import SparkDataset

        return SparkDataset.from_spark_frame(
            self._read_spark_frame(options),
            schema=options.target,
        )

    # ==================================================================
    # Python-native
    # ==================================================================

    def read_pylist(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "list[dict]":
        return self._read_pylist(self.check_options(options, overrides=locals()))

    def _read_pylist(self, options: O) -> "list[dict]":
        return self._read_arrow_table(options).to_pylist()

    def write_pylist(
        self,
        data: Iterable[dict],
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_pylist(data, self.check_options(options, overrides=locals()))

    def _write_pylist(self, data: Iterable[dict], options: O) -> None:
        rows = self._normalize_records(data)
        if not rows:
            return
        self._write_arrow_table(pa.Table.from_pylist(rows), options)

    def read_pydict(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "dict[str, list]":
        return self._read_pydict(self.check_options(options, overrides=locals()))

    def _read_pydict(self, options: O) -> "dict[str, list]":
        return self._read_arrow_table(options).to_pydict()

    def write_pydict(
        self,
        data: "dict[str, list]",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_pydict(data, self.check_options(options, overrides=locals()))

    def _write_pydict(self, data: "dict[str, list]", options: O) -> None:
        self._write_arrow_table(pa.Table.from_pydict(data), options)

    def read_record_iterator(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "Iterator[Mapping[str, Any]]":
        """Stream rows as plain ``dict``. True streaming — the full
        table never materializes; ``batch.to_pylist()`` does the
        column→row rotation in pyarrow C++ once per batch."""
        return self._read_record_iterator(
            self.check_options(options, overrides=locals())
        )

    def _read_record_iterator(self, options: O) -> "Iterator[Mapping[str, Any]]":
        for batch in self._read_arrow_batches(options):
            yield from batch.to_pylist()

    def read_records(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "Iterator[Any]":
        """Stream rows as :class:`yggdrasil.data.record.Record`. Lower
        per-row allocation than :meth:`read_pylist` for stable-schema
        sources — the underlying :class:`Schema` is materialized once
        and shared by reference across every record."""
        return self._read_records(
            self.check_options(options, overrides=locals())
        )

    def _read_records(self, options: O) -> "Iterator[Any]":
        from yggdrasil.data.record import Record
        yield from Record.from_arrow_batches(self._read_arrow_batches(options))

    def write_records(
        self,
        records: "Iterable[Any]",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_records(
            records, self.check_options(options, overrides=locals()),
        )

    def _write_records(self, records: "Iterable[Any]", options: O) -> None:
        """Bucket records by ``options.row_size`` and delegate to
        :meth:`_write_arrow_batches`. The first record's schema becomes
        the writer's target — heterogeneous-schema streams are silently
        re-aligned via dict-order iteration, so callers that need
        strictness should pre-validate. Subclasses with a row-native
        sink (SQL bulk-insert, Spark createDataFrame) should override
        to skip the row → Arrow round-trip.
        """
        from yggdrasil.data.record import Record

        chunk_size = max(1, getattr(options, "row_size", None) or 1024)
        chunk_rows: "list[dict]" = []
        chunk_schema: "pa.Schema | None" = None
        inner = options.copy(sync_metadata=False) if options.sync_metadata else options

        def _flush() -> None:
            if not chunk_rows:
                return
            batch = pa.RecordBatch.from_pylist(chunk_rows, schema=chunk_schema)
            self._write_arrow_batches([batch], inner)
            chunk_rows.clear()

        for rec in records:
            if isinstance(rec, Record):
                if chunk_schema is None:
                    chunk_schema = rec.schema.to_arrow_schema()
                chunk_rows.append(rec.to_dict())
            elif isinstance(rec, Mapping):
                chunk_rows.append(dict(rec))
            else:
                raise TypeError(
                    f"_write_records expected Record or Mapping rows; "
                    f"got {type(rec).__name__}: {rec!r}"
                )
            if len(chunk_rows) >= chunk_size:
                _flush()
        _flush()

    def unique(
        self,
        by: "str | Any | Iterable[Any]",
    ) -> "Tabular":
        """Drop duplicate rows on *by*; keep first occurrence per key tuple.

        Parameters
        ----------
        by
            One or more column references — :class:`str` column names,
            :class:`yggdrasil.data.Field` instances (resolved via
            :attr:`Field.name`), or any iterable mixing the two. Empty
            / ``None`` is a no-op — returns ``self``.

        Returns
        -------
        Tabular
            A new holder carrying the deduped rows. Spark-shaped
            inputs (anything whose :meth:`_native_spark_frame`
            exposes a :class:`pyspark.sql.DataFrame`) return a fresh
            :class:`yggdrasil.spark.tabular.Dataset` over the
            spark-side dedup; everything else collects through Arrow
            and returns an :class:`yggdrasil.arrow.tabular.ArrowTabular`.
        """
        keys = _coerce_column_keys(by)
        if not keys:
            return self
        return self._unique(keys=keys)

    def _unique(self, *, keys: list[str]) -> "Tabular":
        """Typed-argument dedup hook.

        Default routes on :meth:`_native_spark_frame`:

        * holds a Spark frame → :func:`yggdrasil.spark.ops.dedup_spark_dataframe`
          wrapped in a fresh :class:`Dataset`,
        * otherwise → :meth:`read_arrow_table` →
          :func:`yggdrasil.arrow.ops.dedup_arrow_table` →
          :class:`ArrowTabular`.

        Subclasses that already speak a typed engine path can override
        to skip the round trip entirely; the default already handles
        the two engines the codebase ships with.
        """
        return _default_unique(self, keys=keys)

    def resample(
        self,
        on: "str | Any",
        sampling: "int | float | Any",
        *,
        partition_by: "str | Any | Iterable[Any] | None" = None,
        fill_strategy: "str | None" = "ffill",
    ) -> "Tabular":
        """Align rows to a fixed time grid on *on*; one row per bucket.

        Parameters
        ----------
        on
            The time column to resample on — column name
            (:class:`str`) or :class:`yggdrasil.data.Field`.
        sampling
            Bucket size. Accepted shapes:

            * :class:`int` / :class:`float` — seconds (floats are
              rounded to the nearest integer second).
            * :class:`datetime.timedelta` — total seconds.
            * :class:`str` — ISO-8601 duration (``"PT1H"``,
              ``"P1D"``, ``"PT15M"``) parsed via
              :func:`yggdrasil.data.types.primitive.temporal._parse_iso_duration`.

            ``sampling <= 0`` is a short-circuit — returns ``self``.
        partition_by
            Entity columns the resample is independent on. ``None`` /
            empty → flat global timeline. Same coercion as
            :meth:`unique`'s ``by``.
        fill_strategy
            How to fill nulls left by the bucket's "first" aggregation.
            ``"ffill"`` (default), ``"bfill"``, or ``"none"`` /
            ``None`` to disable. See
            :func:`yggdrasil.arrow.ops.fill_arrow_table` for the
            full semantics.

        Returns
        -------
        Tabular
            Spark-shaped holders return a :class:`Dataset` over the
            spark-side resample; everything else returns an
            :class:`ArrowTabular` over the arrow-side resample.
        """
        time_column = _coerce_column_name(on)
        sampling_seconds = _coerce_sampling_seconds(sampling)
        if sampling_seconds <= 0:
            return self
        part_cols = _coerce_column_keys(partition_by) if partition_by else []
        return self._resample(
            time_column=time_column,
            sampling_seconds=sampling_seconds,
            partition_by=part_cols,
            fill_strategy=fill_strategy,
        )

    def _resample(
        self,
        *,
        time_column: str,
        sampling_seconds: int,
        partition_by: list[str],
        fill_strategy: "str | None",
    ) -> "Tabular":
        """Typed-argument resample hook.

        Same routing model as :meth:`_unique` — Spark-shaped
        holders run :func:`yggdrasil.spark.ops.resample_spark_dataframe`
        and return a fresh :class:`Dataset`; everything else routes
        through :func:`yggdrasil.arrow.ops.resample_arrow_table` and
        returns an :class:`ArrowTabular`.
        """
        return _default_resample(
            self,
            time_column=time_column,
            sampling_seconds=sampling_seconds,
            partition_by=partition_by,
            fill_strategy=fill_strategy,
        )

    # ==================================================================
    # Projection / row filter
    #
    # Same public / typed-hook split as ``unique`` / ``resample``: the
    # public method accepts flexible input and coerces, the private
    # ``_select`` / ``_drop`` / ``_filter`` see canonical typed
    # arguments and engine-route on :meth:`_native_spark_frame`.
    # ==================================================================

    def select(
        self,
        *columns: "str | Any",
    ) -> "Tabular":
        """Project to *columns* and return a new Tabular.

        Each entry is a column reference — :class:`str`, a
        :class:`yggdrasil.data.Field` (resolved via
        :attr:`Field.name`), or an iterable mixing both. The result
        preserves the caller's order, which matches both
        :meth:`pyarrow.Table.select` and
        :meth:`pyspark.sql.DataFrame.select` semantics.

        Raises :class:`ValueError` on an empty selection — a zero-
        column projection is almost always a caller mistake; pass
        :class:`Schema.empty` projections through the cast surface
        instead.
        """
        cols = _flatten_column_args(columns)
        if not cols:
            raise ValueError(
                f"{type(self).__name__}.select needs at least one column; "
                "pass column names ('a', 'b'), Field objects, or "
                "iterables of either."
            )
        return self._select(columns=cols)

    def _select(self, *, columns: list[str]) -> "Tabular":
        """Typed-argument projection hook.

        Spark-native holders return a fresh :class:`Dataset` carrying
        ``frame.select(*columns)``; everything else collects through
        :meth:`read_arrow_table`, projects via
        :meth:`pa.Table.select`, and wraps the result in an
        :class:`ArrowTabular`.
        """
        return _default_select(self, columns=columns)

    def drop(
        self,
        *columns: "str | Any",
    ) -> "Tabular":
        """Return a new Tabular with the named columns removed.

        Columns missing from the source are silently ignored —
        matches Spark's :meth:`DataFrame.drop` and pyarrow's
        :meth:`Table.drop_columns` (when filtered to existing
        names). An empty argument list is a no-op that returns
        ``self``.
        """
        cols = _flatten_column_args(columns)
        if not cols:
            return self
        return self._drop(columns=cols)

    def _drop(self, *, columns: list[str]) -> "Tabular":
        """Typed-argument drop hook.

        Same routing as :meth:`_select`. Missing columns in the
        source are filtered out before the underlying engine's
        ``drop`` runs so neither pyarrow nor pyspark raises on
        an absent reference.
        """
        return _default_drop(self, columns=columns)

    def filter(
        self,
        predicate: "PredicateLike",
    ) -> "Tabular":
        """Drop rows where *predicate* is false.

        ``predicate`` accepts every shape
        :meth:`yggdrasil.execution.expr.Expression.from_`
        recognises:

        * a SQL predicate string (``"x > 0 AND y IS NOT NULL"``),
          parsed by the in-tree SQL parser;
        * a yggdrasil :class:`Predicate` node
          (``col("x") > 0``, :func:`is_in`, :func:`between`, …);
        * a native engine expression —
          :class:`pyarrow.compute.Expression`,
          :class:`polars.Expr`, or :class:`pyspark.sql.Column` —
          lifted via the matching backend.

        The predicate is parsed once and dispatched to the typed
        :meth:`_filter` hook; the engine-side filter then runs in
        its native kernel (Arrow C++, Spark Catalyst) so the row
        scan stays vectorised.
        """
        pred = _coerce_predicate(predicate)
        return self._filter(predicate=pred)

    def _filter(self, *, predicate: "Predicate") -> "Tabular":
        """Typed-argument row-filter hook.

        ``predicate`` is always a yggdrasil :class:`Predicate` at
        this point (the public :meth:`filter` did the lift). Spark-
        native holders compile to a :class:`pyspark.sql.Column` and
        return a fresh :class:`Dataset` via
        :meth:`DataFrame.filter`; everything else compiles to a
        :class:`pyarrow.compute.Expression` and runs the C++
        kernel via :meth:`Predicate.filter_arrow_table`.
        """
        return _default_filter(self, predicate=predicate)

    def cast(
        self,
        options: "O | None" = None,
        **kwargs
    ) -> "Tabular":
        """Cast rows, returning a new :class:`Tabular`.

        Accepts a :class:`Schema` or :class:`CastOptions`. When
        *options* is given, reads to arrow and casts each batch
        through :meth:`CastOptions.cast_arrow_batch`.
        """
        options = CastOptions.check(options, **kwargs)
        return self._cast(options)

    def _cast(self, options: O) -> "Tabular":
        from yggdrasil.arrow.tabular import ArrowTabular
        return ArrowTabular.from_arrow_batches(
            self.read_arrow_batches(options)
        )

    # ==================================================================
    # ``to_*`` aliases — pandas-style spelling for the ``read_*`` surface.
    # ==================================================================

    to_arrow = read_arrow_table
    to_arrow_batches = read_arrow_batches
    to_arrow_table = read_arrow_table
    to_arrow_batch_reader = read_arrow_batch_reader
    to_arrow_dataset = read_arrow_dataset

    to_table = read_table

    to_polars = read_polars_frame
    to_polars_frame = read_polars_frame

    to_pandas = read_pandas_frame
    to_pandas_frame = read_pandas_frame

    to_spark = read_spark_frame
    to_spark_frame = read_spark_frame
    to_spark_dataset = read_spark_dataset

    to_pylist = read_pylist
    to_pydict = read_pydict
    to_record_iterator = read_record_iterator
    to_records = read_records

    # ==================================================================
    # Lazy execution plan
    # ==================================================================

    def lazy(self) -> "LazyTabular":
        """Return a :class:`LazyTabular` wrapping this source.

        Transformations on the returned object (``select``, ``filter``,
        ``join``, …) accumulate in an :class:`ExecutionPlan` without
        touching data.  Any ``read_*`` call materialises the plan.
        """
        from yggdrasil.plan.lazy import LazyTabular

        return LazyTabular(self)