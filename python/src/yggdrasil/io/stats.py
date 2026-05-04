"""Format-agnostic columnar statistics, persisted as Arrow IPC.

Parquet ships with min/max/null_count/distinct_count per row group
inside the file footer. CSV, JSON, NDJSON, IPC, XLSX, XML do not —
predicate pushdown, partition pruning, and rowcount-only queries
are all blind once the data lands in those formats.

This module is the single canonical place for columnar stats across
every :class:`yggdrasil.io.buffer.TabularIO` leaf and folder. The
on-disk encoding is **Arrow IPC** (file format): self-describing,
fast to mmap, schema-rich, supports nested types and per-column
metadata. One IPC file can carry stats for an entire folder
(per-file rows + an aggregate row), or for a single file as a
sidecar.

Three entry points
------------------

- :meth:`Stats.compute` — scan an Arrow source (Table, batch
  iterable, or :class:`TabularIO`) and emit a :class:`Stats`.
- :meth:`Stats.merge` — combine multiple :class:`Stats` instances.
  Min/max collapse via Arrow ``min_max`` semantics; counts add;
  distinct counts go to ``None`` when any input was unset (a sum
  would overcount across overlapping inputs).
- :meth:`Stats.read` / :meth:`Stats.write` — Arrow IPC round-trip
  against bytes or a :class:`yggdrasil.io.fs.Path`.

Encoding
--------

A :class:`Stats` is serialised as a single :class:`pyarrow.Table`
with one row per ``(source, column)`` pair. ``source`` is the
file/identifier the stats belong to (``None`` rows are aggregates
across all sources). The schema is fixed so any consumer can
``pa.ipc.read_table(...)`` the file and reason about it without
import-side type registration.

::

    pa.schema([
        pa.field("source", pa.string()),       # nullable; null = aggregate
        pa.field("column", pa.string()),
        pa.field("arrow_type", pa.string()),   # str(arrow_type)
        pa.field("num_values", pa.int64()),
        pa.field("null_count", pa.int64()),
        pa.field("distinct_count", pa.int64()), # nullable
        pa.field("min_value", pa.string()),    # nullable; str(scalar.as_py())
        pa.field("max_value", pa.string()),    # nullable
        pa.field("byte_size", pa.int64()),
    ])

Schema-level metadata carries ``yggdrasil.stats.version`` and
``yggdrasil.stats.created_at`` so future readers can detect which
schema variant they're looking at.
"""

from __future__ import annotations

import dataclasses
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Union,
)

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.io.types import BytesLike  # noqa: F401  (re-export hook)


if TYPE_CHECKING:
    from yggdrasil.io.buffer.base import TabularIO
    from yggdrasil.io.buffer.bytes_io import BytesIO
    from yggdrasil.io.fs import Path


__all__ = [
    "ColumnStats",
    "Stats",
    "STATS_SCHEMA",
    "STATS_VERSION",
    "STATS_FILENAME",
]


#: On-wire schema version. Bump on incompatible schema changes; the
#: reader cross-checks this against the stored metadata and refuses
#: to load mismatched files (forward-compat is opt-in via the caller).
STATS_VERSION: str = "1"

#: Default sidecar filename for folder-level stats (under ``.ygg/``).
STATS_FILENAME: str = "stats.arrow"


# ---------------------------------------------------------------------------
# On-disk schema
# ---------------------------------------------------------------------------


def _build_stats_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source", pa.string(), nullable=True),
            pa.field("column", pa.string(), nullable=False),
            pa.field("arrow_type", pa.string(), nullable=False),
            pa.field("num_values", pa.int64(), nullable=False),
            pa.field("null_count", pa.int64(), nullable=False),
            pa.field("distinct_count", pa.int64(), nullable=True),
            pa.field("min_value", pa.string(), nullable=True),
            pa.field("max_value", pa.string(), nullable=True),
            pa.field("byte_size", pa.int64(), nullable=False),
        ],
        metadata={
            b"yggdrasil.stats.version": STATS_VERSION.encode("ascii"),
        },
    )


STATS_SCHEMA: pa.Schema = _build_stats_schema()


# ---------------------------------------------------------------------------
# ColumnStats
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ColumnStats:
    """Per-column statistics for a single source.

    Built by :meth:`Stats.compute` from an Arrow source. ``min_value``
    / ``max_value`` carry the Python representation of the bounds
    (via ``pa.Scalar.as_py``) — losing some fidelity for exotic
    types (decimal128, fixed-size binary) but cheap to compare and
    serialise. The original Arrow type is kept in :attr:`arrow_type`
    so consumers can cast back when they need a strict scalar.
    """

    name: str
    arrow_type: pa.DataType
    num_values: int
    null_count: int
    byte_size: int
    min_value: Any = None
    max_value: Any = None
    distinct_count: int | None = None

    @property
    def has_nulls(self) -> bool:
        return self.null_count > 0

    @property
    def is_empty(self) -> bool:
        return self.num_values == 0

    def merge_with(self, other: "ColumnStats") -> "ColumnStats":
        """Combine two stats for the same column, across two sources.

        Distinct count goes to ``None`` if either input is unset —
        summing distinct counts would overcount when the inputs
        share values. Re-run :meth:`Stats.compute(distinct=True)`
        on the union if you need it post-merge.
        """
        if self.name != other.name:
            raise ValueError(
                f"Cannot merge ColumnStats for different columns: "
                f"{self.name!r} vs {other.name!r}"
            )

        # Merge min/max using natural ordering on the Python repr.
        # Equal types are the only case we trust; mismatched types
        # collapse to whichever side has a non-None value.
        new_min = _merge_bound(self.min_value, other.min_value, op="min")
        new_max = _merge_bound(self.max_value, other.max_value, op="max")

        if self.distinct_count is None or other.distinct_count is None:
            distinct = None
        else:
            # Best we can do without a re-scan: lower bound on the
            # union is the larger of the two; upper bound is the sum.
            # Picking the upper bound preserves the "this could be
            # at most N" semantic readers expect.
            distinct = self.distinct_count + other.distinct_count

        return ColumnStats(
            name=self.name,
            arrow_type=self.arrow_type if self.arrow_type == other.arrow_type
            else self.arrow_type,
            num_values=self.num_values + other.num_values,
            null_count=self.null_count + other.null_count,
            byte_size=self.byte_size + other.byte_size,
            min_value=new_min,
            max_value=new_max,
            distinct_count=distinct,
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Stats:
    """Columnar statistics for one or more sources.

    Either:

    - **Single-source** — :attr:`sources` has one entry, the union
      of every :class:`ColumnStats` covers that source.
    - **Multi-source** — :attr:`sources` lists one identifier per
      contributing file (folder children, partition leaves, …) and
      :attr:`columns` carries one :class:`ColumnStats` per
      ``(source, column)`` pair *plus* an aggregate per column at
      ``source=None``.

    Build via :meth:`compute` (scan a TabularIO / Table / batches),
    :meth:`merge` (combine several existing :class:`Stats`), or
    :meth:`read` (load an Arrow IPC sidecar).
    """

    schema: pa.Schema
    sources: tuple[str | None, ...]
    columns: tuple[ColumnStats, ...]
    per_source_columns: Mapping[str | None, tuple[ColumnStats, ...]] = (
        dataclasses.field(default_factory=dict)
    )

    # ----- Construction -----

    @classmethod
    def compute(
        cls,
        source: Any,
        *,
        name: str | None = None,
        columns: Sequence[str] | None = None,
        distinct: bool = False,
        with_aggregate: bool = True,
    ) -> "Stats":
        """Compute statistics for *source*.

        :param source: a :class:`pyarrow.Table`, :class:`pyarrow.RecordBatch`,
            an iterable of record batches, or anything
            :class:`TabularIO`-shaped (something with
            ``read_arrow_table``).
        :param name: identifier stored as ``source`` in the encoded
            row(s). ``None`` (default) marks the row as the aggregate.
        :param columns: subset to compute. ``None`` means all
            columns from the source's schema.
        :param distinct: when ``True``, also compute
            :attr:`ColumnStats.distinct_count` via
            :func:`pyarrow.compute.count_distinct`. ``False`` (default)
            skips the distinct scan — it doubles the column-wise cost
            and most callers don't need it.
        :param with_aggregate: ignored on single-source compute (we
            *are* the aggregate). Honoured by :meth:`merge`.
        """
        del with_aggregate  # only relevant in merge path

        table = _coerce_to_arrow_table(source)
        if columns is not None:
            keep = [c for c in columns if c in table.column_names]
            table = table.select(keep)

        col_stats = tuple(
            _compute_column(table.schema.field(i), table.column(i),
                            distinct=distinct)
            for i in range(table.num_columns)
        )
        sources = (name,)
        per_source: dict[str | None, tuple[ColumnStats, ...]] = {
            name: col_stats,
        }
        return cls(
            schema=table.schema,
            sources=sources,
            columns=col_stats,
            per_source_columns=per_source,
        )

    @classmethod
    def merge(
        cls,
        items: Sequence["Stats"],
        *,
        with_aggregate: bool = True,
    ) -> "Stats":
        """Combine several :class:`Stats` into one.

        Per-source rows are preserved; the merged value also carries
        an aggregate row per column (``source=None``) when
        ``with_aggregate=True``. The aggregate row's min/max use
        natural ordering on the Python repr — see
        :meth:`ColumnStats.merge_with`.
        """
        if not items:
            raise ValueError("merge() requires at least one Stats input")

        # Build a unified schema by Arrow's column-union rules.
        merged_schema = items[0].schema
        for item in items[1:]:
            merged_schema = pa.unify_schemas([merged_schema, item.schema])

        # Combine sources in stable order.
        sources_seen: list[str | None] = []
        per_source: dict[str | None, tuple[ColumnStats, ...]] = {}
        for stats in items:
            for src, cols in stats.per_source_columns.items():
                if src in per_source:
                    # Two inputs claim the same source identifier;
                    # treat the second as additive. Rare unless the
                    # caller is merging accidental duplicates.
                    per_source[src] = tuple(
                        a.merge_with(b)
                        for a, b in zip(per_source[src], cols)
                    )
                else:
                    sources_seen.append(src)
                    per_source[src] = cols

        # Aggregate across every per-source column.
        aggregate: tuple[ColumnStats, ...] = ()
        if with_aggregate:
            agg_by_name: dict[str, ColumnStats] = {}
            for cols in per_source.values():
                for col in cols:
                    existing = agg_by_name.get(col.name)
                    if existing is None:
                        agg_by_name[col.name] = col
                    else:
                        agg_by_name[col.name] = existing.merge_with(col)
            aggregate = tuple(
                agg_by_name[f.name]
                for f in merged_schema
                if f.name in agg_by_name
            )

        if with_aggregate:
            sources_seen.append(None)
            per_source[None] = aggregate

        # ``columns`` is the aggregate when present, else the merge
        # of all source columns flattened.
        flat = aggregate if with_aggregate else tuple(
            c for cols in per_source.values() for c in cols
        )

        return cls(
            schema=merged_schema,
            sources=tuple(sources_seen),
            columns=flat,
            per_source_columns=per_source,
        )

    # ----- Lookup helpers -----

    def column(self, name: str, source: str | None = None) -> ColumnStats:
        """Return the :class:`ColumnStats` for *name* under *source*.

        ``source=None`` (default) returns the aggregate row for the
        column. Raises :class:`KeyError` when the (source, column)
        pair is absent.
        """
        cols = self.per_source_columns.get(source)
        if cols is None:
            raise KeyError(
                f"No stats stored for source={source!r}. Known: "
                f"{list(self.per_source_columns)!r}"
            )
        for c in cols:
            if c.name == name:
                return c
        raise KeyError(
            f"No stats for column={name!r} under source={source!r}. "
            f"Known columns: {[c.name for c in cols]!r}"
        )

    def __iter__(self) -> Iterator[ColumnStats]:
        return iter(self.columns)

    @property
    def num_rows(self) -> int:
        """Total row count across every source.

        For an aggregate-only :class:`Stats` (single source or merged
        with aggregate), this is the sum of ``num_values`` for any
        column (they're equal across columns by Arrow's tabular
        invariant); the implementation reads the first column for
        the aggregate row.
        """
        for col in self.columns:
            return col.num_values
        return 0

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    # ----- Arrow IPC round-trip -----

    def to_arrow_table(self) -> pa.Table:
        """Encode the stats as a :class:`pyarrow.Table` matching :data:`STATS_SCHEMA`."""
        cols: dict[str, list[Any]] = {
            "source": [], "column": [], "arrow_type": [],
            "num_values": [], "null_count": [], "distinct_count": [],
            "min_value": [], "max_value": [], "byte_size": [],
        }
        for src, col_stats in self.per_source_columns.items():
            for c in col_stats:
                cols["source"].append(src)
                cols["column"].append(c.name)
                cols["arrow_type"].append(str(c.arrow_type))
                cols["num_values"].append(int(c.num_values))
                cols["null_count"].append(int(c.null_count))
                cols["distinct_count"].append(
                    None if c.distinct_count is None else int(c.distinct_count)
                )
                cols["min_value"].append(
                    None if c.min_value is None else _stringify(c.min_value)
                )
                cols["max_value"].append(
                    None if c.max_value is None else _stringify(c.max_value)
                )
                cols["byte_size"].append(int(c.byte_size))

        meta = {
            b"yggdrasil.stats.version": STATS_VERSION.encode("ascii"),
            b"yggdrasil.stats.created_at": str(int(time.time())).encode("ascii"),
        }
        schema = STATS_SCHEMA.with_metadata(meta)
        arrays = [
            pa.array(cols[f.name], type=f.type) for f in STATS_SCHEMA
        ]
        return pa.Table.from_arrays(arrays, schema=schema)

    @classmethod
    def from_arrow_table(cls, table: pa.Table) -> "Stats":
        """Inverse of :meth:`to_arrow_table`. Validates the schema."""
        cls._check_schema(table.schema)

        per_source: dict[str | None, list[ColumnStats]] = {}
        sources_in_order: list[str | None] = []
        # Walk row-wise; the table is small (one row per (src, col))
        # so a Python loop is fine and keeps the parsing readable.
        rows = table.to_pylist()
        for row in rows:
            src = row.get("source")
            arrow_type = _parse_arrow_type(row["arrow_type"])
            col = ColumnStats(
                name=row["column"],
                arrow_type=arrow_type,
                num_values=int(row["num_values"]),
                null_count=int(row["null_count"]),
                byte_size=int(row["byte_size"]),
                distinct_count=(
                    None if row.get("distinct_count") is None
                    else int(row["distinct_count"])
                ),
                min_value=row.get("min_value"),
                max_value=row.get("max_value"),
            )
            if src not in per_source:
                per_source[src] = []
                sources_in_order.append(src)
            per_source[src].append(col)

        # Build a "logical" schema from the recorded columns. Use
        # the aggregate row when present, otherwise the first
        # source's columns.
        logical_columns: tuple[ColumnStats, ...]
        if None in per_source:
            logical_columns = tuple(per_source[None])
        elif sources_in_order:
            logical_columns = tuple(per_source[sources_in_order[0]])
        else:
            logical_columns = ()

        logical_schema = pa.schema([
            pa.field(c.name, c.arrow_type)
            for c in logical_columns
        ])

        return cls(
            schema=logical_schema,
            sources=tuple(sources_in_order),
            columns=logical_columns,
            per_source_columns={
                src: tuple(per_source[src]) for src in sources_in_order
            },
        )

    @staticmethod
    def _check_schema(schema: pa.Schema) -> None:
        version_bytes = (schema.metadata or {}).get(
            b"yggdrasil.stats.version"
        )
        if version_bytes is None:
            return  # Tolerate older schemas without metadata.
        version = version_bytes.decode("ascii", errors="replace")
        if version != STATS_VERSION:
            raise ValueError(
                f"Stats version mismatch: file has {version!r}, "
                f"this code expects {STATS_VERSION!r}. Upgrade or "
                f"recompute the sidecar."
            )

    def to_ipc(self) -> bytes:
        """Serialise to Arrow IPC file bytes."""
        sink = pa.BufferOutputStream()
        table = self.to_arrow_table()
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
        return sink.getvalue().to_pybytes()

    @classmethod
    def from_ipc(cls, source: "bytes | bytearray | memoryview | pa.Buffer") -> "Stats":
        """Inverse of :meth:`to_ipc`."""
        buf = source if isinstance(source, pa.Buffer) else pa.py_buffer(source)
        with pa.ipc.open_file(pa.BufferReader(buf)) as reader:
            table = reader.read_all()
        return cls.from_arrow_table(table)

    def write(
        self,
        target: "Union[Path, BytesIO, str]",
    ) -> None:
        """Persist as Arrow IPC at *target*.

        ``target`` may be a :class:`yggdrasil.io.fs.Path`, a string
        path/URL, or any :class:`BytesIO`. Writes are atomic via
        the underlying ``write_bytes`` (folder writes go through
        stage+rename; raw paths get a single write).
        """
        from yggdrasil.io.fs import Path as _Path  # local — avoid cycle
        from yggdrasil.io.buffer.bytes_io import BytesIO as _BytesIO

        payload = self.to_ipc()
        if isinstance(target, _BytesIO):
            target.write_bytes(payload)
            return
        path = (
            target if isinstance(target, _Path)
            else _Path.from_(target)
        )
        path.write_bytes(payload, mode="wb", parents=True)

    @classmethod
    def read(
        cls,
        source: "Union[Path, BytesIO, str, bytes, bytearray, memoryview]",
    ) -> "Stats":
        """Inverse of :meth:`write`. EAFP — missing files raise
        :class:`FileNotFoundError`."""
        from yggdrasil.io.fs import Path as _Path
        from yggdrasil.io.buffer.bytes_io import BytesIO as _BytesIO

        if isinstance(source, (bytes, bytearray, memoryview)):
            return cls.from_ipc(source)
        if isinstance(source, _BytesIO):
            return cls.from_ipc(source.to_bytes())

        path = (
            source if isinstance(source, _Path)
            else _Path.from_(source)
        )
        return cls.from_ipc(path.read_bytes())


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _coerce_to_arrow_table(source: Any) -> pa.Table:
    """Convert *source* to a :class:`pyarrow.Table`.

    Accepts: ``pa.Table``, ``pa.RecordBatch``, an iterable of
    batches/tables, anything with a ``read_arrow_table()`` method
    (every :class:`TabularIO`).
    """
    if isinstance(source, pa.Table):
        return source
    if isinstance(source, pa.RecordBatch):
        return pa.Table.from_batches([source])
    if hasattr(source, "read_arrow_table"):
        return source.read_arrow_table()
    if isinstance(source, Iterable):
        batches = []
        tables: list[pa.Table] = []
        for item in source:
            if isinstance(item, pa.Table):
                tables.append(item)
            elif isinstance(item, pa.RecordBatch):
                batches.append(item)
            else:
                raise TypeError(
                    f"Stats.compute: iterable item {type(item)!r} not "
                    "supported. Expected pa.Table or pa.RecordBatch."
                )
        if batches and tables:
            tables.append(pa.Table.from_batches(batches))
        elif batches:
            tables = [pa.Table.from_batches(batches)]
        if not tables:
            return pa.Table.from_pylist([])
        if len(tables) == 1:
            return tables[0]
        return pa.concat_tables(tables, promote_options="default")
    raise TypeError(
        f"Stats.compute: source type {type(source)!r} not supported. "
        "Expected pa.Table, pa.RecordBatch, batch iterable, or any "
        "TabularIO-shaped object with read_arrow_table()."
    )


def _compute_column(
    field: pa.Field,
    column: "pa.ChunkedArray | pa.Array",
    *,
    distinct: bool,
) -> ColumnStats:
    """Compute one column's stats. The Arrow compute kernels do most
    of the heavy lifting — kept inline so subclasses can override
    if they have backend-specific shortcuts (e.g. read from a
    Parquet footer)."""
    array = column

    if isinstance(array, pa.ChunkedArray):
        num_values = array.length()
        null_count = array.null_count
        nbytes = sum(_chunk_nbytes(c) for c in array.chunks)
    else:
        num_values = len(array)
        null_count = array.null_count
        nbytes = _chunk_nbytes(array)

    min_value: Any = None
    max_value: Any = None
    if num_values > null_count and _supports_min_max(field.type):
        try:
            mm = pc.min_max(array)
            scalar = mm.as_py()
            if isinstance(scalar, dict):
                min_value = scalar.get("min")
                max_value = scalar.get("max")
        except Exception:
            # Some types (nested, extension) don't support min_max;
            # leave bounds at None — null_count and num_values still
            # give callers something useful.
            pass

    distinct_count: int | None = None
    if distinct and _supports_count_distinct(field.type):
        try:
            distinct_count = int(pc.count_distinct(array).as_py())
        except Exception:
            distinct_count = None

    return ColumnStats(
        name=field.name,
        arrow_type=field.type,
        num_values=num_values,
        null_count=null_count,
        byte_size=int(nbytes),
        min_value=min_value,
        max_value=max_value,
        distinct_count=distinct_count,
    )


def _chunk_nbytes(array: pa.Array) -> int:
    """Estimated byte size for *array*. ``Array.nbytes`` is exact for
    contiguous arrays; ChunkedArray callers sum across chunks."""
    nb = getattr(array, "nbytes", None)
    if nb is not None:
        return int(nb)
    # Buffer-walk fallback for backends that lack ``.nbytes``.
    total = 0
    for buf in array.buffers():
        if buf is not None:
            total += buf.size
    return total


def _supports_min_max(arrow_type: pa.DataType) -> bool:
    """Heuristic: skip nested/binary-list types where min_max raises."""
    if pa.types.is_struct(arrow_type) or pa.types.is_list(arrow_type) \
            or pa.types.is_large_list(arrow_type) \
            or pa.types.is_fixed_size_list(arrow_type) \
            or pa.types.is_map(arrow_type) \
            or pa.types.is_union(arrow_type):
        return False
    return True


def _supports_count_distinct(arrow_type: pa.DataType) -> bool:
    """Same set as min_max — distinct over nested types is undefined."""
    return _supports_min_max(arrow_type)


def _merge_bound(a: Any, b: Any, *, op: str) -> Any:
    """Pick the smaller / larger of two bound values.

    Either side may be ``None`` (no contribution); ``op`` is
    ``"min"`` or ``"max"``. Comparison falls back to string
    comparison when the natural ordering raises (mismatched types
    after a merge), so a heterogeneous merge still produces *some*
    bound rather than tanking the merge.
    """
    if a is None:
        return b
    if b is None:
        return a
    try:
        return min(a, b) if op == "min" else max(a, b)
    except TypeError:
        sa, sb = str(a), str(b)
        return min(sa, sb) if op == "min" else max(sa, sb)


def _stringify(value: Any) -> str:
    """Render a Python value to its short string form for the IPC sidecar.

    Bytes/bytearray are rendered as a hex prefix (capped at 32
    bytes, ``…`` for longer) — readable for binary mins/maxes
    without bloating the sidecar with unprintable chunks.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        b = bytes(value)
        if len(b) <= 32:
            return f"hex:{b.hex()}"
        return f"hex:{b[:32].hex()}…"
    return str(value)


def _parse_arrow_type(text: str) -> pa.DataType:
    """Best-effort parse of a stringified Arrow type back to a
    :class:`pa.DataType`. Falls back to ``pa.string()`` for shapes
    we can't easily reverse — the stats sidecar's source-of-truth
    for type identity is :attr:`ColumnStats.arrow_type` *before*
    the IPC trip; readers that need a strict type should re-resolve
    via the original source's schema.
    """
    table = {
        "int8": pa.int8(), "int16": pa.int16(),
        "int32": pa.int32(), "int64": pa.int64(),
        "uint8": pa.uint8(), "uint16": pa.uint16(),
        "uint32": pa.uint32(), "uint64": pa.uint64(),
        "float": pa.float32(), "double": pa.float64(),
        "bool": pa.bool_(),
        "string": pa.string(), "utf8": pa.string(),
        "large_string": pa.large_string(), "large_utf8": pa.large_string(),
        "binary": pa.binary(), "large_binary": pa.large_binary(),
        "null": pa.null(),
        "date32[day]": pa.date32(), "date64[ms]": pa.date64(),
    }
    if text in table:
        return table[text]
    if text.startswith("timestamp"):
        # Best-effort: the str repr is "timestamp[unit, tz=...]"
        try:
            return pa.timestamp("ns")
        except Exception:
            pass
    return pa.string()
