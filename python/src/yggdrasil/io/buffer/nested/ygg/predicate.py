"""Predicate model: file pruning + per-file row-index resolution.

A :class:`Predicate` encodes a query over the table's primary-key /
partition columns and exposes two methods:

- :meth:`Predicate.matches_entry` — given a :class:`ManifestEntry`'s
  partition values + per-column statistics (min/max/null_count),
  return True if the file *might* contain matching rows. Used to
  prune the manifest before opening any data file.
- :meth:`Predicate.row_mask` — given a fully-loaded :class:`pa.Table`,
  return an Arrow boolean mask of matching rows. The caller derives
  an ``int64`` row-index array from the mask via
  :func:`row_indices` and uses it to ``.take(...)`` a filtered slice.

The two-stage design is the whole point of the protocol: the
file-pruning stage walks small, in-memory metadata; the row-mask
stage only runs on files the pruner couldn't rule out.

Built-in predicates
-------------------

- :class:`Eq` — column equals a single value.
- :class:`In` — column is one of N values.
- :class:`Between` — column lies in ``[lo, hi]`` (each side optional
  ⇒ open-ended).
- :class:`And` — conjunction of N predicates.

All are dataclasses; build them inline or via the convenience
constructors :func:`eq`, :func:`is_in`, :func:`between`.

Limitations
-----------

We deliberately don't implement OR yet — every real query I've seen
on a ygg table is a conjunction over key columns. Adding ``Or`` is
a one-class addition; resist the urge until something needs it.

NULL handling: a stat row with ``null_count > 0`` *plus*
``min == None == max`` (i.e. all-null column) is treated as
"unknown range, can't prune." Predicates default to "row matches
NULL ⇒ False" — Arrow's standard semantics for equality and IN.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Sequence

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from .manifest import ColumnStats, ManifestEntry


__all__ = [
    "Predicate",
    "Eq",
    "In",
    "Between",
    "And",
    "eq",
    "is_in",
    "between",
    "row_indices",
    "filter_table",
]


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class Predicate:
    """Base class — concrete subclasses implement the two methods.

    Not abstract in the strict ``ABC`` sense because we want
    composition with :class:`And` to be a plain dataclass. Treat
    this as a duck-typed interface.
    """

    def matches_entry(self, entry: "ManifestEntry") -> bool:
        """True if *entry*'s file might contain matching rows.

        Default: True (no pruning). Subclasses tighten.
        """
        return True

    def row_mask(self, table: pa.Table) -> pa.Array:
        """Boolean mask over *table*'s rows. ``True`` ⇒ matching."""
        # Default: every row matches. Subclasses tighten.
        return pa.array([True] * table.num_rows, type=pa.bool_())

    # Convenience: combine via ``&``.
    def __and__(self, other: "Predicate") -> "And":
        if isinstance(self, And):
            return And(self.parts + (other,))
        if isinstance(other, And):
            return And((self,) + other.parts)
        return And((self, other))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stat_for(
    entry: "ManifestEntry", column: str,
) -> "ColumnStats | None":
    """Fetch the per-column stats for *column* from *entry*."""
    return entry.stats.get(column) if entry.stats else None


def _partition_value_for(
    entry: "ManifestEntry", column: str,
) -> Any:
    """Fetch the partition value for *column* from *entry*, or
    raise :class:`KeyError` if the column isn't a partition.
    """
    if column not in entry.partition_values:
        raise KeyError(column)
    return entry.partition_values[column]


def _value_in_range(
    value: Any, lo: Any, hi: Any, *, null_count: int, num_rows: int | None,
) -> bool:
    """True if a single equality-target value can lie in ``[lo, hi]``.

    Handles the all-null / unknown-bound edge cases that the
    pruner needs to fail open on (i.e. "can't rule it out, must
    read the file").
    """
    if value is None:
        # Equality against NULL — Arrow's three-valued logic returns
        # NULL, which doesn't match. The file matches only if it has
        # rows our predicate can flag, which it can't. Prune.
        return False
    if lo is None or hi is None:
        # Stat unknown — fail open.
        return True
    return lo <= value <= hi


def _ranges_overlap(
    lo_a: Any, hi_a: Any, lo_b: Any, hi_b: Any,
) -> bool:
    """True if ``[lo_a, hi_a]`` overlaps ``[lo_b, hi_b]``.

    Either side may be ``None`` to mean "open" — i.e. unbounded
    on that side. Returns True conservatively when stats are
    unknown.
    """
    if lo_a is None or hi_a is None or lo_b is None or hi_b is None:
        return True
    return lo_a <= hi_b and lo_b <= hi_a


# ---------------------------------------------------------------------------
# Eq
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Eq(Predicate):
    """``column == value``."""

    column: str
    value: Any

    def matches_entry(self, entry: "ManifestEntry") -> bool:
        # Partition column: the value is in the entry directly.
        if self.column in entry.partition_values:
            pv = entry.partition_values[self.column]
            # Partition values are strings on the wire; coerce the
            # predicate value to string for the comparison so an
            # ``Eq("year", 2024)`` matches ``year=2024`` correctly.
            return pv == _coerce_to_partition_string(self.value)

        # Stats column: range-prune.
        s = _stat_for(entry, self.column)
        if s is None:
            return True
        return _value_in_range(
            self.value, s.min, s.max,
            null_count=s.null_count, num_rows=entry.num_rows,
        )

    def row_mask(self, table: pa.Table) -> pa.Array:
        if self.column not in table.column_names:
            return pa.array([False] * table.num_rows, type=pa.bool_())
        # ``equal`` returns NULL for NULL inputs; ``fill_null(False)``
        # collapses that into "doesn't match" — the natural read of
        # ``column == value``.
        eq = pc.equal(table[self.column], self.value)
        return pc.fill_null(eq, False)


# ---------------------------------------------------------------------------
# In
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class In(Predicate):
    """``column IN (values)``."""

    column: str
    values: tuple[Any, ...]

    def matches_entry(self, entry: "ManifestEntry") -> bool:
        if not self.values:
            return False  # IN () matches nothing.

        if self.column in entry.partition_values:
            pv = entry.partition_values[self.column]
            wanted = {_coerce_to_partition_string(v) for v in self.values}
            return pv in wanted

        s = _stat_for(entry, self.column)
        if s is None:
            return True
        if s.min is None or s.max is None:
            return True
        # Any value in the IN set must lie inside [min, max] for
        # the file to be relevant.
        for v in self.values:
            if v is None:
                continue
            if s.min <= v <= s.max:
                return True
        return False

    def row_mask(self, table: pa.Table) -> pa.Array:
        if self.column not in table.column_names:
            return pa.array([False] * table.num_rows, type=pa.bool_())
        if not self.values:
            return pa.array([False] * table.num_rows, type=pa.bool_())
        col = table[self.column]
        try:
            value_set = pa.array(list(self.values), type=col.type)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            value_set = pa.array(list(self.values))
        mask = pc.is_in(col, value_set=value_set)
        return pc.fill_null(mask, False)


# ---------------------------------------------------------------------------
# Between
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Between(Predicate):
    """``lo <= column <= hi``. Either bound may be ``None`` for open."""

    column: str
    lo: Any
    hi: Any

    def matches_entry(self, entry: "ManifestEntry") -> bool:
        if self.column in entry.partition_values:
            pv = entry.partition_values[self.column]
            if pv is None:
                return False
            lo_s = (
                _coerce_to_partition_string(self.lo)
                if self.lo is not None else None
            )
            hi_s = (
                _coerce_to_partition_string(self.hi)
                if self.hi is not None else None
            )
            if lo_s is not None and pv < lo_s:
                return False
            if hi_s is not None and pv > hi_s:
                return False
            return True

        s = _stat_for(entry, self.column)
        if s is None:
            return True
        return _ranges_overlap(self.lo, self.hi, s.min, s.max)

    def row_mask(self, table: pa.Table) -> pa.Array:
        if self.column not in table.column_names:
            return pa.array([False] * table.num_rows, type=pa.bool_())
        col = table[self.column]
        mask = pa.array([True] * table.num_rows, type=pa.bool_())
        if self.lo is not None:
            mask = pc.and_(mask, pc.greater_equal(col, self.lo))
        if self.hi is not None:
            mask = pc.and_(mask, pc.less_equal(col, self.hi))
        return pc.fill_null(mask, False)


# ---------------------------------------------------------------------------
# And
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class And(Predicate):
    """Conjunction. ``parts`` may be empty (matches everything)."""

    parts: tuple[Predicate, ...]

    def matches_entry(self, entry: "ManifestEntry") -> bool:
        return all(p.matches_entry(entry) for p in self.parts)

    def row_mask(self, table: pa.Table) -> pa.Array:
        if not self.parts:
            return pa.array([True] * table.num_rows, type=pa.bool_())
        out = self.parts[0].row_mask(table)
        for p in self.parts[1:]:
            out = pc.and_(out, p.row_mask(table))
        return pc.fill_null(out, False)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def eq(column: str, value: Any) -> Eq:
    """Build an :class:`Eq` predicate."""
    return Eq(column=column, value=value)


def is_in(column: str, values: Sequence[Any]) -> In:
    """Build an :class:`In` predicate."""
    return In(column=column, values=tuple(values))


def between(column: str, lo: Any = None, hi: Any = None) -> Between:
    """Build a :class:`Between` predicate. Either bound may be ``None``."""
    return Between(column=column, lo=lo, hi=hi)


# ---------------------------------------------------------------------------
# Index resolution
# ---------------------------------------------------------------------------


def row_indices(predicate: Predicate, table: pa.Table) -> pa.Int64Array:
    """Return the int64 array of matching row positions in *table*.

    The returned array is the canonical "int64 array index" the
    protocol promises: feed it to :meth:`pa.Table.take` to get the
    filtered slice, or pass it to a downstream consumer that wants
    row coordinates rather than rows.
    """
    mask = predicate.row_mask(table)
    # ``indices_nonzero`` is the cheap path — it walks the bitmap
    # once and emits int64 positions in one shot, which is exactly
    # what the caller wants for ``.take(...)``.
    indices = pc.indices_nonzero(mask)
    if indices.type != pa.int64():
        indices = indices.cast(pa.int64())
    return indices


def filter_table(predicate: Predicate, table: pa.Table) -> pa.Table:
    """Apply *predicate* to *table*, returning the matching rows.

    Convenience wrapper over :func:`row_indices` + :meth:`pa.Table.take`.
    """
    return table.take(row_indices(predicate, table))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_to_partition_string(value: Any) -> str | None:
    """Match Hive's wire convention: every partition value is a string.

    A predicate value typed against the *logical* dtype (``int``,
    ``date``, etc.) needs to be stringified before comparison
    against ``entry.partition_values``. Mirrors the encoder side
    in :mod:`yggdrasil.io.buffer.nested.folder_io`.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
