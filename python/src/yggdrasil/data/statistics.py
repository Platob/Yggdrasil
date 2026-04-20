"""Lightweight per-field statistics descriptors.

:class:`DataStatisticsConfig` describes which column-level KPIs a caller
wants computed while a table streams through a pipeline — typically the
:class:`~yggdrasil.io.buffer.media_io.MediaIO` read/write path, or a
SQL-engine insertion planner that picks between bulk-insert, upsert,
partition-pruned append, or ``MERGE`` based on the resulting numbers.

The config is deliberately flag-based and frozen so it can be shared
across threads and cached by callers without defensive copies.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable, Union

__all__ = ["DataStatisticsConfig", "FieldKey"]


# A stats entry targets either a single column (str) or a tuple of
# columns treated as one compound key (used to collect distinct value
# tuples for partition pruning).
FieldKey = Union[str, tuple[str, ...]]


# Flags that only make sense on a single scalar column. When ``field``
# is a tuple, we force these to ``False`` — there is no sensible scalar
# ``min``/``max``/``sum`` for a tuple.
_SINGLE_COLUMN_ONLY_FLAGS: tuple[str, ...] = (
    "min",
    "max",
    "sum",
    "mean",
    "byte_size",
)


@dataclass(frozen=True, slots=True)
class DataStatisticsConfig:
    """Declare which KPIs to collect for a column or a compound key.

    Each flag toggles one statistic. Flags are conservative by default —
    cheap counting/extrema statistics are on, anything that requires a
    second pass or extra buffering is off. A caller passes a list of
    these through
    :class:`~yggdrasil.io.buffer.media_options.MediaOptions.statistics`
    and downstream code (e.g. a SQL engine insertion planner) decides
    the strategy from the resulting numbers.

    A :attr:`field` may be either a single column name (``"country"``)
    or a tuple of column names (``("year", "month")``) treated as one
    compound key. Tuple keys are how you capture the set of distinct
    ``(year, month)`` pairs seen in a batch so the target engine can
    prune partitions and issue writes only against the partitions that
    actually received rows.

    Parameters
    ----------
    field:
        Column name, or tuple of column names for a compound key.
        A list is accepted and normalized to a tuple; a tuple of length
        one collapses back to a plain string to keep the canonical form
        unique.
    count:
        Total number of rows observed (including nulls). Useful to
        decide bulk-load vs. row-by-row insertion.
    null_count:
        Number of null values (per-column, or per-component for tuple
        keys — any-null counts as null). Helps validate ``NOT NULL``
        targets and pick ``COALESCE`` / default-value behavior on write.
    distinct_count:
        Number of distinct values (or distinct tuples for a compound
        key). Drives ``MERGE`` vs. plain ``INSERT`` decisions and
        index/clustering hints. Off by default — typically requires
        hashing or a second pass.
    distinct_values:
        Capture the **set of distinct values** (or distinct tuples for
        a compound key) in addition to the count. This is the partition
        pruning hook: pass ``field=("year", "month"), distinct_values=True``
        and the insertion planner can target only the partitions that
        actually received rows. Off by default — memory cost grows with
        cardinality, so the caller opts in.
    min, max:
        Minimum / maximum value. Drives partition pruning, range checks,
        and target-table stat refreshes. Ignored for tuple keys (use a
        per-column config for per-component ranges).
    sum, mean:
        Numeric aggregates. Off by default — only meaningful for numeric
        scalar columns. Ignored for tuple keys.
    byte_size:
        Total serialized byte size for the column. Useful to size COPY
        staging files and pick chunking boundaries. Ignored for tuple
        keys.
    """

    field: FieldKey
    count: bool = True
    null_count: bool = True
    distinct_count: bool = False
    distinct_values: bool = False
    min: bool = True
    max: bool = True
    sum: bool = False
    mean: bool = False
    byte_size: bool = False

    def __post_init__(self) -> None:
        normalized = self._normalize_field(self.field)
        if normalized is not self.field:
            object.__setattr__(self, "field", normalized)

        for f in fields(self):
            if f.name == "field":
                continue
            value = getattr(self, f.name)
            if not isinstance(value, bool):
                raise TypeError(
                    f"DataStatisticsConfig.{f.name} must be bool, "
                    f"got {type(value).__name__}"
                )

        # Compound keys have no scalar min/max/sum/mean/byte_size — zero
        # them out rather than silently computing something misleading.
        # Caller can still declare a per-column config alongside if they
        # want per-component ranges.
        if isinstance(self.field, tuple):
            for flag in _SINGLE_COLUMN_ONLY_FLAGS:
                if getattr(self, flag):
                    object.__setattr__(self, flag, False)

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    @property
    def is_compound(self) -> bool:
        """``True`` when :attr:`field` targets more than one column."""
        return isinstance(self.field, tuple)

    @property
    def field_names(self) -> tuple[str, ...]:
        """Always-tuple view of :attr:`field`, handy for iteration."""
        return self.field if isinstance(self.field, tuple) else (self.field,)

    # ------------------------------------------------------------------
    # Parsing / normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_field(value: Any) -> FieldKey:
        """Accept str / tuple / list for :attr:`field` and canonicalize."""
        if isinstance(value, str):
            if not value:
                raise ValueError("DataStatisticsConfig.field must be a non-empty str")
            return value

        if isinstance(value, (tuple, list)):
            items = tuple(value)
            if not items:
                raise ValueError(
                    "DataStatisticsConfig.field tuple must contain at least one column name"
                )
            for i, item in enumerate(items):
                if not isinstance(item, str):
                    raise TypeError(
                        "DataStatisticsConfig.field tuple must contain only str, "
                        f"got {type(item).__name__} at index {i}"
                    )
                if not item:
                    raise ValueError(
                        f"DataStatisticsConfig.field tuple has an empty string at index {i}"
                    )
            seen: set[str] = set()
            for name in items:
                if name in seen:
                    raise ValueError(
                        f"DataStatisticsConfig.field tuple contains duplicate column {name!r}"
                    )
                seen.add(name)

            # Single-column tuple collapses to a plain str — one canonical
            # form per semantic target.
            return items[0] if len(items) == 1 else items

        raise TypeError(
            "DataStatisticsConfig.field must be a str or a tuple/list of str, "
            f"got {type(value).__name__}"
        )

    @classmethod
    def coerce(cls, value: Any) -> "DataStatisticsConfig":
        """Accept a config, a field name / tuple, or a dict payload.

        - A :class:`DataStatisticsConfig` is returned unchanged.
        - ``"col"`` → ``DataStatisticsConfig(field="col")``.
        - ``("a", "b")`` or ``["a", "b"]`` → compound-key config.
        - A dict is forwarded to the constructor; requires a ``field``
          key.

        Anything else raises :class:`TypeError` with a message that
        spells out what was passed.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(field=value)
        if isinstance(value, (tuple, list)):
            return cls(field=value)
        if isinstance(value, dict):
            if "field" not in value:
                raise ValueError(
                    "DataStatisticsConfig dict payload requires a 'field' key; "
                    f"got keys={sorted(value)}"
                )
            return cls(**value)
        raise TypeError(
            "DataStatisticsConfig entries must be a DataStatisticsConfig, "
            "a column name (str), a tuple/list of column names, or a dict "
            f"with a 'field' key — got {type(value).__name__}"
        )

    @classmethod
    def coerce_many(
        cls,
        values: Iterable[Any] | None,
    ) -> list["DataStatisticsConfig"] | None:
        """Normalize an iterable of configs / field names / tuples / dicts.

        Returns ``None`` when *values* is ``None`` (no stats requested).
        Raises :class:`ValueError` when the same ``field`` key is
        declared twice — silent last-wins merging would make
        insertion-strategy bugs hard to trace back. A single-column
        entry (``"a"``) and a compound-key entry (``("a", "b")``) are
        considered different targets.
        """
        if values is None:
            return None
        if isinstance(values, (str, bytes, dict, cls)):
            raise TypeError(
                "statistics must be a list of DataStatisticsConfig (or dicts / field "
                f"names / tuples), not a single {type(values).__name__}. Wrap it in a list."
            )

        try:
            items = list(values)
        except TypeError as e:
            raise TypeError(
                "statistics must be an iterable of DataStatisticsConfig entries"
            ) from e

        out: list[DataStatisticsConfig] = []
        seen: set[FieldKey] = set()
        for entry in items:
            cfg = cls.coerce(entry)
            key = cfg.field
            if key in seen:
                raise ValueError(
                    f"Duplicate statistics entry for field {key!r}. "
                    "Merge the flags into a single DataStatisticsConfig instead."
                )
            seen.add(key)
            out.append(cfg)
        return out
