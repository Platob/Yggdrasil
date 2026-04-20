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
from typing import Any, Iterable

__all__ = ["DataStatisticsConfig"]


@dataclass(frozen=True, slots=True)
class DataStatisticsConfig:
    """Declare which KPIs to collect for a single column.

    Each flag toggles one statistic. Flags are conservative by default —
    the cheap counting/extrema statistics are on, anything that requires
    a second pass or extra buffering is off. A caller passes a list of
    these through
    :class:`~yggdrasil.io.buffer.media_options.MediaOptions.statistics`
    and downstream code (e.g. a SQL engine insertion planner) decides
    the strategy from the resulting numbers.

    Parameters
    ----------
    field:
        Column name to profile. Must be a non-empty string.
    count:
        Total number of rows observed for this column (including nulls).
        Useful to decide bulk-load vs. row-by-row insertion.
    null_count:
        Number of null values. Helps validate ``NOT NULL`` targets and
        pick ``COALESCE`` / default-value behavior on write.
    distinct_count:
        Number of distinct values. Drives ``MERGE`` vs. plain ``INSERT``
        decisions and index/clustering hints. Off by default because it
        typically requires hashing or a second pass.
    min, max:
        Minimum / maximum value. Drives partition pruning, range checks,
        and target-table stat refreshes.
    sum, mean:
        Numeric aggregates. Off by default — only meaningful for numeric
        columns and caller pays the scan cost.
    byte_size:
        Total serialized byte size for the column. Useful to size COPY
        staging files and pick chunking boundaries.
    """

    field: str
    count: bool = True
    null_count: bool = True
    distinct_count: bool = False
    min: bool = True
    max: bool = True
    sum: bool = False
    mean: bool = False
    byte_size: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.field, str):
            raise TypeError(
                "DataStatisticsConfig.field must be a non-empty str, "
                f"got {type(self.field).__name__}"
            )
        if not self.field:
            raise ValueError("DataStatisticsConfig.field must be a non-empty str")

        for f in fields(self):
            if f.name == "field":
                continue
            value = getattr(self, f.name)
            if not isinstance(value, bool):
                raise TypeError(
                    f"DataStatisticsConfig.{f.name} must be bool, "
                    f"got {type(value).__name__}"
                )

    @classmethod
    def coerce(cls, value: Any) -> "DataStatisticsConfig":
        """Accept a config, a field name, or a dict payload.

        ``"col"`` → ``DataStatisticsConfig(field="col")``; a dict is
        forwarded to the constructor; an existing instance is returned
        unchanged. Anything else raises :class:`TypeError` with a
        message that spells out what was passed.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
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
            f"a column name (str), or a dict with a 'field' key — got {type(value).__name__}"
        )

    @classmethod
    def coerce_many(
        cls,
        values: Iterable[Any] | None,
    ) -> list["DataStatisticsConfig"] | None:
        """Normalize an iterable of configs / field names / dicts.

        Returns ``None`` when *values* is ``None`` (no stats requested).
        Raises :class:`ValueError` when the same ``field`` is declared
        twice — silent last-wins merging would make insertion-strategy
        bugs hard to trace back.
        """
        if values is None:
            return None
        if isinstance(values, (str, bytes, dict, cls)):
            raise TypeError(
                "statistics must be a list of DataStatisticsConfig (or dicts / field "
                f"names), not a single {type(values).__name__}. Wrap it in a list."
            )

        try:
            items = list(values)
        except TypeError as e:
            raise TypeError(
                "statistics must be an iterable of DataStatisticsConfig entries"
            ) from e

        out: list[DataStatisticsConfig] = []
        seen: set[str] = set()
        for entry in items:
            cfg = cls.coerce(entry)
            if cfg.field in seen:
                raise ValueError(
                    f"Duplicate statistics entry for field {cfg.field!r}. "
                    "Merge the flags into a single DataStatisticsConfig instead."
                )
            seen.add(cfg.field)
            out.append(cfg)
        return out
