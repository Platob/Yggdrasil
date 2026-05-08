"""Persist-mode enum — caching disposition for :class:`Tabular.cache`.

Declares the storage tier a caller wants when materializing a
Tabular for repeated reads. The base :class:`Tabular.cache` is a
no-op that returns ``self``; concrete backends (Spark, Arrow,
Polars, …) interpret the mode against whatever caching primitive
they actually have — Spark's ``StorageLevel``, an in-memory Arrow
buffer, an mmap spill file, etc.

Members are deliberately coarse — ``MEMORY`` / ``DISK`` /
``MEMORY_AND_DISK`` / ``OFF_HEAP`` cover what every backend can
either honor or sensibly approximate. ``AUTO`` lets the
implementation pick (the typical default); ``NONE`` is the
explicit "don't cache" / "uncache" signal.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Optional, Union

__all__ = ["PersistMode", "PersistModeLike"]


PersistModeLike = Union["PersistMode", str, int, None]


class PersistMode(IntEnum):
    """Caching tier for :meth:`Tabular.cache`."""

    AUTO = 0
    NONE = 1
    MEMORY = 2
    DISK = 3
    MEMORY_AND_DISK = 4
    OFF_HEAP = 5

    @classmethod
    def from_(
        cls,
        value: PersistModeLike,
        default: Optional["PersistMode"] = None,
    ) -> "PersistMode":
        """Normalize *value* into a :class:`PersistMode`.

        Accepts a member, an integer code, an alias string
        (``"memory"`` / ``"disk"`` / ``"mem+disk"`` / ``"off_heap"``
        / ``"none"`` / ``"auto"``), or ``None`` (returns *default*
        if supplied, else :attr:`AUTO`). Unknown strings raise
        :class:`ValueError`; non-string non-int inputs raise
        :class:`TypeError`.
        """
        if isinstance(value, cls):
            return value

        if value is None:
            return default if default is not None else cls.AUTO

        if isinstance(value, bool):
            raise TypeError(
                f"PersistMode.from_ expected a string or PersistMode, got "
                f"bool: {value!r}"
            )

        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(
                    f"Cannot parse {value!r} as a PersistMode. Accepted "
                    f"integer codes: {sorted(int(m) for m in cls)}."
                )

        if not isinstance(value, str):
            raise TypeError(
                f"PersistMode.from_ expected a string or PersistMode, got "
                f"{type(value).__name__}: {value!r}"
            )

        normalized = value.strip().lower().replace("-", "_").replace("+", "_and_")
        if not normalized:
            return default if default is not None else cls.AUTO

        hit = _STR_MAPPING.get(normalized)
        if hit is not None:
            return hit

        try:
            return cls[normalized.upper()]
        except KeyError:
            raise ValueError(
                f"Cannot parse {value!r} as a PersistMode. Accepted "
                f"values: {sorted(m.name for m in cls)} or aliases "
                f"like {sorted(_STR_MAPPING)}."
            )

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """``True`` when :meth:`from_` would succeed for *value*."""
        try:
            cls.from_(value)
            return True
        except (TypeError, ValueError):
            return False

    @property
    def caches(self) -> bool:
        """``True`` for any tier that actually retains data."""
        return self not in (PersistMode.AUTO, PersistMode.NONE)


_STR_MAPPING: dict[str, PersistMode] = {
    "": PersistMode.AUTO,
    "auto": PersistMode.AUTO,
    "default": PersistMode.AUTO,

    "none": PersistMode.NONE,
    "no": PersistMode.NONE,
    "off": PersistMode.NONE,
    "uncache": PersistMode.NONE,
    "unpersist": PersistMode.NONE,

    "mem": PersistMode.MEMORY,
    "memory": PersistMode.MEMORY,
    "memory_only": PersistMode.MEMORY,
    "in_memory": PersistMode.MEMORY,
    "ram": PersistMode.MEMORY,

    "disk": PersistMode.DISK,
    "disk_only": PersistMode.DISK,
    "spill": PersistMode.DISK,

    "memory_and_disk": PersistMode.MEMORY_AND_DISK,
    "mem_and_disk": PersistMode.MEMORY_AND_DISK,
    "both": PersistMode.MEMORY_AND_DISK,

    "off_heap": PersistMode.OFF_HEAP,
    "offheap": PersistMode.OFF_HEAP,
}
