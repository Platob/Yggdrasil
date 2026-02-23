# yggdrasil/io/enums/file_format.py
"""
FileFormat enumeration
======================
Canonical file-format identifiers used throughout Yggdrasil's I/O layer.

Each member's *value* is the canonical file extension string (lower-case,
no leading dot).  A rich ``parse_*`` API normalises the many aliases that
appear in the wild (``"pq"``, ``"feather"``, ``"ndjson"``, ``"delta"`` …).

Adding a new format
-------------------
1. Add a member below with the canonical extension as its value.
2. Add its aliases to :data:`_STR_ALIASES` at the bottom of the file.
3. Update any reader/writer dispatch tables in ``path.py``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class FileFormat(str, Enum):
    """Supported file formats for Yggdrasil path I/O."""

    CSV       = "csv"
    PARQUET   = "parquet"
    JSON      = "json"
    AVRO      = "avro"
    ORC       = "orc"
    ARROW_IPC = "ipc"
    EXCEL     = "xlsx"
    BINARY    = "bin"
    DELTA     = "delta"

    # ------------------------------------------------------------------
    # NOTE: ClassVar fields cannot be declared inside a str-Enum body in
    # Python < 3.11 — the enum metaclass treats every plain assignment as a
    # potential member.  The default is therefore exposed as a class-level
    # attribute *after* class creation (see bottom of file).
    # ------------------------------------------------------------------

    @property
    def is_default(self) -> bool:
        """Return ``True`` if this is the default format (PARQUET)."""
        return self is FileFormat.default()

    @classmethod
    def default(cls) -> "FileFormat":
        """Return the default ``FileFormat`` (PARQUET)."""
        return cls.PARQUET

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @classmethod
    def parse_any(
        cls,
        value: Any,
        default: Optional["FileFormat"] = None,
    ) -> "FileFormat":
        """
        Coerce *value* to a ``FileFormat``, with an optional *default*.

        Resolution order
        ----------------
        1. ``None``               → *default* if given, else ``PARQUET``.
        2. Already a member       → returned as-is.
        3. ``str``                → :meth:`parse_str`.
        4. Anything else          → *default* if given, else ``ValueError``.

        Parameters
        ----------
        value:
            Raw value to resolve.
        default:
            Fallback when *value* is ``None`` or an unrecognised type.

        Raises
        ------
        ValueError
            When *value* is an unrecognised type and no *default* is provided.
        """
        if value is None:
            return default if default is not None else cls.default()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.parse_str(value, default=default)

        if default is not None:
            return default
        raise ValueError(
            f"Cannot convert {type(value).__name__!r} to {cls.__name__}"
        )

    @classmethod
    def parse_str(
        cls,
        value: str,
        default: Optional["FileFormat"] = None,
    ) -> "FileFormat":
        """
        Parse a string (extension, alias, or member name) to a ``FileFormat``.

        Lookup order
        ------------
        1. ``None`` / empty string → *default* if given, else ``PARQUET``.
        2. Alias table             → :data:`_STR_ALIASES` (lower-cased, stripped).
        3. Member name             → case-insensitive (e.g. ``"ARROW_IPC"``).
        4. Member value            → exact lower-case value (e.g. ``"ipc"``).

        Raises
        ------
        ValueError
            When no match is found.
        """
        if not value:
            return default if default is not None else cls.default()

        v = value.strip().lower()
        if not v:
            return default if default is not None else cls.default()

        # 1. Alias table (fastest, covers the common cases).
        hit = _STR_ALIASES.get(v)
        if hit is not None:
            return hit

        # 2. Member name (e.g. "PARQUET", "arrow_ipc").
        try:
            return cls[v.upper()]
        except KeyError:
            pass

        # 3. Member value (e.g. "parquet", "ipc").
        try:
            return cls(v)
        except ValueError:
            pass

        raise ValueError(f"Unknown {cls.__name__}: {value!r}")


# ---------------------------------------------------------------------------
# Alias table
# ---------------------------------------------------------------------------
# Defined *after* the class so FileFormat members are fully constructed.
# Keys are already lower-cased; values are FileFormat members.

_STR_ALIASES: dict[str, FileFormat] = {
    # PARQUET
    "pq":           FileFormat.PARQUET,
    "parq":         FileFormat.PARQUET,
    "parquet":      FileFormat.PARQUET,

    # CSV
    "csv":          FileFormat.CSV,
    "tsv":          FileFormat.CSV,   # tab-separated; callers must set delimiter
    "txt":          FileFormat.CSV,   # plain text tables

    # JSON
    "json":         FileFormat.JSON,
    "jsonl":        FileFormat.JSON,
    "ndjson":       FileFormat.JSON,
    "ldjson":       FileFormat.JSON,

    # ARROW IPC / Feather v2
    "arrow":        FileFormat.ARROW_IPC,
    "ipc":          FileFormat.ARROW_IPC,
    "feather":      FileFormat.ARROW_IPC,
    "arrows":       FileFormat.ARROW_IPC,  # streaming IPC

    # EXCEL
    "xls":          FileFormat.EXCEL,
    "xlsx":         FileFormat.EXCEL,
    "xlsm":         FileFormat.EXCEL,
    "excel":        FileFormat.EXCEL,

    # ORC
    "orc":          FileFormat.ORC,

    # AVRO
    "avro":         FileFormat.AVRO,

    # BINARY
    "bin":          FileFormat.BINARY,
    "binary":       FileFormat.BINARY,
    "bytes":        FileFormat.BINARY,

    # DELTA
    "delta":        FileFormat.DELTA,
    "delta_rs":     FileFormat.DELTA,   # deltalake / delta-rs naming
    "deltatable":   FileFormat.DELTA,
}


__all__ = ["FileFormat"]