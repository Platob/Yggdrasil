"""
models.py — Delta log value objects.

All dataclasses in this module are parsed directly from Delta log JSON actions
or checkpoint Parquet rows.  They are intentionally kept free of I/O so they
can be constructed, compared, and tested without a live filesystem.

Classes
-------
DeltaProtocol   — ``protocol`` action; reader/writer version negotiation.
DeletionVector  — per-file roaring bitmap of logically deleted rows.
DeltaMetadata   — ``metaData`` action; schema, partition columns, configuration.
DeltaStats      — aggregate statistics derived from the active file set.
DeltaFile       — one active Parquet file as tracked by the Delta log.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import pyarrow.parquet as pq

from ._uri import strip_uri_scheme

if TYPE_CHECKING:
    from pyarrow.fs import FileSystem

__all__ = [
    "DeltaProtocol",
    "DeletionVector",
    "DeltaMetadata",
    "DeltaStats",
    "DeltaFile",
]


# ---------------------------------------------------------------------------
# DeltaProtocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaProtocol:
    """Parsed ``protocol`` action from the Delta log.

    Controls which reader and writer versions are required to interact with
    the table.  ``reader_features`` and ``writer_features`` are populated for
    tables that use protocol version 3+ feature flags (e.g.
    ``deletionVectors``, ``columnMapping``).

    Attributes:
        min_reader_version: Minimum Delta reader version required.
        min_writer_version: Minimum Delta writer version required.
        reader_features:    Named reader features enabled on this table.
        writer_features:    Named writer features enabled on this table.
    """
    min_reader_version: int
    min_writer_version: int
    reader_features: frozenset[str] = field(default_factory=frozenset)
    writer_features: frozenset[str] = field(default_factory=frozenset)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_action(cls, d: dict) -> DeltaProtocol:
        """Parse a ``protocol`` action dict from the Delta log."""
        return cls(
            min_reader_version=d.get("minReaderVersion", 1),
            min_writer_version=d.get("minWriterVersion", 1),
            reader_features=frozenset(d.get("readerFeatures") or []),
            writer_features=frozenset(d.get("writerFeatures") or []),
        )

    def to_action(self) -> dict[str, Any]:
        """Serialise to a Delta log ``protocol`` action dict."""
        d: dict[str, Any] = {
            "minReaderVersion": self.min_reader_version,
            "minWriterVersion": self.min_writer_version,
        }
        if self.reader_features:
            d["readerFeatures"] = sorted(self.reader_features)
        if self.writer_features:
            d["writerFeatures"] = sorted(self.writer_features)
        return {"protocol": d}

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    @property
    def supports_deletion_vectors(self) -> bool:
        """``True`` when the ``deletionVectors`` reader feature is enabled."""
        return "deletionVectors" in self.reader_features

    def unsupported_reader_features(
        self, supported: frozenset[str]
    ) -> frozenset[str]:
        """Return reader features present in this protocol but not in *supported*."""
        return self.reader_features - supported


# ---------------------------------------------------------------------------
# DeletionVector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeletionVector:
    """Deletion vector attached to a :class:`DeltaFile`.

    A deletion vector is a roaring bitmap that identifies rows within a
    Parquet file that have been logically deleted without rewriting the file.
    The bitmap is stored in one of three ways:

    * ``"i"`` — inline, base85-encoded directly in the log.
    * ``"u"`` — UUID-based path relative to the table root
                (``_delta_log/<uuid>.bin``).
    * ``"p"`` — absolute path to the DV file.

    Attributes:
        storage_type:      ``"i"``, ``"u"``, or ``"p"``.
        path_or_inline_dv: UUID string, absolute path, or inline base85 bitmap.
        offset:            Byte offset within the DV file (``"u"`` only).
        size_in_bytes:     Byte length of the serialised bitmap.
        cardinality:       Number of rows marked as deleted.
    """
    storage_type: str
    path_or_inline_dv: str
    offset: int
    size_in_bytes: int
    cardinality: int

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> DeletionVector:
        """Parse a ``deletionVector`` sub-dict from an ``add`` action."""
        return cls(
            storage_type=d["storageType"],
            path_or_inline_dv=d["pathOrInlineDv"],
            offset=d.get("offset") or 0,
            size_in_bytes=d.get("sizeInBytes", 0),
            cardinality=d.get("cardinality", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a ``deletionVector`` sub-dict for an ``add`` action."""
        return {
            "storageType":    self.storage_type,
            "pathOrInlineDv": self.path_or_inline_dv,
            "offset":         self.offset,
            "sizeInBytes":    self.size_in_bytes,
            "cardinality":    self.cardinality,
        }

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve_path(self, table_root: str) -> Optional[str]:
        """Return the absolute path to the DV file, or ``None`` for inline DVs.

        Args:
            table_root: The table's ``storage_location`` URI or bare path.

        Returns:
            Absolute path for ``"u"`` and ``"p"`` storage types;
            ``None`` for ``"i"`` (inline).
        """
        if self.storage_type == "i":
            return None
        if self.storage_type == "p":
            return self.path_or_inline_dv
        # "u" — UUID-based; Delta spec stores without hyphens as filename stem.
        root = table_root.rstrip("/")
        return f"{root}/_delta_log/{self.path_or_inline_dv}.bin"

    # ------------------------------------------------------------------
    # Bitmap deserialisation
    # ------------------------------------------------------------------

    def read_deleted_rows(self, fs: "FileSystem", table_root: str) -> set[int]:
        """Deserialise the deletion vector and return deleted row indices.

        Supports all three storage types.  The Delta roaring bitmap format
        prepends a 4-byte magic number (``0x52 0x4F 0x41 0x52``, i.e.
        ``"ROAR"``) before the standard portable roaring bitmap payload.

        Args:
            fs:          PyArrow ``FileSystem`` used to read ``"u"``/``"p"`` DVs.
            table_root:  The table's storage location URI.

        Returns:
            ``set[int]`` of 0-based row indices that are logically deleted.

        Raises:
            ImportError: If ``pyroaring`` is not installed.
            ValueError:  If the DV format is unrecognised.

        Note:
            Install the optional dependency with::

                pip install pyroaring
        """
        try:
            from pyroaring import BitMap  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "Reading deletion vectors requires 'pyroaring'.\n"
                "Install it with:  pip install pyroaring"
            )

        _MAGIC = b"\x52\x4F\x41\x52"  # "ROAR"

        if self.storage_type == "i":
            import base64
            raw = base64.b85decode(self.path_or_inline_dv)
        else:
            path = self.resolve_path(table_root)
            bare = strip_uri_scheme(path)
            with fs.open_input_file(bare) as fh:
                fh.seek(self.offset)
                raw = fh.read(self.size_in_bytes)

        if raw[:4] == _MAGIC:
            raw = raw[4:]

        return set(BitMap.deserialize(raw))


# ---------------------------------------------------------------------------
# DeltaMetadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaMetadata:
    """Parsed ``metaData`` action from the Delta log.

    Carries the table schema (as a Delta JSON ``schemaString``), partition
    columns, table properties (``configuration``), and housekeeping fields.

    The ``schema`` property lazily parses ``schema_string`` into a dict so the
    cost is paid only when field names are actually needed.

    Attributes:
        id:                Stable UUID for this table.
        name:              Human-readable table name (optional).
        description:       Table comment / description (optional).
        format_provider:   Storage format; always ``"parquet"`` in practice.
        format_options:    Format-specific options (usually empty).
        schema_string:     Delta JSON schema blob.
        partition_columns: Ordered list of partition column names.
        configuration:     ``TBLPROPERTIES`` dict.
        created_time:      Creation timestamp in milliseconds since epoch.
    """
    id: str
    name: Optional[str]
    description: Optional[str]
    format_provider: str
    format_options: dict[str, str]
    schema_string: str
    partition_columns: list[str]
    configuration: dict[str, str]
    created_time: Optional[int]

    _schema: Optional[dict] = field(
        default=None, repr=False, compare=False, hash=False
    )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_action(cls, d: dict) -> DeltaMetadata:
        """Parse a ``metaData`` action dict from the Delta log."""
        fmt = d.get("format", {})
        return cls(
            id=d.get("id", ""),
            name=d.get("name"),
            description=d.get("description"),
            format_provider=fmt.get("provider", "parquet"),
            format_options=fmt.get("options") or {},
            schema_string=d.get("schemaString", "{}"),
            partition_columns=d.get("partitionColumns") or [],
            configuration=d.get("configuration") or {},
            created_time=d.get("createdTime"),
        )

    def to_action(self) -> dict[str, Any]:
        """Serialise to a Delta log ``metaData`` action dict."""
        return {
            "metaData": {
                "id":               self.id,
                "name":             self.name,
                "description":      self.description,
                "format": {
                    "provider": self.format_provider,
                    "options":  self.format_options,
                },
                "schemaString":     self.schema_string,
                "partitionColumns": self.partition_columns,
                "configuration":    self.configuration,
                "createdTime":      self.created_time,
            }
        }

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    @property
    def schema(self) -> dict:
        """Lazily parsed Delta schema dict (from ``schemaString``)."""
        if self._schema is None:
            object.__setattr__(self, "_schema", json.loads(self.schema_string))
        return self._schema

    @property
    def column_names(self) -> list[str]:
        """Column names in schema field order."""
        return [f["name"] for f in self.schema.get("fields", [])]

    @property
    def delta_columns(self) -> list[dict]:
        """Raw Delta schema field dicts: ``{name, type, nullable, metadata}``."""
        return self.schema.get("fields", [])

    def is_partitioned(self) -> bool:
        """``True`` when the table has at least one partition column."""
        return len(self.partition_columns) > 0

    # ------------------------------------------------------------------
    # Immutable update
    # ------------------------------------------------------------------

    def with_updates(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        configuration: Optional[dict[str, str]] = None,
        partition_columns: Optional[list[str]] = None,
        schema_string: Optional[str] = None,
    ) -> DeltaMetadata:
        """Return a new :class:`DeltaMetadata` with the supplied fields replaced.

        Only the fields explicitly passed are changed; all others are preserved.
        ``configuration`` is *merged* (not replaced) so existing properties
        survive unless explicitly overridden.
        """
        return DeltaMetadata(
            id=self.id,
            name=name if name is not None else self.name,
            description=description if description is not None else self.description,
            format_provider=self.format_provider,
            format_options=self.format_options,
            schema_string=schema_string if schema_string is not None else self.schema_string,
            partition_columns=partition_columns if partition_columns is not None else self.partition_columns,
            configuration={**self.configuration, **(configuration or {})},
            created_time=self.created_time,
        )


# ---------------------------------------------------------------------------
# DeltaStats
# ---------------------------------------------------------------------------

@dataclass
class DeltaStats:
    """Aggregate statistics derived from the active file set.

    Computed by :meth:`DeltaTable.stats` over the in-memory file index.

    Note:
        ``total_records`` reflects the count stored in file-level stats and
        does **not** subtract rows masked by deletion vectors.

    Attributes:
        num_files:     Number of active Parquet files.
        total_bytes:   Sum of file sizes in bytes.
        total_records: Sum of ``numRecords`` across all files.
    """
    num_files: int = 0
    total_bytes: int = 0
    total_records: int = 0

    def __repr__(self) -> str:
        mb = self.total_bytes / 1024 / 1024
        return (
            f"DeltaStats("
            f"files={self.num_files}, "
            f"size={mb:.1f} MB, "
            f"records={self.total_records:,})"
        )


# ---------------------------------------------------------------------------
# DeltaFile
# ---------------------------------------------------------------------------

@dataclass
class DeltaFile:
    """One active Parquet file as tracked by the Delta transaction log.

    ``path`` is either absolute (``s3://…``) or relative to the table root.
    ``partition_values`` contains string-typed values exactly as written in the
    Delta log (before any type coercion). ``stats`` is the raw per-file
    statistics dict (``numRecords``, ``minValues``, ``maxValues``,
    ``nullCount``).

    Attributes:
        path:             Relative or absolute path to the Parquet file.
        size:             File size in bytes.
        partition_values: ``{column: value}`` as strings from the log.
        modification_time: Last-modified epoch milliseconds.
        data_change:      ``True`` for logical data changes; ``False`` for
                          compaction / Z-ordering rewrites.
        stats:            Raw Delta file statistics dict.
        tags:             Arbitrary string tags attached to the file.
        deletion_vector:  Non-``None`` when rows in this file are logically
                          deleted without a file rewrite.
    """
    path: str
    size: int
    partition_values: dict[str, str]
    modification_time: int
    data_change: bool
    stats: dict[str, Any]
    tags: dict[str, str]
    deletion_vector: Optional[DeletionVector] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_add_action(cls, d: dict) -> DeltaFile:
        """Parse a ``DeltaFile`` from an ``add`` action dict."""
        raw_stats = d.get("stats")
        if isinstance(raw_stats, str):
            try:
                raw_stats = json.loads(raw_stats)
            except Exception:
                raw_stats = {}

        dv: Optional[DeletionVector] = None
        if dv_dict := d.get("deletionVector"):
            try:
                dv = DeletionVector.from_dict(dv_dict)
            except Exception:
                pass

        return cls(
            path=d["path"],
            size=d.get("size", 0),
            partition_values=d.get("partitionValues") or {},
            modification_time=d.get("modificationTime", 0),
            data_change=d.get("dataChange", True),
            stats=raw_stats or {},
            tags=d.get("tags") or {},
            deletion_vector=dv,
        )

    @classmethod
    def from_parquet(
        cls,
        path: str,
        *,
        partition_values: Optional[dict[str, str]] = None,
        fs: Optional["FileSystem"] = None,
        data_change: bool = True,
        tags: Optional[dict[str, str]] = None,
    ) -> DeltaFile:
        """Construct a :class:`DeltaFile` from a Parquet file in storage.

        Reads the file footer via *fs* to determine byte size and row count so
        Delta stats are accurate without a full data scan.

        Args:
            path:             URI or bare path to the Parquet file.  The URI
                              scheme (if any) is preserved in the returned
                              ``DeltaFile.path``.
            partition_values: ``{column: value}`` as strings, e.g.
                              ``{"date": "2024-01-15", "commodity": "crude_oil"}``.
            fs:               PyArrow ``FileSystem`` for footer reads.
                              Pass ``None`` to skip — stats will be empty.
            data_change:      ``True`` (default) for a logical data change.
            tags:             Optional arbitrary string tags.

        Returns:
            A fully populated :class:`DeltaFile` with no deletion vector.
        """
        size = 0
        num_records = 0

        if fs is not None:
            bare = strip_uri_scheme(path)
            try:
                info = fs.get_file_info(bare)
                size = info.size or 0
            except Exception:
                pass
            try:
                meta = pq.read_metadata(bare, filesystem=fs)
                num_records = meta.num_rows
            except Exception:
                pass

        stats: dict[str, Any] = {"numRecords": num_records} if num_records else {}

        return cls(
            path=path,
            size=size,
            partition_values=partition_values or {},
            modification_time=int(time.time() * 1000),
            data_change=data_change,
            stats=stats,
            tags=tags or {},
            deletion_vector=None,
        )

    # ------------------------------------------------------------------
    # Stats properties
    # ------------------------------------------------------------------

    @property
    def has_deletion_vector(self) -> bool:
        """``True`` when this file has a deletion vector attached."""
        return self.deletion_vector is not None

    @property
    def num_records(self) -> Optional[int]:
        """Row count from file-level stats, or ``None`` if unavailable."""
        return self.stats.get("numRecords")

    @property
    def min_values(self) -> dict[str, Any]:
        """Per-column minimum values from file-level stats."""
        return self.stats.get("minValues") or {}

    @property
    def max_values(self) -> dict[str, Any]:
        """Per-column maximum values from file-level stats."""
        return self.stats.get("maxValues") or {}

    @property
    def null_count(self) -> dict[str, int]:
        """Per-column null counts from file-level stats."""
        return self.stats.get("nullCount") or {}

    # ------------------------------------------------------------------
    # Predicate pushdown helpers
    # ------------------------------------------------------------------

    def matches_partition(self, filters: dict[str, str]) -> bool:
        """``True`` when this file satisfies all partition equality filters.

        Comparison is string-based, as partition values are stored as strings
        in the Delta log.

        Args:
            filters: ``{column: value}`` equality conditions (AND semantics).
        """
        return all(
            self.partition_values.get(col) == val
            for col, val in filters.items()
        )

    def matches_stats(
        self, column: str, lo: Any = None, hi: Any = None
    ) -> bool:
        """Conservative min/max range check for *column* over ``[lo, hi]``.

        Returns ``False`` only when the file is *provably* outside the range
        (``file_max < lo`` or ``file_min > hi``).  Missing stats or
        incomparable types are treated conservatively as "keep the file".

        Args:
            column: Column name to check.
            lo:     Inclusive lower bound, or ``None`` for open.
            hi:     Inclusive upper bound, or ``None`` for open.
        """
        file_min = self.min_values.get(column)
        file_max = self.max_values.get(column)

        try:
            if lo is not None and file_max is not None and file_max < lo:
                return False
        except TypeError:
            pass

        try:
            if hi is not None and file_min is not None and file_min > hi:
                return False
        except TypeError:
            pass

        return True