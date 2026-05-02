"""Manifest: the snapshot record that names every live data file.

A ygg manifest is a single Arrow IPC file with:

- **Schema-level ``metadata``** (Arrow's ``Schema.metadata`` /
  ``custom_metadata``) holding the table-scoped fields: timestamp,
  table id, partition columns, primary key columns, and the
  embedded data schema as serialized Arrow IPC schema bytes.
- **One row per live data file** in the body, with strongly-typed
  columns for path / size / mtime / num_rows + JSON-encoded
  partition values + JSON-encoded per-column statistics
  (min / max / null count) for the declared primary key columns.

The "metadata in the schema, files in the body" split lets a reader
fetch table-level info in one footer parse without paying for the
file list, and conversely scan the file list in batches without
re-parsing scalar metadata per row.

The per-file stats column is the input to the predicate
prefilter (:mod:`.predicate`): walking N file rows, evaluating
range overlaps, and pruning entries before any data file is
opened. The body lives as JSON in a single string column rather
than a struct of typed Arrow columns because:

- Dynamic per-table column set: declare ``primary_key_columns =
  ["id", "ts"]`` and the manifest carries stats for both; declare
  no key columns and it carries none. A static struct schema can't
  do that without rewriting the manifest schema per table.
- Most tables track stats for ≤ 4 columns, so the JSON cost is a
  rounding error against the per-file row.
"""

from __future__ import annotations

import dataclasses
import io as _stdio
import json
from datetime import date, datetime, time
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.ipc as ipc

from yggdrasil.data.schema import Schema

from .constants import (
    DEFAULT_ENGINE_INFO,
    META_KEY_DATA_SCHEMA,
    META_KEY_ENGINE_INFO,
    META_KEY_PARTITION_COLUMNS,
    META_KEY_PRIMARY_KEY_COLUMNS,
    META_KEY_TABLE_ID,
    META_KEY_TIMESTAMP,
    PROTOCOL_VERSION,
)


__all__ = [
    "ColumnStats",
    "ManifestEntry",
    "Manifest",
    "MANIFEST_BODY_SCHEMA",
    "encode_manifest",
    "decode_manifest",
]


# ---------------------------------------------------------------------------
# File-list body schema
# ---------------------------------------------------------------------------


#: Arrow schema for the manifest body — one row per live data file.
#:
#: ``partition_values`` and ``stats`` are JSON-encoded strings; the
#: rest are strongly typed. ``num_rows`` is nullable to accommodate
#: writers that can't produce a row count cheaply (e.g. streaming
#: codec leaves where the row count requires a full re-decode).
MANIFEST_BODY_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("path", pa.string(), nullable=False),
        pa.field("size", pa.int64(), nullable=False),
        pa.field("modification_time", pa.int64(), nullable=False),
        pa.field("num_rows", pa.int64(), nullable=True),
        pa.field("partition_values", pa.string(), nullable=False),
        pa.field("stats", pa.string(), nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# Manifest dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ColumnStats:
    """Per-column statistics for one file.

    All three fields are best-effort: a writer that can't compute
    a min/max (all-null column, unsupported dtype) leaves them
    ``None`` and the predicate pruner falls back to "can't rule
    out, must read."

    :param min: smallest non-null value in the column. ``None``
        when the column is all-null or stats weren't computed.
    :param max: largest non-null value. Same null semantics.
    :param null_count: number of NULL rows in the column. ``-1``
        when the writer didn't compute one.
    """

    min: Any
    max: Any
    null_count: int = -1


@dataclasses.dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One live data file in the snapshot.

    Mirrors a Delta ``AddFile`` minus the parts we don't need
    (deletion vectors, base_row_id). All paths are forward-slash
    separated and *relative* to the table root — the only form
    that survives table relocation cleanly.

    :param path: forward-slash path relative to the table root.
    :param size: file size in bytes at the time the manifest was
        written. Informational; readers don't validate.
    :param modification_time: unix epoch milliseconds. Set by the
        writer; used by external tooling for staleness checks.
    :param num_rows: row count. ``None`` when the writer couldn't
        compute one cheaply.
    :param partition_values: ``{column: value | None}`` mapping.
        Empty dict for non-partitioned tables.
    :param stats: ``{column: ColumnStats}`` for the table's
        declared primary key columns. Empty dict when no key
        columns are declared or stats weren't computed.
    """

    path: str
    size: int
    modification_time: int
    num_rows: int | None
    partition_values: Mapping[str, str | None]
    stats: Mapping[str, ColumnStats]


@dataclasses.dataclass(frozen=True, slots=True)
class Manifest:
    """A complete snapshot — table metadata plus the live file list.

    :param timestamp: write-time unix epoch milliseconds.
    :param table_id: stable UUID identifying the logical table.
    :param partition_columns: ordered list of partition column
        names. Order matches the on-disk directory layering.
    :param primary_key_columns: column names tracked in each
        entry's ``stats``. Used by the predicate pruner.
    :param data_schema: the table's data schema (i.e. the schema
        of the data files, *not* the schema of this manifest body).
    :param engine_info: free-form string identifying the writer.
    :param entries: one :class:`ManifestEntry` per live data file.
    :param protocol_version: layout/format version. Readers refuse
        versions above what they implement.
    """

    timestamp: int
    table_id: str
    partition_columns: tuple[str, ...]
    primary_key_columns: tuple[str, ...]
    data_schema: Schema
    engine_info: str
    entries: tuple[ManifestEntry, ...]
    protocol_version: int = PROTOCOL_VERSION

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def empty(
        cls,
        *,
        timestamp: int,
        table_id: str,
        partition_columns: Sequence[str] = (),
        primary_key_columns: Sequence[str] = (),
        data_schema: Schema | None = None,
        engine_info: str = DEFAULT_ENGINE_INFO,
    ) -> "Manifest":
        """Build a manifest with no data files (i.e. an empty table)."""
        return cls(
            timestamp=timestamp,
            table_id=table_id,
            partition_columns=tuple(partition_columns),
            primary_key_columns=tuple(primary_key_columns),
            data_schema=data_schema if data_schema is not None else Schema.empty(),
            engine_info=engine_info,
            entries=(),
        )

    def with_entries(
        self, entries: Sequence[ManifestEntry],
    ) -> "Manifest":
        """Return a copy with ``entries`` replaced."""
        return dataclasses.replace(self, entries=tuple(entries))


# ---------------------------------------------------------------------------
# Schema metadata round-trip helpers
# ---------------------------------------------------------------------------


def _serialize_data_schema(schema: Schema) -> bytes:
    """Serialize *schema* as Arrow IPC schema-stream bytes.

    Arrow ships a stable wire format for a bare schema (no
    batches): ``write_schema`` to a buffer. Round-trips losslessly
    including field metadata, nullability, nested types, and
    extension types. JSON would mash several of those flat.
    """
    arrow_schema = schema.to_arrow_schema() if hasattr(schema, "to_arrow_schema") else schema
    buf = _stdio.BytesIO()
    with ipc.new_stream(buf, arrow_schema):
        # Streaming schema-only: no batches written.
        pass
    return buf.getvalue()


def _deserialize_data_schema(blob: bytes) -> Schema:
    """Deserialize Arrow IPC schema-stream bytes into a :class:`Schema`."""
    if not blob:
        return Schema.empty()
    reader = ipc.open_stream(_stdio.BytesIO(blob))
    return Schema.from_arrow(reader.schema)


# ---------------------------------------------------------------------------
# Stats / partition-value JSON helpers
# ---------------------------------------------------------------------------


def _coerce_scalar_for_json(v: Any) -> Any:
    """Stringify a scalar so :func:`json.dumps` will accept it.

    JSON natively handles ``int``, ``float``, ``bool``, ``str``, and
    ``None``. Datetime / date / time / pyarrow scalars get their
    ISO 8601 string form. Anything weirder falls back to ``str(v)``;
    the predicate pruner can still compare strings against strings,
    so we never break the prefilter contract — we just lose a bit
    of precision.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, (datetime, date, time)):
        return v.isoformat()
    # pyarrow scalars: unwrap to Python.
    if hasattr(v, "as_py"):
        try:
            return _coerce_scalar_for_json(v.as_py())
        except Exception:
            pass
    return str(v)


def _encode_stats(
    stats: Mapping[str, ColumnStats],
) -> str:
    """JSON-encode the stats dict for a manifest row."""
    if not stats:
        return "{}"
    out: dict[str, dict[str, Any]] = {}
    for col, s in stats.items():
        out[col] = {
            "min": _coerce_scalar_for_json(s.min),
            "max": _coerce_scalar_for_json(s.max),
            "null_count": int(s.null_count),
        }
    return json.dumps(out, separators=(",", ":"), sort_keys=True)


def _decode_stats(raw: str) -> dict[str, ColumnStats]:
    """Inverse of :func:`_encode_stats`. Empty input ⇒ empty dict."""
    if not raw or raw == "{}":
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Manifest entry has malformed stats JSON {raw!r}: {e}"
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Manifest entry stats must decode to an object; got "
            f"{type(parsed).__name__} from {raw!r}."
        )
    out: dict[str, ColumnStats] = {}
    for col, payload in parsed.items():
        if not isinstance(payload, dict):
            raise ValueError(
                f"Manifest entry stats[{col!r}] must be an object; got "
                f"{type(payload).__name__}."
            )
        out[col] = ColumnStats(
            min=payload.get("min"),
            max=payload.get("max"),
            null_count=int(payload.get("null_count", -1)),
        )
    return out


def _entries_to_arrays(
    entries: Sequence[ManifestEntry],
) -> dict[str, list]:
    """Lay out entries as one column per :data:`MANIFEST_BODY_SCHEMA` field."""
    paths: list[str] = []
    sizes: list[int] = []
    mtimes: list[int] = []
    rowcounts: list[int | None] = []
    pvalues: list[str] = []
    stats_jsons: list[str] = []
    for e in entries:
        paths.append(e.path)
        sizes.append(int(e.size))
        mtimes.append(int(e.modification_time))
        rowcounts.append(None if e.num_rows is None else int(e.num_rows))
        pvalues.append(
            json.dumps(
                {k: e.partition_values[k] for k in sorted(e.partition_values)},
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        stats_jsons.append(_encode_stats(e.stats or {}))
    return {
        "path": paths,
        "size": sizes,
        "modification_time": mtimes,
        "num_rows": rowcounts,
        "partition_values": pvalues,
        "stats": stats_jsons,
    }


def _arrays_to_entries(table: pa.Table) -> tuple[ManifestEntry, ...]:
    """Inverse of :func:`_entries_to_arrays`."""
    if table.num_rows == 0:
        return ()
    paths = table.column("path").to_pylist()
    sizes = table.column("size").to_pylist()
    mtimes = table.column("modification_time").to_pylist()
    rowcounts = table.column("num_rows").to_pylist()
    pvalues_raw = table.column("partition_values").to_pylist()
    stats_raw = (
        table.column("stats").to_pylist()
        if "stats" in table.column_names else [""] * table.num_rows
    )

    out: list[ManifestEntry] = []
    for path, size, mtime, num_rows, pv_raw, stats_str in zip(
        paths, sizes, mtimes, rowcounts, pvalues_raw, stats_raw,
    ):
        pv: Mapping[str, str | None]
        if not pv_raw:
            pv = {}
        else:
            try:
                parsed = json.loads(pv_raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Manifest entry has malformed partition_values JSON "
                    f"for path {path!r}: {pv_raw!r}. {e}"
                ) from e
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Manifest entry partition_values must decode to a "
                    f"JSON object; got {type(parsed).__name__} for path "
                    f"{path!r}."
                )
            pv = parsed
        out.append(ManifestEntry(
            path=path,
            size=int(size) if size is not None else 0,
            modification_time=int(mtime) if mtime is not None else 0,
            num_rows=None if num_rows is None else int(num_rows),
            partition_values=pv,
            stats=_decode_stats(stats_str or ""),
        ))
    return tuple(out)


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------


def encode_manifest(manifest: Manifest) -> bytes:
    """Encode *manifest* as Arrow IPC file bytes.

    Single record batch (one row per entry). All non-row-shaped
    information lives on the schema's ``custom_metadata`` so the
    body schema is identical regardless of how much per-table
    state we accumulate over time.
    """
    columns = _entries_to_arrays(manifest.entries)
    arrays = [pa.array(columns[f.name], type=f.type) for f in MANIFEST_BODY_SCHEMA]

    schema_md: dict[bytes, bytes] = {
        META_KEY_TIMESTAMP.encode(): str(manifest.timestamp).encode(),
        META_KEY_TABLE_ID.encode(): manifest.table_id.encode(),
        META_KEY_PARTITION_COLUMNS.encode(): json.dumps(
            list(manifest.partition_columns), separators=(",", ":"),
        ).encode(),
        META_KEY_PRIMARY_KEY_COLUMNS.encode(): json.dumps(
            list(manifest.primary_key_columns), separators=(",", ":"),
        ).encode(),
        META_KEY_DATA_SCHEMA.encode(): _serialize_data_schema(manifest.data_schema),
        META_KEY_ENGINE_INFO.encode(): manifest.engine_info.encode(),
        b"ygg.protocol_version": str(manifest.protocol_version).encode(),
    }

    body_schema = MANIFEST_BODY_SCHEMA.with_metadata(schema_md)
    batch = pa.RecordBatch.from_arrays(arrays, schema=body_schema)

    sink = _stdio.BytesIO()
    with ipc.new_file(sink, body_schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def decode_manifest(blob: bytes) -> Manifest:
    """Decode Arrow IPC file bytes into a :class:`Manifest`.

    Validates the schema-level metadata is shaped as we expect; a
    missing required key raises :class:`ValueError` with the key
    name. Unknown ``ygg.*`` keys are tolerated — callers may stamp
    extra metadata as long as the required ones are present.
    """
    if not blob:
        raise ValueError("Cannot decode empty manifest blob.")

    reader = ipc.open_file(_stdio.BytesIO(blob))
    body = reader.read_all()
    md = body.schema.metadata or {}

    def _get(key: str) -> bytes:
        v = md.get(key.encode())
        if v is None:
            raise ValueError(
                f"Manifest is missing required metadata key {key!r}. "
                "This is not a yggdrasil ygg manifest, or it was "
                "written by an incompatible version."
            )
        return v

    timestamp = int(_get(META_KEY_TIMESTAMP).decode())
    table_id = _get(META_KEY_TABLE_ID).decode()

    partition_columns_raw = _get(META_KEY_PARTITION_COLUMNS).decode()
    try:
        partition_columns = tuple(json.loads(partition_columns_raw))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Manifest {META_KEY_PARTITION_COLUMNS!r} must be a JSON "
            f"array of strings; got {partition_columns_raw!r}. {e}"
        ) from e

    pk_raw = md.get(META_KEY_PRIMARY_KEY_COLUMNS.encode(), b"[]").decode()
    try:
        primary_key_columns = tuple(json.loads(pk_raw))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Manifest {META_KEY_PRIMARY_KEY_COLUMNS!r} must be a JSON "
            f"array of strings; got {pk_raw!r}. {e}"
        ) from e

    data_schema = _deserialize_data_schema(_get(META_KEY_DATA_SCHEMA))
    engine_info = _get(META_KEY_ENGINE_INFO).decode()

    proto_raw = md.get(b"ygg.protocol_version")
    protocol_version = int(proto_raw.decode()) if proto_raw else PROTOCOL_VERSION

    entries = _arrays_to_entries(body)

    return Manifest(
        timestamp=timestamp,
        table_id=table_id,
        partition_columns=partition_columns,
        primary_key_columns=primary_key_columns,
        data_schema=data_schema,
        engine_info=engine_info,
        entries=entries,
        protocol_version=protocol_version,
    )
