"""Manifest: the snapshot record that names every live data file.

A ygg manifest is a single Arrow IPC file with:

- **Schema-level ``metadata``** (Arrow's ``Schema.metadata`` /
  ``custom_metadata``) holding the table-scoped fields: version,
  timestamp, table id, partition columns, the data schema (as
  serialized Arrow IPC schema bytes), and the engine that wrote
  the snapshot.
- **One row per live data file** in the body, with columns
  describing the file (path, byte size, modification time, row
  count, JSON-serialized partition values).

The "metadata in the schema, files in the body" split lets a reader
fetch table-level info in one footer parse without paying for the
file list, and conversely scan the file list in batches without
re-parsing scalar metadata per row.

This module is intentionally small: it defines the dataclasses that
describe a manifest, the Arrow schema that backs the on-disk
representation, and the serialize / deserialize round-trip. The
write-then-rename and pointer-bump dance lives in
:mod:`yggdrasil.io.buffer.nested.ygg.commit`.
"""

from __future__ import annotations

import dataclasses
import io as _stdio
import json
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.ipc as ipc

from yggdrasil.data.schema import Schema

from .constants import (
    DEFAULT_ENGINE_INFO,
    META_KEY_DATA_SCHEMA,
    META_KEY_ENGINE_INFO,
    META_KEY_PARTITION_COLUMNS,
    META_KEY_TABLE_ID,
    META_KEY_TIMESTAMP,
    META_KEY_VERSION,
    PROTOCOL_VERSION,
)


__all__ = [
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
#: ``partition_values`` is a JSON-encoded ``{column: value | null}`` map
#: stored as a string. Could be a native Arrow ``map<string, string>``;
#: JSON wins on two counts: it's lossless across writers that don't
#: agree on Arrow map physical layout, and a JSON cell is dirt-cheap
#: to inspect by humans tailing the file with hex tools.
MANIFEST_BODY_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("path", pa.string(), nullable=False),
        pa.field("size", pa.int64(), nullable=False),
        pa.field("modification_time", pa.int64(), nullable=False),
        pa.field("num_rows", pa.int64(), nullable=True),
        pa.field("partition_values", pa.string(), nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# Manifest dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One live data file in a snapshot.

    Mirrors a Delta ``AddFile`` minus the parts we don't need
    (stats, base_row_id, deletion vectors). All paths are
    forward-slash separated and *relative* to the table root —
    that's the only form that survives table relocation cleanly.

    :param path: forward-slash path relative to the table root.
    :param size: file size in bytes at the time the manifest was
        written. Informational; readers don't validate.
    :param modification_time: unix epoch milliseconds. Set by the
        writer; used by external tooling for staleness checks.
    :param num_rows: optional row count. ``None`` when the writer
        couldn't compute one cheaply.
    :param partition_values: ``{column: value | None}`` mapping.
        Empty dict for non-partitioned tables.
    """

    path: str
    size: int
    modification_time: int
    num_rows: int | None
    partition_values: Mapping[str, str | None]


@dataclasses.dataclass(frozen=True, slots=True)
class Manifest:
    """A complete snapshot — table metadata plus the live file list.

    :param version: monotonic snapshot version, starting at 0.
    :param timestamp: write-time unix epoch milliseconds.
    :param table_id: stable UUID identifying the logical table.
        Survives schema changes; carried forward by every commit.
    :param partition_columns: ordered list of partition column
        names. Order matches the on-disk directory layering.
    :param data_schema: the table's data schema (i.e. the schema of
        the data files, *not* the schema of this manifest body).
    :param engine_info: free-form string identifying the writer.
    :param entries: one :class:`ManifestEntry` per live data file.
    :param protocol_version: layout/format version. Readers refuse
        versions above what they implement.
    """

    version: int
    timestamp: int
    table_id: str
    partition_columns: tuple[str, ...]
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
        version: int,
        timestamp: int,
        table_id: str,
        partition_columns: Sequence[str] = (),
        data_schema: Schema | None = None,
        engine_info: str = DEFAULT_ENGINE_INFO,
    ) -> "Manifest":
        """Build a manifest with no data files (i.e. an empty table)."""
        return cls(
            version=version,
            timestamp=timestamp,
            table_id=table_id,
            partition_columns=tuple(partition_columns),
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


def _entries_to_arrays(
    entries: Sequence[ManifestEntry],
) -> dict[str, list]:
    """Lay out entries as one column per :data:`MANIFEST_BODY_SCHEMA` field."""
    paths: list[str] = []
    sizes: list[int] = []
    mtimes: list[int] = []
    rowcounts: list[int | None] = []
    pvalues: list[str] = []
    for e in entries:
        paths.append(e.path)
        sizes.append(int(e.size))
        mtimes.append(int(e.modification_time))
        rowcounts.append(None if e.num_rows is None else int(e.num_rows))
        # Sort keys for stable diffs across writers.
        pvalues.append(
            json.dumps(
                {k: e.partition_values[k] for k in sorted(e.partition_values)},
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
    return {
        "path": paths,
        "size": sizes,
        "modification_time": mtimes,
        "num_rows": rowcounts,
        "partition_values": pvalues,
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

    out: list[ManifestEntry] = []
    for path, size, mtime, num_rows, pv_raw in zip(
        paths, sizes, mtimes, rowcounts, pvalues_raw,
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
        META_KEY_VERSION.encode(): str(manifest.version).encode(),
        META_KEY_TIMESTAMP.encode(): str(manifest.timestamp).encode(),
        META_KEY_TABLE_ID.encode(): manifest.table_id.encode(),
        META_KEY_PARTITION_COLUMNS.encode(): json.dumps(
            list(manifest.partition_columns), separators=(",", ":"),
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

    version = int(_get(META_KEY_VERSION).decode())
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

    data_schema = _deserialize_data_schema(_get(META_KEY_DATA_SCHEMA))
    engine_info = _get(META_KEY_ENGINE_INFO).decode()

    proto_raw = md.get(b"ygg.protocol_version")
    protocol_version = int(proto_raw.decode()) if proto_raw else PROTOCOL_VERSION

    entries = _arrays_to_entries(body)

    return Manifest(
        version=version,
        timestamp=timestamp,
        table_id=table_id,
        partition_columns=partition_columns,
        data_schema=data_schema,
        engine_info=engine_info,
        entries=entries,
        protocol_version=protocol_version,
    )
