"""Delta log action dataclasses and their parse / serialize functions.

Each Delta commit file is a sequence of newline-delimited JSON
objects, each object having exactly one top-level key naming the
action kind. The action keys we handle here:

- ``commitInfo`` — informational; engine, timestamp, operation. We
  emit one per commit; we read but don't act on them during replay.
- ``protocol`` — minReaderVersion / minWriterVersion + opt-in
  feature names. Drives the refusal surface.
- ``metaData`` — table schema, partition columns, properties.
  There's exactly one live Metadata at any commit version (the
  most recent one wins).
- ``add`` — adds a parquet file to the live set. Carries
  partition values, size, modification time, optional stats, and
  an optional ``deletionVector``.
- ``remove`` — drops a previously-added file from the live set.
  May carry the file's old DV info (``extendedFileMetadata=True``).
- ``txn`` — application-level transaction id, used for idempotent
  re-writes. We read but don't apply them during replay; on write,
  we accept a Txn from the caller and pass it through unchanged.
- ``domainMetadata`` — opaque per-domain string blobs. We surface
  them through the replay result so subclasses can interpret;
  default behavior is pass-through round-trip.

All dataclasses here are public (no leading underscore) and
``frozen=True, slots=True`` for cheap construction at log-replay
scale.
"""

from __future__ import annotations

import dataclasses
import json
import time
import urllib.parse
from typing import Any, Mapping, Sequence

from yggdrasil.data.schema import Schema

from .deletion_vector import DeletionVectorDescriptor
from .schema_codec import (
    delta_schema_string_to_schema,
    schema_to_delta_schema_string,
)


__all__ = [
    "Protocol",
    "Metadata",
    "AddFile",
    "RemoveFile",
    "CommitInfo",
    "Txn",
    "DomainMetadata",
    "parse_action",
    "serialize_action",
    "parse_protocol",
    "parse_metadata",
    "parse_add",
    "parse_remove",
    "serialize_protocol",
    "serialize_metadata",
    "serialize_add",
    "serialize_remove",
    "serialize_commit_info",
]


# ---------------------------------------------------------------------------
# Action dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Protocol:
    """Reader/writer protocol gate.

    Tables emit a Protocol action whenever their feature requirements
    change. The most recent Protocol on the log governs.
    """
    min_reader_version: int
    min_writer_version: int
    reader_features: tuple[str, ...] = ()
    writer_features: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class Metadata:
    """Table-level schema and configuration."""
    id: str
    schema: Schema
    partition_columns: tuple[str, ...]
    configuration: Mapping[str, str]
    name: str | None = None
    description: str | None = None
    created_time: int | None = None
    format_provider: str = "parquet"
    format_options: Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class AddFile:
    """One parquet file in the live data set.

    :attr:`path` is decoded (the wire form is URL-encoded). Always
    relative to the table root, forward-slash separators.

    :attr:`deletion_vector` carries DV info when the file has
    deletes recorded against it; ``None`` for files with no
    deletes.
    """
    path: str
    partition_values: Mapping[str, str | None]
    size: int
    modification_time: int
    data_change: bool
    stats: str | None = None
    tags: Mapping[str, str] | None = None
    deletion_vector: DeletionVectorDescriptor | None = None
    base_row_id: int | None = None
    default_row_commit_version: int | None = None
    clustering_provider: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class RemoveFile:
    """Removal of a file from the live data set.

    Carries the removed file's metadata when ``extended_file_metadata``
    is True — required for DV-bearing tables, since the next reader
    needs to know what DV (if any) was on the file being removed.
    """
    path: str
    deletion_timestamp: int | None
    data_change: bool
    extended_file_metadata: bool = False
    partition_values: Mapping[str, str | None] | None = None
    size: int | None = None
    tags: Mapping[str, str] | None = None
    deletion_vector: DeletionVectorDescriptor | None = None
    base_row_id: int | None = None
    default_row_commit_version: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class CommitInfo:
    """Diagnostic record of who/why/when of a commit. Informational only."""
    timestamp: int
    operation: str
    operation_parameters: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    engine_info: str | None = None
    isolation_level: str = "Serializable"
    is_blind_append: bool = False
    user_metadata: Mapping[str, Any] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Txn:
    """Application-level transaction id for idempotent re-writes.

    Writers re-running a job can stamp ``app_id`` + ``version``; if
    a previous run already committed at that ``app_id``, the
    duplicate is detected at commit time. We read these but don't
    yet enforce — single-writer assumption.
    """
    app_id: str
    version: int
    last_updated: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class DomainMetadata:
    """Opaque per-domain string blob.

    Used by various Delta features (CDC, row tracking, ...) to
    associate metadata with the table outside the schema.
    Round-trip preserved.
    """
    domain: str
    configuration: str
    removed: bool = False


# ---------------------------------------------------------------------------
# Parse — wire dict → dataclass
# ---------------------------------------------------------------------------


def parse_action(action: Mapping[str, Any]) -> Any:
    """Dispatch a single-key action dict to the right parse helper.

    Returns a dataclass instance (Protocol/Metadata/AddFile/...) for
    known kinds, ``None`` for kinds we deliberately skip
    (``commitInfo`` during replay), and raises for unknown kinds.
    Raising on unknown is critical: a future action might
    semantically change live state, and silently skipping would
    return wrong rows.
    """
    if "add" in action:
        return parse_add(action["add"])
    if "remove" in action:
        return parse_remove(action["remove"])
    if "metaData" in action:
        return parse_metadata(action["metaData"])
    if "protocol" in action:
        return parse_protocol(action["protocol"])
    if "txn" in action:
        return parse_txn(action["txn"])
    if "domainMetadata" in action:
        return parse_domain_metadata(action["domainMetadata"])
    if "commitInfo" in action:
        return parse_commit_info(action["commitInfo"])
    if "cdc" in action:
        # Change Data Feed action — not part of the live data set,
        # used by CDF readers. Skip during replay; readers that want
        # CDF use a separate code path.
        return None

    kind = next(iter(action), "<empty>")
    raise ValueError(
        f"Unknown Delta action kind {kind!r}. yggdrasil refuses to "
        "replay tables containing actions it doesn't understand."
    )


def parse_protocol(raw: Mapping[str, Any]) -> Protocol:
    return Protocol(
        min_reader_version=int(raw.get("minReaderVersion", 1)),
        min_writer_version=int(raw.get("minWriterVersion", 1)),
        reader_features=tuple(raw.get("readerFeatures") or ()),
        writer_features=tuple(raw.get("writerFeatures") or ()),
    )


def parse_metadata(raw: Mapping[str, Any]) -> Metadata:
    schema_string = raw.get("schemaString")
    if not schema_string:
        raise ValueError("Delta metaData action is missing schemaString.")
    schema = delta_schema_string_to_schema(schema_string)

    fmt = raw.get("format") or {}
    return Metadata(
        id=str(raw.get("id", "")),
        schema=schema,
        partition_columns=tuple(raw.get("partitionColumns") or ()),
        configuration=dict(raw.get("configuration") or {}),
        name=raw.get("name"),
        description=raw.get("description"),
        created_time=raw.get("createdTime"),
        format_provider=str(fmt.get("provider", "parquet")),
        format_options=dict(fmt.get("options") or {}),
    )


def parse_add(raw: Mapping[str, Any]) -> AddFile:
    dv_raw = raw.get("deletionVector")
    dv = DeletionVectorDescriptor.from_json(dv_raw) if dv_raw else None

    return AddFile(
        path=urllib.parse.unquote(str(raw["path"])),
        partition_values=dict(raw.get("partitionValues") or {}),
        size=int(raw.get("size", 0)),
        modification_time=int(raw.get("modificationTime", 0)),
        data_change=bool(raw.get("dataChange", True)),
        stats=raw.get("stats"),
        tags=dict(raw["tags"]) if raw.get("tags") is not None else None,
        deletion_vector=dv,
        base_row_id=raw.get("baseRowId"),
        default_row_commit_version=raw.get("defaultRowCommitVersion"),
        clustering_provider=raw.get("clusteringProvider"),
    )


def parse_remove(raw: Mapping[str, Any]) -> RemoveFile:
    dv_raw = raw.get("deletionVector")
    dv = DeletionVectorDescriptor.from_json(dv_raw) if dv_raw else None

    return RemoveFile(
        path=urllib.parse.unquote(str(raw["path"])),
        deletion_timestamp=raw.get("deletionTimestamp"),
        data_change=bool(raw.get("dataChange", True)),
        extended_file_metadata=bool(raw.get("extendedFileMetadata", False)),
        partition_values=(
            dict(raw["partitionValues"])
            if raw.get("partitionValues") is not None else None
        ),
        size=int(raw["size"]) if raw.get("size") is not None else None,
        tags=dict(raw["tags"]) if raw.get("tags") is not None else None,
        deletion_vector=dv,
        base_row_id=raw.get("baseRowId"),
        default_row_commit_version=raw.get("defaultRowCommitVersion"),
    )


def parse_commit_info(raw: Mapping[str, Any]) -> CommitInfo:
    return CommitInfo(
        timestamp=int(raw.get("timestamp", 0)),
        operation=str(raw.get("operation", "")),
        operation_parameters=dict(raw.get("operationParameters") or {}),
        engine_info=raw.get("engineInfo"),
        isolation_level=str(raw.get("isolationLevel", "Serializable")),
        is_blind_append=bool(raw.get("isBlindAppend", False)),
        user_metadata=raw.get("userMetadata"),
    )


def parse_txn(raw: Mapping[str, Any]) -> Txn:
    return Txn(
        app_id=str(raw["appId"]),
        version=int(raw["version"]),
        last_updated=raw.get("lastUpdated"),
    )


def parse_domain_metadata(raw: Mapping[str, Any]) -> DomainMetadata:
    return DomainMetadata(
        domain=str(raw["domain"]),
        configuration=str(raw.get("configuration", "")),
        removed=bool(raw.get("removed", False)),
    )


# ---------------------------------------------------------------------------
# Serialize — dataclass → wire dict
# ---------------------------------------------------------------------------


def serialize_action(value: Any) -> Mapping[str, Any]:
    """Wrap a dataclass instance in its single-key action envelope."""
    if isinstance(value, AddFile):
        return {"add": serialize_add(value)}
    if isinstance(value, RemoveFile):
        return {"remove": serialize_remove(value)}
    if isinstance(value, Metadata):
        return {"metaData": serialize_metadata(value)}
    if isinstance(value, Protocol):
        return {"protocol": serialize_protocol(value)}
    if isinstance(value, CommitInfo):
        return {"commitInfo": serialize_commit_info(value)}
    if isinstance(value, Txn):
        return {"txn": serialize_txn(value)}
    if isinstance(value, DomainMetadata):
        return {"domainMetadata": serialize_domain_metadata(value)}
    raise TypeError(
        f"Cannot serialize Delta action: unknown type {type(value).__name__}."
    )


def serialize_protocol(p: Protocol) -> Mapping[str, Any]:
    out: dict[str, Any] = {
        "minReaderVersion": p.min_reader_version,
        "minWriterVersion": p.min_writer_version,
    }
    if p.reader_features:
        out["readerFeatures"] = list(p.reader_features)
    if p.writer_features:
        out["writerFeatures"] = list(p.writer_features)
    return out


def serialize_metadata(m: Metadata) -> Mapping[str, Any]:
    return {
        "id": m.id,
        "format": {"provider": m.format_provider, "options": dict(m.format_options)},
        "schemaString": schema_to_delta_schema_string(m.schema),
        "partitionColumns": list(m.partition_columns),
        "configuration": dict(m.configuration),
        **({"name": m.name} if m.name else {}),
        **({"description": m.description} if m.description else {}),
        "createdTime": m.created_time if m.created_time is not None else int(time.time() * 1000),
    }


def serialize_add(a: AddFile) -> Mapping[str, Any]:
    out: dict[str, Any] = {
        "path": _encode_path(a.path),
        "partitionValues": dict(a.partition_values),
        "size": a.size,
        "modificationTime": a.modification_time,
        "dataChange": a.data_change,
    }
    if a.stats is not None:
        out["stats"] = a.stats
    if a.tags is not None:
        out["tags"] = dict(a.tags)
    if a.deletion_vector is not None:
        out["deletionVector"] = a.deletion_vector.to_json()
    if a.base_row_id is not None:
        out["baseRowId"] = a.base_row_id
    if a.default_row_commit_version is not None:
        out["defaultRowCommitVersion"] = a.default_row_commit_version
    if a.clustering_provider is not None:
        out["clusteringProvider"] = a.clustering_provider
    return out


def serialize_remove(r: RemoveFile) -> Mapping[str, Any]:
    out: dict[str, Any] = {
        "path": _encode_path(r.path),
        "dataChange": r.data_change,
        "extendedFileMetadata": r.extended_file_metadata,
    }
    if r.deletion_timestamp is not None:
        out["deletionTimestamp"] = r.deletion_timestamp
    if r.partition_values is not None:
        out["partitionValues"] = dict(r.partition_values)
    if r.size is not None:
        out["size"] = r.size
    if r.tags is not None:
        out["tags"] = dict(r.tags)
    if r.deletion_vector is not None:
        out["deletionVector"] = r.deletion_vector.to_json()
    if r.base_row_id is not None:
        out["baseRowId"] = r.base_row_id
    if r.default_row_commit_version is not None:
        out["defaultRowCommitVersion"] = r.default_row_commit_version
    return out


def serialize_commit_info(c: CommitInfo) -> Mapping[str, Any]:
    out: dict[str, Any] = {
        "timestamp": c.timestamp,
        "operation": c.operation,
        "operationParameters": dict(c.operation_parameters),
        "isolationLevel": c.isolation_level,
        "isBlindAppend": c.is_blind_append,
    }
    if c.engine_info is not None:
        out["engineInfo"] = c.engine_info
    if c.user_metadata is not None:
        out["userMetadata"] = c.user_metadata
    return out


def serialize_txn(t: Txn) -> Mapping[str, Any]:
    out: dict[str, Any] = {"appId": t.app_id, "version": t.version}
    if t.last_updated is not None:
        out["lastUpdated"] = t.last_updated
    return out


def serialize_domain_metadata(d: DomainMetadata) -> Mapping[str, Any]:
    return {
        "domain": d.domain,
        "configuration": d.configuration,
        "removed": d.removed,
    }


# ---------------------------------------------------------------------------
# Path encoding
# ---------------------------------------------------------------------------

def _encode_path(path: str) -> str:
    """URL-encode a relative table path for an action's ``path`` field.

    Spec: "URI-style escaping: each path segment is escaped per
    RFC 3986 unreserved + sub-delims rules; '/' separators are
    preserved." We pass ``safe="/="`` to keep slashes and the
    ``=`` of Hive partition segments unescaped — matches the
    reference implementations' output.
    """
    return urllib.parse.quote(path, safe="/=")
