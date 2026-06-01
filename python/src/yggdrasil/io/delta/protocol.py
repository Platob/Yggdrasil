"""Delta transaction-log action types.

Every line of a Delta JSON commit is exactly one action — a dict with
a single top-level key naming the action type. The same shapes appear
inside parquet checkpoints, expanded into typed columns.

Action types modelled here:

- ``protocol``        — reader/writer version + named features
- ``metaData``        — schema, partition columns, configuration
- ``add``             — parquet file added
- ``remove``          — parquet file logically removed
- ``txn``             — idempotent-write marker
- ``domainMetadata``  — per-domain named blob
- ``commitInfo``      — operational provenance (opaque to replay)

``cdc``, ``checkpointMetadata``, ``sidecar``, and the rest of the zoo
are treated as opaque — the log iterator exposes the raw payload so
callers that need them can pull them out themselves.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Dict, List, Mapping, Optional

__all__ = [
    "AddFile",
    "CommitInfo",
    "DeletionVectorDescriptor",
    "DeltaAction",
    "DomainMetadata",
    "Metadata",
    "Protocol",
    "RemoveFile",
    "Txn",
    "parse_action",
]


# ---------------------------------------------------------------------------
# Action key dispatch
# ---------------------------------------------------------------------------


class DeltaAction:
    """Marker base for every Delta action dataclass.

    Subclasses register themselves under their JSON key on the
    class-level :data:`ACTIONS` map via :meth:`__init_subclass__`.
    """

    json_key: ClassVar[str] = ""
    ACTIONS: ClassVar[Dict[str, type]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.json_key:
            DeltaAction.ACTIONS[cls.json_key] = cls

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DeltaAction":
        raise NotImplementedError

    def to_action(self) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DeletionVectorDescriptor — embedded inside add/remove actions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class DeletionVectorDescriptor:
    """Pointer to a deletion-vector blob.

    Three storage shapes:

    - ``"i"`` — inline (Z85-encoded bytes in ``pathOrInlineDv``).
    - ``"u"`` — uuid-based sidecar (``deletion_vector_<uuid>.bin``).
    - ``"p"`` — absolute path (table-relative).
    """

    storage_type: str
    path_or_inline_dv: str
    size_in_bytes: int
    cardinality: int = 0
    offset: Optional[int] = None

    @classmethod
    def from_payload(
        cls,
        payload: Optional[Mapping[str, Any]],
    ) -> "DeletionVectorDescriptor | None":
        if not payload:
            return None
        return cls(
            storage_type=str(
                payload.get("storageType") or payload.get("storage_type") or ""
            ),
            path_or_inline_dv=str(
                payload.get("pathOrInlineDv") or payload.get("path_or_inline_dv") or ""
            ),
            size_in_bytes=int(
                payload.get("sizeInBytes") or payload.get("size_in_bytes") or 0
            ),
            cardinality=int(payload.get("cardinality") or 0),
            offset=(
                int(payload["offset"]) if payload.get("offset") is not None else None
            ),
        )

    def to_payload(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "storageType": self.storage_type,
            "pathOrInlineDv": self.path_or_inline_dv,
            "sizeInBytes": int(self.size_in_bytes),
            "cardinality": int(self.cardinality),
        }
        if self.offset is not None:
            out["offset"] = int(self.offset)
        return out


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Protocol(DeltaAction):
    """Reader / writer version requirements + named features."""

    json_key: ClassVar[str] = "protocol"

    min_reader_version: int = 1
    min_writer_version: int = 2
    reader_features: List[str] = dataclasses.field(default_factory=list)
    writer_features: List[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Protocol":
        return cls(
            min_reader_version=int(payload.get("minReaderVersion") or 1),
            min_writer_version=int(payload.get("minWriterVersion") or 2),
            reader_features=list(payload.get("readerFeatures") or []),
            writer_features=list(payload.get("writerFeatures") or []),
        )

    def to_action(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "minReaderVersion": int(self.min_reader_version),
            "minWriterVersion": int(self.min_writer_version),
        }
        if self.reader_features:
            body["readerFeatures"] = list(self.reader_features)
        if self.writer_features:
            body["writerFeatures"] = list(self.writer_features)
        return {"protocol": body}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Metadata(DeltaAction):
    """Table identity + schema + partitioning + configuration."""

    json_key: ClassVar[str] = "metaData"

    id: str = ""
    name: Optional[str] = None
    description: Optional[str] = None
    format_provider: str = "parquet"
    format_options: Dict[str, str] = dataclasses.field(default_factory=dict)
    schema_string: str = ""
    partition_columns: List[str] = dataclasses.field(default_factory=list)
    configuration: Dict[str, str] = dataclasses.field(default_factory=dict)
    created_time: Optional[int] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Metadata":
        fmt = payload.get("format") or {}
        return cls(
            id=str(payload.get("id") or ""),
            name=payload.get("name"),
            description=payload.get("description"),
            format_provider=str(fmt.get("provider") or "parquet"),
            format_options=dict(fmt.get("options") or {}),
            schema_string=str(payload.get("schemaString") or ""),
            partition_columns=list(payload.get("partitionColumns") or []),
            configuration=dict(payload.get("configuration") or {}),
            created_time=(
                int(payload["createdTime"])
                if payload.get("createdTime") is not None
                else None
            ),
        )

    def to_action(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "id": self.id,
            "format": {
                "provider": self.format_provider,
                "options": dict(self.format_options),
            },
            "schemaString": self.schema_string,
            "partitionColumns": list(self.partition_columns),
            "configuration": dict(self.configuration),
        }
        if self.name is not None:
            body["name"] = self.name
        if self.description is not None:
            body["description"] = self.description
        if self.created_time is not None:
            body["createdTime"] = int(self.created_time)
        return {"metaData": body}


# ---------------------------------------------------------------------------
# AddFile / RemoveFile
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class AddFile(DeltaAction):
    """A parquet data file added to the table at a given commit version."""

    json_key: ClassVar[str] = "add"

    path: str = ""
    partition_values: Dict[str, Optional[str]] = dataclasses.field(default_factory=dict)
    size: int = 0
    modification_time: int = 0
    data_change: bool = True
    stats: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    deletion_vector: Optional[DeletionVectorDescriptor] = None
    base_row_id: Optional[int] = None
    default_row_commit_version: Optional[int] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AddFile":
        return cls(
            path=str(payload.get("path") or ""),
            partition_values=dict(payload.get("partitionValues") or {}),
            size=int(payload.get("size") or 0),
            modification_time=int(payload.get("modificationTime") or 0),
            data_change=bool(payload.get("dataChange", True)),
            stats=payload.get("stats"),
            tags=(dict(payload["tags"]) if payload.get("tags") else None),
            deletion_vector=DeletionVectorDescriptor.from_payload(
                payload.get("deletionVector")
            ),
            base_row_id=(
                int(payload["baseRowId"])
                if payload.get("baseRowId") is not None
                else None
            ),
            default_row_commit_version=(
                int(payload["defaultRowCommitVersion"])
                if payload.get("defaultRowCommitVersion") is not None
                else None
            ),
        )

    def to_action(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "path": self.path,
            "partitionValues": dict(self.partition_values),
            "size": int(self.size),
            "modificationTime": int(self.modification_time),
            "dataChange": bool(self.data_change),
        }
        if self.stats is not None:
            body["stats"] = self.stats
        if self.tags:
            body["tags"] = dict(self.tags)
        if self.deletion_vector is not None:
            body["deletionVector"] = self.deletion_vector.to_payload()
        if self.base_row_id is not None:
            body["baseRowId"] = int(self.base_row_id)
        if self.default_row_commit_version is not None:
            body["defaultRowCommitVersion"] = int(self.default_row_commit_version)
        return {"add": body}

    def __hash__(self) -> int:
        return hash(self.path)


@dataclasses.dataclass(slots=True)
class RemoveFile(DeltaAction):
    """A previously-added parquet file logically removed at this commit."""

    json_key: ClassVar[str] = "remove"

    path: str = ""
    deletion_timestamp: Optional[int] = None
    data_change: bool = True
    extended_file_metadata: bool = False
    partition_values: Optional[Dict[str, Optional[str]]] = None
    size: Optional[int] = None
    deletion_vector: Optional[DeletionVectorDescriptor] = None
    base_row_id: Optional[int] = None
    default_row_commit_version: Optional[int] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RemoveFile":
        return cls(
            path=str(payload.get("path") or ""),
            deletion_timestamp=(
                int(payload["deletionTimestamp"])
                if payload.get("deletionTimestamp") is not None
                else None
            ),
            data_change=bool(payload.get("dataChange", True)),
            extended_file_metadata=bool(payload.get("extendedFileMetadata") or False),
            partition_values=(
                dict(payload["partitionValues"])
                if payload.get("partitionValues") is not None
                else None
            ),
            size=(int(payload["size"]) if payload.get("size") is not None else None),
            deletion_vector=DeletionVectorDescriptor.from_payload(
                payload.get("deletionVector")
            ),
            base_row_id=(
                int(payload["baseRowId"])
                if payload.get("baseRowId") is not None
                else None
            ),
            default_row_commit_version=(
                int(payload["defaultRowCommitVersion"])
                if payload.get("defaultRowCommitVersion") is not None
                else None
            ),
        )

    def to_action(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "path": self.path,
            "dataChange": bool(self.data_change),
        }
        if self.deletion_timestamp is not None:
            body["deletionTimestamp"] = int(self.deletion_timestamp)
        if self.extended_file_metadata:
            body["extendedFileMetadata"] = True
        if self.partition_values is not None:
            body["partitionValues"] = dict(self.partition_values)
        if self.size is not None:
            body["size"] = int(self.size)
        if self.deletion_vector is not None:
            body["deletionVector"] = self.deletion_vector.to_payload()
        if self.base_row_id is not None:
            body["baseRowId"] = int(self.base_row_id)
        if self.default_row_commit_version is not None:
            body["defaultRowCommitVersion"] = int(self.default_row_commit_version)
        return {"remove": body}


# ---------------------------------------------------------------------------
# Txn
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Txn(DeltaAction):
    """Application-level idempotency record."""

    json_key: ClassVar[str] = "txn"

    app_id: str = ""
    version: int = 0
    last_updated: Optional[int] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Txn":
        return cls(
            app_id=str(payload.get("appId") or ""),
            version=int(payload.get("version") or 0),
            last_updated=(
                int(payload["lastUpdated"])
                if payload.get("lastUpdated") is not None
                else None
            ),
        )

    def to_action(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "appId": self.app_id,
            "version": int(self.version),
        }
        if self.last_updated is not None:
            body["lastUpdated"] = int(self.last_updated)
        return {"txn": body}


# ---------------------------------------------------------------------------
# DomainMetadata
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class DomainMetadata(DeltaAction):
    """Named per-domain configuration blob."""

    json_key: ClassVar[str] = "domainMetadata"

    domain: str = ""
    configuration: str = ""
    removed: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DomainMetadata":
        return cls(
            domain=str(payload.get("domain") or ""),
            configuration=str(payload.get("configuration") or ""),
            removed=bool(payload.get("removed") or False),
        )

    def to_action(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "domain": self.domain,
            "configuration": self.configuration,
            "removed": bool(self.removed),
        }
        return {"domainMetadata": body}


# ---------------------------------------------------------------------------
# CommitInfo
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class CommitInfo(DeltaAction):
    """Free-form commit metadata. Opaque to replay."""

    json_key: ClassVar[str] = "commitInfo"

    payload: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CommitInfo":
        return cls(payload=dict(payload))

    def to_action(self) -> Dict[str, Any]:
        return {"commitInfo": dict(self.payload)}


def parse_action(line: Mapping[str, Any]) -> "DeltaAction | None":
    """Dispatch one JSON commit line to its typed action.

    Returns ``None`` for unknown action keys — matches the spec's
    forward-compatibility guidance.
    """
    if not line:
        return None
    try:
        key = next(iter(line))
    except StopIteration:
        return None
    cls = DeltaAction.ACTIONS.get(key)
    if cls is None:
        return None
    body = line[key] or {}
    return cls.from_payload(body)
