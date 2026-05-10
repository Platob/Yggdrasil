"""Delta Lake support — :class:`DeltaIO` plus the log replay machinery.

The package is layered so callers reach for the shape they need
without pulling the whole stack:

- :mod:`yggdrasil.io.nested.delta.protocol` — small dataclasses for
  every action on the Delta transaction log (``Protocol``,
  ``Metadata``, ``AddFile``, ``RemoveFile``, ``CommitInfo``, ``Txn``,
  ``DomainMetadata``, ``DeletionVectorDescriptor``).
- :mod:`yggdrasil.io.nested.delta.deletion_vector` — Roaring-bitmap
  decoder + encoder used to mask soft-deleted rows out of a parquet
  read and to mint fresh DVs on the write side.
- :mod:`yggdrasil.io.nested.delta.log` — :class:`DeltaLog` parser
  that resolves a table version through ``_delta_log/_last_checkpoint``
  (V1 + V2), applies JSON commits on top, and surfaces the raw action
  stream.
- :mod:`yggdrasil.io.nested.delta.snapshot` — :class:`Snapshot`
  materializer that collapses the action stream into the active file
  set + metadata + protocol + per-file deletion-vector map.
- :mod:`yggdrasil.io.nested.delta.checkpoint` — V1 + V2 checkpoint
  writers (single parquet vs manifest+sidecars) and the
  ``_last_checkpoint`` pointer updater.
- :mod:`yggdrasil.io.nested.delta.delta_io` — :class:`DeltaIO`, the
  :class:`yggdrasil.io.nested.FolderIO` that runs the snapshot and
  surfaces the active files as Arrow batches with predicate /
  partition pruning, deletion-vector masking, and a write path that
  emits parquet parts plus a fresh commit. Engine fan-out (Polars /
  pandas / Spark) flows through the inherited :class:`Tabular`
  surface, so reads and writes work in any registered engine.

Everything routes through :class:`yggdrasil.io.path.Path`, so the
same :class:`DeltaIO` works against a local folder, an S3 prefix, or
a DBFS path without code change. Metadata fetches are always
coalesced through the path's ``iterdir`` and a single per-instance
log cache — the hottest reads (``_last_checkpoint`` + the tail of
``_delta_log``) collapse to one round trip per snapshot resolution.
"""

from __future__ import annotations

from yggdrasil.io.nested.delta.checkpoint import (
    update_last_checkpoint,
    write_checkpoint,
)
from yggdrasil.io.nested.delta.delta_io import DeltaIO, DeltaOptions
from yggdrasil.io.nested.delta.deletion_vector import (
    DeletionVector,
    decode_deletion_vector,
    encode_inline_deletion_vector,
    mask_batch_with_dv,
    write_uuid_deletion_vector,
)
from yggdrasil.io.nested.delta.log import DeltaLog, LogSegment
from yggdrasil.io.nested.delta.protocol import (
    AddFile,
    CommitInfo,
    DeletionVectorDescriptor,
    DeltaAction,
    DomainMetadata,
    Metadata,
    Protocol,
    RemoveFile,
    Txn,
    parse_action,
)
from yggdrasil.io.nested.delta.schema_codec import (
    arrow_schema_to_spark_json,
    schema_to_spark_json,
    spark_json_to_arrow_schema,
    spark_json_to_schema,
)
from yggdrasil.io.nested.delta.snapshot import Snapshot

__all__ = [
    "AddFile",
    "CommitInfo",
    "DeletionVector",
    "DeletionVectorDescriptor",
    "DeltaAction",
    "DeltaIO",
    "DeltaLog",
    "DeltaOptions",
    "DomainMetadata",
    "LogSegment",
    "Metadata",
    "Protocol",
    "RemoveFile",
    "Snapshot",
    "Txn",
    "arrow_schema_to_spark_json",
    "decode_deletion_vector",
    "encode_inline_deletion_vector",
    "mask_batch_with_dv",
    "parse_action",
    "schema_to_spark_json",
    "spark_json_to_arrow_schema",
    "spark_json_to_schema",
    "update_last_checkpoint",
    "write_checkpoint",
    "write_uuid_deletion_vector",
]
