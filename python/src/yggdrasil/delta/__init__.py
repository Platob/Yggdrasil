"""Delta Lake support — :class:`DeltaIO` and the log replay machinery.

The package is layered so callers reach for the shape they need without
pulling the whole stack:

- :mod:`yggdrasil.delta.protocol` — small dataclasses for every action
  on the Delta transaction log (``Protocol``, ``Metadata``, ``AddFile``,
  ``RemoveFile``, ``CommitInfo``, ``DeletionVectorDescriptor``).
- :mod:`yggdrasil.delta.deletion_vector` — Roaring-bitmap decoder used
  to mask soft-deleted rows out of a parquet read. Pure Python; no
  optional dependency.
- :mod:`yggdrasil.delta.log` — :class:`DeltaLog` parser that resolves a
  table version through ``_delta_log/_last_checkpoint`` (V1 + V2),
  applies JSON commits on top, and surfaces the raw action stream.
- :mod:`yggdrasil.delta.snapshot` — :class:`Snapshot` materializer that
  collapses the action stream into the active file set + metadata +
  protocol + per-file deletion-vector map.
- :mod:`yggdrasil.delta.io` — :class:`DeltaIO`, a
  :class:`yggdrasil.io.nested.FolderIO` that runs the snapshot and
  surfaces the active files as Arrow batches with predicate /
  partition pruning, deletion-vector masking, and a write path that
  emits parquet parts plus a fresh commit.

Everything routes through :class:`yggdrasil.io.path.Path`, so the same
:class:`DeltaIO` works against a local folder, an S3 prefix, or a DBFS
path without code change. Metadata fetches are always coalesced through
the path's ``iterdir`` and a single per-instance log cache — the
hottest reads (``_last_checkpoint`` + the tail of ``_delta_log``)
collapse to one round trip per snapshot resolution.
"""

from __future__ import annotations

from yggdrasil.delta.deletion_vector import DeletionVector, DeletionVectorDescriptor
from yggdrasil.delta.io import DeltaIO, DeltaOptions
from yggdrasil.delta.log import DeltaLog, LogSegment
from yggdrasil.delta.protocol import (
    AddFile,
    CommitInfo,
    DeltaAction,
    Metadata,
    Protocol,
    RemoveFile,
)
from yggdrasil.delta.snapshot import Snapshot

__all__ = [
    "AddFile",
    "CommitInfo",
    "DeletionVector",
    "DeletionVectorDescriptor",
    "DeltaAction",
    "DeltaIO",
    "DeltaLog",
    "DeltaOptions",
    "LogSegment",
    "Metadata",
    "Protocol",
    "RemoveFile",
    "Snapshot",
]
