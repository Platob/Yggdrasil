"""Back-compat shim — Delta support lives at :mod:`yggdrasil.io.nested.delta`.

The whole package was relocated under ``yggdrasil.io.nested`` to sit
alongside the other :class:`Folder` leaves (folders, zip archives,
ygg folders). This module keeps the old import paths
(``yggdrasil.delta``, ``yggdrasil.delta.io``, ``yggdrasil.delta.log``,
…) working so existing callers don't break — every name here is just
re-exported from its new home.

New code should import from :mod:`yggdrasil.io.nested.delta`
directly; this shim exists to avoid a flag-day rename.
"""

from __future__ import annotations

from yggdrasil.io.nested.delta import (
    AddFile,
    CommitInfo,
    DeletionVector,
    DeletionVectorDescriptor,
    DeltaAction,
    DeltaFolder,
    DeltaLog,
    DeltaOptions,
    DomainMetadata,
    LogSegment,
    Metadata,
    Protocol,
    RemoveFile,
    Snapshot,
    Txn,
    parse_action,
)

__all__ = [
    "AddFile",
    "CommitInfo",
    "DeletionVector",
    "DeletionVectorDescriptor",
    "DeltaAction",
    "DeltaFolder",
    "DeltaLog",
    "DeltaOptions",
    "DomainMetadata",
    "LogSegment",
    "Metadata",
    "Protocol",
    "RemoveFile",
    "Snapshot",
    "Txn",
    "parse_action",
]
