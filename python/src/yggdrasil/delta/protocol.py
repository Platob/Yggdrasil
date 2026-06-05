"""Back-compat shim — Delta action types live at :mod:`yggdrasil.io.delta.protocol`."""

from __future__ import annotations

from yggdrasil.io.delta.protocol import (
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
