"""Delta Lake support — :class:`DeltaFolder` plus the log replay machinery.

Reworked implementation with full Delta read/write protocol:

- :mod:`~.protocol` — action dataclasses (Protocol, Metadata, AddFile, etc.)
- :mod:`~.deletion_vector` — Roaring-bitmap + simple-list DV encode/decode
- :mod:`~.log` — transaction-log parser with V1+V2 checkpoint support
- :mod:`~.snapshot` — action-stream reduction into table state
- :mod:`~.checkpoint` — V1+V2 checkpoint writers with multi-sidecar V2
- :mod:`~.delta_folder` — the Folder subclass orchestrating everything
- :mod:`~.schema_codec` — Arrow <-> Spark schema conversion
"""

from __future__ import annotations

from yggdrasil.io.delta.checkpoint import (
    update_last_checkpoint,
    write_checkpoint,
)
from yggdrasil.io.delta.delta_folder import (
    ConcurrentDeltaCommitError,
    DeltaFolder,
    DeltaOptions,
)
from yggdrasil.io.delta.deletion_vector import (
    DeletionVector,
    decode_deletion_vector,
    encode_inline_deletion_vector,
    mask_batch_with_dv,
    write_uuid_deletion_vector,
)
from yggdrasil.io.delta.log import DeltaLog, LogSegment
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
from yggdrasil.io.delta.schema_codec import (
    arrow_schema_to_spark_json,
    schema_to_spark_json,
    spark_json_to_arrow_schema,
    spark_json_to_schema,
)
from yggdrasil.io.delta.snapshot import Snapshot

__all__ = [
    "AddFile",
    "CommitInfo",
    "ConcurrentDeltaCommitError",
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
