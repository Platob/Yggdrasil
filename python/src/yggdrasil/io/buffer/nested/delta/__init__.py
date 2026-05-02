"""Delta Lake IO for yggdrasil.

Pure-pyarrow Delta Lake reader/writer with deletion-vector support
and v2 checkpoints. Wraps no external Delta library — the
implementation is internal to this package.

Top-level entry points
----------------------

- :class:`DeltaIO` — :class:`PartitionedFolderIO` subclass; the
  thing you instantiate to read/write a Delta table.
- :class:`DeltaOptions` — :class:`PartitionedOptions` subclass
  with Delta-specific knobs.

Module layout (for reading or extending the package):

- :mod:`.constants` — protocol versions, refused/tolerated
  feature names, filename regexes.
- :mod:`.actions` — :class:`Protocol`, :class:`Metadata`,
  :class:`AddFile`, :class:`RemoveFile`, etc., plus parse and
  serialize helpers.
- :mod:`.schema_codec` — :class:`Schema` ↔ Delta Spark-JSON
  schemaString.
- :mod:`.deletion_vector` — :class:`DeletionVectorDescriptor`,
  Z85 codec, roaring bitmap framing.
- :mod:`.replay` — log replay, v1 + v2 checkpoint loading.
- :mod:`.commit` — atomic commit-file writer.
- :mod:`.io` — :class:`DeltaIO` itself; ties it together.

Test burden flag
----------------

DV writes (in particular UPSERT) and v2 checkpoint reads are the
two areas with the highest risk of subtle protocol mismatches
against reference implementations (delta-rs, Spark). Run round-trip
tests against `deltalake` Python package outputs before trusting
any production write path.
"""

from __future__ import annotations

from .actions import (
    AddFile,
    CommitInfo,
    DomainMetadata,
    Metadata,
    Protocol,
    RemoveFile,
    Txn,
)
from .deletion_vector import (
    MAX_INLINE_DV_BYTES,
    DeletionVectorDescriptor,
)
from .io import DeltaIO, DeltaOptions
from .replay import ReplayResult, replay_log


__all__ = [
    # The everyday API.
    "DeltaIO",
    "DeltaOptions",
    # Action types — exposed for callers that want to inspect the
    # wire actions directly (e.g. to stamp a Txn before commit, or
    # to read a CommitInfo for diagnostics).
    "Protocol",
    "Metadata",
    "AddFile",
    "RemoveFile",
    "CommitInfo",
    "Txn",
    "DomainMetadata",
    # DV-side types — most callers don't touch these, but auditors
    # of UPSERT correctness will want them.
    "DeletionVectorDescriptor",
    "MAX_INLINE_DV_BYTES",
    # Replay surface.
    "ReplayResult",
    "replay_log",
]
