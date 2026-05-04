"""Protocol constants and on-disk file conventions for Delta.

Pulled into a separate module so the rest of the package can import
``constants.X`` without dragging in any heavy modules.

A short tour of what lives here:

- **Protocol versions** we accept on read and emit on write. The
  Delta protocol uses ``minReaderVersion`` / ``minWriterVersion``
  integers plus, since version 3 / 7 respectively, opt-in named
  feature lists (``readerFeatures`` / ``writerFeatures``). We
  enumerate which features we *implement* — anything outside that
  set we refuse, rather than read or write half-correctly.

- **Filename patterns** for commit files (`NNNNNNNNNNNNNNNNNNNN.json`),
  v1 single-part checkpoints (`NN…NN.checkpoint.parquet`), v1
  multi-part checkpoints (refused), and v2 UUID-bearing checkpoints
  (`NN…NN.checkpoint.<uuid>.parquet`).

- **Sidecar / DV directories** under ``_delta_log/`` (v2-spec
  directory layout: sidecars under ``_delta_log/_sidecars/``, DV
  ``.bin`` files conventionally next to data parquet but
  technically anywhere under the table root).

These are facts about the format, not policy. Keep them dumb.
"""

from __future__ import annotations

import re
from typing import FrozenSet, Pattern


# ---------------------------------------------------------------------------
# Reader / writer protocol versions
# ---------------------------------------------------------------------------

#: Minimum reader version in the integer-only protocol (pre-table-features).
#: We always support this much. Tables claiming a higher minReaderVersion
#: must list opt-in reader features and every feature must be in our
#: SUPPORTED_READER_FEATURES set.
SUPPORTED_READER_VERSION_LEGACY: int = 1

#: Reader version that triggers the table-features opt-in model. From this
#: version onward, only the explicitly-listed features are required.
READER_VERSION_FEATURES: int = 3

#: Writer version that triggers table-features opt-in.
WRITER_VERSION_FEATURES: int = 7

#: Default writer version for fresh tables (no features needed). Version 2
#: is the lowest with invariants support — version 1 is pre-history.
DEFAULT_FRESH_WRITER_VERSION: int = 2

#: Maximum writer version we'll write under, in the legacy-integer protocol.
#: We jump to 7 directly when we need named features.
MAX_LEGACY_WRITER_VERSION: int = 6


# ---------------------------------------------------------------------------
# Feature catalog — what we implement vs refuse
# ---------------------------------------------------------------------------

#: Reader features we implement. Anything else in a table's
#: ``readerFeatures`` causes a refusal at read time.
SUPPORTED_READER_FEATURES: FrozenSet[str] = frozenset({
    "deletionVectors",
    "v2Checkpoint",
    "columnMapping",
    "timestampNtz",
})

#: Writer features we implement. Used on write when we emit our own
#: Protocol action under the table-features model. Subset / superset of
#: the reader set is normal — DVs are read+write, but a hypothetical
#: read-only feature might appear here without a writer counterpart.
SUPPORTED_WRITER_FEATURES: FrozenSet[str] = frozenset({
    "deletionVectors",
    "v2Checkpoint",
    "appendOnly",
    "invariants",
})

#: Reader features we know about but don't support. Listing them
#: explicitly produces a clearer error message than "unknown feature".
KNOWN_REFUSED_READER_FEATURES: FrozenSet[str] = frozenset({
    "rowTracking",
    "typeWidening",
    "typeWidening-preview",
    "vacuumProtocolCheck",
    "icebergCompatV1",
    "icebergCompatV2",
    "variantType",
    "variantType-preview",
})

#: Writer features we tolerate on *read* (we never emit them). Listed
#: explicitly because some are widespread and refusing them would lock
#: out lots of real tables.
TOLERATED_WRITER_FEATURES: FrozenSet[str] = frozenset({
    "appendOnly",
    "invariants",
    "checkConstraints",
    "generatedColumns",
    "changeDataFeed",
    "identityColumns",
    "domainMetadata",
    "columnMapping",
    "timestampNtz",
    "rowTracking",
})


# ---------------------------------------------------------------------------
# Filename patterns
# ---------------------------------------------------------------------------

#: ``NNNNNNNNNNNNNNNNNNNN.json`` — 20-digit zero-padded version + .json.
COMMIT_FILE_RE: Pattern = re.compile(r"^(\d{20})\.json$")

#: ``NN…NN.checkpoint.parquet`` — v1 single-part checkpoint.
CHECKPOINT_V1_FILE_RE: Pattern = re.compile(
    r"^(\d{20})\.checkpoint\.parquet$"
)

#: ``NN…NN.checkpoint.<n>.<m>.parquet`` — v1 multi-part. Rare in practice;
#: we read these by reading every part in order, but emit single-part
#: only on write.
CHECKPOINT_V1_MULTIPART_RE: Pattern = re.compile(
    r"^(\d{20})\.checkpoint\.(\d+)\.(\d+)\.parquet$"
)

#: ``NN…NN.checkpoint.<uuid>.parquet`` — v2 manifest. The UUID
#: distinguishes simultaneous v2 checkpoint attempts; multiple
#: ``.checkpoint.<uuid>.parquet`` for the same version can coexist
#: until cleanup. ``_last_checkpoint`` arbitrates.
CHECKPOINT_V2_FILE_RE: Pattern = re.compile(
    r"^(\d{20})\.checkpoint\.([0-9a-fA-F-]{36})\.parquet$"
)


# ---------------------------------------------------------------------------
# Directory conventions
# ---------------------------------------------------------------------------

#: Subdirectory under ``_delta_log/`` for v2 checkpoint sidecars.
SIDECARS_DIR_NAME: str = "_sidecars"

#: Suggested subdirectory for deletion-vector .bin files relative to the
#: table root. The spec doesn't mandate this — DVs can live anywhere
#: under the root — but writers conventionally collocate them. We
#: write here; we read from wherever the AddFile points.
DV_DIR_NAME: str = "deletion_vectors"


# ---------------------------------------------------------------------------
# Engine identification
# ---------------------------------------------------------------------------

#: Default value for ``CommitInfo.engineInfo``. Identifies our writer in
#: the commit log so other tools and humans know where commits came
#: from. Versioned so future changes are recognizable.
DEFAULT_ENGINE_INFO: str = "yggdrasil-delta/0.1"
