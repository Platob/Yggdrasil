"""On-disk conventions for the ygg folder protocol.

Pulled into a separate module so the rest of the package can import
``constants.X`` without dragging in pyarrow at import time.

Layout
------

A ygg table is a folder. Data lives at the root (or under
``key=value/`` partition prefixes); metadata lives in a single
side-folder named :data:`META_DIR_NAME` (``_ygg/`` by default)::

    table_root/
      _ygg/
        _LATEST                    # plain text: latest version number
        versions/
          v00000000000.arrow       # Arrow IPC manifest for version 0
          v00000000001.arrow       # ... for version 1
      part-00000.arrow             # data files (Arrow IPC by default)
      year=2025/part-00000.arrow   # partitioned data files

Why Arrow IPC for the manifest
------------------------------

Delta uses newline-delimited JSON. Iceberg uses Avro. Both reparse
strings on every replay. We use **Arrow IPC** because:

- The footer indexes record batches; opening a manifest is a stat +
  fseek, not a byte-by-byte parse.
- Schema-level ``custom_metadata`` carries the table-level fields
  (table id, version, partition columns, embedded data schema) so
  one open == both the file list and the table metadata.
- The manifest itself is the same on-disk format as the data, so a
  caller already wired for :class:`ArrowIPCIO` reads it the same
  way it reads everything else.

Versioning model
----------------

A ygg table is a sequence of immutable per-version snapshots.
Reading version N is one Arrow IPC open of ``versions/vN.arrow``;
no log replay, no checkpoint dance. Writes mint a new snapshot
file, then atomically swap ``_LATEST`` to point at it. Old
snapshots are kept for time-travel until a vacuum removes them.
"""

from __future__ import annotations

import re
from typing import Pattern


__all__ = [
    "META_DIR_NAME",
    "VERSIONS_DIR_NAME",
    "LATEST_POINTER_NAME",
    "MANIFEST_VERSION_RE",
    "DEFAULT_DATA_EXTENSION",
    "DEFAULT_ENGINE_INFO",
    "MANIFEST_META_PREFIX",
    "META_KEY_VERSION",
    "META_KEY_TIMESTAMP",
    "META_KEY_TABLE_ID",
    "META_KEY_PARTITION_COLUMNS",
    "META_KEY_DATA_SCHEMA",
    "META_KEY_ENGINE_INFO",
    "PROTOCOL_VERSION",
]


# ---------------------------------------------------------------------------
# Directory / filename conventions
# ---------------------------------------------------------------------------

#: Side-folder under the table root for ygg metadata. Hidden from
#: :meth:`NestedIO._is_ignored_path` so iter_children doesn't try
#: to read it as data.
META_DIR_NAME: str = "_ygg"

#: Sub-folder of :data:`META_DIR_NAME` that holds per-version
#: snapshot manifests. One file per version, never rewritten.
VERSIONS_DIR_NAME: str = "versions"

#: Tiny pointer file under :data:`META_DIR_NAME` whose content is
#: the latest committed version number as ASCII digits. Updated by
#: write-then-rename for atomicity (single-writer assumption).
LATEST_POINTER_NAME: str = "_LATEST"

#: ``vNNNNNNNNNNN.arrow`` — 11-digit zero-padded version + .arrow.
#: 11 digits comfortably accommodates 100B versions; if you hit
#: that you have other problems.
MANIFEST_VERSION_RE: Pattern = re.compile(r"^v(\d{11})\.arrow$")


def manifest_filename(version: int) -> str:
    """Build the manifest filename for *version*."""
    if version < 0:
        raise ValueError(
            f"Manifest version must be >= 0; got {version!r}."
        )
    return f"v{version:011d}.arrow"


# ---------------------------------------------------------------------------
# Data file conventions
# ---------------------------------------------------------------------------

#: Default extension for child data files. The folder writer mints
#: ``part-NNNNN.arrow`` filenames using this extension when no
#: explicit ``child_media_type`` is set on the options.
DEFAULT_DATA_EXTENSION: str = "arrow"


# ---------------------------------------------------------------------------
# Manifest custom-metadata keys
# ---------------------------------------------------------------------------

#: Common prefix for every ygg-owned key in the manifest's
#: schema-level ``metadata`` dict. Lets a future reader filter
#: ygg-owned keys from caller-stamped ones at a glance.
MANIFEST_META_PREFIX: str = "ygg."

META_KEY_VERSION: str = MANIFEST_META_PREFIX + "version"
META_KEY_TIMESTAMP: str = MANIFEST_META_PREFIX + "timestamp"
META_KEY_TABLE_ID: str = MANIFEST_META_PREFIX + "table_id"
META_KEY_PARTITION_COLUMNS: str = MANIFEST_META_PREFIX + "partition_columns"
META_KEY_DATA_SCHEMA: str = MANIFEST_META_PREFIX + "data_schema"
META_KEY_ENGINE_INFO: str = MANIFEST_META_PREFIX + "engine_info"


# ---------------------------------------------------------------------------
# Engine identification
# ---------------------------------------------------------------------------

#: Default value embedded in each manifest's metadata. Identifies
#: our writer so other tools and humans can trace commits back.
DEFAULT_ENGINE_INFO: str = "yggdrasil-ygg/0.1"

#: Protocol version stamped on fresh tables. Bump on incompatible
#: layout / manifest-schema changes; readers can refuse versions
#: above what they implement.
PROTOCOL_VERSION: int = 1
