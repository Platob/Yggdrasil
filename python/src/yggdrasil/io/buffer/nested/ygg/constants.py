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
        manifest.arrow             # current snapshot manifest (Arrow IPC)
      part-00000.arrow             # data files (Arrow IPC by default)
      year=2025/part-00000.arrow   # partitioned data files

There is exactly one manifest file at any given time. Commits
rewrite it via stage-and-rename (POSIX rename is an atomic
overwrite, so observers either see the old or the new manifest,
never a half-written state). There is no version history and no
time-travel — overwrite is a hard delete of everything that came
before. Use upstream backups if you need recoverability.

Why Arrow IPC for the manifest
------------------------------

Delta uses newline-delimited JSON. Iceberg uses Avro. Both reparse
strings on every replay. We use **Arrow IPC** because:

- The footer indexes record batches; opening a manifest is a stat +
  fseek, not a byte-by-byte parse.
- Schema-level ``custom_metadata`` carries the table-level fields
  (table id, partition columns, primary key columns, embedded data
  schema) so one open == both the file list and the table metadata.
- The body of the manifest is itself a typed Arrow table — the
  per-file column stats live as a JSON-encoded string column for
  flexibility, but the structural columns (path, size, num_rows)
  stay strongly typed for cheap predicate evaluation against many
  files at once.

Statistics + predicate prefilter
--------------------------------

Each manifest entry carries optional :class:`ColumnStats` for each
primary key column declared at write time (and partitions, which
are also tracked via ``partition_values``). At read time, a
:class:`Predicate` walks the manifest, prunes files whose
``[min, max]`` ranges can't satisfy the predicate, and within each
surviving file scans for matching rows — returning an
``int64`` array of row indices that the caller takes against the
file. Predicate-less reads take the legacy fast path: stream
batches with no per-row work.
"""

from __future__ import annotations

__all__ = [
    "META_DIR_NAME",
    "MANIFEST_FILE_NAME",
    "DEFAULT_DATA_EXTENSION",
    "DEFAULT_ENGINE_INFO",
    "MANIFEST_META_PREFIX",
    "META_KEY_TIMESTAMP",
    "META_KEY_TABLE_ID",
    "META_KEY_PARTITION_COLUMNS",
    "META_KEY_PRIMARY_KEY_COLUMNS",
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

#: Name of the manifest file under :data:`META_DIR_NAME`. Single
#: file, rewritten in place on every commit via stage-and-rename.
MANIFEST_FILE_NAME: str = "manifest.arrow"


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

META_KEY_TIMESTAMP: str = MANIFEST_META_PREFIX + "timestamp"
META_KEY_TABLE_ID: str = MANIFEST_META_PREFIX + "table_id"
META_KEY_PARTITION_COLUMNS: str = MANIFEST_META_PREFIX + "partition_columns"
META_KEY_PRIMARY_KEY_COLUMNS: str = MANIFEST_META_PREFIX + "primary_key_columns"
META_KEY_DATA_SCHEMA: str = MANIFEST_META_PREFIX + "data_schema"
META_KEY_ENGINE_INFO: str = MANIFEST_META_PREFIX + "engine_info"


# ---------------------------------------------------------------------------
# Engine identification
# ---------------------------------------------------------------------------

#: Default value embedded in each manifest's metadata. Identifies
#: our writer so other tools and humans can trace commits back.
DEFAULT_ENGINE_INFO: str = "yggdrasil-ygg/0.2"

#: Protocol version stamped on fresh tables. Bump on incompatible
#: layout / manifest-schema changes; readers can refuse versions
#: above what they implement.
PROTOCOL_VERSION: int = 2
