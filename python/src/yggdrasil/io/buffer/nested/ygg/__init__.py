"""Ygg folder protocol — Arrow IPC manifest, single-snapshot, predicate-prunable.

A ygg table is a folder of data files (Arrow IPC by default) plus
a small ``_ygg/`` metadata side-folder that holds exactly one Arrow
IPC manifest file. Reads are O(1) (one mmap-friendly footer parse),
writes are atomic via stage-and-rename, ``OVERWRITE`` is a hard
delete of every file referenced by the previous manifest.

It plays the same role as Delta or Iceberg in our package — a
folder-of-files protocol with managed metadata — but trades log
replay + version history for one mutable manifest, and JSON / Avro
for Arrow IPC. There is no time travel; recoverability is the
caller's job.

Per-file statistics + predicate prefilter
-----------------------------------------

Each manifest entry carries optional :class:`ColumnStats`
(``min`` / ``max`` / ``null_count``) for the table's declared
primary key columns. At read time, a :class:`Predicate` walks the
manifest, prunes files whose ``[min, max]`` ranges can't satisfy
the predicate, and within each surviving file scans for matching
rows — returning an ``int64`` row-index array that the caller
takes against the file. Predicate-less reads take the legacy fast
path: stream batches with no per-row work.

Top-level entry points
----------------------

- :class:`YggIO` — :class:`PartitionedFolderIO` subclass; the
  thing you instantiate to read/write a ygg table.
- :class:`YggOptions` — :class:`PartitionedOptions` subclass with
  ygg-specific knobs (``primary_key_columns``, ``predicate``).
- :func:`eq` / :func:`is_in` / :func:`between` — convenience
  predicate constructors.
- :func:`row_indices` / :func:`filter_table` — apply a predicate
  to a fully-loaded :class:`pa.Table`.

Module layout
-------------

- :mod:`.constants` — directory / filename conventions, manifest
  metadata keys, protocol version.
- :mod:`.manifest` — :class:`Manifest`, :class:`ManifestEntry`,
  :class:`ColumnStats`, Arrow IPC encode / decode.
- :mod:`.predicate` — :class:`Predicate` interface +
  :class:`Eq` / :class:`In` / :class:`Between` / :class:`And`,
  plus the ``int64`` row-index resolver.
- :mod:`.commit` — single-file manifest writer / reader.
- :mod:`.io` — :class:`YggIO` itself; ties it together.
"""

from __future__ import annotations

from .commit import (
    manifest_path,
    read_manifest,
    write_manifest,
)
from .constants import (
    DEFAULT_DATA_EXTENSION,
    DEFAULT_ENGINE_INFO,
    MANIFEST_FILE_NAME,
    META_DIR_NAME,
    PROTOCOL_VERSION,
)
from .io import YggIO, YggOptions
from .manifest import (
    MANIFEST_BODY_SCHEMA,
    ColumnStats,
    Manifest,
    ManifestEntry,
    decode_manifest,
    encode_manifest,
)
from .predicate import (
    And,
    Between,
    Eq,
    In,
    Predicate,
    between,
    eq,
    filter_table,
    is_in,
    row_indices,
)


__all__ = [
    # Everyday API.
    "YggIO",
    "YggOptions",
    # Predicate model + helpers.
    "Predicate",
    "Eq",
    "In",
    "Between",
    "And",
    "eq",
    "is_in",
    "between",
    "row_indices",
    "filter_table",
    # Manifest dataclasses + codec.
    "Manifest",
    "ManifestEntry",
    "ColumnStats",
    "MANIFEST_BODY_SCHEMA",
    "encode_manifest",
    "decode_manifest",
    # Commit helpers.
    "write_manifest",
    "read_manifest",
    "manifest_path",
    # Conventions.
    "META_DIR_NAME",
    "MANIFEST_FILE_NAME",
    "DEFAULT_DATA_EXTENSION",
    "DEFAULT_ENGINE_INFO",
    "PROTOCOL_VERSION",
]
