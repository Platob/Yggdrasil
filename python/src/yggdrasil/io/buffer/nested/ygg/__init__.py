"""Ygg folder protocol — Arrow IPC manifest, snapshot-per-version.

A ygg table is a folder of data files (Arrow IPC by default) plus
a small ``_ygg/`` metadata side-folder that holds one Arrow IPC
manifest per committed version. Reads are O(1) (one manifest open),
writes are atomic per version (stage-and-rename + pointer flip).

It plays the same role as Delta or Iceberg in our package — a
folder-of-files protocol with versioned metadata — but trades log
replay for snapshot-per-version manifests, and JSON / Avro for
Arrow IPC. Reads are dramatically cheaper; writes pay the price
of rewriting the live file list per commit.

Top-level entry points
----------------------

- :class:`YggIO` — :class:`PartitionedFolderIO` subclass; the
  thing you instantiate to read/write a ygg table.
- :class:`YggOptions` — :class:`PartitionedOptions` subclass with
  ygg-specific knobs.

Module layout
-------------

- :mod:`.constants` — directory / filename conventions, manifest
  metadata keys, protocol version.
- :mod:`.manifest` — :class:`Manifest`, :class:`ManifestEntry`,
  Arrow IPC encode / decode.
- :mod:`.commit` — manifest writer + ``_LATEST`` pointer flip.
- :mod:`.io` — :class:`YggIO` itself; ties it together.
"""

from __future__ import annotations

from .commit import (
    latest_pointer_path,
    manifest_path_for_version,
    read_latest_version,
    versions_dir,
    write_manifest,
)
from .constants import (
    DEFAULT_DATA_EXTENSION,
    DEFAULT_ENGINE_INFO,
    LATEST_POINTER_NAME,
    META_DIR_NAME,
    MANIFEST_VERSION_RE,
    PROTOCOL_VERSION,
    VERSIONS_DIR_NAME,
    manifest_filename,
)
from .io import YggIO, YggOptions
from .manifest import (
    MANIFEST_BODY_SCHEMA,
    Manifest,
    ManifestEntry,
    decode_manifest,
    encode_manifest,
)


__all__ = [
    # Everyday API.
    "YggIO",
    "YggOptions",
    # Manifest dataclasses + codec — exposed for callers that want
    # to inspect or build manifests directly.
    "Manifest",
    "ManifestEntry",
    "MANIFEST_BODY_SCHEMA",
    "encode_manifest",
    "decode_manifest",
    # Commit / pointer helpers — diagnostics, vacuum tools, tests.
    "write_manifest",
    "manifest_path_for_version",
    "versions_dir",
    "latest_pointer_path",
    "read_latest_version",
    # Conventions.
    "META_DIR_NAME",
    "VERSIONS_DIR_NAME",
    "LATEST_POINTER_NAME",
    "MANIFEST_VERSION_RE",
    "DEFAULT_DATA_EXTENSION",
    "DEFAULT_ENGINE_INFO",
    "PROTOCOL_VERSION",
    "manifest_filename",
]
