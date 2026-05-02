"""Manifest writer + reader.

A ygg commit is one file operation: rewrite ``_ygg/manifest.arrow``
via stage-and-rename. POSIX ``rename(src, dst)`` overwrites ``dst``
atomically, so any reader either sees the previous manifest or the
new one — never a half-written intermediate.

Atomicity model (single writer)
-------------------------------

Single writer per table. The writer:

1. Encodes the new manifest into a TTL-encoded staging file under
   ``_ygg/`` (uses :meth:`Path.make_staging`).
2. ``rename`` it onto ``_ygg/manifest.arrow``. The previous
   manifest, if any, is silently replaced.

There is no version history, no pointer file, no log replay. Old
data files referenced by the previous manifest must be removed by
the caller of this module — that's an :class:`YggIO` concern, not
a manifest concern.
"""

from __future__ import annotations

from yggdrasil.io.enums import MediaTypes
from yggdrasil.io.fs import Path

from .constants import MANIFEST_FILE_NAME, META_DIR_NAME
from .manifest import Manifest, decode_manifest, encode_manifest


__all__ = [
    "manifest_path",
    "write_manifest",
    "read_manifest",
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def manifest_path(table_root: Path) -> Path:
    """Return the manifest file path for *table_root*."""
    return table_root / META_DIR_NAME / MANIFEST_FILE_NAME


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_manifest(table_root: Path) -> Manifest | None:
    """Read the live manifest, or return ``None`` if absent.

    Raises :class:`ValueError` on a corrupt manifest — that's
    distinct from "no manifest yet" and the caller has to act
    differently.
    """
    target = manifest_path(table_root)
    if not target.exists():
        return None
    blob = target.read_bytes()
    if not blob:
        return None
    return decode_manifest(blob)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_manifest(table_root: Path, manifest: Manifest) -> Path:
    """Atomically (re)write the manifest under *table_root*.

    Returns the final manifest file path on success. Always
    overwrites — there is no "version already exists" check
    because there are no versions.

    :raises OSError: any I/O failure during write or rename. The
        staging file is cleaned up on the error path; partial
        commits do not appear in the table.
    """
    target = manifest_path(table_root)
    target.parent.mkdir(parents=True, exist_ok=True)

    blob = encode_manifest(manifest)

    staging = target.parent.make_staging(media_type=MediaTypes.ARROW_IPC)
    try:
        staging.write_bytes(blob)
        staging.rename(target)
    except Exception:
        try:
            staging.remove(allow_not_found=True)
        except Exception:
            pass
        raise

    return target
