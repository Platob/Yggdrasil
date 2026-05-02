"""Manifest writer + ``_LATEST`` pointer flip.

A ygg commit is two file operations:

1. Write the new manifest to ``_ygg/versions/vNNNN.arrow``.
2. Atomically swap ``_ygg/_LATEST`` to point at the new version.

Both operations stage to a ``tmp-…`` sibling first and rename into
place, so a partially-written manifest is never linkable as a
committed version.

Atomicity model (single writer)
-------------------------------

POSIX ``rename(src, dst)`` is atomic w.r.t. observers, but
overwrites ``dst`` silently. We refuse the manifest rename when
its target already exists — single writer, monotonic versions.
The pointer swap is the *only* destructive rename in the path,
and it overwrites the previous pointer by design.

The reader sequence is:

- Read ``_LATEST`` to learn the live version.
- Open ``_ygg/versions/v<N>.arrow`` with that version.

If the reader reads ``_LATEST`` between our manifest write and our
pointer swap, it sees the previous version — fine, that's still a
valid live snapshot. If it reads after the pointer swap, it sees
the new version. There is no half-state where ``_LATEST`` advertises
a manifest that doesn't exist (we write the manifest first).
"""

from __future__ import annotations


from yggdrasil.io.enums import MediaTypes
from yggdrasil.io.fs import Path

from .constants import (
    LATEST_POINTER_NAME,
    META_DIR_NAME,
    VERSIONS_DIR_NAME,
    manifest_filename,
)
from .manifest import Manifest, encode_manifest


__all__ = [
    "write_manifest",
    "manifest_path_for_version",
    "versions_dir",
    "latest_pointer_path",
    "read_latest_version",
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def versions_dir(table_root: Path) -> Path:
    """Return the ``_ygg/versions/`` directory under *table_root*."""
    return table_root / META_DIR_NAME / VERSIONS_DIR_NAME


def latest_pointer_path(table_root: Path) -> Path:
    """Return the ``_ygg/_LATEST`` pointer file path."""
    return table_root / META_DIR_NAME / LATEST_POINTER_NAME


def manifest_path_for_version(table_root: Path, version: int) -> Path:
    """Return the manifest file path for *version*."""
    return versions_dir(table_root) / manifest_filename(version)


# ---------------------------------------------------------------------------
# Pointer file
# ---------------------------------------------------------------------------


def read_latest_version(table_root: Path) -> int:
    """Read the live version number from ``_ygg/_LATEST``.

    Returns ``-1`` when the pointer file is missing — that's the
    "no commits yet" state, distinct from version 0 (an empty
    table that's been initialized).
    """
    pointer = latest_pointer_path(table_root)
    if not pointer.exists():
        return -1
    raw = pointer.read_bytes(raise_error=False)
    if not raw:
        return -1
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return -1
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(
            f"Malformed ygg pointer {pointer!r}: expected an integer "
            f"version, got {text!r}. The pointer file should contain "
            "a single ASCII integer (the live version number)."
        ) from e


def _write_pointer_atomic(table_root: Path, version: int) -> None:
    """Rewrite ``_ygg/_LATEST`` to *version* via stage + rename.

    The pointer is small (a handful of bytes); we still go through
    a staging file because POSIX rename is the only way to update
    it without exposing a half-written intermediate to a concurrent
    reader.
    """
    pointer = latest_pointer_path(table_root)
    pointer.parent.mkdir(parents=True, exist_ok=True)

    staging = pointer.parent.make_staging(media_type=None)
    body = f"{version}\n".encode("ascii")
    try:
        staging.write_bytes(body)
        staging.rename(pointer)
    except Exception:
        try:
            staging.remove(allow_not_found=True)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Public commit entry point
# ---------------------------------------------------------------------------


def write_manifest(table_root: Path, manifest: Manifest) -> Path:
    """Atomically commit *manifest* under *table_root*.

    Returns the final manifest file path on success.

    :raises FileExistsError: target version's manifest file already
        exists. ygg is single-writer; another writer raced this
        commit, or the same writer attempted to commit the same
        version twice. The caller can re-read the latest pointer
        and retry with a higher version.
    :raises OSError: any I/O failure during write or rename.
    """
    target = manifest_path_for_version(table_root, manifest.version)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        raise FileExistsError(
            f"ygg manifest version {manifest.version} already exists "
            f"at {target!r}. ygg is single-writer; another writer "
            "beat this commit, or the same writer attempted to "
            "commit the same version twice. Re-read the latest "
            "pointer and retry with a higher version."
        )

    blob = encode_manifest(manifest)

    staging = target.parent.make_staging(media_type=MediaTypes.ARROW_IPC)
    try:
        staging.write_bytes(blob)
        if target.exists():
            raise FileExistsError(
                f"ygg manifest version {manifest.version} appeared at "
                f"{target!r} between our existence check and rename. "
                "Concurrent writer detected."
            )
        staging.rename(target)
    except Exception:
        try:
            staging.remove(allow_not_found=True)
        except Exception:
            pass
        raise

    # Pointer flip. If this fails, the manifest is on disk but
    # unreachable — a future commit can simply use a higher version
    # and overwrite the pointer; the orphan manifest is harmless
    # until vacuumed.
    _write_pointer_atomic(table_root, manifest.version)

    return target
