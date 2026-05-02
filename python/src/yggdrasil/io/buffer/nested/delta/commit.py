"""Commit-file writer for Delta.

A Delta commit is one JSON file under ``_delta_log/`` named
``NN…NN.json`` with N = the commit version, content =
newline-delimited JSON Action objects.

Atomicity model (single writer)
-------------------------------

We assume a single writer per table. The atomicity story is:

1. Write the commit body to a TTL-encoded staging file under
   ``_delta_log/`` (uses :meth:`Path.make_staging`).
2. Rename the staging file to the target ``NN…NN.json`` name.
   If a file already exists at the target, refuse the rename
   and surface the conflict — single-writer assumption broken.

This is *not* a true compare-and-swap. POSIX ``rename(src, dst)``
overwrites ``dst`` atomically; we explicitly check existence
before the rename to refuse rather than overwrite. There's a
TOCTOU race where two processes could both see "version N
absent" simultaneously and both rename — the second one wins
silently. For multi-writer correctness, the writer needs the
atomic-create-only-if-absent semantics that most cloud object
stores expose (e.g. S3 ``IfNoneMatch``); plumbing that through
yggdrasil's :class:`Path` is out of scope here.

Failure handling
----------------

Failed staging or rename: the staging file's TTL ensures
eventual cleanup; we also explicitly remove on the error path.
The caller (``DeltaIO`` write methods) is responsible for
cleaning up parquet data files written for this commit when the
commit itself fails — that lives in ``io.py``, not here.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from yggdrasil.io.enums import MediaTypes
from yggdrasil.io.fs import Path

from .actions import serialize_action


__all__ = [
    "write_commit",
    "build_commit_body",
    "commit_path_for_version",
]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def commit_path_for_version(log_dir: Path, version: int) -> Path:
    """Build the commit file path for *version* under *log_dir*."""
    return log_dir / f"{version:020d}.json"


def build_commit_body(actions: Sequence[Any]) -> bytes:
    """Serialize a list of action dataclasses to commit-file bytes.

    One JSON object per line, no trailing space. ``json.dumps`` with
    ``separators=(",", ":")`` produces the compact form Delta
    convention prefers (smaller commit files, faster parse).
    """
    lines: list[str] = []
    for action in actions:
        # ``serialize_action`` accepts both pre-wrapped dicts (for
        # caller convenience: pass a raw {"commitInfo": {...}}) and
        # dataclass instances. Detect and route.
        if isinstance(action, Mapping) and len(action) == 1 and not isinstance(action, dict):
            envelope = dict(action)
        elif isinstance(action, dict):
            envelope = action
        else:
            envelope = serialize_action(action)
        lines.append(json.dumps(envelope, separators=(",", ":")))
    body = "\n".join(lines) + "\n"
    return body.encode("utf-8")


def write_commit(
    log_dir: Path,
    version: int,
    actions: Sequence[Any],
) -> Path:
    """Atomically write a commit file at *version*.

    Returns the final commit file path on success.

    :raises FileExistsError: target version's commit file already
        exists. The caller should re-replay (a concurrent writer
        beat us to it) and retry with a higher version, or surface
        the conflict.
    :raises OSError: any I/O failure during write or rename. The
        staging file is cleaned up on error path; partial commits
        do not appear in the log.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    target = commit_path_for_version(log_dir, version)

    if target.exists():
        raise FileExistsError(
            f"Delta commit version {version} already exists at "
            f"{target!r}. yggdrasil DeltaIO is single-writer; another "
            "writer raced this commit, or the same writer attempted "
            "to commit the same version twice."
        )

    body = build_commit_body(actions)

    staging = log_dir.make_staging(media_type=MediaTypes.JSON)
    try:
        staging.write_bytes(body)
        # One more existence check after the staging write — narrows
        # the TOCTOU window; doesn't close it. A real CAS would do
        # this in one syscall.
        if target.exists():
            raise FileExistsError(
                f"Delta commit version {version} appeared at "
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

    return target
