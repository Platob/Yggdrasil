"""Filename helpers for the Delta ``_delta_log`` directory.

Every file under ``_delta_log`` is named after a 20-zero-padded
version number plus a suffix that identifies its kind:

- ``00000000000000000007.json`` — version 7 commit.
- ``00000000000000000010.checkpoint.parquet`` — V1 single-file
  checkpoint for version 10.
- ``00000000000000000010.checkpoint.0000000001.0000000003.parquet`` —
  V1 multi-part checkpoint (rare; parts indexed against ``size``).
- ``00000000000000000010.checkpoint.<uuid>.json`` — V2 manifest.
- ``00000000000000000010.checkpoint.<uuid>.parquet`` — V2 sidecar
  (lives under ``_delta_log/_sidecars/`` in modern writers).
- ``_last_checkpoint`` — JSON pointer at the most recent checkpoint.

This module owns the parsing + formatting of those names. Keeping the
regex and the format helpers in one place means the writer, the log
reader, and the checkpoint writer never disagree on padding or suffix
shape.
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = [
    "LOG_DIR_NAME",
    "LAST_CHECKPOINT_NAME",
    "SIDECARS_DIR_NAME",
    "format_commit_name",
    "format_checkpoint_v1_name",
    "format_checkpoint_v2_manifest_name",
    "format_checkpoint_v2_sidecar_name",
    "version_from_log_name",
]


LOG_DIR_NAME = "_delta_log"
LAST_CHECKPOINT_NAME = "_last_checkpoint"
SIDECARS_DIR_NAME = "_sidecars"

#: 20-zero-padded version. ``20 == len(str(2**63))`` rounded up — every
#: int64 version fits without overflow.
_VERSION_FMT = "{:020d}"

#: Matches every recognised name. ``version`` always present;
#: ``uuid``/``part`` populated for the relevant checkpoint shapes.
_VERSION_RE = re.compile(
    r"^(?P<version>\d{20})"
    r"(?:"
    r"\.checkpoint(?:\.(?P<uuid>[0-9A-Fa-f-]+))?(?:\.(?P<part>[0-9]+))?(?:\.parquet|\.json)"
    r"|\.json"
    r")$"
)


def format_commit_name(version: int) -> str:
    """``00000000000000000007.json`` for ``version=7``."""
    return f"{_VERSION_FMT.format(int(version))}.json"


def format_checkpoint_v1_name(version: int) -> str:
    """``...10.checkpoint.parquet`` for the single-parquet V1 layout."""
    return f"{_VERSION_FMT.format(int(version))}.checkpoint.parquet"


def format_checkpoint_v2_manifest_name(version: int, uuid: str) -> str:
    """``...10.checkpoint.<uuid>.json`` — the V2 manifest filename."""
    return f"{_VERSION_FMT.format(int(version))}.checkpoint.{uuid}.json"


def format_checkpoint_v2_sidecar_name(uuid: str) -> str:
    """``<uuid>.parquet`` — the V2 sidecar filename, under ``_sidecars/``."""
    return f"{uuid}.parquet"


def version_from_log_name(name: str) -> Optional[int]:
    """Extract the version embedded in a log filename, or ``None``.

    Returns ``None`` for files we don't recognise (``_last_checkpoint``,
    ``.crc`` companions, the ``_sidecars`` directory, and any other
    metadata file Delta-tooling sprinkles in beside the canonical set).
    """
    m = _VERSION_RE.match(name)
    if m is None:
        return None
    return int(m.group("version"))
