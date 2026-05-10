"""Back-compat shim — :class:`DeltaLog` lives at :mod:`yggdrasil.io.nested.delta.log`."""

from __future__ import annotations

from yggdrasil.io.nested.delta._names import (
    LAST_CHECKPOINT_NAME,
    LOG_DIR_NAME,
    SIDECARS_DIR_NAME,
    format_checkpoint_v1_name,
    format_checkpoint_v2_manifest_name,
    format_commit_name,
    version_from_log_name,
)
from yggdrasil.io.nested.delta.log import DeltaLog, LogSegment

__all__ = [
    "DeltaLog",
    "LogSegment",
    "LOG_DIR_NAME",
    "LAST_CHECKPOINT_NAME",
    "SIDECARS_DIR_NAME",
    "format_commit_name",
    "format_checkpoint_v1_name",
    "format_checkpoint_v2_manifest_name",
    "version_from_log_name",
]
