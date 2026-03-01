# yggdrasil/pyutils/dynamic_buffer/_config.py
"""Spill-to-disk buffer configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from yggdrasil.io.path import AbstractDataPath

__all__ = ["BufferConfig", "DEFAULT_CONFIG"]


@dataclass(frozen=True, slots=True)
class BufferConfig:
    """Immutable configuration for :class:`~dynamic_buffer.BytesIO`.

    Attributes
    ----------
    spill_bytes:
        Maximum number of bytes to hold in memory before migrating the buffer
        to a temporary file on disk.  Defaults to 128 MiB.
    prefix:
        Filename prefix for spill files.  Defaults to ``"tmp-"``.
    suffix:
        Filename suffix / extension for spill files.  Defaults to ``".bin"``.
    keep_spilled_file:
        When *True*, the spill file is **not** deleted when the buffer is
        closed or garbage-collected.  Useful for post-mortem inspection or
        hand-off to another process.
    tmp_dir:
        Directory in which spill files are created.  When *None* the OS
        default temp directory is used (e.g. ``/tmp`` on Linux).
    """

    spill_bytes: int = 128 * 1024 * 1024
    keep_spilled_file: bool = False
    prefix: str = "tmp-"
    suffix: str = ".bin"
    tmp_dir: Optional["AbstractDataPath"] = None

    @classmethod
    def default(cls) -> "BufferConfig":
        """Return the shared module-level default configuration."""
        return DEFAULT_CONFIG


DEFAULT_CONFIG = BufferConfig(
    spill_bytes=128 * 1024 * 1024, # 128 MiB
    prefix="tmp-",
    suffix=".bin",
    keep_spilled_file=False,
    tmp_dir=None
)