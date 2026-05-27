from __future__ import annotations

from yggdrasil.path.path import Path


class File(Path):
    """Path that is known to be a file — skips is_dir/is_file stat calls."""

    def is_file(self) -> bool:
        return True

    def is_dir(self) -> bool:
        return False
