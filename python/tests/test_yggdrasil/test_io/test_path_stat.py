"""Tests for yggdrasil.io.path_stat."""

from __future__ import annotations

import pytest

from yggdrasil.io.path_stat import PathKind, PathStats


class TestPathKind:
    def test_members_are_lower_snake_strings(self):
        assert PathKind.MISSING.value == "missing"
        assert PathKind.FILE.value == "file"
        assert PathKind.DIRECTORY.value == "directory"
        assert PathKind.SYMLINK.value == "symlink"

    def test_str_subclass(self):
        assert isinstance(PathKind.FILE, str)
        assert PathKind.FILE == "file"


class TestPathStatsDefaults:
    def test_defaults(self):
        stats = PathStats()
        assert stats.size == 0
        assert stats.mtime == 0.0
        assert stats.kind is PathKind.MISSING
        assert stats.mode == 0

    def test_st_aliases(self):
        stats = PathStats(size=42, mtime=12.5, mode=0o755)
        assert stats.st_size == 42
        assert stats.st_mtime == 12.5
        assert stats.st_mode == 0o755

    def test_subscript_access(self):
        # os.stat_result-style indexing: (mode, ?, ?, ?, ?, ?, size, ?, mtime, ?)
        stats = PathStats(size=10, mtime=1.0, mode=0o644)
        assert stats[0] == 0o644
        assert stats[6] == 10
        assert stats[8] == 1.0


class TestPathStatsImmutability:
    def test_frozen_dataclass(self):
        stats = PathStats(size=1)
        with pytest.raises((AttributeError, TypeError)):
            stats.size = 2  # type: ignore[misc]


class TestPathStatsWith:
    def test_with_copy_returns_new_instance(self):
        original = PathStats(size=1, mtime=1.0, kind=PathKind.FILE, mode=0o644)
        updated = original.with_(size=99, copy=True)
        assert original.size == 1
        assert updated.size == 99
        # Other fields preserved
        assert updated.mtime == 1.0
        assert updated.kind is PathKind.FILE
        assert updated.mode == 0o644

    def test_with_inplace_mutates_via_object_setattr(self):
        stats = PathStats()
        result = stats.with_(size=10, kind=PathKind.FILE, mode=0o755)
        assert result is stats
        assert stats.size == 10
        assert stats.kind is PathKind.FILE
        assert stats.mode == 0o755

    def test_with_none_does_not_overwrite(self):
        stats = PathStats(size=5)
        stats.with_(size=None)
        assert stats.size == 5
