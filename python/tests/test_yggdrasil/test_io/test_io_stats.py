"""Tests for yggdrasil.io.io_stats — the unified ``stat`` quad + media_type."""

from __future__ import annotations

from yggdrasil.io.io_stats import IOKind, IOStats


class TestIOKind:
    def test_int_backed(self):
        assert isinstance(IOKind.FILE, int)
        assert IOKind.MISSING == 0
        assert IOKind.FILE == 1
        assert IOKind.DIRECTORY == 2

    def test_members_present(self):
        # Spot-check the full enum surface.
        assert IOKind.SYMLINK is IOKind(IOKind.SYMLINK.value)
        for name in (
            "MISSING", "FILE", "DIRECTORY", "SYMLINK",
            "SOCKET", "FIFO", "CHAR_DEVICE", "BLOCK_DEVICE",
        ):
            assert hasattr(IOKind, name)


class TestIOStatsDefaults:
    def test_defaults(self):
        s = IOStats()
        assert s.size == 0
        assert s.mtime == 0.0
        assert s.kind is IOKind.MISSING
        assert s.mode == 0
        assert s.media_type is None

    def test_st_aliases(self):
        s = IOStats(size=42, mtime=12.5, mode=0o755)
        assert s.st_size == 42
        assert s.st_mtime == 12.5
        assert s.st_mode == 0o755

    def test_subscript_access(self):
        # os.stat_result layout: (mode, ?, ?, ?, ?, ?, size, ?, mtime, ?)
        s = IOStats(size=10, mtime=1.0, mode=0o644)
        assert s[0] == 0o644
        assert s[6] == 10
        assert s[8] == 1.0


class TestIOStatsKindHelpers:
    def test_exists(self):
        assert not IOStats(kind=IOKind.MISSING).exists
        assert IOStats(kind=IOKind.FILE).exists
        assert IOStats(kind=IOKind.DIRECTORY).exists

    def test_is_file(self):
        assert IOStats(kind=IOKind.FILE).is_file
        assert not IOStats(kind=IOKind.DIRECTORY).is_file

    def test_is_dir(self):
        assert IOStats(kind=IOKind.DIRECTORY).is_dir
        assert not IOStats(kind=IOKind.FILE).is_dir


class TestIOStatsWith:
    def test_with_default_returns_new_instance(self):
        original = IOStats(
            size=1, mtime=1.0, kind=IOKind.FILE, mode=0o644,
        )
        updated = original.with_(size=99)
        assert updated is not original
        assert original.size == 1
        assert updated.size == 99
        # Other fields preserved.
        assert updated.mtime == 1.0
        assert updated.kind is IOKind.FILE
        assert updated.mode == 0o644

    def test_with_inplace_mutates_self(self):
        s = IOStats()
        result = s.with_(size=10, kind=IOKind.FILE, mode=0o755, inplace=True)
        assert result is s
        assert s.size == 10
        assert s.kind is IOKind.FILE
        assert s.mode == 0o755

    def test_with_omitted_fields_carry_over(self):
        s = IOStats(size=5)
        out = s.with_()
        assert out is not s
        assert out.size == 5

    def test_with_clears_media_type_explicitly(self):
        from yggdrasil.io.enums import MediaTypes

        s = IOStats(media_type=MediaTypes.JSON)
        s.with_(media_type=None, inplace=True)
        assert s.media_type is None

    def test_copy_returns_independent_instance(self):
        original = IOStats(size=4, kind=IOKind.FILE)
        copy = original.copy()
        assert copy is not original
        assert (copy.size, copy.kind) == (4, IOKind.FILE)
        copy.size = 99
        assert original.size == 4

    def test_copy_overrides_only_named_fields(self):
        original = IOStats(size=4, mtime=1.5, mode=0o600)
        copy = original.copy(size=99, media_type=None)
        assert copy.size == 99
        assert copy.mtime == 1.5
        assert copy.mode == 0o600
        assert copy.media_type is None


class TestIOStatsIteration:
    def test_iter_yields_full_quad_plus_media_type(self):
        s = IOStats(
            size=10, mtime=2.5, kind=IOKind.FILE, mode=0o600, media_type=None,
        )
        size, mtime, kind, mode, media_type = s
        assert (size, mtime, kind, mode, media_type) == (
            10, 2.5, IOKind.FILE, 0o600, None,
        )
