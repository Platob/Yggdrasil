"""Tests for yggdrasil.io.io_stats — the unified ``stat`` quad + media_type."""

from __future__ import annotations

from yggdrasil.io.io_stats import IOKind, IOStats, format_bytes


class TestFormatBytes:
    def test_sub_step_stays_exact_bytes(self):
        assert format_bytes(0) == "0 B"
        assert format_bytes(512) == "512 B"
        assert format_bytes(1023) == "1023 B"

    def test_binary_units_are_iec_1024_step(self):
        assert format_bytes(1024) == "1.0 KiB"
        assert format_bytes(1536) == "1.5 KiB"
        assert format_bytes(1048576) == "1.0 MiB"
        assert format_bytes(1073741824) == "1.0 GiB"
        assert format_bytes(5 * 1024 ** 4) == "5.0 TiB"

    def test_si_units_are_1000_step(self):
        assert format_bytes(1500, binary=False) == "1.5 kB"
        assert format_bytes(1_000_000, binary=False) == "1.0 MB"

    def test_none_is_unknown(self):
        assert format_bytes(None) == "unknown"

    def test_negative_keeps_sign(self):
        assert format_bytes(-2048) == "-2.0 KiB"


class TestIOKind:
    def test_int_backed(self):
        assert isinstance(IOKind.FILE, int)
        assert IOKind.MISSING == 0
        assert IOKind.FILE == 1
        assert IOKind.DIRECTORY == 2
        assert IOKind.MEMORY == 3

    def test_members_present(self):
        for name in ("MISSING", "FILE", "DIRECTORY", "MEMORY"):
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
        from yggdrasil.enums import MediaTypes

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

    def test_field_is_not_part_of_the_stat_tuple(self):
        # ``field`` is the schema extension — it must stay out of the
        # ``os.stat_result``-style positional iteration / __getitem__.
        from yggdrasil.data.schema import Schema

        s = IOStats(size=1, field=Schema.empty())
        assert len(list(s)) == 5
        assert s[6] == 1  # positional size slot unaffected


class TestIOStatsField:
    """IOStats carries an optional cached schema in ``field``."""

    def _schema(self):
        from yggdrasil.data.schema import Schema
        import pyarrow as pa

        return Schema.from_arrow(pa.schema([("a", pa.int64())]))

    def test_default_field_is_none(self):
        assert IOStats().field is None

    def test_copy_carries_field(self):
        sch = self._schema()
        s = IOStats(size=3, field=sch)
        assert s.copy(size=9).field is sch

    def test_copy_can_override_field(self):
        s = IOStats(field=self._schema())
        assert s.copy(field=None).field is None

    def test_with_inplace_sets_field(self):
        sch = self._schema()
        s = IOStats()
        assert s.with_(field=sch, inplace=True) is s
        assert s.field is sch

    def test_repr_shows_field_only_when_set(self):
        assert "field=" not in repr(IOStats(size=1))
        assert "field=" in repr(IOStats(size=1, field=self._schema()))
