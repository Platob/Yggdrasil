"""Tests for :class:`yggdrasil.io.parquet_file.ParquetFile`."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.path.memory import Memory
from yggdrasil.path.local_path import LocalPath
from yggdrasil.io.parquet_file import ParquetFile


class TestRegistration:

    def test_class_for_media_type_parquet(self) -> None:
        assert Holder.class_for_media_type("parquet") is ParquetFile

    def test_class_for_media_type_mime(self) -> None:
        assert (
            Holder.class_for_media_type("application/vnd.apache.parquet")
            is ParquetFile
        )

    def test_path_dispatches_via_extension(self, tmp_path) -> None:
        from yggdrasil.io.base import IO

        b = IO(path=str(tmp_path / "x.parquet"))
        assert isinstance(b, ParquetFile)

    def test_open_local_path_dispatches(self, tmp_path) -> None:
        lp = LocalPath(str(tmp_path / "x.parquet"))
        cursor = lp.open("rb", auto_open=False)
        assert isinstance(cursor, ParquetFile)


class TestMemoryRoundTrip:

    @pytest.fixture
    def table(self) -> pa.Table:
        return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})

    def test_write_then_read_arrow_table(self, table) -> None:
        mem = Memory()
        leaf = ParquetFile(holder=mem, owns_holder=False)
        leaf.write_arrow_table(table)
        assert mem.size > 0

        leaf2 = ParquetFile(holder=mem, owns_holder=False)
        assert leaf2.read_arrow_table().equals(table)

    def test_dispatch_via_stamped_media(self, table) -> None:
        from yggdrasil.enums import MediaType, MimeTypes
        from yggdrasil.io.base import IO

        mem = Memory()
        mem.media_type = MediaType(MimeTypes.PARQUET)
        writer = IO(holder=mem, owns_holder=False)
        assert isinstance(writer, ParquetFile)
        writer.write_arrow_table(table)

        reader = IO(holder=mem, owns_holder=False)
        assert reader.read_arrow_table().equals(table)


class TestLocalPathRoundTrip:

    def test_write_and_read_back(self, tmp_path) -> None:
        table = pa.table({"x": [1, 2, 3]})
        path = LocalPath(str(tmp_path / "out.parquet"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(table)
        with path.open("rb") as cursor:
            assert cursor.read_arrow_table().equals(table)

    def test_open_returns_parquet_file_cursor(self, tmp_path) -> None:
        path = LocalPath(str(tmp_path / "x.parquet"))
        cursor = path.open("rb", auto_open=False)
        assert isinstance(cursor, ParquetFile)
        assert cursor.parent is path


class TestOptions:

    def test_options_class(self) -> None:
        from yggdrasil.io.parquet_file import ParquetOptions

        assert ParquetFile.options_class() is ParquetOptions


class TestWriteArrowTableBypassesBatchHook:
    """``ParquetFile._write_arrow_table`` should route the
    "replace the buffer wholesale" shapes straight through
    ``pq.write_table`` and only fall through to
    ``_write_arrow_batches`` for read-modify-rewrite merge cases
    and the guarded ``IGNORE`` / ``ERROR_IF_EXISTS`` paths."""

    @staticmethod
    def _counting_patch(monkeypatch):
        from yggdrasil.io.parquet_file import ParquetFile

        calls = {"n": 0}
        original = ParquetFile._write_arrow_batches

        def counting(self, batches, options):
            calls["n"] += 1
            return original(self, batches, options)

        monkeypatch.setattr(ParquetFile, "_write_arrow_batches", counting)
        return calls

    def test_overwrite_on_empty_skips_batch_hook(self, monkeypatch) -> None:
        from yggdrasil.io.parquet_file import ParquetFile

        calls = self._counting_patch(monkeypatch)
        table = pa.table({"id": list(range(1000))})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(table)

        assert calls["n"] == 0
        assert ParquetFile(
            holder=mem, owns_holder=False,
        ).read_arrow_table().equals(table)

    def test_explicit_overwrite_on_nonempty_skips_batch_hook(
        self, monkeypatch,
    ) -> None:
        from yggdrasil.enums import Mode
        from yggdrasil.io.parquet_file import ParquetFile

        seed = pa.table({"id": [1, 2, 3]})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        # Buffer is non-empty now; explicit OVERWRITE must still
        # bypass.
        calls = self._counting_patch(monkeypatch)
        replacement = pa.table({"id": [99]})
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(
            replacement, mode=Mode.OVERWRITE,
        )
        assert calls["n"] == 0

        out = ParquetFile(holder=mem, owns_holder=False).read_arrow_table()
        assert out.column("id").to_pylist() == [99]

    def test_truncate_routes_to_fast_path(self, monkeypatch) -> None:
        from yggdrasil.enums import Mode
        from yggdrasil.io.parquet_file import ParquetFile

        seed = pa.table({"id": [1, 2, 3]})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        calls = self._counting_patch(monkeypatch)
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [42]}), mode=Mode.TRUNCATE,
        )
        assert calls["n"] == 0

        out = ParquetFile(holder=mem, owns_holder=False).read_arrow_table()
        assert out.column("id").to_pylist() == [42]

    def test_append_to_nonempty_uses_batch_hook(self, monkeypatch) -> None:
        from yggdrasil.enums import Mode
        from yggdrasil.io.parquet_file import ParquetFile

        seed = pa.table({"id": [1, 2, 3]})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        calls = self._counting_patch(monkeypatch)
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [4, 5]}), mode=Mode.APPEND,
        )
        assert calls["n"] >= 1

        out = ParquetFile(holder=mem, owns_holder=False).read_arrow_table()
        assert sorted(out.column("id").to_pylist()) == [1, 2, 3, 4, 5]

    def test_append_to_empty_skips_batch_hook(self, monkeypatch) -> None:
        """APPEND on an empty buffer reduces to OVERWRITE — the
        merge logic has nothing to merge against, so the fast path
        is safe and the batch hook is overhead."""
        from yggdrasil.enums import Mode
        from yggdrasil.io.parquet_file import ParquetFile

        calls = self._counting_patch(monkeypatch)
        table = pa.table({"id": [1, 2, 3]})
        ParquetFile(holder=Memory(), owns_holder=False).write_arrow_table(
            table, mode=Mode.APPEND,
        )
        assert calls["n"] == 0

    def test_auto_on_nonempty_uses_batch_hook(self, monkeypatch) -> None:
        """AUTO with no ``match_by`` resolves to APPEND semantics
        on a non-empty buffer — the existing data must survive, so
        the batch hook owns the rewrite."""
        from yggdrasil.io.parquet_file import ParquetFile

        seed = pa.table({"id": [1, 2, 3]})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        calls = self._counting_patch(monkeypatch)
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [4, 5]}),
        )  # default mode = AUTO
        assert calls["n"] >= 1

        out = ParquetFile(holder=mem, owns_holder=False).read_arrow_table()
        assert sorted(out.column("id").to_pylist()) == [1, 2, 3, 4, 5]

    def test_ignore_on_nonempty_uses_batch_hook(self, monkeypatch) -> None:
        """IGNORE on a non-empty buffer must NOT clobber existing
        data — that's the batch hook's size-check + return."""
        from yggdrasil.enums import Mode
        from yggdrasil.io.parquet_file import ParquetFile

        seed = pa.table({"id": [1, 2, 3]})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(seed)
        original_bytes = mem.to_bytes()

        calls = self._counting_patch(monkeypatch)
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [99]}), mode=Mode.IGNORE,
        )
        assert calls["n"] >= 1
        # Bytes unchanged — IGNORE skipped the write.
        assert mem.to_bytes() == original_bytes

    def test_error_if_exists_on_nonempty_uses_batch_hook(
        self, monkeypatch,
    ) -> None:
        from yggdrasil.enums import Mode
        from yggdrasil.io.parquet_file import ParquetFile

        seed = pa.table({"id": [1, 2, 3]})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        calls = self._counting_patch(monkeypatch)
        with pytest.raises(FileExistsError):
            ParquetFile(holder=mem, owns_holder=False).write_arrow_table(
                pa.table({"id": [99]}), mode=Mode.ERROR_IF_EXISTS,
            )
        assert calls["n"] >= 1

    def test_upsert_with_match_by_uses_batch_hook(self, monkeypatch) -> None:
        """UPSERT (or AUTO+match_by) needs the read-modify-rewrite
        path so the keyed dedup actually fires."""
        from yggdrasil.enums import Mode
        from yggdrasil.io.parquet_file import ParquetFile

        seed = pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        calls = self._counting_patch(monkeypatch)
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [2, 4], "v": ["B", "d"]}),
            mode=Mode.UPSERT, match_by=["id"],
        )
        assert calls["n"] >= 1

        out = ParquetFile(holder=mem, owns_holder=False).read_arrow_table()
        pairs = sorted(
            zip(out.column("id").to_pylist(), out.column("v").to_pylist())
        )
        assert pairs == [(1, "a"), (2, "B"), (3, "c"), (4, "d")]

    def test_compression_and_row_group_options_pass_through(self) -> None:
        """The fast path must thread the parquet writer options the
        same way the batch path does — compression, row_group_size,
        statistics, dictionary."""
        from yggdrasil.io.parquet_file import (
            ParquetFile, ParquetOptions,
        )

        # 10k rows so multiple row groups are possible.
        table = pa.table({"id": list(range(10_000))})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(
            table,
            options=ParquetOptions(
                compression="snappy",
                row_group_size=1_000,
                write_statistics=True,
                use_dictionary=True,
            ),
        )

        # Inspect via pyarrow directly to confirm the options landed.
        import pyarrow.parquet as _pq
        reader = _pq.ParquetFile(pa.BufferReader(mem.to_bytes()))
        assert reader.num_row_groups == 10  # 10_000 / 1_000
        meta = reader.metadata.row_group(0).column(0)
        assert meta.compression.lower() == "snappy"

    def test_target_schema_cast_applied_on_fast_path(self) -> None:
        """When ``options.target`` reshapes the input, the fast path
        must still apply the cast — otherwise the bytes encode the
        wrong dtype."""
        from yggdrasil.data.options import CastOptions
        from yggdrasil.data.data_field import Field
        from yggdrasil.io.parquet_file import ParquetFile

        # int64 source, int32 target — the fast path should cast.
        source = pa.table({"id": pa.array([1, 2, 3], type=pa.int64())})
        target = Field.from_(pa.schema([pa.field("id", pa.int32())]))

        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(
            source, options=CastOptions(target=target),
        )

        # Read back the raw schema to confirm the cast committed.
        import pyarrow.parquet as _pq
        schema = _pq.ParquetFile(pa.BufferReader(mem.to_bytes())).schema_arrow
        assert schema.field("id").type == pa.int32()

    def test_empty_table_fast_path(self) -> None:
        """An empty pa.Table on the fast path must still produce a
        valid parquet file with the table's schema."""
        from yggdrasil.io.parquet_file import ParquetFile

        empty = pa.table({"id": pa.array([], type=pa.int64())})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(empty)

        reread = ParquetFile(holder=mem, owns_holder=False).read_arrow_table()
        assert reread.num_rows == 0
        assert reread.schema.field("id").type == pa.int64()

    def test_cursor_opened_in_overwrite_mode_takes_fast_path(
        self, monkeypatch, tmp_path,
    ) -> None:
        """``path.open("wb")`` gives a cursor with parent.mode = OVERWRITE
        — ``holder_is_overwrite`` is True, so the override's
        ``has_existing`` check short-circuits to False even when the
        underlying file already contains bytes. The fast path runs."""
        from yggdrasil.io.parquet_file import ParquetFile

        path = LocalPath(str(tmp_path / "x.parquet"))
        # Pre-populate so the file is non-empty on disk.
        with path.open("wb") as cursor:
            cursor.write_arrow_table(pa.table({"id": [1, 2, 3]}))

        calls = self._counting_patch(monkeypatch)
        # Re-open with "wb" — cursor.holder_is_overwrite must skip
        # the merge path even though path.size > 0 on disk.
        with path.open("wb") as cursor:
            assert isinstance(cursor, ParquetFile)
            cursor.write_arrow_table(pa.table({"id": [99]}))
        assert calls["n"] == 0

        with path.open("rb") as cursor:
            assert cursor.read_arrow_table().column("id").to_pylist() == [99]


class TestPandasIndexRoundTrip(__import__(
    "yggdrasil.pandas.tests", fromlist=["PandasTestCase"],
).PandasTestCase):
    """ParquetFile round-trips every non-default pandas index shape.

    The base ``_write_pandas_frame`` preserves the index only when at
    least one level is named, which silently dropped non-default
    ``RangeIndex`` and unnamed non-range ``Index`` values. The Parquet
    override uses ``preserve_index=None`` so pyarrow's auto mode
    stamps the ``b"pandas"`` schema-metadata channel — ``to_pandas()``
    rebuilds the index from it on the read side.
    """

    def _roundtrip(self, df):
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(df)
        return ParquetFile(holder=mem, owns_holder=False).read_pandas_frame()

    def test_default_range_index(self) -> None:
        df = self.df({"a": [1, 2, 3]})
        result = self._roundtrip(df)
        self.assertFrameEqual(result, df, check_index=True)
        assert isinstance(result.index, self.pd.RangeIndex)
        assert result.index.start == 0

    def test_non_default_range_index(self) -> None:
        df = self.df({"a": [1, 2, 3]}, index=self.pd.RangeIndex(5, 8))
        result = self._roundtrip(df)
        self.assertFrameEqual(result, df, check_index=True)
        assert list(result.index) == [5, 6, 7]

    def test_named_range_index(self) -> None:
        df = self.df({"a": [10, 20, 30]})
        df.index.name = "i"
        result = self._roundtrip(df)
        self.assertFrameEqual(result, df, check_index=True)
        assert result.index.name == "i"

    def test_datetime_index(self) -> None:
        df = self.df(
            {"v": [1, 2, 3]},
            index=self.pd.date_range("2024-01-01", periods=3, name="ts"),
        )
        result = self._roundtrip(df)
        # ``freq`` round-trips as None — parquet's pandas metadata
        # carries the values, not the index's freq descriptor.
        self.assertFrameEqual(result, df, check_index=True, check_freq=False)
        assert isinstance(result.index, self.pd.DatetimeIndex)
        assert result.index.name == "ts"

    def test_multi_index(self) -> None:
        idx = self.pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=["k1", "k2"],
        )
        df = self.df({"v": [10, 20, 30, 40]}, index=idx)
        result = self._roundtrip(df)
        self.assertFrameEqual(result, df, check_index=True)
        assert isinstance(result.index, self.pd.MultiIndex)
        assert result.index.names == ["k1", "k2"]

    def test_unnamed_non_range_index(self) -> None:
        df = self.df({"a": [1, 2, 3]}, index=[10, 20, 30])
        result = self._roundtrip(df)
        self.assertFrameEqual(result, df, check_index=True)
        assert list(result.index) == [10, 20, 30]

    def test_local_path_roundtrip(self) -> None:
        df = self.df({"a": [1, 2, 3]}, index=self.pd.RangeIndex(100, 103))
        path = LocalPath(str(self.tmp_path / "indexed.parquet"))
        with path.open("wb") as cursor:
            cursor.write_pandas_frame(df)
        with path.open("rb") as cursor:
            result = cursor.read_pandas_frame()
        self.assertFrameEqual(result, df, check_index=True)
        assert list(result.index) == [100, 101, 102]

    def test_target_bound_write_drops_index(self) -> None:
        """A bound target schema strictly defines columns — index drops out."""
        from yggdrasil.data.schema import Schema
        from yggdrasil.io.parquet_file import ParquetOptions

        df = self.df({"a": [1, 2, 3]}, index=self.pd.RangeIndex(5, 8))
        target = Schema.from_arrow(pa.schema([("a", pa.int64())]))
        opts = ParquetOptions(target=target)
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(df, options=opts)
        result = ParquetFile(holder=mem, owns_holder=False).read_pandas_frame()
        assert isinstance(result.index, self.pd.RangeIndex)
        assert result.index.start == 0

    def test_reads_parquet_written_by_pandas(self) -> None:
        """Cross-tool: pandas.to_parquet → ParquetFile.read_pandas_frame restores index."""
        import io

        df = self.df(
            {"a": [1, 2, 3]},
            index=self.pd.Index([100, 200, 300], name="custom_idx"),
        )
        buf = io.BytesIO()
        df.to_parquet(buf)
        mem = Memory()
        mem.write(buf.getvalue())
        result = ParquetFile(holder=mem, owns_holder=False).read_pandas_frame()
        self.assertFrameEqual(result, df, check_index=True)
        assert result.index.name == "custom_idx"

    def test_reads_multi_index_parquet_written_by_pandas(self) -> None:
        """Cross-tool: pandas MultiIndex → ParquetFile restores all levels."""
        import io

        idx = self.pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["k1", "k2"],
        )
        df = self.df({"v": [10, 20, 30]}, index=idx)
        buf = io.BytesIO()
        df.to_parquet(buf)
        mem = Memory()
        mem.write(buf.getvalue())
        result = ParquetFile(holder=mem, owns_holder=False).read_pandas_frame()
        self.assertFrameEqual(result, df, check_index=True)
        assert result.index.names == ["k1", "k2"]

    def test_schema_carries_index_key_tags(self) -> None:
        """Each materialised index level lands as a tagged Field in the schema."""
        import pyarrow.parquet as pq
        from io import BytesIO
        from yggdrasil.data.data_field import Field

        df = self.df(
            {"v": [10, 20, 30]},
            index=self.pd.MultiIndex.from_tuples(
                [("a", 1), ("a", 2), ("b", 1)], names=["k1", "k2"],
            ),
        )
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(df)

        schema = pq.ParquetFile(BytesIO(mem.to_bytes())).schema_arrow

        assert b"pandas" not in (schema.metadata or {})

        assert schema.field("k1").metadata[Field._TAG_KEY_INDEX_KEY] == b"true"
        assert schema.field("k1").metadata[Field._TAG_KEY_INDEX_KEY_LEVEL] == b"0"
        assert schema.field("k2").metadata[Field._TAG_KEY_INDEX_KEY] == b"true"
        assert schema.field("k2").metadata[Field._TAG_KEY_INDEX_KEY_LEVEL] == b"1"
        assert schema.field("v").metadata is None

        ygg_schema = Field.from_arrow_schema(schema)
        children_by_name = {f.name: f for f in ygg_schema.fields}
        assert children_by_name["k1"].index_key is True
        assert children_by_name["k1"].index_key_level == 0
        assert children_by_name["k2"].index_key is True
        assert children_by_name["k2"].index_key_level == 1
        assert not children_by_name["v"].index_key

    def test_default_range_index_skips_index_column(self) -> None:
        """Default RangeIndex(0, N) doesn't materialise — no synthetic column."""
        import pyarrow.parquet as pq
        from io import BytesIO
        from yggdrasil.data.data_field import Field

        df = self.df({"a": [1, 2, 3]})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(df)

        schema = pq.ParquetFile(BytesIO(mem.to_bytes())).schema_arrow
        assert schema.names == ["a"]
        for name in schema.names:
            meta = schema.field(name).metadata or {}
            assert Field._TAG_KEY_INDEX_KEY not in meta


class TestPandasAppendUpsert(__import__(
    "yggdrasil.pandas.tests", fromlist=["PandasTestCase"],
).PandasTestCase):
    """APPEND / UPSERT round-trip pandas frames + merge with the on-disk schema.

    The Parquet writer's merge modes read the existing footer's
    schema, bind it as the target, and route incoming batches through
    the cast — the index_key Field tags survive that hop, so
    a chain of pandas writes lands with the index reconstructed on
    read.
    """

    def _new_mem(self) -> Memory:
        return Memory()

    def _read(self, mem: Memory):
        return ParquetFile(holder=mem, owns_holder=False).read_pandas_frame()

    def test_append_pandas_frames(self) -> None:
        """APPEND concatenates rows; the named index round-trips through both writes."""
        from yggdrasil.enums import Mode
        from yggdrasil.io.parquet_file import ParquetOptions

        df1 = self.df(
            {"k": ["a", "b", "c"], "v": [1, 2, 3]},
            index=self.pd.Index([10, 20, 30], name="i"),
        )
        df2 = self.df(
            {"k": ["d", "e"], "v": [4, 5]},
            index=self.pd.Index([40, 50], name="i"),
        )

        mem = self._new_mem()
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(df1)
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(
            df2, options=ParquetOptions(mode=Mode.APPEND),
        )

        result = self._read(mem)
        expected = self.pd.concat([df1, df2])
        self.assertFrameEqual(result, expected, check_index=True)
        assert result.index.name == "i"
        assert list(result.index) == [10, 20, 30, 40, 50]

    def test_append_preserves_index_tag_in_schema(self) -> None:
        """After an APPEND, the on-disk schema still carries the index tag."""
        import pyarrow.parquet as pq
        from io import BytesIO
        from yggdrasil.enums import Mode
        from yggdrasil.data.data_field import Field
        from yggdrasil.io.parquet_file import ParquetOptions

        df1 = self.df({"v": [1, 2]}, index=self.pd.Index([10, 20], name="i"))
        df2 = self.df({"v": [3, 4]}, index=self.pd.Index([30, 40], name="i"))

        mem = self._new_mem()
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(df1)
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(
            df2, options=ParquetOptions(mode=Mode.APPEND),
        )

        schema = pq.ParquetFile(BytesIO(mem.to_bytes())).schema_arrow
        # b'pandas' is stripped — the yggdrasil tag is still the source of truth.
        assert b"pandas" not in (schema.metadata or {})
        assert schema.field("i").metadata[Field._TAG_KEY_INDEX_KEY_LEVEL] == b"0"

    def test_upsert_pandas_frames_by_key(self) -> None:
        """UPSERT with match_by replaces existing rows; incoming wins."""
        from yggdrasil.enums import Mode
        from yggdrasil.io.parquet_file import ParquetOptions

        df1 = self.df(
            {"k": ["a", "b", "c"], "v": [1, 2, 3]},
        )
        # Updates 'b' (v=2 → v=99) and inserts 'd'.
        df2 = self.df({"k": ["b", "d"], "v": [99, 100]})

        mem = self._new_mem()
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(df1)
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(
            df2, options=ParquetOptions(mode=Mode.UPSERT, match_by=["k"]),
        )

        result = self._read(mem)
        by_k = dict(zip(result["k"], result["v"]))
        assert by_k == {"a": 1, "b": 99, "c": 3, "d": 100}

    def test_append_round_trips_multi_index(self) -> None:
        """MultiIndex levels survive an APPEND — every level stays tagged."""
        from yggdrasil.enums import Mode
        from yggdrasil.io.parquet_file import ParquetOptions

        idx1 = self.pd.MultiIndex.from_tuples(
            [("a", 1), ("a", 2)], names=["k1", "k2"],
        )
        idx2 = self.pd.MultiIndex.from_tuples(
            [("b", 1), ("b", 2)], names=["k1", "k2"],
        )
        df1 = self.df({"v": [10, 20]}, index=idx1)
        df2 = self.df({"v": [30, 40]}, index=idx2)

        mem = self._new_mem()
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(df1)
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(
            df2, options=ParquetOptions(mode=Mode.APPEND),
        )

        result = self._read(mem)
        assert isinstance(result.index, self.pd.MultiIndex)
        assert result.index.names == ["k1", "k2"]
        assert list(result.index) == [
            ("a", 1), ("a", 2), ("b", 1), ("b", 2),
        ]
        assert list(result["v"]) == [10, 20, 30, 40]

    def test_append_mode_auto_into_empty_buffer(self) -> None:
        """APPEND against an empty buffer collapses to OVERWRITE and tags as usual."""
        import pyarrow.parquet as pq
        from io import BytesIO
        from yggdrasil.enums import Mode
        from yggdrasil.data.data_field import Field
        from yggdrasil.io.parquet_file import ParquetOptions

        df = self.df({"v": [1, 2]}, index=self.pd.Index([10, 20], name="i"))

        mem = self._new_mem()
        # First write under APPEND mode against an empty buffer — the
        # parquet writer's _MERGE_MODES branch sees size == 0 and
        # falls back to OVERWRITE, so the tag must still land.
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(
            df, options=ParquetOptions(mode=Mode.APPEND),
        )

        schema = pq.ParquetFile(BytesIO(mem.to_bytes())).schema_arrow
        assert schema.field("i").metadata[Field._TAG_KEY_INDEX_KEY_LEVEL] == b"0"

        result = self._read(mem)
        self.assertFrameEqual(result, df, check_index=True)

    def test_upsert_round_trips_index_with_match_by(self) -> None:
        """UPSERT preserves the pandas index after merging with existing rows."""
        from yggdrasil.enums import Mode
        from yggdrasil.io.parquet_file import ParquetOptions

        # Existing rows keyed by 'k'; the named index 'i' rides along
        # as a regular index column. After upsert by 'k', the index
        # must still surface on the result frame.
        df1 = self.df(
            {"k": ["a", "b", "c"], "v": [1, 2, 3]},
            index=self.pd.Index([100, 200, 300], name="i"),
        )
        df2 = self.df(
            {"k": ["b"], "v": [99]},
            index=self.pd.Index([999], name="i"),
        )

        mem = self._new_mem()
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(df1)
        ParquetFile(holder=mem, owns_holder=False).write_pandas_frame(
            df2, options=ParquetOptions(mode=Mode.UPSERT, match_by=["k"]),
        )

        result = self._read(mem)
        assert result.index.name == "i"
        # Existing rows 'a' (idx 100) and 'c' (idx 300) win on no match;
        # 'b' is replaced by the incoming row's value + index (999).
        by_k = {k: (v, i) for k, v, i in zip(
            result["k"], result["v"], result.index,
        )}
        assert by_k == {"a": (1, 100), "b": (99, 999), "c": (3, 300)}


class TestReadArrowTableBypassesBatchHook:
    """``ParquetFile._read_arrow_table`` routes through a single
    :meth:`pq.ParquetFile.read` C++ call instead of streaming
    ``_read_arrow_batches`` and re-stitching via
    ``pa.Table.from_batches``. The bypass holds for non-empty
    files; the empty-file edge falls back to the base class for
    schema synthesis."""

    @staticmethod
    def _counting_patch(monkeypatch):
        from yggdrasil.io.parquet_file import ParquetFile
        calls = {"n": 0}
        original = ParquetFile._read_arrow_batches

        def counting(self, options):
            calls["n"] += 1
            return original(self, options)

        monkeypatch.setattr(ParquetFile, "_read_arrow_batches", counting)
        return calls

    def test_read_arrow_table_skips_batch_hook(self, monkeypatch) -> None:
        from yggdrasil.io.parquet_file import ParquetFile

        table = pa.table({"id": list(range(1000))})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(table)

        calls = self._counting_patch(monkeypatch)
        out = ParquetFile(holder=mem, owns_holder=False).read_arrow_table()
        assert calls["n"] == 0
        assert out.equals(table)

    def test_row_limit_applied_on_fast_path(self) -> None:
        from yggdrasil.io.parquet_file import (
            ParquetFile, ParquetOptions,
        )

        table = pa.table({"id": list(range(1000))})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(table)

        out = ParquetFile(holder=mem, owns_holder=False).read_arrow_table(
            options=ParquetOptions(row_limit=42),
        )
        assert out.num_rows == 42
        assert out.column("id").to_pylist() == list(range(42))

    def test_target_projection_pushed_down(self, monkeypatch) -> None:
        """``options.target`` with a subset of columns should drive
        the parquet column projection — fewer bytes off disk, no
        column drop on the Python side."""
        from yggdrasil.data.options import CastOptions
        from yggdrasil.data.data_field import Field
        from yggdrasil.io.parquet_file import ParquetFile

        seed = pa.table({"id": [1, 2, 3], "skip": ["a", "b", "c"], "keep": [10, 20, 30]})
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        target = Field.from_(pa.schema([
            pa.field("id", pa.int64()),
            pa.field("keep", pa.int64()),
        ]))
        out = ParquetFile(holder=mem, owns_holder=False).read_arrow_table(
            options=CastOptions(target=target),
        )
        assert out.column_names == ["id", "keep"]
        assert out.column("keep").to_pylist() == [10, 20, 30]

    def test_empty_file_falls_back_to_base(self, monkeypatch) -> None:
        from yggdrasil.io.parquet_file import ParquetFile

        mem = Memory()  # size == 0
        # The base path runs because the fast path's
        # ``size == 0`` guard bails out — base in turn calls
        # _read_arrow_batches which yields nothing, then synthesises
        # the empty table.
        calls = self._counting_patch(monkeypatch)
        out = ParquetFile(holder=mem, owns_holder=False).read_arrow_table()
        assert calls["n"] >= 1
        assert out.num_rows == 0


class TestFastPathEquivalence:
    """The fast path (``writer.write_table``) must produce *the same
    table* on read-back as the slow path (``writer.write_batch`` loop)
    for the same input. Byte equality is too strict — pyarrow can
    reorder row groups internally or stamp different write metadata
    — but the Arrow Table that comes back has to match."""

    @pytest.fixture
    def fixtures(self):
        # Numeric only (no string compression variance).
        numeric = pa.table({
            "id": pa.array(list(range(10_000)), type=pa.int64()),
            "x": pa.array([float(i) / 7.0 for i in range(10_000)], type=pa.float64()),
        })
        # Mixed types — strings + bools.
        mixed = pa.table({
            "id": pa.array(list(range(5_000)), type=pa.int64()),
            "v": pa.array([f"row-{i}" for i in range(5_000)], type=pa.string()),
            "flag": pa.array([i % 2 == 0 for i in range(5_000)], type=pa.bool_()),
        })
        return {"numeric": numeric, "mixed": mixed}

    def _write_fast(self, table) -> bytes:
        mem = Memory()
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(table)
        return bytes(mem.to_bytes())

    def _write_slow(self, table) -> bytes:
        # Force the batch path by writing through _write_arrow_batches.
        from yggdrasil.io.parquet_file import ParquetOptions
        mem = Memory()
        leaf = ParquetFile(holder=mem, owns_holder=False)
        leaf._write_arrow_batches(iter(table.to_batches()), ParquetOptions())
        return bytes(mem.to_bytes())

    def test_numeric_round_trip_matches(self, fixtures) -> None:
        table = fixtures["numeric"]
        fast = self._write_fast(table)
        slow = self._write_slow(table)
        # Read both back via raw pyarrow — independent of yggdrasil.
        import pyarrow.parquet as _pq
        fast_table = _pq.read_table(pa.BufferReader(fast))
        slow_table = _pq.read_table(pa.BufferReader(slow))
        assert fast_table.equals(slow_table)
        assert fast_table.equals(table)

    def test_mixed_round_trip_matches(self, fixtures) -> None:
        table = fixtures["mixed"]
        fast = self._write_fast(table)
        slow = self._write_slow(table)
        import pyarrow.parquet as _pq
        fast_table = _pq.read_table(pa.BufferReader(fast))
        slow_table = _pq.read_table(pa.BufferReader(slow))
        assert fast_table.equals(slow_table)
        assert fast_table.equals(table)


class TestThriftFooterLimits:
    """Parquet footers with large metadata exceed pyarrow's default thrift
    deserialization limits ("Couldn't deserialize thrift: Exceeded size
    limit"). Reads must pass generous limits so such files open."""

    def test_read_passes_thrift_limits_to_parquet_open(self) -> None:
        from unittest.mock import patch
        import pyarrow as pa
        from yggdrasil.io import parquet_file as pf_mod

        table = pa.table({"a": [1, 2, 3]})
        mem = Memory()
        ParquetFile(parent=mem).write_arrow_table(table)

        real = pf_mod.pq.ParquetFile
        seen = {}

        def _spy(source, *args, **kwargs):
            seen.update(kwargs)
            return real(source, *args, **kwargs)

        with patch.object(pf_mod.pq, "ParquetFile", side_effect=_spy):
            out = ParquetFile(parent=mem).read_arrow_table()

        assert out.num_rows == 3
        assert seen.get("thrift_string_size_limit") == pf_mod._THRIFT_LIMITS["thrift_string_size_limit"]
        assert seen.get("thrift_container_size_limit") == pf_mod._THRIFT_LIMITS["thrift_container_size_limit"]
        # generous enough to clear the 100 MB string default
        assert pf_mod._THRIFT_LIMITS["thrift_string_size_limit"] > 100_000_000
