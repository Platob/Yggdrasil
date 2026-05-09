"""Behavior tests for :class:`yggdrasil.io.primitive.arrow_ipc_io.ArrowIPCIO`.

`ArrowIPCIO` is a :class:`BytesIO` leaf that auto-registers under
:data:`MimeTypes.ARROW_IPC`. Tests pin:

* round-trip of Arrow tables / batches via the :class:`Tabular`
  convenience surface (``read_arrow_table`` / ``write_arrow_table``,
  pandas / polars / pylist views);
* persistence to / from :class:`LocalPath` and back, both via
  ``with path.open() as bio: bio.write_arrow_table(...)`` and via
  the ``ArrowIPCIO(holder=...)`` shape;
* mode dispatch — OVERWRITE / APPEND / IGNORE / ERROR_IF_EXISTS /
  AUTO behave the way the docstring claims;
* :meth:`Tabular.for_holder` resolves a holder with the IPC media
  type to :class:`ArrowIPCIO`.
"""
from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.arrow_ipc_io import ArrowIPCIO, ArrowIPCOptions
from yggdrasil.io.tabular import Tabular


@pytest.fixture
def table() -> pa.Table:
    return pa.table(
        {"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "v": [0.5, 1.5, 2.5, 3.5]}
    )


class TestRegistration:

    def test_mime_type_is_arrow_ipc(self) -> None:
        assert ArrowIPCIO.mime_type is MimeTypes.ARROW_IPC

    def test_registry_resolves_to_arrow_ipc_io(self) -> None:
        cls = Tabular.class_for_media_type(MimeTypes.ARROW_IPC)
        assert cls is ArrowIPCIO

    def test_options_class(self) -> None:
        assert ArrowIPCIO.options_class() is ArrowIPCOptions


class TestRoundTripInMemory:

    def test_arrow_table_round_trip(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        loaded = io.read_arrow_table()
        assert loaded.equals(table)

    def test_arrow_batches_round_trip(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_batches(table.to_batches())
        out = list(io.read_arrow_batches())
        assert sum(b.num_rows for b in out) == table.num_rows

    def test_collect_schema(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        schema = io.collect_schema()
        assert set(schema.field_names()) == {"id", "name", "v"}

    def test_read_pandas_frame(self, table) -> None:
        pd = pytest.importorskip("pandas")
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        df = io.read_pandas_frame()
        pd.testing.assert_frame_equal(df, table.to_pandas())

    def test_read_polars_frame(self, table) -> None:
        pl = pytest.importorskip("polars")
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        df = io.read_polars_frame()
        assert df.equals(pl.from_arrow(table))

    def test_read_pylist(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        rows = io.read_pylist()
        assert rows == table.to_pylist()


class TestEmptyBuffer:

    def test_read_yields_no_batches(self) -> None:
        io = ArrowIPCIO()
        assert list(io.read_arrow_batches()) == []

    def test_collect_schema_returns_empty(self) -> None:
        from yggdrasil.data.schema import Schema
        io = ArrowIPCIO()
        assert io.collect_schema() == Schema.empty()

    def test_read_table_empty(self) -> None:
        io = ArrowIPCIO()
        assert io.read_arrow_table().num_rows == 0


class TestHolderBacked:

    def test_round_trip_via_local_path(self, tmp_path, table) -> None:
        target = LocalPath(str(tmp_path / "data.arrow"))
        io = ArrowIPCIO(holder=target, owns_holder=False)
        io.write_arrow_table(table)
        assert target.size > 0

        # Read back through a fresh ArrowIPCIO over the same holder.
        reader = ArrowIPCIO(holder=target, owns_holder=False)
        loaded = reader.read_arrow_table()
        assert loaded.equals(table)

    def test_round_trip_via_memory_holder(self, table) -> None:
        mem = Memory()
        io = ArrowIPCIO(holder=mem, owns_holder=False)
        io.write_arrow_table(table)
        assert mem.size > 0

        # Fresh reader over the same memory.
        reader = ArrowIPCIO(holder=mem, owns_holder=False)
        assert reader.read_arrow_table().equals(table)


class TestModes:

    def test_overwrite_truncates(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        first_size = io.size

        smaller = pa.table({"id": [1], "name": ["x"], "v": [0.5]})
        io.write_arrow_table(smaller, options=ArrowIPCOptions(mode=Mode.OVERWRITE))
        assert io.size < first_size
        assert io.read_arrow_table().equals(smaller)

    def test_append_concatenates(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        more = pa.table({"id": [5, 6], "name": ["e", "f"], "v": [4.5, 5.5]})
        io.write_arrow_batches(more.to_batches(), options=ArrowIPCOptions(mode=Mode.APPEND))
        loaded = io.read_arrow_table()
        assert loaded.num_rows == table.num_rows + more.num_rows
        # Existing batches first, then new.
        assert loaded.column("id").to_pylist() == [1, 2, 3, 4, 5, 6]

    def test_ignore_skips_when_non_empty(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        before = io.size
        io.write_arrow_batches(
            pa.table({"id": [99], "name": ["z"], "v": [9.5]}).to_batches(),
            options=ArrowIPCOptions(mode=Mode.IGNORE),
        )
        assert io.size == before

    def test_ignore_writes_when_empty(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_batches(table.to_batches(), options=ArrowIPCOptions(mode=Mode.IGNORE))
        assert io.read_arrow_table().equals(table)

    def test_error_if_exists_raises_when_non_empty(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        with pytest.raises(FileExistsError):
            io.write_arrow_batches(
                table.to_batches(), options=ArrowIPCOptions(mode=Mode.ERROR_IF_EXISTS),
            )

    def test_error_if_exists_passes_when_empty(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_batches(
            table.to_batches(), options=ArrowIPCOptions(mode=Mode.ERROR_IF_EXISTS),
        )
        assert io.read_arrow_table().equals(table)


class TestKeyedMerge:
    """``options.match_by_names`` drives key-aware APPEND / UPSERT."""

    def test_append_with_keys_drops_incoming_duplicates(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        # Rows id=2, 3 collide with existing → dropped; id=5 is new.
        more = pa.table(
            {"id": [2, 3, 5], "name": ["X", "Y", "e"], "v": [-1.0, -2.0, 4.5]}
        )
        io.write_arrow_batches(
            more.to_batches(),
            options=ArrowIPCOptions(mode=Mode.APPEND, match_by_names=["id"]),
        )
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3, 4, 5]
        # Existing rows win for the colliding keys.
        assert loaded.column("name").to_pylist() == ["a", "b", "c", "d", "e"]

    def test_upsert_with_keys_replaces_existing(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        more = pa.table(
            {"id": [2, 3, 5], "name": ["X", "Y", "e"], "v": [-1.0, -2.0, 4.5]}
        )
        io.write_arrow_batches(
            more.to_batches(),
            options=ArrowIPCOptions(mode=Mode.UPSERT, match_by_names=["id"]),
        )
        loaded = io.read_arrow_table()
        # Surviving existing (id=1, 4) first, then all incoming.
        assert loaded.column("id").to_pylist() == [1, 4, 2, 3, 5]
        assert loaded.column("name").to_pylist() == ["a", "d", "X", "Y", "e"]
        assert loaded.column("v").to_pylist() == [0.5, 3.5, -1.0, -2.0, 4.5]

    def test_merge_behaves_like_upsert(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        more = pa.table({"id": [3], "name": ["Z"], "v": [9.0]})
        io.write_arrow_batches(
            more.to_batches(),
            options=ArrowIPCOptions(mode=Mode.MERGE, match_by_names=["id"]),
        )
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 4, 3]
        assert loaded.column("name").to_pylist() == ["a", "b", "d", "Z"]

    def test_upsert_without_keys_falls_back_to_append(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        more = pa.table({"id": [2], "name": ["X"], "v": [-1.0]})
        io.write_arrow_batches(
            more.to_batches(), options=ArrowIPCOptions(mode=Mode.UPSERT),
        )
        # No keys → degrades to plain concatenation; the duplicate id=2
        # row is appended as-is.
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3, 4, 2]
        assert loaded.column("name").to_pylist() == ["a", "b", "c", "d", "X"]

    def test_upsert_with_keys_into_empty_writes_payload(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_batches(
            table.to_batches(),
            options=ArrowIPCOptions(mode=Mode.UPSERT, match_by_names=["id"]),
        )
        assert io.read_arrow_table().equals(table)

    def test_append_with_composite_keys(self) -> None:
        base = pa.table(
            {"a": [1, 1, 2], "b": ["x", "y", "x"], "v": [10, 20, 30]}
        )
        io = ArrowIPCIO()
        io.write_arrow_table(base)
        more = pa.table(
            # (1, "x") collides → dropped under APPEND; (2, "y") is new.
            {"a": [1, 2], "b": ["x", "y"], "v": [-1, 40]}
        )
        io.write_arrow_batches(
            more.to_batches(),
            options=ArrowIPCOptions(mode=Mode.APPEND, match_by_names=["a", "b"]),
        )
        loaded = io.read_arrow_table()
        assert loaded.column("a").to_pylist() == [1, 1, 2, 2]
        assert loaded.column("b").to_pylist() == ["x", "y", "x", "y"]
        assert loaded.column("v").to_pylist() == [10, 20, 30, 40]


class TestExternalWriterPattern:
    """`with path.open() as b: pyarrow.ipc.RecordBatchFileWriter(b, ...)` flow."""

    def test_pyarrow_writer_into_path_open(self, tmp_path, table) -> None:
        import pyarrow.ipc as ipc

        target = LocalPath(str(tmp_path / "data.arrow"))
        with target.open("wb") as bio:
            with ipc.new_file(bio, table.schema) as writer:
                writer.write_table(table)

        # Read back through ArrowIPCIO over the same path.
        reader = ArrowIPCIO(holder=target, owns_holder=False)
        assert reader.read_arrow_table().equals(table)


class TestCompressionOption:

    def test_lz4_round_trip(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(
            table, options=ArrowIPCOptions(compression="lz4"),
        )
        assert io.read_arrow_table().equals(table)

    def test_zstd_round_trip(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(
            table, options=ArrowIPCOptions(compression="zstd"),
        )
        assert io.read_arrow_table().equals(table)


class TestTabularForHolder:
    """`Tabular.for_holder` dispatches a holder to the right leaf."""

    def test_holder_with_explicit_media_type(self, tmp_path, table) -> None:
        # Seed the holder's IOStats with the IPC media type so dispatch
        # has something to look at without needing magic bytes.
        from yggdrasil.data.enums.media_type import MediaType
        target = LocalPath(str(tmp_path / "x.arrow"))
        target.media_type = MediaType.from_(MimeTypes.ARROW_IPC)

        leaf = Tabular.for_holder(target)
        assert isinstance(leaf, ArrowIPCIO)


class TestOptionsCopyability:
    """``ArrowIPCOptions`` is a frozen dataclass; ``dataclasses.replace`` works."""

    def test_replace_preserves_other_fields(self) -> None:
        base = ArrowIPCOptions(compression="lz4", row_size=1024)
        flipped = dataclasses.replace(base, compression="zstd")
        assert flipped.compression == "zstd"
        assert flipped.row_size == 1024
