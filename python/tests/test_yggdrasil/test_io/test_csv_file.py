"""Tests for :class:`yggdrasil.io.csv_file.CSVFile`."""

from __future__ import annotations

import pyarrow as pa

from yggdrasil.io.holder import Holder
from yggdrasil.path.memory import Memory
from yggdrasil.path.local_path import LocalPath
from yggdrasil.io.csv_file import CSVFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        assert Holder.class_for_media_type("text/csv") is CSVFile
        assert Holder.class_for_media_type("csv") is CSVFile

    def test_path_dispatches(self, tmp_path) -> None:
        from yggdrasil.io.base import IO

        b = IO(path=str(tmp_path / "x.csv"))
        assert isinstance(b, CSVFile)


class TestMemoryRoundTrip:

    def test_write_then_read(self) -> None:
        table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        mem = Memory()
        CSVFile(holder=mem, owns_holder=False).write_arrow_table(table)
        assert mem.size > 0
        got = CSVFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.column("id").to_pylist() == [1, 2, 3]
        assert got.column("name").to_pylist() == ["a", "b", "c"]


class TestLocalPathRoundTrip:

    def test_write_and_read(self, tmp_path) -> None:
        table = pa.table({"x": [10, 20, 30]})
        path = LocalPath(str(tmp_path / "out.csv"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(table)
        with path.open("rb") as cursor:
            assert cursor.read_arrow_table().column("x").to_pylist() == [10, 20, 30]

    def test_text_layout(self, tmp_path) -> None:
        # CSV is text; ensure the on-disk shape parses back correctly.
        path = LocalPath(str(tmp_path / "out.csv"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(pa.table({"a": [1, 2], "b": ["x", "y"]}))
        text = (tmp_path / "out.csv").read_text()
        lines = text.strip().splitlines()
        # Header is the column names, in some quoting style.
        assert {c.strip('"') for c in lines[0].split(",")} == {"a", "b"}
        # 2 data rows.
        assert len(lines) == 3


class TestEmptyOverwriteWritesValidFile:
    """An empty input under OVERWRITE through the generic path route
    (the route every non-leaf path — VolumePath, S3Path, LocalPath —
    takes) must persist a valid, schema-bearing CSV (header only),
    not a 0-byte stub."""

    def test_empty_table_writes_header(self, tmp_path) -> None:
        from yggdrasil.enums import Mode

        empty = pa.table(
            {"id": pa.array([], type=pa.int64()),
             "amount": pa.array([], type=pa.float64())}
        )
        path = LocalPath(str(tmp_path / "empty.csv"))
        path.write_table(empty, mode=Mode.OVERWRITE)

        assert path.size > 0
        text = (tmp_path / "empty.csv").read_text()
        assert {c.strip('"') for c in text.strip().splitlines()[0].split(",")} == {
            "id", "amount",
        }

    def test_empty_batches_overwrite_uses_bound_schema(self) -> None:
        from yggdrasil.enums import Mode
        from yggdrasil.data.options import CastOptions

        schema = pa.schema([("id", pa.int64()), ("amount", pa.float64())])
        mem = Memory()
        leaf = CSVFile(holder=mem, owns_holder=False)
        leaf._write_arrow_batches(
            iter([]), leaf.check_options(CastOptions(mode=Mode.OVERWRITE, target=schema)),
        )
        assert mem.size > 0
        header = mem.to_bytes().decode().strip().splitlines()[0]
        assert {c.strip('"') for c in header.split(",")} == {"id", "amount"}


class TestAutoDefaultsToOverwrite:
    """``Mode.AUTO`` (the default) replaces the file for a bare write —
    matching the JSON / Excel / Zip leaves. ``match_by`` still upserts."""

    def test_auto_replaces_existing(self) -> None:
        from yggdrasil.enums import Mode
        from yggdrasil.data.options import CastOptions

        mem = Memory()
        CSVFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [1, 2, 3]}), CastOptions(mode=Mode.OVERWRITE),
        )
        CSVFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [9]}),  # default mode = AUTO
        )
        out = CSVFile(holder=mem, owns_holder=False).read_arrow_table()
        assert out.column("id").to_pylist() == [9]
