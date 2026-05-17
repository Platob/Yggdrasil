"""Tests for :class:`yggdrasil.io.primitive.arrow_ipc_file.ArrowIPCFile`."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        from yggdrasil.data.enums import MimeTypes

        assert Holder.class_for_media_type(MimeTypes.ARROW_IPC) is ArrowIPCFile

    def test_path_dispatches_arrow_ext(self, tmp_path) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        b = BytesIO(path=str(tmp_path / "x.arrow"))
        assert isinstance(b, ArrowIPCFile)


class TestMemoryRoundTrip:

    def test_write_then_read_table(self) -> None:
        table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(table)
        got = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.equals(table)

    def test_write_then_read_batches(self) -> None:
        batch_a = pa.record_batch({"x": [1, 2]})
        batch_b = pa.record_batch({"x": [3, 4]})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_batches(
            iter([batch_a, batch_b]),
        )
        got = list(ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_batches())
        combined = pa.Table.from_batches(got)
        assert combined.column("x").to_pylist() == [1, 2, 3, 4]


class TestLocalPathRoundTrip:

    def test_write_and_read(self, tmp_path) -> None:
        table = pa.table({"x": [10, 20, 30]})
        path = LocalPath(str(tmp_path / "out.arrow"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(table)
        with path.open("rb") as cursor:
            assert cursor.read_arrow_table().equals(table)
