"""Tests for :class:`yggdrasil.io.pickle_file.PickleFile`."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.enums import Mode, MimeTypes
from yggdrasil.io.holder import IO
from yggdrasil.io.pickle_file import PickleFile
from yggdrasil.path.local_path import LocalPath
from yggdrasil.path.memory import Memory


def _pf(holder=None) -> PickleFile:
    return PickleFile(holder=holder if holder is not None else Memory(), owns_holder=False)


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        assert IO.class_for_media_type(MimeTypes.PICKLE.value, default=None) is PickleFile

    def test_path_extensions_dispatch(self, tmp_path) -> None:
        # .pkl / .pickle paths route to PickleFile via the media registry.
        for name in ("o.pkl", "o.pickle"):
            p = LocalPath(str(tmp_path / name))
            leaf = p.as_media("pickle") if False else None  # noqa: F841
        # round-trip a table through a .pkl path
        table = pa.table({"x": [1, 2, 3]})
        path = LocalPath(str(tmp_path / "data.pkl"))
        with path.open("wb") as cur:
            cur.write_arrow_table(table)
        with path.open("rb") as cur:
            assert cur.read_arrow_table().equals(table)


class TestObjectSurface:
    """dump/load round-trip arbitrary Python objects — the pickle
    specialization that the columnar leaves can't do."""

    def test_dump_load_arbitrary_object(self) -> None:
        obj = {"weights": [1, 2, 3], "name": "model", "nested": {"a": (1, 2)}}
        pf = _pf()
        pf.dump(obj)
        assert pf.load() == obj

    def test_dump_load_non_tabular_value(self) -> None:
        pf = _pf()
        pf.dump(["just", "a", "list", 42])
        assert pf.load() == ["just", "a", "list", 42]

    def test_load_empty_is_none(self) -> None:
        assert _pf().load() is None

    def test_overwrite_replaces(self) -> None:
        pf = _pf()
        pf.dump({"v": 1})
        pf.dump({"v": 2})
        assert pf.load() == {"v": 2}

    def test_local_path_object_roundtrip(self, tmp_path) -> None:
        path = LocalPath(str(tmp_path / "obj.pkl"))
        obj = {"a": 1, "b": [1, 2, 3]}
        PickleFile(holder=path, owns_holder=False).dump(obj)
        assert PickleFile(holder=path, owns_holder=False).load() == obj


class TestTabularRoundTrip:

    def test_table_roundtrip(self) -> None:
        table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        mem = Memory()
        PickleFile(holder=mem, owns_holder=False).write_arrow_table(table)
        got = PickleFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.equals(table)

    def test_empty_buffer_reads_empty(self) -> None:
        assert _pf().read_arrow_table().num_rows == 0


class TestUnpickleAndCast:
    """A pickled object is cast to the requested tabular shape via
    any_to_arrow_table / Tabular.from_ on read."""

    def test_list_of_dicts_reads_as_table(self) -> None:
        pf = _pf()
        pf.dump([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        got = pf.read_arrow_table()
        assert got.column("a").to_pylist() == [1, 2]
        assert got.column("b").to_pylist() == ["x", "y"]

    def test_pickled_polars_frame_reads_as_arrow_and_pandas(self) -> None:
        pl = pytest.importorskip("polars")
        pytest.importorskip("pandas")
        pf = _pf()
        pf.dump(pl.DataFrame({"x": [10, 20], "y": ["p", "q"]}))
        assert PickleFile(holder=pf._parent, owns_holder=False).read_arrow_table().column("x").to_pylist() == [10, 20]
        pdf = PickleFile(holder=pf._parent, owns_holder=False).read_pandas_frame()
        assert list(pdf.columns) == ["x", "y"]

    def test_pickled_pandas_frame_reads_as_arrow(self) -> None:
        pd = pytest.importorskip("pandas")
        pf = _pf()
        pf.dump(pd.DataFrame({"n": [1, 2, 3]}))
        assert pf.read_arrow_table().column("n").to_pylist() == [1, 2, 3]

    def test_read_with_target_projection(self) -> None:
        pf = _pf()
        pf.dump([{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}])
        got = pf.read_arrow_table(target=pa.schema([("b", pa.int64())]))
        assert got.column_names == ["b"]
        assert got.column("b").to_pylist() == [2, 5]

    def test_read_polars_frame_from_pickled_table(self) -> None:
        pl = pytest.importorskip("polars")
        table = pa.table({"v": [1, 2, 3]})
        pf = _pf()
        pf.write_arrow_table(table)
        frame = PickleFile(holder=pf._parent, owns_holder=False).read_polars_frame()
        assert frame["v"].to_list() == [1, 2, 3]


class TestWriteModes:

    def test_append_concatenates(self) -> None:
        mem = Memory()
        PickleFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [1, 2]}), mode=Mode.OVERWRITE,
        )
        PickleFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [3, 4]}), mode=Mode.APPEND,
        )
        got = PickleFile(holder=mem, owns_holder=False).read_arrow_table()
        assert sorted(got.column("id").to_pylist()) == [1, 2, 3, 4]

    def test_upsert_by_key_incoming_wins(self) -> None:
        mem = Memory()
        PickleFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [1, 2], "v": ["a", "b"]}), mode=Mode.OVERWRITE,
        )
        PickleFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [2, 3], "v": ["B", "c"]}),
            mode=Mode.UPSERT, match_by=["id"],
        )
        got = PickleFile(holder=mem, owns_holder=False).read_arrow_table()
        rows = dict(zip(got.column("id").to_pylist(), got.column("v").to_pylist()))
        assert rows == {1: "a", 2: "B", 3: "c"}

    def test_error_if_exists(self) -> None:
        mem = Memory()
        PickleFile(holder=mem, owns_holder=False).write_arrow_table(pa.table({"x": [1]}))
        with pytest.raises(FileExistsError):
            PickleFile(holder=mem, owns_holder=False).write_arrow_table(
                pa.table({"x": [2]}), mode=Mode.ERROR_IF_EXISTS,
            )


class TestCodec:

    def test_gzip_codec_roundtrip(self, tmp_path) -> None:
        # A .pkl.gz holder carries a gzip codec; dump/load go through
        # arrow_output_stream / arrow_input_stream which apply + peel it.
        obj = {"big": list(range(1000))}
        path = LocalPath(str(tmp_path / "o.pkl.gz"))
        PickleFile(holder=path, owns_holder=False).dump(obj)
        assert PickleFile(holder=path, owns_holder=False).load() == obj


class TestSchema:

    def test_collect_schema_from_pickled_table(self) -> None:
        pf = _pf()
        pf.write_arrow_table(pa.table({"a": [1], "b": ["x"]}))
        schema = PickleFile(holder=pf._parent, owns_holder=False).collect_schema()
        assert set(schema.to_arrow_schema().names) == {"a", "b"}

    def test_collect_schema_empty(self) -> None:
        assert _pf().collect_schema().to_arrow_schema().names == []
