"""Round-trip tests for the new BytesIO + Holder + LocalPath stack
under the patterns external libraries expect.

The user-facing contract is:

    with holder.open() as b:
        external_lib.write_to(b, ...)

    with holder.open("rb") as b:
        external_lib.read_from(b)

These tests exercise that pattern across the formats the library
actually plumbs through — JSON, CSV (pandas + polars), Arrow IPC,
Parquet, zip — over both :class:`Memory` and :class:`LocalPath`
holders. They protect:

* the durable commit on ``with`` exit;
* cursor semantics (after close, ``read_bytes`` returns whatever
  the writer produced);
* zero-extra-conversion ergonomics — no manual ``.to_arrow()`` or
  ``BytesIO`` adapter needed at the call site.
"""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.memory import Memory
from yggdrasil.path.local_path import LocalPath


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_pandas():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})


@pytest.fixture
def df_polars():
    pl = pytest.importorskip("polars")
    return pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})


# ---------------------------------------------------------------------------
# Pandas through holder.open
# ---------------------------------------------------------------------------


class TestPandasIntoLocalPath:

    def test_to_csv_round_trip(self, tmp_path, df_pandas) -> None:
        pd = pytest.importorskip("pandas")
        target = LocalPath(str(tmp_path / "data.csv"))
        with target.open("wb") as b:
            df_pandas.to_csv(b, index=False)

        # Read back via holder.open in read mode.
        with target.open("rb") as b:
            loaded = pd.read_csv(b)
        pd.testing.assert_frame_equal(loaded, df_pandas)

    def test_to_json_round_trip(self, tmp_path, df_pandas) -> None:
        pd = pytest.importorskip("pandas")
        target = LocalPath(str(tmp_path / "data.json"))
        with target.open("wb") as b:
            df_pandas.to_json(b, orient="records", lines=False)
        loaded = pd.read_json(io.BytesIO(target.read_bytes()), orient="records")
        pd.testing.assert_frame_equal(loaded, df_pandas)

    def test_to_parquet_round_trip(self, tmp_path, df_pandas) -> None:
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        target = LocalPath(str(tmp_path / "data.parquet"))
        with target.open("wb") as b:
            df_pandas.to_parquet(b, engine="pyarrow")
        # Read directly via pyarrow on the materialized file.
        table = pq.read_table(target.os_path)
        pd.testing.assert_frame_equal(table.to_pandas(), df_pandas)


class TestPandasIntoMemory:

    def test_to_csv_round_trip_via_memory_holder(self, df_pandas) -> None:
        pd = pytest.importorskip("pandas")
        mem = Memory()
        b = BytesIO(holder=mem, owns_holder=False, mode="wb+")
        with b:
            df_pandas.to_csv(b, index=False)
        loaded = pd.read_csv(io.BytesIO(mem.read_bytes()))
        pd.testing.assert_frame_equal(loaded, df_pandas)


# ---------------------------------------------------------------------------
# Polars through holder.open
# ---------------------------------------------------------------------------


class TestPolarsIntoLocalPath:

    def test_write_csv_round_trip(self, tmp_path, df_polars) -> None:
        pl = pytest.importorskip("polars")
        target = LocalPath(str(tmp_path / "data.csv"))
        with target.open("wb") as b:
            df_polars.write_csv(b)

        with target.open("rb") as b:
            loaded = pl.read_csv(b)
        assert loaded.equals(df_polars)

    def test_write_parquet_round_trip(self, tmp_path, df_polars) -> None:
        pl = pytest.importorskip("polars")
        target = LocalPath(str(tmp_path / "data.parquet"))
        with target.open("wb") as b:
            df_polars.write_parquet(b)
        loaded = pl.read_parquet(target.os_path)
        assert loaded.equals(df_polars)

    def test_write_ipc_round_trip(self, tmp_path, df_polars) -> None:
        pl = pytest.importorskip("polars")
        target = LocalPath(str(tmp_path / "data.arrow"))
        with target.open("wb") as b:
            df_polars.write_ipc(b)
        loaded = pl.read_ipc(target.os_path)
        assert loaded.equals(df_polars)


# ---------------------------------------------------------------------------
# Arrow IPC round-trip through holder.open
# ---------------------------------------------------------------------------


class TestArrowIPC:

    def test_arrow_table_round_trip(self, tmp_path) -> None:
        pa = pytest.importorskip("pyarrow")
        ipc = pa.ipc

        table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        target = LocalPath(str(tmp_path / "data.arrow"))
        with target.open("wb") as b:
            with ipc.new_file(b, table.schema) as writer:
                writer.write_table(table)

        with target.open("rb") as b:
            with ipc.open_file(b) as reader:
                loaded = reader.read_all()
        assert loaded.equals(table)


# ---------------------------------------------------------------------------
# Zip archive round-trip — exercises seek-during-write
# ---------------------------------------------------------------------------


class TestZipArchive:

    def test_zip_write_inside_with(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "bundle.zip"))
        with target.open("wb") as b:
            with zipfile.ZipFile(b, "w") as zf:
                zf.writestr("a.txt", "alpha")
                zf.writestr("b.txt", "beta")

        # Read back through the same pattern.
        with target.open("rb") as b:
            with zipfile.ZipFile(b) as zf:
                assert sorted(zf.namelist()) == ["a.txt", "b.txt"]
                assert zf.read("a.txt") == b"alpha"


# ---------------------------------------------------------------------------
# JSON / NDJSON
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:

    def test_dump_load(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "doc.json"))
        payload = {"x": [1, 2, 3], "ok": True}
        with target.open("wb") as b:
            b.write(json.dumps(payload).encode("utf-8"))

        with target.open("rb") as b:
            loaded = json.load(io.TextIOWrapper(b, encoding="utf-8"))
        assert loaded == payload

    def test_ndjson_lines(self, tmp_path) -> None:
        target = LocalPath(str(tmp_path / "stream.ndjson"))
        rows = [{"id": i, "name": f"row-{i}"} for i in range(5)]
        with target.open("wb") as b:
            for row in rows:
                b.write((json.dumps(row) + "\n").encode("utf-8"))

        with target.open("rb") as b:
            loaded = [json.loads(line) for line in b if line.strip()]
        assert loaded == rows


# ---------------------------------------------------------------------------
# Cross-format: pandas → CSV → polars → ArrowIPC → pandas
# ---------------------------------------------------------------------------


class TestCrossFormatChain:

    def test_pandas_csv_then_polars_ipc_then_pandas(self, tmp_path) -> None:
        pd = pytest.importorskip("pandas")
        pl = pytest.importorskip("polars")

        original = pd.DataFrame({"id": [1, 2, 3], "v": [0.5, 1.5, 2.5]})
        csv_target = LocalPath(str(tmp_path / "step1.csv"))
        with csv_target.open("wb") as b:
            original.to_csv(b, index=False)

        with csv_target.open("rb") as b:
            df = pl.read_csv(b)

        ipc_target = LocalPath(str(tmp_path / "step2.arrow"))
        with ipc_target.open("wb") as b:
            df.write_ipc(b)

        with ipc_target.open("rb") as b:
            roundtrip = pl.read_ipc(b).to_pandas()
        pd.testing.assert_frame_equal(roundtrip, original)
