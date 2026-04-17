from __future__ import annotations

from pathlib import Path

import pytest
from yggdrasil.arrow.lib import pyarrow as pa
from yggdrasil.io.buffer.databricks_path_io import DatabricksPathIO
from yggdrasil.io.buffer.local_path_io import LocalPathIO
from yggdrasil.io.buffer.path_io import PathIO, PathOptions


def _write_parquet(path: Path, table: pa.Table) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


@pytest.fixture()
def sample_table() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "group": pa.array(["a", "b", "a"], type=pa.string()),
            "value": pa.array([10, 20, 30], type=pa.int64()),
        }
    )


def test_pathio_reads_single_parquet_file(tmp_path: Path, sample_table: pa.Table) -> None:
    path = tmp_path / "data.parquet"
    _write_parquet(path, sample_table)

    data = LocalPathIO.make(path)
    out = data.read_arrow_table()

    assert out.schema == sample_table.schema
    assert out.to_pylist() == sample_table.to_pylist()


def test_pathio_selects_columns_and_filters_rows(tmp_path: Path, sample_table: pa.Table) -> None:
    path = tmp_path / "data.parquet"
    _write_parquet(path, sample_table)

    out = LocalPathIO.make(path).read_arrow_table(
        options=PathOptions(columns=["id", "group"]),
        filter={"group": "a"},
    )

    assert out.column_names == ["id", "group"]
    assert out.to_pylist() == [
        {"id": 1, "group": "a"},
        {"id": 3, "group": "a"},
    ]


def test_pathio_filters_on_unselected_column(tmp_path: Path, sample_table: pa.Table) -> None:
    path = tmp_path / "data.parquet"
    _write_parquet(path, sample_table)

    out = LocalPathIO.make(path).read_arrow_table(
        columns=["id"],
        filter={"group": "a"},
    )

    assert out.column_names == ["id"]
    assert out.to_pylist() == [
        {"id": 1},
        {"id": 3},
    ]


def test_pathio_reads_directory_with_partitions(tmp_path: Path) -> None:
    table = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "value": pa.array([100, 200], type=pa.int64()),
        }
    )
    _write_parquet(tmp_path / "group=a" / "part-0.parquet", table.slice(0, 1))
    _write_parquet(tmp_path / "group=b" / "part-1.parquet", table.slice(1, 1))

    out = LocalPathIO.make(tmp_path).read_arrow_table(filter={"group": "b"})

    assert out.to_pylist() == [{"id": 2, "value": 200, "group": "b"}]


def test_pathio_iter_files_skips_hidden_by_default(tmp_path: Path, sample_table: pa.Table) -> None:
    visible = tmp_path / "visible.parquet"
    hidden = tmp_path / ".hidden.parquet"
    _write_parquet(visible, sample_table)
    _write_parquet(hidden, sample_table)

    files = list(LocalPathIO.make(tmp_path).iter_files())

    assert [file.path for file in files] == [visible]


def test_pathio_is_abstract(tmp_path: Path, sample_table: pa.Table) -> None:
    path = tmp_path / "abstract.parquet"
    _write_parquet(path, sample_table)

    with pytest.raises(TypeError):
        PathIO.make(path)


class _FakeDatabricksPath:
    def __init__(self, name: str, payload: bytes, *, is_dir: bool = False, children=None):
        self._name = name
        self._payload = payload
        self._is_dir = is_dir
        self._children = list(children or [])
        self.parts = [part for part in name.split("/") if part]

    @property
    def name(self) -> str:
        return self.parts[-1] if self.parts else ""

    def is_file(self):
        return not self._is_dir

    def is_dir(self):
        return self._is_dir

    def read_bytes(self):
        return self._payload

    def iterdir(self):
        return iter(self._children)

    def rglob(self, pattern: str):
        del pattern
        for child in self._children:
            if child.is_dir():
                yield from child.rglob("*")
            else:
                yield child

    def __str__(self) -> str:
        return self._name


def test_databricks_pathio_reads_parquet_from_fake_path(tmp_path: Path, sample_table: pa.Table) -> None:
    import pyarrow.parquet as pq

    local = tmp_path / "remote.parquet"
    pq.write_table(sample_table, local)
    fake = _FakeDatabricksPath("/Volumes/cat/schema/vol/remote.parquet", local.read_bytes())

    out = DatabricksPathIO.make(fake).read_arrow_table()

    assert out.to_pylist() == sample_table.to_pylist()


def test_databricks_pathio_filters_rows_without_pyarrow_dataset(tmp_path: Path, sample_table: pa.Table) -> None:
    import pyarrow.parquet as pq

    local = tmp_path / "remote.parquet"
    pq.write_table(sample_table, local)
    fake = _FakeDatabricksPath("/Volumes/cat/schema/vol/remote.parquet", local.read_bytes())

    out = DatabricksPathIO.make(fake).read_arrow_table(filter={"group": "a"})

    assert out.to_pylist() == [
        {"id": 1, "group": "a", "value": 10},
        {"id": 3, "group": "a", "value": 30},
    ]


def test_databricks_pathio_iter_files_for_directory(tmp_path: Path, sample_table: pa.Table) -> None:
    import pyarrow.parquet as pq

    local = tmp_path / "remote.parquet"
    pq.write_table(sample_table, local)
    child = _FakeDatabricksPath("/dbfs/tmp/remote.parquet", local.read_bytes())
    root = _FakeDatabricksPath("/dbfs/tmp", b"", is_dir=True, children=[child])

    files = list(DatabricksPathIO.make(root).iter_files())

    assert [file.path for file in files] == [child]
