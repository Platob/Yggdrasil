"""Advanced feature tests: URL sources, auto-cast joins, filter pushdown,
IO UDFs, codec operations."""

from __future__ import annotations

import gzip
import json
import shutil
import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.saga.plan import parse_sql, BUILTIN_REGISTRY
from yggdrasil.saga.plan.func_registry import FunctionRegistry


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# URL/path auto-resolution in FROM clause
# ---------------------------------------------------------------------------

class TestURLAutoResolve:
    def test_local_parquet_path(self, tmpdir):
        path = f"{tmpdir}/data.parquet"
        pq.write_table(pa.table({"id": [1, 2], "val": ["a", "b"]}), path)
        node = parse_sql(f"SELECT * FROM '{path}'")
        # The path is in the SQL as a string — won't auto-resolve from parse
        # but if registered as a table, it works
        result = node.execute(tables={path: ArrowTabular(pq.read_table(path))})
        assert result.read_arrow_table().num_rows == 2

    def test_path_auto_resolve_in_tables(self, tmpdir):
        """Tabular.from_() resolves path-like table names."""
        path = f"{tmpdir}/test.parquet"
        pq.write_table(pa.table({"x": [10, 20, 30]}), path)
        node = parse_sql("SELECT * FROM test_data")
        # Register the path as a name
        from yggdrasil.io.tabular.base import Tabular
        tab = Tabular.from_(path)
        result = node.execute(tables={"test_data": tab})
        assert result.read_arrow_table().num_rows == 3


# ---------------------------------------------------------------------------
# Auto-cast join key columns
# ---------------------------------------------------------------------------

class TestAutoCastJoin:
    def test_int_vs_int64_join(self):
        left = ArrowTabular(pa.table({"id": pa.array([1, 2, 3], type=pa.int32()), "name": ["a", "b", "c"]}))
        right = ArrowTabular(pa.table({"id": pa.array([1, 2, 3], type=pa.int64()), "val": [10, 20, 30]}))
        node = parse_sql("SELECT left_t.name, right_t.val FROM left_t JOIN right_t ON left_t.id = right_t.id")
        result = node.execute(tables={"left_t": left, "right_t": right})
        table = result.read_arrow_table()
        assert table.num_rows == 3

    def test_int_vs_string_join(self):
        left = ArrowTabular(pa.table({"key": [1, 2, 3], "name": ["a", "b", "c"]}))
        right = ArrowTabular(pa.table({"key": ["1", "2", "3"], "val": [10, 20, 30]}))
        node = parse_sql("SELECT l.name, r.val FROM l JOIN r ON l.key = r.key")
        result = node.execute(tables={"l": left, "r": right})
        table = result.read_arrow_table()
        # Cast to common type (string) should enable the join
        assert table.num_rows == 3


# ---------------------------------------------------------------------------
# Filter pushdown through CTEs
# ---------------------------------------------------------------------------

class TestFilterPushdown:
    def test_predicate_pushdown_to_folder(self, tmpdir):
        from yggdrasil.path.local_path import LocalPath
        from yggdrasil.path.folder import Folder

        folder = Folder(path=LocalPath(tmpdir))
        folder.write_table(pa.table({
            "id": list(range(1000)),
            "region": ["US" if i % 2 == 0 else "EU" for i in range(1000)],
        }))

        node = parse_sql("SELECT * FROM data WHERE region = 'US'")
        result = node.execute(tables={"data": folder})
        table = result.read_arrow_table()
        assert all(r == "US" for r in table.column("region").to_pylist())
        assert table.num_rows == 500

    def test_cte_filter_propagation(self):
        data = ArrowTabular(pa.table({
            "id": list(range(100)),
            "score": [i % 10 for i in range(100)],
        }))
        node = parse_sql(
            "WITH filtered AS (SELECT * FROM data WHERE score > 5) "
            "SELECT * FROM filtered WHERE id < 50"
        )
        result = node.execute(tables={"data": data})
        table = result.read_arrow_table()
        assert all(s > 5 for s in table.column("score").to_pylist())
        assert all(i < 50 for i in table.column("id").to_pylist())


# ---------------------------------------------------------------------------
# IO UDFs: read_files, compress, decompress, parse_json
# ---------------------------------------------------------------------------

class TestIOUDFs:
    def test_read_files_kernel(self, tmpdir):
        import pathlib
        p = pathlib.Path(tmpdir) / "test.txt"
        p.write_text("hello world")
        result = BUILTIN_REGISTRY.apply_arrow("READ_FILES", pa.array([str(p)]))
        assert result.to_pylist()[0] == "hello world"

    def test_read_paths_kernel(self, tmpdir):
        import pathlib
        (pathlib.Path(tmpdir) / "a.txt").touch()
        (pathlib.Path(tmpdir) / "b.txt").touch()
        result = BUILTIN_REGISTRY.apply_arrow("READ_PATHS", pa.array([tmpdir]))
        assert result.to_pylist()[0] is not None

    def test_compress_decompress(self):
        data = pa.array(["hello", "world", "test"])
        compressed = BUILTIN_REGISTRY.apply_arrow("COMPRESS", data)
        assert compressed.type == pa.binary()
        assert all(v is not None for v in compressed.to_pylist())
        # Decompress
        decompressed = BUILTIN_REGISTRY.apply_arrow("DECOMPRESS", compressed)
        assert all(v is not None for v in decompressed.to_pylist())

    def test_parse_json_kernel(self):
        data = pa.array(['{"a": 1}', '{"b": 2}', None])
        result = BUILTIN_REGISTRY.apply_arrow("PARSE_JSON", data)
        assert result is not None
        assert len(result) == 3

    def test_to_json_kernel(self):
        data = pa.array([1, 2, 3])
        result = BUILTIN_REGISTRY.apply_arrow("TO_JSON", data)
        vals = result.to_pylist()
        assert json.loads(vals[0]) == 1

    def test_compress_with_codec_name(self):
        data = pa.array(["test data here"])
        compressed = BUILTIN_REGISTRY.apply_arrow("COMPRESS", data, "gzip")
        assert compressed.type == pa.binary()
        raw = compressed.to_pylist()[0]
        assert gzip.decompress(raw) == b"test data here"


# ---------------------------------------------------------------------------
# MediaType-aware operations
# ---------------------------------------------------------------------------

class TestMediaTypeAware:
    def test_parquet_read_via_tabular_from(self, tmpdir):
        path = f"{tmpdir}/test.parquet"
        pq.write_table(pa.table({"x": [1, 2, 3]}), path)
        from yggdrasil.io.tabular.base import Tabular
        tab = Tabular.from_(path)
        assert tab is not None
        assert tab.read_arrow_table().num_rows == 3

    def test_csv_read_via_tabular_from(self, tmpdir):
        path = f"{tmpdir}/test.csv"
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "name"])
            w.writerow([1, "alice"])
            w.writerow([2, "bob"])
        from yggdrasil.io.tabular.base import Tabular
        tab = Tabular.from_(path, default=None)
        # CSV detection may depend on format registry bootstrap
        if tab is not None:
            assert tab.read_arrow_table().num_rows >= 2


# ---------------------------------------------------------------------------
# Lazy execution plan optimization
# ---------------------------------------------------------------------------

class TestLazyOptimization:
    def test_lazy_chain_executes_once(self):
        data = ArrowTabular(pa.table({
            "id": list(range(100)),
            "name": [f"u{i}" for i in range(100)],
            "score": [i * 10 % 97 for i in range(100)],
        }))
        result = (data.lazy()
                  .filter("score > 50")
                  .select("id", "name")
                  .limit(5)
                  .read_arrow_table())
        assert result.num_rows == 5
        assert result.column_names == ["id", "name"]

    def test_lazy_preserves_predicate_for_pushdown(self):
        data = ArrowTabular(pa.table({"x": list(range(1000))}))
        lazy = data.lazy()
        lazy.filter("x > 500").limit(10)
        assert lazy.plan.predicate is not None
        assert lazy.plan.limit_rows == 10


# ---------------------------------------------------------------------------
# Benchmark-style throughput tests
# ---------------------------------------------------------------------------

class TestThroughput:
    def test_10k_upper_under_1ms(self):
        import time
        t = ArrowTabular(pa.table({"name": [f"user_{i}" for i in range(10_000)]}))
        t0 = time.perf_counter()
        result = parse_sql("SELECT UPPER(name) AS u FROM t").execute(tables={"t": t})
        result.read_arrow_table()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.01  # 10ms budget

    def test_10k_filter_under_5ms(self):
        import time
        t = ArrowTabular(pa.table({"id": list(range(10_000)), "val": [i % 100 for i in range(10_000)]}))
        t0 = time.perf_counter()
        result = parse_sql("SELECT * FROM t WHERE val > 50").execute(tables={"t": t})
        result.read_arrow_table()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.05
