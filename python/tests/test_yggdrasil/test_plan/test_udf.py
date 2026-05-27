"""UDF integration tests — Arrow kernel execution and Spark registration."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.plan import parse_sql, BUILTIN_REGISTRY
from yggdrasil.plan.func_registry import FunctionRegistry


@pytest.fixture
def users():
    return ArrowTabular(pa.table({
        "id": [1, 2, 3, 4, 5],
        "name": ["alice", "BOB", "Carol", "dave", "EVE"],
        "score": [90, -80, 95, -70, 85],
        "region": ["US", "EU", "US", "EU", "US"],
    }))


# ---------------------------------------------------------------------------
# Built-in Arrow kernel execution via plan
# ---------------------------------------------------------------------------

class TestBuiltinKernelExecution:
    def test_upper(self, users):
        result = parse_sql("SELECT UPPER(name) AS uname FROM t").execute(tables={"t": users})
        assert result.read_arrow_table().column("uname").to_pylist() == ["ALICE", "BOB", "CAROL", "DAVE", "EVE"]

    def test_lower(self, users):
        result = parse_sql("SELECT LOWER(name) AS lname FROM t").execute(tables={"t": users})
        assert result.read_arrow_table().column("lname").to_pylist() == ["alice", "bob", "carol", "dave", "eve"]

    def test_abs(self, users):
        result = parse_sql("SELECT ABS(score) AS val FROM t").execute(tables={"t": users})
        assert result.read_arrow_table().column("val").to_pylist() == [90, 80, 95, 70, 85]

    def test_ceil_floor(self):
        t = ArrowTabular(pa.table({"x": [1.2, 2.7, -0.5]}))
        result = parse_sql("SELECT CEIL(x) AS c FROM t").execute(tables={"t": t})
        assert result.read_arrow_table().column("c").to_pylist() == [2.0, 3.0, 0.0]
        result = parse_sql("SELECT FLOOR(x) AS f FROM t").execute(tables={"t": t})
        assert result.read_arrow_table().column("f").to_pylist() == [1.0, 2.0, -1.0]

    def test_sqrt(self):
        t = ArrowTabular(pa.table({"x": [4.0, 9.0, 16.0]}))
        result = parse_sql("SELECT SQRT(x) AS s FROM t").execute(tables={"t": t})
        vals = result.read_arrow_table().column("s").to_pylist()
        assert vals == [2.0, 3.0, 4.0]

    def test_length(self, users):
        result = parse_sql("SELECT LENGTH(name) AS len FROM t").execute(tables={"t": users})
        assert result.read_arrow_table().column("len").to_pylist() == [5, 3, 5, 4, 3]

    def test_column_plus_function(self, users):
        result = parse_sql("SELECT id, UPPER(name) AS uname FROM t").execute(tables={"t": users})
        table = result.read_arrow_table()
        assert table.column_names == ["id", "uname"]
        assert table.column("id").to_pylist() == [1, 2, 3, 4, 5]

    def test_coalesce(self):
        t = ArrowTabular(pa.table({"a": [1, None, 3], "b": [10, 20, 30]}))
        result = parse_sql("SELECT COALESCE(a, b) AS val FROM t").execute(tables={"t": t})
        assert result.read_arrow_table().column("val").to_pylist() == [1, 20, 3]

    def test_ifnull(self):
        t = ArrowTabular(pa.table({"a": [1, None, 3], "b": [10, 20, 30]}))
        result = parse_sql("SELECT IFNULL(a, b) AS val FROM t").execute(tables={"t": t})
        assert result.read_arrow_table().column("val").to_pylist() == [1, 20, 3]

    def test_year_extraction(self):
        t = ArrowTabular(pa.table({
            "ts": pa.array([1704067200, 1706745600, 1709424000], type=pa.timestamp("s")),
        }))
        result = parse_sql("SELECT YEAR(ts) AS yr FROM t").execute(tables={"t": t})
        years = result.read_arrow_table().column("yr").to_pylist()
        assert all(isinstance(y, int) for y in years)

    def test_trig_functions(self):
        t = ArrowTabular(pa.table({"x": [0.0, 1.0, 3.14159]}))
        for fn in ("SIN", "COS", "TAN", "EXP", "LN"):
            result = parse_sql(f"SELECT {fn}(x) AS val FROM t").execute(tables={"t": t})
            assert result.read_arrow_table().num_rows == 3


# ---------------------------------------------------------------------------
# Custom UDF registration and execution
# ---------------------------------------------------------------------------

class TestCustomUDF:
    def test_register_and_apply(self):
        reg = BUILTIN_REGISTRY.copy()
        reg.register_udf("DOUBLE_IT", lambda a: pc.multiply(a, 2))
        assert reg.is_known("DOUBLE_IT")
        arr = reg.apply_arrow("DOUBLE_IT", pa.array([1, 2, 3]))
        assert arr.to_pylist() == [2, 4, 6]

    def test_register_string_udf(self):
        reg = BUILTIN_REGISTRY.copy()
        reg.register_udf("SHOUT", lambda a: pc.binary_join_element_wise(pc.utf8_upper(a), pa.scalar("!!!"), pa.scalar("")))
        assert reg.is_known("SHOUT")

    def test_udf_metadata(self):
        reg = BUILTIN_REGISTRY.copy()
        meta = reg.register_udf("MY_FN", lambda a: a, min_args=1, max_args=2)
        assert meta.name == "MY_FN"
        assert meta.category == "udf"
        assert meta.min_args == 1
        assert meta.kernel is not None

    def test_copy_preserves_udfs(self):
        reg = BUILTIN_REGISTRY.copy()
        reg.register_udf("CUSTOM", lambda a: a)
        reg2 = reg.copy()
        assert reg2.is_known("CUSTOM")


# ---------------------------------------------------------------------------
# Registry direct API
# ---------------------------------------------------------------------------

class TestRegistryAPI:
    def test_apply_arrow_builtin(self):
        arr = BUILTIN_REGISTRY.apply_arrow("UPPER", pa.array(["hello", "world"]))
        assert arr.to_pylist() == ["HELLO", "WORLD"]

    def test_apply_arrow_unknown(self):
        result = BUILTIN_REGISTRY.apply_arrow("NONEXISTENT_FN", pa.array([1]))
        assert result is None

    def test_apply_arrow_abs(self):
        arr = BUILTIN_REGISTRY.apply_arrow("ABS", pa.array([-1, -2, 3]))
        assert arr.to_pylist() == [1, 2, 3]

    def test_apply_table(self):
        table = pa.table({"name": ["alice", "bob"]})
        arr = BUILTIN_REGISTRY.apply_table("UPPER", table, ["name"])
        assert arr.to_pylist() == ["ALICE", "BOB"]

    def test_kerneled_function_count(self):
        with_kernel = sum(1 for m in BUILTIN_REGISTRY._functions.values() if m.kernel is not None)
        assert with_kernel >= 40


# ---------------------------------------------------------------------------
# Spark UDF registration
# ---------------------------------------------------------------------------

class TestSparkUDFRegistration:
    @pytest.fixture(scope="class")
    def spark(self):
        try:
            from pyspark.sql import SparkSession
            session = (SparkSession.builder.master("local[1]").appName("udf_test")
                       .config("spark.ui.enabled", "false")
                       .config("spark.driver.memory", "512m")
                       .getOrCreate())
            yield session
            session.stop()
        except ImportError:
            pytest.skip("pyspark not available")

    def test_spark_builtin_upper(self, spark):
        df = spark.createDataFrame([("hello",), ("world",)], ["name"])
        df.createOrReplaceTempView("names")
        result = spark.sql("SELECT UPPER(name) AS uname FROM names")
        rows = [r.uname for r in result.collect()]
        assert rows == ["HELLO", "WORLD"]

    def test_spark_plan_with_function(self, spark):
        """Spark SQL passthrough executes function via Catalyst."""
        from yggdrasil.spark.tabular import SparkDataset
        df = spark.createDataFrame(
            [(1, "alice"), (2, "bob")], ["id", "name"]
        )
        ds = SparkDataset(frame=df)
        node = parse_sql("SELECT id, UPPER(name) AS uname FROM t")
        # Spark passthrough won't fire (no GROUP BY / SET ops),
        # but Arrow kernel execution handles UPPER
        result = node.execute(tables={"t": ds})
        table = result.read_arrow_table()
        assert table.column("uname").to_pylist() == ["ALICE", "BOB"]


# ---------------------------------------------------------------------------
# Nested type constructors
# ---------------------------------------------------------------------------

class TestNestedConstructors:
    def test_struct(self):
        r = BUILTIN_REGISTRY.apply_arrow("STRUCT", pa.array([1, 2]), pa.array(["a", "b"]))
        assert r.to_pylist() == [{"c0": 1, "c1": "a"}, {"c0": 2, "c1": "b"}]

    def test_named_struct(self):
        r = BUILTIN_REGISTRY.apply_arrow("NAMED_STRUCT",
            "id", pa.array([1, 2]), "name", pa.array(["a", "b"]))
        assert r.to_pylist() == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]

    def test_array_construct(self):
        r = BUILTIN_REGISTRY.apply_arrow("ARRAY",
            pa.array([1, 2]), pa.array([3, 4]), pa.array([5, 6]))
        assert r.to_pylist() == [[1, 3, 5], [2, 4, 6]]

    def test_map_construct(self):
        r = BUILTIN_REGISTRY.apply_arrow("MAP",
            "x", pa.array([1, 2]), "y", pa.array([3, 4]))
        assert r.to_pylist() == [{"x": 1, "y": 3}, {"x": 2, "y": 4}]

    def test_map_from_arrays(self):
        r = BUILTIN_REGISTRY.apply_arrow("MAP_FROM_ARRAYS",
            pa.array([["a", "b"]], type=pa.list_(pa.utf8())),
            pa.array([[1, 2]], type=pa.list_(pa.int64())))
        vals = r.to_pylist()
        assert vals[0]["a"] == 1 and vals[0]["b"] == 2

    def test_map_keys(self):
        m = pa.array([[("a", 1), ("b", 2)]], type=pa.map_(pa.utf8(), pa.int64()))
        r = BUILTIN_REGISTRY.apply_arrow("MAP_KEYS", m)
        assert r.to_pylist() == [["a", "b"]]

    def test_map_values(self):
        m = pa.array([[("a", 1), ("b", 2)]], type=pa.map_(pa.utf8(), pa.int64()))
        r = BUILTIN_REGISTRY.apply_arrow("MAP_VALUES", m)
        assert r.to_pylist() == [[1, 2]]

    def test_get_field(self):
        s = pc.make_struct(pa.array([10, 20]), pa.array(["x", "y"]),
                           field_names=["id", "name"])
        r = BUILTIN_REGISTRY.apply_arrow("GET_FIELD", s, "name")
        assert r.to_pylist() == ["x", "y"]


# ---------------------------------------------------------------------------
# Explode / posexplode table-level operations
# ---------------------------------------------------------------------------

class TestExplodeTable:
    def test_explode(self):
        from yggdrasil.plan.func_registry import explode_table
        t = pa.table({"id": [1, 2], "vals": [[10, 20], [30, 40, 50]]})
        r = explode_table(t, "vals")
        assert r.num_rows == 5
        assert r.column("id").to_pylist() == [1, 1, 2, 2, 2]
        assert r.column("vals").to_pylist() == [10, 20, 30, 40, 50]

    def test_explode_with_rename(self):
        from yggdrasil.plan.func_registry import explode_table
        t = pa.table({"id": [1, 2], "items": [[10, 20], [30]]})
        r = explode_table(t, "items", out_col="item")
        assert "item" in r.column_names
        assert "items" not in r.column_names
        assert r.column("item").to_pylist() == [10, 20, 30]

    def test_posexplode(self):
        from yggdrasil.plan.func_registry import posexplode_table
        t = pa.table({"id": [1, 2], "vals": [[10, 20], [30]]})
        r = posexplode_table(t, "vals", out_col="val")
        assert r.num_rows == 3
        assert r.column("pos").to_pylist() == [0, 1, 0]
        assert r.column("val").to_pylist() == [10, 20, 30]

    def test_explode_preserves_columns(self):
        from yggdrasil.plan.func_registry import explode_table
        t = pa.table({"id": [1, 2], "name": ["a", "b"], "vals": [[10, 20], [30]]})
        r = explode_table(t, "vals")
        assert set(r.column_names) == {"id", "name", "vals"}
        assert r.column("name").to_pylist() == ["a", "a", "b"]


# ---------------------------------------------------------------------------
# Collection operations
# ---------------------------------------------------------------------------

class TestCollectionOps:
    def test_size(self):
        assert BUILTIN_REGISTRY.apply_arrow("SIZE", pa.array([[1, 2], [3]])).to_pylist() == [2, 1]

    def test_flatten(self):
        assert BUILTIN_REGISTRY.apply_arrow("FLATTEN", pa.array([[1, 2], [3, 4]])).to_pylist() == [1, 2, 3, 4]

    def test_sort_array(self):
        assert BUILTIN_REGISTRY.apply_arrow("SORT_ARRAY", pa.array([[3, 1, 2]])).to_pylist() == [[1, 2, 3]]

    def test_array_distinct(self):
        r = BUILTIN_REGISTRY.apply_arrow("ARRAY_DISTINCT", pa.array([[1, 2, 2, 3]]))
        assert set(r.to_pylist()[0]) == {1, 2, 3}

    def test_array_contains(self):
        assert BUILTIN_REGISTRY.apply_arrow("ARRAY_CONTAINS", pa.array([[1, 2, 3]]), 2).to_pylist() == [True]
        assert BUILTIN_REGISTRY.apply_arrow("ARRAY_CONTAINS", pa.array([[1, 2, 3]]), 5).to_pylist() == [False]

    def test_array_min_max(self):
        assert BUILTIN_REGISTRY.apply_arrow("ARRAY_MIN", pa.array([[3, 1, 2]])).to_pylist() == [1]
        assert BUILTIN_REGISTRY.apply_arrow("ARRAY_MAX", pa.array([[3, 1, 2]])).to_pylist() == [3]

    def test_string_cast(self):
        assert BUILTIN_REGISTRY.apply_arrow("STRING", pa.array([1, 2, 3])).to_pylist() == ["1", "2", "3"]

    def test_bigint_cast(self):
        assert BUILTIN_REGISTRY.apply_arrow("BIGINT", pa.array(["1", "2"])).to_pylist() == [1, 2]

    def test_md5(self):
        r = BUILTIN_REGISTRY.apply_arrow("MD5", pa.array(["hello"]))
        assert r.to_pylist()[0] == "5d41402abc4b2a76b9719d911017c592"

    def test_total_kerneled(self):
        count = sum(1 for m in BUILTIN_REGISTRY._functions.values() if m.kernel)
        assert count >= 85
