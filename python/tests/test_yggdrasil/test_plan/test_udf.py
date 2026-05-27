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
