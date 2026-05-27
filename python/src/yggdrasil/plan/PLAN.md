# Yggdrasil Plan Module

SQL parsing, lazy execution plans, and Arrow-native UDF execution for Tabular data.

## Quick Start

```python
from yggdrasil.plan import parse_sql, SelectPlan, LazyTabular, BUILTIN_REGISTRY
from yggdrasil.arrow.tabular import ArrowTabular
import pyarrow as pa

# Create a table
users = ArrowTabular(pa.table({
    "id": [1, 2, 3, 4, 5],
    "name": ["alice", "bob", "carol", "dave", "eve"],
    "region": ["US", "EU", "US", "EU", "US"],
    "score": [90, 80, 95, 70, 85],
}))

# Parse and execute SQL
node = parse_sql("SELECT name, score FROM users WHERE score > 80 ORDER BY score DESC")
result = node.execute(tables={"users": users})
print(result.read_arrow_table().to_pylist())
# [{'name': 'carol', 'score': 95}, {'name': 'alice', 'score': 90}, {'name': 'eve', 'score': 85}]
```

## SQL Parsing

### Basic SELECT

```python
parse_sql("SELECT * FROM t")
parse_sql("SELECT a, b, c FROM t")
parse_sql("SELECT DISTINCT region FROM users")
parse_sql("SELECT a + b AS total, c * 2 AS doubled FROM t")
parse_sql("SELECT 1 AS one, 'hello' AS greeting, NULL AS empty")
parse_sql("SELECT t.* FROM t")  # qualified star
```

### WHERE Clauses

```python
parse_sql("SELECT * FROM t WHERE id > 10")
parse_sql("SELECT * FROM t WHERE a > 1 AND b < 2 OR c = 3")
parse_sql("SELECT * FROM t WHERE id IN (1, 2, 3)")
parse_sql("SELECT * FROM t WHERE id NOT IN (4, 5)")
parse_sql("SELECT * FROM t WHERE id BETWEEN 10 AND 20")
parse_sql("SELECT * FROM t WHERE name LIKE '%test%'")
parse_sql("SELECT * FROM t WHERE name ILIKE '%john%'")  # case-insensitive
parse_sql("SELECT * FROM t WHERE x IS NULL")
parse_sql("SELECT * FROM t WHERE x IS NOT NULL")
parse_sql("SELECT * FROM t WHERE NOT (a > 1)")
```

### JOIN

```python
parse_sql("SELECT * FROM a INNER JOIN b ON a.id = b.id")
parse_sql("SELECT * FROM a LEFT JOIN b ON a.id = b.id")
parse_sql("SELECT * FROM a RIGHT OUTER JOIN b ON a.id = b.ref_id")
parse_sql("SELECT * FROM a FULL OUTER JOIN b ON a.id = b.id")
parse_sql("SELECT * FROM a CROSS JOIN b")
parse_sql("SELECT * FROM a JOIN b ON a.id = b.id LEFT JOIN c ON b.id = c.id")  # multi-join
parse_sql("SELECT * FROM a JOIN (SELECT * FROM b WHERE active) sub ON a.id = sub.id")  # subquery
```

### GROUP BY / HAVING

```python
parse_sql("SELECT region, COUNT(*) AS cnt FROM users GROUP BY region")
parse_sql("SELECT region, SUM(score), AVG(score), MIN(score), MAX(score) FROM users GROUP BY region")
parse_sql("SELECT region, COUNT(*) AS cnt FROM users GROUP BY region HAVING COUNT(*) > 2")
parse_sql("SELECT COUNT(*) AS total FROM users")  # aggregate without GROUP BY
parse_sql("SELECT YEAR(dt), COUNT(*) FROM events GROUP BY YEAR(dt)")  # expression in GROUP BY
```

### ORDER BY / LIMIT / OFFSET

```python
parse_sql("SELECT * FROM t ORDER BY score DESC")
parse_sql("SELECT * FROM t ORDER BY region ASC, score DESC")
parse_sql("SELECT * FROM t ORDER BY score DESC NULLS LAST")
parse_sql("SELECT * FROM t LIMIT 10")
parse_sql("SELECT * FROM t LIMIT 10 OFFSET 20")
parse_sql("SELECT * FROM t ORDER BY id DESC LIMIT 5")
```

### UNION / INTERSECT / EXCEPT

```python
parse_sql("SELECT a FROM t1 UNION ALL SELECT a FROM t2")
parse_sql("SELECT a FROM t1 UNION SELECT a FROM t2")  # distinct
parse_sql("SELECT a FROM t1 INTERSECT SELECT a FROM t2")
parse_sql("SELECT a FROM t1 EXCEPT SELECT a FROM t2")
```

### WITH (CTEs)

```python
parse_sql("""
    WITH top_users AS (
        SELECT * FROM users WHERE score > 80
    )
    SELECT name, score FROM top_users ORDER BY score DESC
""")

parse_sql("""
    WITH a AS (SELECT 1 AS x),
         b AS (SELECT 2 AS y)
    SELECT * FROM a JOIN b ON a.x = b.y
""")
```

### Function Calls

```python
# String
parse_sql("SELECT UPPER(name), LOWER(name), LENGTH(name) FROM t")
parse_sql("SELECT TRIM(name), REVERSE(name), INITCAP(name) FROM t")
parse_sql("SELECT CONCAT(first, ' ', last) FROM t", dialect="databricks")
parse_sql("SELECT SUBSTRING(name, 1, 3) FROM t", dialect="databricks")
parse_sql("SELECT REGEXP_REPLACE(s, '[0-9]+', 'X') FROM t", dialect="databricks")

# Math
parse_sql("SELECT ABS(x), CEIL(x), FLOOR(x), ROUND(x, 2) FROM t")
parse_sql("SELECT SQRT(x), POWER(x, 2), LN(x), EXP(x) FROM t")
parse_sql("SELECT SIN(x), COS(x), TAN(x), ATAN2(y, x) FROM t")

# Null handling
parse_sql("SELECT COALESCE(a, b, 0) FROM t")
parse_sql("SELECT NVL(a, 0), IFNULL(a, 0), NULLIF(a, b) FROM t")

# Date/time
parse_sql("SELECT YEAR(ts), MONTH(ts), DAY(ts) FROM t")
parse_sql("SELECT DATE_TRUNC('month', ts) FROM t", dialect="databricks")
parse_sql("SELECT DATE_ADD(dt, 7), DATEDIFF(a, b) FROM t", dialect="databricks")
parse_sql("SELECT EXTRACT(YEAR FROM ts) FROM t")
parse_sql("SELECT CURRENT_TIMESTAMP(), CURRENT_DATE() FROM t")
parse_sql("SELECT ts + INTERVAL '7' DAY FROM t")

# Aggregate
parse_sql("SELECT COUNT(*), COUNT(DISTINCT id) FROM t")
parse_sql("SELECT SUM(amount), AVG(score), MIN(id), MAX(id) FROM t")
parse_sql("SELECT COLLECT_LIST(name) FROM t GROUP BY region", dialect="databricks")
parse_sql("SELECT APPROX_COUNT_DISTINCT(id) FROM t", dialect="databricks")

# Window
parse_sql("SELECT ROW_NUMBER() OVER (PARTITION BY region ORDER BY score DESC) FROM t", dialect="databricks")
parse_sql("SELECT RANK() OVER (ORDER BY score DESC) FROM t", dialect="databricks")
parse_sql("SELECT LAG(val, 1) OVER (ORDER BY ts) FROM t", dialect="databricks")
parse_sql("SELECT SUM(val) OVER (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
          dialect="databricks")

# Conditional
parse_sql("SELECT CASE WHEN score > 90 THEN 'A' WHEN score > 80 THEN 'B' ELSE 'C' END FROM t")
parse_sql("SELECT IF(a > 0, 'positive', 'negative') FROM t", dialect="databricks")

# Type casting
parse_sql("SELECT CAST(id AS BIGINT), CAST(name AS STRING) FROM t")
parse_sql("SELECT CAST(price AS DECIMAL(10, 2)) FROM t")
parse_sql("SELECT TRY_CAST(x AS INT) FROM t", dialect="databricks")
parse_sql("SELECT TIMESTAMP '2024-01-01 00:00:00', DATE '2024-01-01'")
```

### LATERAL VIEW / EXPLODE

```python
parse_sql("""
    SELECT id, val
    FROM t
    LATERAL VIEW EXPLODE(arr) vals AS val
""", dialect="databricks")

parse_sql("""
    SELECT id, v1, v2
    FROM t
    LATERAL VIEW EXPLODE(arr1) t1 AS v1
    LATERAL VIEW EXPLODE(arr2) t2 AS v2
""", dialect="databricks")
```

### INSERT / MERGE

```python
parse_sql("INSERT INTO target SELECT * FROM source")
parse_sql("INSERT INTO t (id, name) VALUES (1, 'alice'), (2, 'bob')")

parse_sql("""
    MERGE INTO target t USING source s ON t.id = s.id
    WHEN MATCHED THEN UPDATE SET t.name = s.name
    WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)
""")
```

### FROM Path/URL Sources

```python
# String literal paths in FROM clause
parse_sql("SELECT * FROM '/data/users.parquet'")
parse_sql("SELECT * FROM 's3://bucket/data.csv'")
```

## Execution

### Against Arrow Tables

```python
users = ArrowTabular(pa.table({...}))
node = parse_sql("SELECT * FROM users WHERE score > 80")
result = node.execute(tables={"users": users})
table = result.read_arrow_table()
```

### Against Spark DataFrames

```python
from yggdrasil.spark.tabular import SparkDataset

spark_users = SparkDataset(frame=spark.createDataFrame(...))
node = parse_sql("SELECT * FROM users WHERE score > 80")
result = node.execute(tables={"users": spark_users})
# Uses Spark-native filter, stays as SparkDataset
```

### Against Folders (disk-backed)

```python
from yggdrasil.path.local_path import LocalPath
from yggdrasil.path.folder import Folder

folder = Folder(path=LocalPath("/data/users"))
node = parse_sql("SELECT * FROM users WHERE region = 'US'")
# Pushes predicate into CastOptions → partition pruning
result = node.execute(tables={"users": folder})
```

### With CTEs

```python
node = parse_sql("""
    WITH active AS (SELECT * FROM users WHERE active = TRUE),
         scored AS (SELECT * FROM active WHERE score > 80)
    SELECT region, COUNT(*) AS cnt FROM scored GROUP BY region
""")
result = node.execute(tables={"users": users_table})
```

### Default on Parse Failure

```python
node = parse_sql("INVALID SQL", default=None)  # returns None
node = parse_sql("SELECT 1", default=None)     # returns SelectNode
```

## Lazy Execution (LazyTabular)

```python
lazy = users.lazy()
lazy.select("id", "name").filter("score > 80").limit(10)
table = lazy.read_arrow_table()  # executes here

# Chain operations
result = (users.lazy()
          .filter("region = 'US'")
          .select("name", "score")
          .limit(5)
          .read_arrow_table())

# Join
result = (users.lazy()
          .join(orders, on="id", how="inner")
          .filter("amount > 20")
          .read_arrow_table())
```

## SelectPlan (Programmatic Builder)

```python
plan = SelectPlan()
plan.select("a", "b").filter("x > 10").limit(100)
result = plan.execute(source_tabular)

# SQL round-trip
sql = plan.to_sql(dialect="databricks")
plan2 = ExecutionPlan.from_sql(sql)
```

## UDF System

### Built-in Arrow Kernels (78 functions)

```python
from yggdrasil.plan import BUILTIN_REGISTRY

# Direct Arrow execution
BUILTIN_REGISTRY.apply_arrow("UPPER", pa.array(["hello"]))    # → ["HELLO"]
BUILTIN_REGISTRY.apply_arrow("ABS", pa.array([-1, -2, 3]))    # → [1, 2, 3]
BUILTIN_REGISTRY.apply_arrow("SQRT", pa.array([4.0, 9.0]))    # → [2.0, 3.0]
BUILTIN_REGISTRY.apply_arrow("COALESCE", pa.array([1, None]), pa.array([10, 20]))  # → [1, 20]
BUILTIN_REGISTRY.apply_arrow("MD5", pa.array(["hello"]))       # → ["5d41402abc..."]

# Via SQL execution
node = parse_sql("SELECT UPPER(name) AS uname, ABS(score) AS abs_score FROM users")
result = node.execute(tables={"users": users})
```

### Custom UDFs

```python
import pyarrow.compute as pc

# Register
BUILTIN_REGISTRY.register_udf("DOUBLE_IT", lambda a: pc.multiply(a, 2))

# Apply directly
BUILTIN_REGISTRY.apply_arrow("DOUBLE_IT", pa.array([1, 2, 3]))  # → [2, 4, 6]

# Register with metadata
BUILTIN_REGISTRY.register_udf("NORMALIZE",
    lambda a: pc.divide(pc.subtract(a, pc.mean(a)), pc.stddev(a)),
    min_args=1, max_args=1)
```

### Collection Operations

```python
from yggdrasil.plan.func_registry import explode_table, posexplode_table

# Explode list columns
t = pa.table({"id": [1, 2], "vals": [[10, 20], [30, 40, 50]]})
explode_table(t, "vals")
# id | vals
# 1  | 10
# 1  | 20
# 2  | 30
# 2  | 40
# 2  | 50

posexplode_table(t, "vals", pos_col="pos", out_col="val")
# id | val | pos
# 1  | 10  | 0
# 1  | 20  | 1
# 2  | 30  | 0
# ...

# Array functions via registry
BUILTIN_REGISTRY.apply_arrow("SIZE", pa.array([[1,2,3],[4,5]]))        # → [3, 2]
BUILTIN_REGISTRY.apply_arrow("FLATTEN", pa.array([[1,2],[3],[4,5]]))   # → [1,2,3,4,5]
BUILTIN_REGISTRY.apply_arrow("SORT_ARRAY", pa.array([[3,1,2]]))       # → [[1,2,3]]
BUILTIN_REGISTRY.apply_arrow("ARRAY_DISTINCT", pa.array([[1,2,2,3]])) # → [[1,2,3]]
BUILTIN_REGISTRY.apply_arrow("ARRAY_CONTAINS", pa.array([[1,2,3]]), 2) # → [True]
```

### IO UDFs

```python
# Read file contents
BUILTIN_REGISTRY.apply_arrow("READ_FILES", pa.array(["/path/to/file.txt"]))

# List directory
BUILTIN_REGISTRY.apply_arrow("READ_PATHS", pa.array(["/data/"]))

# Compress / decompress
data = pa.array(["hello world"])
compressed = BUILTIN_REGISTRY.apply_arrow("COMPRESS", data, "gzip")
BUILTIN_REGISTRY.apply_arrow("DECOMPRESS", compressed, "gzip")

# JSON parse / serialize
BUILTIN_REGISTRY.apply_arrow("PARSE_JSON", pa.array(['{"a": 1}']))
BUILTIN_REGISTRY.apply_arrow("TO_JSON", pa.array([1, 2, 3]))

# Type casting
BUILTIN_REGISTRY.apply_arrow("STRING", pa.array([1, 2, 3]))    # → ["1", "2", "3"]
BUILTIN_REGISTRY.apply_arrow("BIGINT", pa.array(["1", "2"]))   # → [1, 2]
```

### Spark UDF Registration

```python
# Register all kerneled functions in Spark
BUILTIN_REGISTRY.register_in_spark(spark_session)

# Custom UDFs work in both Arrow and Spark
reg = BUILTIN_REGISTRY.copy()
reg.register_udf("MY_FN", lambda a: pc.multiply(a, 2))
reg.register_in_spark(spark)
spark.sql("SELECT MY_FN(val) FROM t")
```

## SQL Dialects

```python
parse_sql("SELECT * FROM t", dialect="ansi")
parse_sql("SELECT * FROM t", dialect="databricks")  # default
parse_sql("SELECT * FROM t", dialect="postgres")
parse_sql("SELECT * FROM t", dialect="mysql")
parse_sql("SELECT * FROM t", dialect="sqlite")
```

Databricks dialect adds recognition for 200+ built-in functions, `LATERAL VIEW`,
`TRY_CAST`, `INTERVAL` literals, and backtick identifier quoting.

## SQL Round-Trip

```python
node = parse_sql("SELECT a, b FROM t WHERE id > 10 LIMIT 5", dialect="databricks")
sql = node.to_sql(dialect="databricks")
# "SELECT `a`, `b` FROM `t` WHERE `id` > 10 LIMIT 5"

node2 = parse_sql(sql, dialect="databricks")  # round-trips cleanly
```

## Function Registry

```python
from yggdrasil.plan import BUILTIN_REGISTRY

len(BUILTIN_REGISTRY)          # 290+ functions
BUILTIN_REGISTRY.is_known("DATE_TRUNC")  # True
meta = BUILTIN_REGISTRY.get("DATE_TRUNC")
# FunctionMeta(name='DATE_TRUNC', category='datetime', min_args=2, max_args=2, kernel=...)

# Categories
# aggregate, window, datetime, string, null, math, collection,
# generator, type, json, higher_order, hash, conditional, io, misc

# Copy for isolated scope
my_reg = BUILTIN_REGISTRY.copy()
my_reg.register_udf("MY_FN", ...)
```

## Performance

| Operation (10k rows) | Time |
|---------------------|------|
| Parse simple SELECT | 35µs |
| Parse complex CTE+window | 170µs |
| Execute SELECT* | 31µs |
| Execute WHERE filter | 670µs |
| Execute GROUP BY | 700µs |
| Execute ORDER BY+LIMIT | 490µs |
| UPPER(name) UDF | 340µs |
| ABS(score) UDF | 145µs |
| 3-column UDF projection | 450µs |
| SQL round-trip | 100µs |
| Registry lookup | 0.2µs |
| Folder SELECT* (disk) | 370µs |
| Folder WHERE (disk) | 1.1ms |
