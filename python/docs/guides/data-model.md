# Data Model — `DataType`, `Field`, `Schema`

Yggdrasil's data layer has three load-bearing types that show up everywhere a column, value, or row crosses a boundary:

| Class | Carries | Lives in |
| --- | --- | --- |
| `DataType` | Type-level info: kind (int / str / list / …), width, precision, nested children, engine projections | `yggdrasil.data.types.base.DataType` |
| `Field` | Column-level info: name, dtype, nullability, metadata | `yggdrasil.data.data_field.Field` |
| `Schema` | Field-level info: an ordered, named tuple of `Field`s | `yggdrasil.data.schema.Schema` |

Plus one global conversion entry point that every engine plugs into:

| Function | Purpose |
| --- | --- |
| `convert(value, target, options=...)` | Single dispatch surface for value → type conversion |
| `register_converter(from_hint, to_hint)` | Decorator to teach the registry a new path |
| `find_converter(from_type, to_hint)` | Lookup-only — what `convert` calls under the hood |

The rest of this page is the worked surface. Pair it with [Casting](casting.md) for tabular round-trips and [Architecture](architecture.md) for the higher-level rules.

---

## `DataType` — the single source of typing truth

`DataType` is the class hierarchy that describes "what kind of value lives in a column". Concrete subclasses cover the usual primitives plus nested and extension shapes:

```text
DataType
├── PrimitiveType
│   ├── NullType                      — only None
│   ├── ObjectType                    — variant / opaque Python object
│   ├── BooleanType
│   ├── BinaryType                    — bytes
│   ├── StringType                    — utf8 / large_utf8 / view
│   ├── NumericType
│   │   ├── IntegerType / Int8..64Type / UInt8..64Type
│   │   ├── FloatingPointType / Float8..64Type
│   │   └── DecimalType
│   └── TemporalType
│       ├── DateType
│       ├── TimeType
│       ├── TimestampType
│       └── DurationType
├── NestedType
│   ├── ArrayType                     — list[T]
│   ├── MapType                       — dict[K, V]
│   └── StructType                    — fixed-shape record
├── UnionType                         — Union[T, U] / Optional[T]
└── (extension shapes: EnumType, SJsonType, BJsonType, DictionaryType, …)
```

Frozen dataclasses. Per-class singletons on default-arg construction (`IntegerType()` returns the same instance every call). Equality / hash track the dataclass fields; `to_dict()` / `from_dict()` round-trip through a stable `{"id": <DataTypeId>, "name": "…", …}` payload.

### Build a DataType

Five entry points, picked by what you have:

```python
from yggdrasil.data.types.base import DataType
from yggdrasil.data import schema, field

# 1. From a Python annotation — the most common.
DataType.from_pytype(int)                    # IntegerType()
DataType.from_pytype(dt.date)                # DateType()
DataType.from_pytype(list[int])              # ArrayType(item_field=Field("item", IntegerType()))
DataType.from_pytype(dict[str, float])       # MapType(key=String, value=Float)
DataType.from_pytype(MyDataclass)            # StructType(fields=...)

# 2. From an Arrow type — when ingesting an existing schema.
import pyarrow as pa
DataType.from_arrow_type(pa.int64())         # Int64Type()
DataType.from_arrow_type(pa.list_(pa.string()))  # ArrayType(item=String)

# 3. From a Polars / Spark dtype.
import polars as pl
DataType.from_polars_type(pl.Int64)          # Int64Type()

# 4. From a DSL string (the schema-DDL parser).
DataType.from_str("int64?")                  # IntegerType, plus nullable=True at the field layer
DataType.from_str("list<string>")            # ArrayType(item=String)

# 5. From an already-serialised dict (round-trip).
DataType.from_dict({"id": 24, "name": "INT64"})
```

### Round-trip Python hints

`from_pytype` parses; `to_pyhint` emits the hint back.

```python
DataType.from_pytype(int).to_pyhint()           # int
DataType.from_pytype(list[list[str]]).to_pyhint()   # list[list[str]]
DataType.from_pytype(MyDataclass).to_pyhint()   # MyDataclass  (preserved via _pyhint_cache)
```

The round-trip uses two layers:

1. **Cached hint** — `from_pytype` stamps the original parsed hint on the resulting instance via `object.__setattr__(self, "_pyhint_cache", hint)` when reconstruction would lose information (user dataclass, `Enum` subclass, `np.int64` aliases). First-write-wins on the shared singletons so `from_pytype(int)` stamping doesn't corrupt a later `from_pytype(np.int64)`.
2. **Default reconstruction** — `_default_pyhint()` (per subclass) builds a canonical hint from the dtype's own state: `IntegerType` → `int`, `ArrayType` → `list[<child.to_pyhint()>]`, `MapType` → `dict[K, V]`, …

The `_pyhint_cache` slot lives outside the dataclass `fields`, so equality / hash / pickle / `to_dict()` are untouched — same pattern the engine-projection caches (`_to_arrow_cached` etc.) already use.

### Centralised typing utilities

`DataType` owns the canonical Python type-hint resolution helpers — every caller in the codebase (`safe_function`, `cast.registry`, `data_field`) routes through these instead of forking its own `get_origin` / `Annotated` / `NewType` / Optional logic:

```python
# Alias prefix table — editable so third-party engines can register their own.
DataType.PYHINT_ALIASES
# {'pa.': 'pyarrow.', 'pl.': 'polars.', 'pd.': 'pandas.', 'np.': 'numpy.',
#  'ps.': 'pyspark.', 'ddf.': 'dask.dataframe.'}

DataType.expand_alias("pa.Table")               # "pyarrow.Table"

DataType.strip_annotated(Annotated[int, "tag"]) # int
DataType.unwrap_newtype(MyNewType)              # int  (chain of NewType -> base)
DataType.normalize_hint(Annotated[NewType("X", int), "x"])  # int

DataType.unwrap_optional(int | None)            # (True, int)
DataType.unwrap_optional(Union[int, str])       # (False, Union[int, str])

DataType.unwrap_nullable_hint(int | None)       # (int, True)        — Field-flavoured
DataType.unwrap_nullable_hint(Union[int, str, None])
# (Union[int, str], True)                        — multi-arm union keeps its shape

DataType.is_runtime_value(42)                   # True
DataType.is_runtime_value(int)                  # False
DataType.is_runtime_value(list[int])            # False

DataType.resolve_str_annotation("pa.Table")     # pyarrow.Table   (alias expansion)
DataType.resolve_str_annotation("Optional[int]")  # typing.Optional[int]

def f(t: "pa.Table") -> "pa.RecordBatch": ...
DataType.resolve_function_annotations(f)
# {'t': pyarrow.Table, 'return': pyarrow.RecordBatch}
```

`resolve_str_annotation` tries func-globals → typing namespace → alias-prefix expansion + dotted `importlib.import_module`. `resolve_function_annotations` does both passes (`inspect.get_annotations(eval_str=True)` then per-string fallback) so PEP-563 stringified short aliases (`pa.Table`) resolve cleanly even when the function's own globals never imported them.

### Engine projections

Every `DataType` has methods that project into the engines yggdrasil targets:

```python
dt = DataType.from_pytype(int)

dt.to_arrow()          # pa.int64()
dt.to_polars()         # pl.Int64
dt.to_spark()          # pyspark.sql.types.LongType()
dt.to_spark_name()     # "BIGINT"
dt.to_dict()           # {"id": 24, "name": "INT64"}
dt.to_json()           # JSON-string of the dict
dt.to_pyhint()         # int
```

These results are cached on the instance via `_to_arrow_cached` / `_to_polars_cached` / `_to_spark_cached` slots (set with `object.__setattr__` since the dataclass is frozen). The cast hot path hits them constantly; the cache keeps every subsequent call at a single `getattr`.

### Nested children

`ArrayType` / `MapType` / `StructType` carry child `Field`s, not bare dtypes — the field shape captures per-column nullability and metadata that the dtype layer alone can't:

```python
from yggdrasil.data import field
from yggdrasil.data.types import ArrayType, MapType, StructType
from yggdrasil.data.types.primitive import IntegerType, StringType

ArrayType(item_field=field("item", IntegerType()))
MapType.from_key_value(
    key_field=field("key", StringType()),
    value_field=field("value", IntegerType()),
)
StructType(fields=(
    field("id", IntegerType(), nullable=False),
    field("name", StringType(), nullable=True),
))
```

`Type.children` returns the child fields so the structural-walk paths (cast / merge / pretty-print) recurse uniformly.

### Merging schemas

`DataType.merge_with(other, mode=Mode.UPSERT, …)` is the cross-engine schema reconciliation. Same-type merges short-circuit (return `self`); cross-family merges widen up the numeric / temporal lattice or fall back to a safe wider type:

```python
from yggdrasil.data.types.primitive import Int32Type, Int64Type, StringType
from yggdrasil.data.enums import Mode

Int32Type().merge_with(Int64Type())           # Int64Type()  (widen)
Int32Type().merge_with(StringType())          # StringType() (widening to safe carrier)
Int32Type().merge_with(Int64Type(), mode=Mode.STRICT)  # raises CastError on mismatch
```

`Schema.merge_with` calls into the per-field `Field.merge_with` which calls into `DataType.merge_with`, so a `Schema.merge_with` walks the whole shape consistently.

### `UnionType` — `Optional` and `Union` at the DataType layer

When you want to keep the union arms visible at the DataType layer (rather than collapsing `Optional[int]` to `IntegerType()` like the default `from_pytype` Optional shortcut does), construct a `UnionType` explicitly:

```python
from yggdrasil.data.types import UnionType, IntegerType, StringType, NullType

u = UnionType(members=(IntegerType(), NullType()))
u.nullable                  # True  — NullType is in members
u.non_null_members          # (IntegerType(),)
u.to_pyhint()               # Optional[int]
u.to_arrow()                # pa.int64()   (delegates to the single non-null member)
```

The user-visible contract is **`to_field()` flattening** — the bridge between the union-rich DataType layer and the nullable-flat Field layer:

| Union shape | `to_field()` result |
| --- | --- |
| `UnionType(Int, Null)` | `Field(dtype=Int, nullable=True)` — drop Null, **unnest** |
| `UnionType(Int, Str, Null)` | `Field(dtype=UnionType(Int, Str), nullable=True)` — drop Null, keep multi-arm |
| `UnionType(Int, Str)` | `Field(dtype=UnionType(Int, Str), nullable=…)` — no Null in union |
| `UnionType(Null,)` | `Field(dtype=NullType, nullable=True)` |
| `UnionType()` | `Field(dtype=NullType, nullable=True)` |

`NullType` membership is the stronger signal of intent — when it's in the union, `to_field` forces `nullable=True` regardless of the `nullable=` argument the caller passed.

Engine projections delegate to a single member: one non-null arm → that arm's projection; multi-arm non-null → `StringType` (matches the legacy `from_pytype` fallback for mixed unions); zero non-null → `NullType`. A field carrying `UnionType(Int, Null)` therefore produces the same Arrow type as the original `IntegerType()` did — no engine-side disruption.

Serialises via `to_dict()` / `from_dict()`; merges by concatenating + deduplicating member tuples.

---

## `Field` — column descriptor

`Field` adds the per-column shape on top of `DataType`: name, dtype, nullability, metadata, and an optional `parent` link so a nested field can find its enclosing schema or struct. Frozen dataclass with lazy `_arrow_field` / `_polars_field` / `_spark_field` caches.

```python
from yggdrasil.data import field
from yggdrasil.data.types.primitive import IntegerType

f = field(
    name="id",
    dtype=IntegerType(byte_size=8),
    nullable=False,
    metadata={"unit": "rows", "doc": "primary key"},
)

f.name                  # "id"
f.dtype                 # Int64Type()
f.nullable              # False
f.metadata              # {"unit": "rows", "doc": "primary key"}
```

### Build a Field

Same five entry points as `DataType`, but at the field shape:

```python
from yggdrasil.data import Field

Field.from_pytype("id", int, nullable=False)
Field.from_pytype("score", float | None)        # nullable=True (from the Optional)
Field.from_pytype("tags", list[str])
Field.from_pytype("user", MyDataclass)          # struct field, dtype carries MyDataclass

Field.from_arrow_field(pa.field("price", pa.float64(), nullable=False))
Field.from_polars_field(...)
Field.from_spark_field(...)

Field.from_str("id:int64?")             # name="id", dtype=Int64, nullable=True
Field.from_dict({"name": "id", "dtype": {"id": 24, "name": "INT64"}, "nullable": False})

# The polymorphic entry — accepts every shape above plus DataType / Schema / DataFrame / etc.
Field.from_any(some_value, name="id")
```

`from_pytype` calls into `DataType.unwrap_nullable_hint` to split `int | None` into `(int, True)` before delegating to `DataType.from_pytype`, so the `nullable` flag tracks the optionality even though the dtype itself stays `IntegerType`.

### Project a Field

```python
f.to_arrow_field()      # pa.Field("id", pa.int64(), nullable=False, metadata=...)
f.to_polars_field()     # polars field shape
f.to_pyspark_field()    # pyspark.sql.types.StructField
f.to_spark_schema()     # one-field StructType (Spark's flavour)
f.to_dict()             # serialisable
f.to_json()             # JSON-string of the dict
```

The cached engine projections are invalidated on structural mutation (the `parent` cascade in `Schema.with_field` and friends) so re-reading them after a `with_dtype` / `with_nullable` rebuild returns the fresh result.

### Metadata round-tripping

`Field.metadata` is a free-form `dict[str, Any]`. The engine projections push the metadata through the appropriate channel:

| Engine | Metadata channel |
| --- | --- |
| Arrow | `pa.field(metadata={...})` — string-keyed bytes |
| Polars | The schema's column-level metadata when the Polars version supports it |
| Spark | `StructField.metadata` (JSON-encoded) — yggdrasil dumps `type_json` for the engine types Spark can't represent natively (Map, Array of Struct, etc.) |

The `from_*_field` ingestion strips yggdrasil-side hints (`name`, `nullable`, `type_json`) out of the metadata so a `from_X → to_X` round trip is metadata-stable.

### Mutating a Field

`Field` is frozen but exposes `with_X` / `with_Y` builders that return a fresh instance:

```python
f.with_name("user_id")
f.with_nullable(True)
f.with_dtype(StringType())
f.with_metadata({"doc": "FK to users.id"})
f.with_metadata({"updated": True}, merge=True)
```

In-place mutation is available via `f.set_metadata(...)`, but the `with_*` form is preferred — the frozen contract holds and the engine caches invalidate cleanly.

---

## `Schema` — ordered field list

`Schema` is an ordered, named tuple of fields. The same metadata / projection / merge contract as `Field`, applied at the row shape.

```python
from yggdrasil.data import schema, field
from yggdrasil.data.types.primitive import IntegerType, StringType, Float64Type

s = schema([
    field("id",    IntegerType(), nullable=False),
    field("name",  StringType()),
    field("score", Float64Type()),
])

s.names                 # ("id", "name", "score")
len(s)                  # 3
s.field("name")         # Field(name='name', dtype=String, nullable=True)
s.field(index=0)        # Field(name='id', ...)

s.to_arrow_schema()
s.to_polars_schema()
s.to_spark_schema()
s.to_dict()
```

### Build a Schema

```python
Schema.from_pytype(MyDataclass)
Schema.from_arrow_schema(pa.schema(...))
Schema.from_polars_schema(...)
Schema.from_spark_schema(...)
Schema.from_str("id:int64?, name:string, score:float64")
Schema.from_fields([...])
Schema.from_any(...)        # polymorphic — DataFrame / dict / iterable of fields
```

### Merge

```python
a = Schema.from_str("id:int64, name:string")
b = Schema.from_str("id:int64, age:int32")
a.merge_with(b)
# Schema(id:int64, name:string?, age:int32?)
# — union of names, widen any mismatched dtypes, widen nullability
```

`mode=Mode.APPEND` widens nullability for any field missing on the other side; `mode=Mode.STRICT` raises when shapes don't match exactly.

---

## `convert` and the cast registry

`convert(value, target_hint, options=…, **kwargs)` is the single dispatch surface every cast in yggdrasil goes through. The path is in `yggdrasil/data/cast/registry.py`.

```python
from yggdrasil.data.cast.registry import convert, register_converter, find_converter

convert("42", int)                  # 42                  — registered str->int
convert(42, int)                    # 42                  — identity (~140 ns)
convert("2024-06-01", dt.date)      # datetime.date(2024, 6, 1)
convert({"id": 1, "name": "x"}, MyDataclass)
convert([1, 2, 3], list[str])       # ['1', '2', '3']     — generic container
convert(record_batch, pl.DataFrame) # zero-copy Arrow→Polars bridge
```

### Dispatch order

`convert` (the user-facing call):

1. **`Any` / `object` target** — identity passthrough (96 ns).
2. **Plain-type identity** — `isinstance(target_hint, type)` and `isinstance(value, target_hint)` → identity, no `unwrap_optional` call (140 ns).
3. **`Optional[T]` unwrap** — only for generic-alias targets or None values.
4. **`None` handling** — `None` if optional, else `default_scalar(target)`.
5. **Registry lookup** — `find_converter(type(value), target)` followed by the converter call.
6. **`Enum` / `dataclass`** — value-to-enum-member resolution; dict-to-dataclass coercion.
7. **Container generics** — `list[T]` / `set[T]` / `tuple[A, B]` / `dict[K, V]` / `Mapping`.
8. **`TypeError`** — no path found.

`find_converter` (the registry lookup):

1. **Exact** — `_registry[(from_type, to_hint)]`. Cached.
2. **Identity** — `from_type == to_hint` or target is `Any` / `object`.
3. **`Any → to_hint` wildcard** — `_any_registry[to_hint]`.
4. **Namespace late-import** — when either type lives in a `polars` / `pandas` / `pyspark` / `pyarrow` namespace, import the matching `yggdrasil.<engine>.cast` module to trigger its registrations, then retry without the namespace probe.
5. **MRO cross-product** — walk `iter_mro(from_type) × iter_mro(to_hint)` looking for any registered pair.
6. **`issubclass` scan** — final pass for odd registered keys (protocols, generic-alias keys, …).

**No auto-composition.** The registry refuses to chain two registered hops (`X → Y → int`) into a synthetic direct cast. The intermediate type used to depend on the order of unrelated registrations, which masked missing direct converters and made every cast load-bearing on the global registration order. Register the direct `X → int` converter when you need it.

### Register a new converter

```python
from decimal import Decimal
from yggdrasil.data.cast.registry import register_converter, convert

@register_converter(str, Decimal)
def _str_to_decimal(value: str, options=None) -> Decimal:
    return Decimal(value.replace(",", "."))

convert("19,95", Decimal)   # Decimal('19.95')
```

Tips:

- The decorator slot is `(from_hint, to_hint)`. Both can be a real type, `Any` (wildcard source), or `object`.
- The converter signature is `(value, options) -> result`. `options` is a `CastOptions` instance or `None`.
- Stay idempotent — `convert(value, X)` on a value that already matches `X` should pass through. The identity check at convert-entry handles the common case; only odd subclass relationships need a defensive isinstance check in the body.
- Register the cross-engine paths next to the existing engine module (e.g. `yggdrasil.polars.cast`) so a single `import yggdrasil.polars.cast` light up every related conversion.

### `CastOptions` — the normalised options carrier

`CastOptions` threads through every cast helper. It holds the target schema, the source hints (when known), the safety / strictness / memory flags, and the row / byte chunk sizes for streaming converters. It's the single options carrier — don't invent a parallel per-call options object.

```python
import pyarrow as pa
from yggdrasil.data.cast.options import CastOptions

opts = CastOptions(
    target=pa.schema([
        pa.field("id",    pa.int64(),   nullable=False),
        pa.field("price", pa.float64(), nullable=False),
    ]),
    safe=True,
    byte_size=128 * 1024 * 1024,
    row_size=10_000,
    strict_match_names=True,
)
```

`CastOptions.check(options, **kwargs)` is the polymorphic normaliser — accepts `None`, a bare `pa.Schema` / `pa.Field` / `pa.DataType` (lifts into a `CastOptions` with `target=` set), or an existing `CastOptions` (returned as-is when no overrides apply).

```python
def my_helper(source, options=None, *, target_field=None):
    options = CastOptions.check(options, target=target_field)
    ...
```

### Engine bridges

Each engine module registers its converters **on import**:

| Module | Helpers |
| --- | --- |
| `yggdrasil.arrow.cast` | `any_to_arrow_table`, `cast_arrow_tabular`, `cast_arrow_record_batch_reader`, `rechunk_arrow_batches` |
| `yggdrasil.polars.cast` | `cast_polars_dataframe`, `cast_polars_lazyframe`, `polars_dataframe_to_arrow_table` |
| `yggdrasil.pandas.cast` | `cast_pandas_dataframe` |
| `yggdrasil.spark.cast` | `cast_spark_dataframe`, `any_to_spark_dataframe`, `spark_dataframe_to_arrow` |

Always reach the optional engines via `yggdrasil.lazy_imports` so base installs stay functional:

```python
from yggdrasil.lazy_imports import polars   # correct
import polars                             # wrong — breaks base installs
```

### Performance shape

Steady-state numbers from `benchmarks/data/bench_registry.py` (3-repeat best, x86_64):

| Path | Time | Notes |
| --- | --- | --- |
| `convert(42, int)` identity | ~140 ns | The dominant per-row-coercion shape |
| `convert(value, Any)` passthrough | ~96 ns | Cheapest exit |
| `convert("42", int)` | ~725 ns | One registered converter call |
| `convert("2024-06-01", dt.date)` | ~1.7 µs | ISO-string parse |
| `convert(dict, MyDataclass)` | ~31 µs | Field-by-field coercion |
| `convert(pa.RecordBatch, pa.Table)` | ~42 µs | `Table.from_batches` |
| `convert(pa.RecordBatch, pl.DataFrame)` | ~163 µs | Zero-copy Arrow→Polars |
| `find_converter` cache hit | ~170 ns | Tuple cache key + dict lookup |
| `find_converter` exact cold | ~410 ns | First-call after process start |
| `find_converter` MRO cold | ~8.2 µs | Worst case (no exact match) |

The benchmark is checked-in; quote before/after numbers when changing the registry hot path.

---

## Putting it together — a typed `Dataset.apply`

```python
import dataclasses
from yggdrasil.spark.tabular import Dataset
from yggdrasil.data import field, schema
from yggdrasil.data.types.primitive import Int64Type, StringType

@dataclasses.dataclass
class Row:
    id: int
    label: str

# Build a schema from a dataclass — DataType walks the annotations.
out_schema = schema(Row)

# Distribute a user function over the cluster.
def make_row(id: int, label: str) -> Row:
    return Row(id=id, label=label.upper())

result = Dataset.parallelize(
    make_row,
    inputs=[{"id": 1, "label": "a"}, {"id": 2, "label": "b"}],
    schema=out_schema,
    spark_session=spark,
)
```

What happened:

1. `schema(Row)` walked `Row.__annotations__` via `DataType.from_pytype` for each field, capturing types and nullability.
2. `Dataset.parallelize` pickled `make_row` and shipped it (with the closure modules) to the executors.
3. On each executor, `build_row_invoker(make_row)` ran once per partition. It inspected `make_row`'s signature, saw `(id: int, label: str)`, and built a per-row callable that spreads dict rows as kwargs with type coercion via `convert()`.
4. The function returned `Row` instances; the typed-cast pipeline cast the outputs against `out_schema` via the registered `dataclass → Arrow` converter.
5. The result is a `Dataset` whose underlying Spark frame matches `out_schema`.

The same shape works for `def f(batch: pa.RecordBatch)` (whole-batch dispatch through `build_batch_invoker`), `def f(value: int)` over a column with a name-matching arg (vectorised column cast via `pa.compute.cast`), or `def f(row)` for the plain "give me whatever you have" shape.

See [Casting](casting.md) for the tabular round-trip surface and the engine bridges.
