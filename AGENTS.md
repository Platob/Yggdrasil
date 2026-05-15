# AGENTS.md

## Mission

Yggdrasil exists to help users **integrate data systems fast, cleanly, and with less glue code**.

When working in this repo, optimize in this order:

1. **User integration success**
2. **Helpful behavior and recovery**
3. **Cross-engine compatibility**
4. **Performance**
5. **Internal neatness**

If there is a tradeoff, prefer the implementation that makes the library easier to use correctly and easier to recover from when the user gets it wrong.

---

## Core rule

This repo is not just about converting types.
It is about making Arrow, Polars, pandas, Spark, Databricks, Python objects, schemas, HTTP payloads, and serialized buffers work together without the user writing a pile of annoying custom code.

A good change should do at least one of these:

- reduce boilerplate for the caller
- preserve more schema intent across boundaries
- improve debugging and recovery
- make a common workflow faster without changing semantics
- make the public API easier to discover and harder to misuse

If it does none of those, it is probably not worth adding.

### Integrate, don't invent

Before writing anything new, find where the behavior already lives and extend it.
A patch that grows an existing module, registers into the existing registry, or adds a field to an existing dataclass is almost always better than a new parallel module, helper, or option object.

Default to:

- modifying the function or class that already does 80% of the job
- adding the converter to `data/cast/registry.py` (or the engine `cast.py` that registers on import) instead of writing a one-off conversion at the call site
- adding a flag to `CastOptions` instead of inventing a new options object
- adding a method to `DataField` / `Schema` / `DataTable` / `StatementResult` instead of writing a sibling helper module
- using `HTTPSession` / `PreparedRequest` / `Response` instead of bypassing the modern HTTP stack
- using `lib.py` guards already in the subsystem instead of writing a new guard pattern

Only create a new module, class, or abstraction when the existing surface genuinely cannot host the behavior — and when you do, wire it back into the existing surface so the next caller finds it through the canonical path.

Red flags that you are inventing instead of integrating:

- a new helper that duplicates 90% of an existing one with one extra branch
- a new options dataclass that overlaps with `CastOptions`
- conversion logic added at a call site that should have been a registered converter
- a private utility that re-implements something already in `pyutils/` or `data/`
- a "v2" of something with no migration path from "v1"

### Reach for `yggdrasil.data.enums` first

Whenever a value belongs to a fixed token set — currencies, time units, byte sizes (`ByteUnit`), media types, MIME types, modes, codecs, geozones, timezones — go through the enum at the API boundary. Use a member directly (`128 * ByteUnit.MIB`) or coerce with the enum's `from_(...)` / `parse_size(...)` / `parse(...)` classmethod. The enum is the single place that knows the canonical token, the accepted aliases, and how to format the value back out.

Default to:

- using `ByteUnit.parse_size(x)` instead of `int(x) * 1024 * 1024` or a regex inline
- using `TimeUnit.from_(x)` instead of a local `{"us": ..., "micros": ...}` dict
- using `Currency` / `MediaType` / `MimeType` / `Mode` / `Codec` / `Geozone` / `Timezone` members and their `from_` classmethods instead of bare strings at call sites
- subclassing `int` (`IntEnum`) or `str` (`StrEnum`) on new enums so members slot into existing typed fields without a cast

If the enum is missing a member or alias the code legitimately needs, **add it to the enum** and let the caller route through it. Do not branch around the enum with a one-off lookup, and do not scatter string constants for what is conceptually a new fixed-vocabulary concept — add a new enum to `yggdrasil.data.enums` (matching the `from_` / `parse*` / `is_valid` shape of the existing ones) instead.

Red flags that you are bypassing the enum layer:

- inline alias tables / regexes for tokens an enum already canonicalizes
- expressions like `n * 1024 * 1024 * 1024` repeated across modules instead of `n * ByteUnit.GIB`
- accepting a string at the boundary, normalizing it ad-hoc, then never round-tripping it
- string constants that mean the same thing across files but are typed out separately

---

## Tone and style rules

### General writing vibe
Use clear, direct, modern language.
Do not write stiff corporate robot text.
Do not write cringe meme sludge either.

Target tone:
- blunt but helpful
- confident but not arrogant
- practical, not academic
- reads like a sharp engineer explaining the fix, not a legal disclaimer

### Comments must sound human
Comments should help the next engineer understand:
- what weird edge case exists
- why the branch exists
- what invariant is being protected
- what the caller gets out of it

Comments should **not** narrate obvious syntax.

Good:
```python
# PyArrow can choke on repeat(size=0) on some versions, so return a typed empty array instead.
```

Better, in repo vibe:
```python
# Yeah, this looks extra, but PyArrow can throw on repeat(size=0).
# Return a typed empty array here so callers do not get random nonsense.
```

Bad:
```python
# create empty array
```

### Error messages should help, not just complain
Error messages should be explicit and useful.
They should usually include:
- what was passed
- what was expected
- what valid values or next steps exist
- suggestions for likely typos or nearby matches when practical

Good:
```python
raise KeyError(
    "No field named 'prcie'. Available fields: ['price', 'volume']. "
    "Did you mean 'price'? Use .field_names() to inspect valid names."
)
```

Bad:
```python
raise KeyError("field not found")
```

### Gen Z style guidance for comments and errors
Yes, you can use a Gen Z tone, but do it with restraint.
The library still needs to feel professional and stable.

Allowed vibe:
- direct
- slightly punchy
- lightly conversational
- small bits of personality

Not allowed:
- slang spam
- forced jokes
- meme formatting
- sarcasm that hides the actual fix
- anything that makes logs or exceptions harder to parse

Good examples:
```python
# Databricks type payloads are messy in the wild, so normalize hard here.
# Better we absorb the chaos than force every caller to do cleanup first.
```

```python
raise ValueError(
    "Conflicting field names: 'foo' and 'bar'. Pick one. "
    "If you meant the same field, pass matching values."
)
```

```python
raise TypeError(
    "Unsupported Databricks SQL type: 'ARRAY<NOPE>'. "
    "Valid nested forms are ARRAY<type>, MAP<key, value>, and STRUCT<...>."
)
```

Not good:
```python
raise ValueError("nah this is busted fr")
```

```python
# no cap pyarrow is tweaking 💀
```

Rule: **sound sharp, not unserious**.

---

## What this project is

Yggdrasil is a **schema-aware data interchange library**.

- PyPI package: `ygg`
- import package: `yggdrasil`

It converts and normalizes:
- Python values and type hints
- Apache Arrow arrays, fields, schemas, tables
- Polars types, series, expressions, frames
- pandas dtypes and objects
- Spark SQL types and DataFrames
- Databricks SQL and API type payloads
- HTTP request and response payloads
- serialized binary and structured formats

The point is to make all of that feel like one coherent system.

---

## Architecture you need to understand first

### Reach for `yggdrasil.data` first
`yggdrasil.data` is the canonical surface for describing and moving frames.
If the task touches a dataframe, schema, field, type, or cross-engine conversion, start here and drop down to `polars` / `pandas` / `pyspark` / `pyarrow` only when the abstraction genuinely does not cover the case.

Prefer, in this order:
- `DataField` / `Field` and `Schema` (names, nullability, metadata, tags, nested structure, engine dtype intent in one place). Use `Field.from_pandas`, `Field.from_polars`, `Field.from_arrow`, `Schema.from_fields` instead of re-building per-engine schemas by hand.
- `DataType` / `DataTypeId` and the `types/` submodules (`primitive`, `nested`, `iso`, `extensions`) for type hints. Don't hand-roll `pa.int64()` / `pl.Int64` / `"bigint"` when a `DataType` produces all of them.
- `DataTable` and `StatementResult` for "execute, then move rows" integrations. New backends should implement these rather than invent a parallel surface.
- `yggdrasil.data.cast.convert(value, target, options=...)` with `CastOptions` for value conversion. Do not call `df.to_pandas()`, `pl.from_arrow(...)`, `spark.createDataFrame(...)` directly from feature code if a registered converter already handles it.
- `yggdrasil.data.enums` (currency, geozone, timezone) for normalized domain values — don't re-parse those strings inline.

Only drop past this layer to add a new converter, write engine-internal glue in `arrow/` / `polars/` / `pandas/` / `spark/`, or implement performance-critical code the abstraction intentionally doesn't cover. When you do, register the new behavior back into `yggdrasil.data` (converter + `DataType` + `Field`/`Schema` support) so the next caller gets it for free.

### Converter registry is the core pattern
Everything flows through:
- `yggdrasil/data/cast/registry.py`

Converters are registered with:
```python
@register_converter(from_hint, to_hint)
```

Dispatch order:
1. exact match
2. identity
3. `Any` wildcard
4. MRO fallback
5. one-hop composition

Before adding conversion code, decide whether it belongs in:
- the registry
- an engine-specific converter module
- a normalization helper used by multiple converters

Do not bolt on random ad hoc conversion logic when it should live in the registry.

### Engine modules register on import
These modules register their own converters on import:
- `arrow/cast.py`
- `polars/cast.py`
- `pandas/cast.py`
- `spark/cast.py`

If a conversion seems like it should exist but does not fire, check import/registration first.

### `CastOptions` is the shared options object
`CastOptions` lives in:
- `data/cast/options.py`

Use it to normalize:
- source hints
- safety behavior
- memory behavior
- conversion options
- schema hints

Do not invent parallel one-off option patterns unless there is a real reason.

### `lib.py` is the optional dependency guard
Every optional-dependency subsystem should import through its `lib.py` guard when that pattern exists.

Correct:
```python
from yggdrasil.polars.lib import polars
```

Wrong:
```python
import polars
```

Base installs must stay lightweight.
Optional dependencies must remain optional.

### `io/` is the real integration boundary
The `io/` package is where integration gets real:
- requests
- responses
- URLs
- buffers
- media types
- codecs
- sessions
- pagination
- caching

Preferred HTTP client for new work:
- `io/http_/session.py` → `HTTPSession`

Use the modern stack unless you have a strong reason not to.

---

## Repo map

### Core conversion and schema
- `data/cast/` — registry, cast options, core converters
- `data/enums/` — enums and normalization helpers
- `arrow/` — Arrow inference and Arrow conversions
- `polars/` — Polars integration
- `pandas/` — pandas integration
- `spark/` — Spark integration
- `dataclasses/` — dataclass helpers and config/cache utilities

### IO and transport
- `io/` — sessions, requests, responses, buffers, codecs, media IO
- `web/` — proxy-related code

### Platform integrations
- `databricks/` — SQL, jobs, clusters, IAM, secrets, account services
- `mongoengine/` — MongoEngine integration
- `fastapi/` — optional API layer

### AI and SQL
- `ai/` — OpenAI-backed sessions and SQL generation

### Serialization
- `pickle/` — custom serialization system
- `blake3/`, `xxhash/` — guarded hash wrappers

### Utilities
- `pyutils/` — retry, parallelization, helper functions
- `environ/` — runtime import/install logic

---

## Engineering priorities

### 1. Make the common path easy
If a user already has a value, schema, field, type string, or framework object, the library should do its best to accept it.

Support practical caller inputs when reasonable:
- Python type hints
- strings
- dict payloads
- Arrow fields and schemas
- Polars and pandas objects
- Spark types
- Databricks `type_text` payloads
- objects with `.schema`, `.type`, `.arrow_schema`, or equivalent

The caller should not need a pre-normalization ritual just to use the library.

### 2. Help the user recover fast
When something fails, do not just reject the input.
Explain:
- what failed
- why it failed
- what values are valid
- what to try next

For lookup-style APIs, suggest valid names or indices.
For parsing APIs, show the unsupported fragment.
For conversion APIs, mention what source and target were involved.

### 3. Preserve the contract surface
Arrow schema and field semantics are part of the user contract.
Preserve when possible:
- field names
- field order
- nullability
- metadata
- tags
- nested structure
- precision/scale
- timezone intent

Do not silently drop meaningful schema information unless the API explicitly documents that loss.

### 4. Prefer ergonomics over abstraction purity
If a small convenience change saves the caller a bunch of repeated glue code, it is usually a good trade.

Good:
```python
field_by("price")
field_by(0)
field_by(name="price", raise_error=False)
field_by(index=0)
```

Bad:
- forcing the caller to pre-convert everything into one exact shape
- making them know internal module boundaries to use public APIs
- surfacing raw low-level exceptions with zero context

### 5. Performance matters, but not at the cost of semantics
Prefer:
- Arrow-native operations
- Polars lazy expressions
- vectorized work
- chunk-preserving operations
- zero-copy or near-zero-copy flows

But correctness and user-visible behavior come first.

#### Benchmark-driven optimization — measure, don't guess

Performance changes go through `python/benchmarks/` — not through
intuition.

The benches are organized to mirror the source tree
(`benchmarks/data/`, `benchmarks/io/primitive/`,
`benchmarks/io/path/`, …). Each module that ships a hot path has
a paired `bench_<name>.py` that exercises the public surface.

The rule for any "I think this is slow" or "this would be faster"
change:

1. **Find or add the bench first.** If the operation you want to
   optimize isn't already covered, write the scenario into the
   matching `benchmarks/<module>/bench_<name>.py`. The bench must
   exercise the exact call shape callers hit — not a
   reduced-to-absurdity micro that doesn't represent real use.
2. **Capture the baseline.** Run `python benchmarks/<file>.py
   --repeat 5` against `main` (or whatever you're branching from)
   and save the numbers somewhere — commit message, PR body,
   scratchpad. Without a baseline you can't tell whether your
   change helped, regressed, or did nothing.
3. **Apply the change.** Keep it minimal. One conceptual change at
   a time so the before/after diff is obvious.
4. **Re-run the same bench.** Same `--repeat`, same machine, same
   process — quote the new `best` and `median` against the
   baseline. A change that improves `mean` while regressing
   `best` is usually a thermal artefact, not a real win.
5. **Validate the rest of the suite didn't regress.** Run
   `python benchmarks/run_all.py --repeat 3` (the full sweep, a
   few minutes) and skim the output. Optimizations frequently
   trade off — speeding up one path can slow down another. Show
   the relevant numbers in the PR / commit.
6. **Document the win.** Mention the before / after in the commit
   message body. Future readers (and reviewers) need to know
   which numbers are load-bearing.

Skipping the bench step is the most common way good intentions
land regressions. "Felt faster" doesn't ship — quoted numbers do.

When the bench is missing for a hot path, **adding the bench
counts as part of the change**. It's how the next contributor
will find their regression.

### 6. Never loop over data rows in Python

Python `for` loops over Arrow / Polars / pandas / Spark values are the
single biggest performance trap in this codebase. A loop over 1 M rows
is 1 M Python frames; the same work in a vectorised kernel runs in
microseconds.

Reach for vectorised primitives in this order **before** considering a
Python loop:

1. **pyarrow.compute kernels** — `pc.cast`, `pc.if_else`,
   `pc.list_element`, `pc.binary_join_element_wise`,
   `pc.replace_substring_regex`, `pc.fill_null`, `pc.equal`,
   `pc.is_in`, `pc.greater`, ... These stay inside the C++ runtime
   and don't cross a Python frame per row.
2. **pyarrow.json / pyarrow.csv** vectorised readers when the
   workload is "decode N rows of `<format>` into a typed Arrow
   array". See `_cast_json.py:_vectorized_parse_json` for the
   one-pass NDJSON shape.
3. **Polars expressions** — `pl.col(...).str.json_decode`,
   `.str.contains`, `.cast`, `.list.eval`,
   `pl.when(...).then(...).otherwise(...)`. The next-best vectorised
   fallback when pyarrow.compute doesn't cover the op. Build a
   polars Series / LazyFrame, run the expression, hand the result
   back to Arrow via `.to_arrow()`.
4. **Numpy ufuncs** for numerical work on numpy-backed buffers.

Only after those exhaust their coverage is a Python loop acceptable,
and even then **only as an explicitly-documented fallback path** —
permissive per-row null-on-failure, or the rare case where pyarrow.json
genuinely can't emit the target type (e.g. `map` arrays). Comment why
the vectorised path doesn't cover the case and what the fallback's
cost is.

Same rule applies to `array.to_pylist()` followed by a Python
comprehension — that's the same trap with one less line. Look for a
vectorised expression first; reach for `to_pylist` only when
materialising into Python objects is the genuine endpoint (e.g. JSON
encoding via `json.dumps`).

The shape of an acceptable per-row fallback already exists at
`_cast_json.py:_parse_via_python` — vectorised C++ NDJSON first,
per-row only when that raises in permissive mode. Mirror that
pattern; don't invent a new one.

### 6a. Never materialise heavy data via `to_pylist` / `to_list` / `tolist`

`pa.Array.to_pylist()`, `pl.Series.to_list()`, and
`pd.Series.tolist()` / `np.ndarray.tolist()` all walk every cell into
a Python object — same per-row cost as a Python `for` loop, just
hidden inside C code. **Do not use them in any data-shaped path**
(cast helpers, frame converters, batch readers, type inference, hot
inner loops). The fact that pyarrow's `to_pylist` runs in C is not a
licence: building 1 M Python ints / dicts / lists allocates 1 M
PyObjects, and the GC churn alone dominates the next operation.

When you need to leave Arrow / Polars / pandas, ride the engine's own
zero-copy bridge instead:

| Need                                       | Use this                              |
| ------------------------------------------ | ------------------------------------- |
| Arrow → pandas Series                      | `Array.to_pandas()` (works for nested types — struct cells surface as dicts, list cells as numpy arrays, no per-row hop) |
| Arrow → polars Series                      | `pl.from_arrow(arr)` (zero-copy)      |
| Arrow → numpy (numeric only)               | `Array.to_numpy(zero_copy_only=True)` |
| Pandas → Arrow                             | `pa.array(series, from_pandas=True)`  |
| Polars → Arrow                             | `series.to_arrow()` (zero-copy)       |
| Pandas → Polars                            | `pl.from_pandas(df)`                  |
| Build a struct from per-child arrays       | `pa.StructArray.from_arrays(arrs, names=..., mask=...).to_pandas()` — one C-bridge pass |

Limited exemptions — and only these — keep `to_pylist` / `to_list`:

1. **Documented per-row fallback** for shapes vectorised engines
   genuinely can't emit (e.g. the permissive `_cast_json.py` path that
   runs after `pyarrow.json.read_json` rejects the row). Comment why
   no vectorised path applies and gate it behind the strict/permissive
   knob.
2. **Genuine row endpoint** — the workload IS "yield one Python
   row to a downstream sink": ndjson writers, kafka producers,
   mongo inserts, JSON HTTP responses, xlsx row writers,
   `Tabular.to_pylist` / `iter_pylist` API methods, pickle one-shot
   serialization. The cost is paid for the user, not in a hot
   transform pipeline.
3. **Diagnostics-only paths** — test assertion formatters, debug
   logging, `repr` output. These run at human scale (KB-MB), never
   on workload data.

If a new call site doesn't fit one of those three buckets, find the
vectorised form first. Same goes for `series.tolist()` /
`df.values.tolist()` / numpy `arr.tolist()` — those are the same
trap dressed in different syntax.

---

## Public API design rules

### Be forgiving on input, strict on meaning
It is good to accept multiple equivalent input shapes.
It is not good to silently accept conflicting arguments.

If arguments conflict, fail explicitly.

Good:
```python
raise ValueError(
    "Conflicting field names: 'foo' and 'bar'. "
    "Pass only one name source, or pass matching values."
)
```

### Convenience is good when it stays predictable
It is fine to support a convenience positional argument like `name_or_index` if:
- explicit keyword overrides still exist
- conflicts are detected
- return behavior stays clear
- type hints stay honest

### Type hints must match runtime truth
If a method can return `None`, annotate `| None`.
Do not fake stricter types just because the happy path returns a value.

### Keyword-only arguments are good for ambiguous options
Use keyword-only arguments when they make calls easier to read and harder to misuse.

---

## Rules for comments

### Comment the weirdness, not the syntax
Focus comments on:
- version quirks
- engine edge cases
- schema invariants
- user-facing behavior
- compatibility hacks

Good:
```python
# Spark schema payloads can arrive half-normalized, so accept both the raw type
# and the wrapper object here. Better this helper deals with it once.
```

Good:
```python
# Keep metadata intact here. Losing it would make downstream schema diffs useless.
```

Bad:
```python
# loop through fields
for field in fields:
```

### Comment style vibe
Keep it sharp and readable.
You can sound a little conversational, but do not turn the repo into a meme graveyard.

Examples of acceptable vibe:
```python
# This branch looks redundant, but it saves callers from a super annoying Arrow edge case.
```

```python
# Databricks payloads are not always clean, so normalize aggressively here.
```

Examples of not acceptable vibe:
```python
# bro this thing is cooked
```

```python
# yolo
```

---

## Rules for error messages

### Every good error should try to answer these
- what did you pass?
- what did the library expect?
- what can you pass instead?
- what should you do next?

### Preferred error shape
Use explicit, practical error messages.

Good:
```python
raise KeyError(
    "No field named 'prcie'. Available fields: ['price', 'volume']. "
    "Did you mean 'price'? Use .field_names() to inspect valid names."
)
```

Good:
```python
raise TypeError(
    "Unsupported Arrow data type: dense_union<...>. "
    "This path currently supports primitive, list, map, and struct-like types."
)
```

Good:
```python
raise ValueError(
    "No field found at index 5. Valid index range: 0..2. "
    "Use .field_names() to inspect available fields."
)
```

### Gen Z style for errors
Allowed:
- direct wording
- short punchy sentence fragments when still clear
- slightly more human phrasing than default enterprise sludge

Examples:
```python
raise ValueError(
    "Conflicting field indexes: 1 and 2. Pick one. "
    "If both point to the same thing in your code, normalize before calling field_by()."
)
```

```python
raise TypeError(
    "Unsupported Databricks SQL type: 'STRUCT<foo BAR>'. "
    "This parser is strict on nested syntax because guessing here would be messy."
)
```

Not allowed:
- meme slang
- emoji
- inside jokes
- sarcasm instead of guidance
- error text that sounds funny but hides the fix

Rule: the message can have attitude, but it still needs to be instantly useful in logs.

---

## Logging guidance

### Prefer `%r` self-representation in logs with explicit simple texts
Format objects through their `__repr__` (`%r`) — not `%s` / `str(...)` —
in every `LOGGER.debug` / `LOGGER.info` / `LOGGER.warning` / `LOGGER.error`
call. The `__repr__` of our long-lived objects (`DatabricksClient`,
`DatabricksPath`, `Volume`, `Schema`, `Table`, `Session`, `URL`,
`HTTPSession`, etc.) carries the full identity (`<VolumePath dbfs+volume://…>`,
`DatabricksClient(host='…', auth_type='pat')`) — `str(obj)` often
collapses to a bare path or hostname, which is ambiguous when two
similar objects are logged in the same line.

The text around the `%r` should be a short, explicit, scannable English
description — say what happened in plain words, not a one-token
shorthand. The reader of `journalctl` shouldn't have to grep the source
to figure out which call emitted the line.

Good:
```python
LOGGER.info("Created temp volume path %r (cleanup in %ds)", path, ttl)
LOGGER.warning("Permission denied listing directory %r: %r", self, exc)
LOGGER.debug("Cached config snapshot for client %r — host=%r", client, host)
```

Bad:
```python
LOGGER.info("path=%s ttl=%d", path, ttl)              # bare %s drops the type tag
LOGGER.warning("perm denied %s %s", self, exc)        # cryptic, no English
LOGGER.debug("config cached")                          # which client? where?
```

Two practical exceptions:
- The `%(asctime)s` / `%(levelname)s` / `%(name)s` slots in the
  `logging` formatter are interpolated by the logging framework with
  primitive types — keep `%s` there.
- When the value is *already* a primitive (`int`, `bool`, a URL string
  that's part of the message structure), `%s` reads more naturally
  than `%r`'s redundant quotes (`size=1234` not `size='1234'`).

Lazy interpolation (`LOGGER.debug("foo %r", obj)`, not
`LOGGER.debug(f"foo {obj!r}")`) is still the rule — the formatter
only renders when the level is enabled, and `repr()` on a heavy
object is exactly the kind of work the level guard exists to avoid.

---

## Optional dependency policy

Optional dependencies must remain optional.

### Rules
- base `ygg` install must work without non-core engines
- guarded imports must remain guarded
- missing dependency errors should say which package/feature is required

If a subsystem uses `lib.py`, follow that pattern.
If it uses an inline runtime guard, stay consistent with that subsystem.

---

## Conversion design rules

Before adding or changing a conversion, answer these:

1. Is this a value conversion, schema conversion, or both?
2. Should this live in the registry?
3. What schema intent must survive?
4. Is the behavior symmetric or intentionally one-way?
5. What happens if the target engine is unavailable?
6. What does the failure message say?

Good conversion work:
- fits the registry model
- preserves names/nullability/metadata where possible
- handles nested structures correctly
- has realistic tests for success and failure
- does not force the caller to know internals

Red flags:
- conversion only works after some non-obvious import side effect
- conversion silently drops metadata
- native path and Python fallback differ semantically
- raw low-level exceptions leak straight to users with no context

---

## HTTP / IO guidance

When working in `io/`:

### Prefer the modern HTTP stack
Use:
- `HTTPSession`
- `PreparedRequest`
- `Response`
- `SendConfig` / `SendManyConfig`

### Preserve observability
Requests and responses should stay easy to inspect and debug.
Preserve things like:
- normalized URL parts
- promoted headers
- remaining headers maps
- body bytes
- payload hashes
- timestamps
- status and timing fields

### Treat buffers as core infrastructure
`BytesIO` and `MediaIO` are not side helpers.
They are central to real interop.
Changes here must preserve:
- spill-to-disk behavior
- codec behavior
- cursor safety
- Arrow / Parquet / JSON / IPC compatibility
- typed IO semantics

### Fail fast on remote resources, retry the real call — don't pre-check
Every call to a remote resource (HTTP, Databricks Files / SQL / Workspace,
S3, MongoDB, Spark cluster, …) costs latency, quota, and a chance to fail
on its own. Treat them as expensive.

Rules:
- **Do the operation, handle the error.** Don't gate a `download` /
  `upload` / `delete` / `read_bytes` on a preceding `exists()` /
  `stat()` / `get_metadata()` / `HEAD` probe. The probe doubles the
  latency, races against concurrent writers, and lies under eventual
  consistency. Catch `NotFound` / `404` / `FileNotFoundError` from the
  real call instead.
- **One round trip per intent.** If you need both "does this exist" and
  the bytes, just fetch the bytes and treat `NotFound` as "no". If you
  need both stat and listing, list and read the entry — don't stat each
  child. Use `_call_ensuring_parents` (or its equivalent) to lazily
  recover from missing-parent errors instead of `mkdir -p` on every
  write.
- **Retry is for transience, not correctness.** Wrap the operation in
  the existing retry policy (`retry_sdk_call`, `_call`, the HTTP send
  retry config) to absorb 5xx / throttling / connect timeouts. Do not
  use a retry loop to paper over a precondition you could just have
  let the server enforce.
- **Fail fast on deterministic errors.** `NotFound`, `AlreadyExists`,
  `PermissionDenied`, `BadRequest` with a stable message — propagate
  immediately. Retrying these wastes the user's time and burns quota.
- **Cache only what's safe to cache.** Stat / metadata results have a
  TTL (see `RemotePath._STAT_CACHE`); reuse them across the same op
  instead of re-issuing the call. Invalidate after mutations
  (`_invalidate_stat_cache`).
- **No probe-then-act loops.** `if path.exists(): path.read_bytes()` is
  a bug pattern, not a safety net — between the two calls another
  client can delete or rotate the object. Read it, catch the miss.

When extending a remote integration, count the round trips your change
adds and justify each one. The default budget is **one** for the
intended action, plus at most one parent-recovery retry.

---

## Databricks guidance

Databricks code should make real payloads easier to work with.
That means:
- supporting practical type strings and wrappers
- parsing nested SQL types robustly
- normalizing inconsistent payload shapes
- showing the exact unsupported fragment in failures

Do not build toy parsers for idealized input only.
Real integration payloads are messier than docs examples.
The library should absorb that mess where it is safe to do so.

---

## AI / SQL guidance

The AI layer should be:
- retry-safe
- cache-aware
- structured in output
- explicit about dialect behavior

When changing AI or SQL code:
- prefer structured parsing over vague text handling
- preserve retry and caching logic
- keep dialect differences visible and testable
- avoid hidden network assumptions

---

## Testing expectations

Run at least the relevant scope:

```bash
cd python
pytest
pytest tests/test_yggdrasil/test_data/
ruff check
black .
```

### Use the engine TestCase base classes for data tests
Any test that touches a dataframe, Arrow object, or engine-side type **must** subclass the matching base class from `yggdrasil.*.tests` instead of importing the engine module at the top of the file.

| Engine | Base class | Module |
| --- | --- | --- |
| Arrow  | `ArrowTestCase`  | `yggdrasil.arrow.tests`  |
| Polars | `PolarsTestCase` | `yggdrasil.polars.tests` |
| pandas | `PandasTestCase` | `yggdrasil.pandas.tests` |
| Spark  | `SparkTestCase`  | `yggdrasil.spark.tests`  |

Why this is the rule:
- Optional deps get skipped with a real install hint. Top-level `import polars` in a test file breaks base installs.
- You get `self.pa` / `self.pl` / `self.pd` / `self.spark`, a per-test `self.tmp_path`, Arrow interop helpers, and built-in frame/schema assertions (`assertFrameEqual`, `assertSchemaEqual`, `assertSeriesEqual`, `SparkTestCase.assertDataFrameEqual` / `assertSparkEqual`).
- `SparkTestCase` shares one process-wide `SparkSession` — don't build your own and don't stop it in `tearDown`.
- Cross-engine tests can multi-inherit (`class TestX(PolarsTestCase, ArrowTestCase):`) or split into sibling classes in the same file — both work; pick whichever keeps the body clean.
- Databricks integration tests keep the `integration` marker **and** subclass the relevant engine TestCase so local runs skip cleanly without `DATABRICKS_HOST`.

Example:

```python
from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.polars.tests import PolarsTestCase

class TestMyCodec(ArrowTestCase):
    def test_roundtrip(self):
        tbl = self.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        out = self.tmp_path / "t.parquet"
        self.write_parquet(tbl, out)
        self.assertFrameEqual(self.read_parquet(out), tbl)

class TestMyCodecPolars(PolarsTestCase):
    def test_from_arrow(self):
        df = self.arrow_to_polars(self.pl.DataFrame({"x": [1, 2]}).to_arrow())
        self.assertFrameEqual(df, {"x": [1, 2]})
```

### Every meaningful change should test
- expected success behavior
- realistic failure behavior
- optional dependency boundaries when relevant
- native vs fallback parity when `rs.py` is involved
- schema preservation when applicable

### Test the user experience, not just internals
Prefer tests that validate:
- what input a real caller passes
- what result they get back
- what exception they see when they misuse the API

This repo wins when callers need less glue code and get less confusing failures.
So test that, not only the internals.

---

## What to read before changing code

Start here:
1. `AGENTS.md`
2. root `README.md`
3. `python/README.md`

Then inspect the subsystem you are touching.

Before implementing a fix, check:
- existing registry behavior
- nearby converters
- option normalization patterns
- tests covering adjacent behavior
- whether the same bug pattern already exists elsewhere

---

## Review checklist before finishing

### User integration
- Does this reduce caller boilerplate?
- Does it accept the shapes users naturally have?
- Does it improve recovery when input is wrong?

### Integration vs invention
- Did you extend the existing module/class/registry instead of creating a parallel one?
- Is every new public symbol actually called by something in this change?
- Did you remove any helper, parameter, or branch that ended up unused?
- Is the diff the smallest change that cleanly solves the problem?

### Semantics
- Are names, metadata, nullability, and nested structure preserved?
- Is lossy behavior explicit?

### API quality
- Are conflicts explicit?
- Are defaults ergonomic?
- Do type hints match runtime truth?

### Compatibility
- Does this still work without optional dependencies unless truly required?
- Does it preserve Python 3.10 compatibility?

### Performance
- Is the path vectorized / Arrow-native / Polars-first where appropriate?
- Did performance work avoid changing semantics?

### Testing
- Is there a realistic success test?
- Is there a realistic failure test?
- Is fallback behavior covered where relevant?

If those answers are shaky, the patch is probably not done.

---

## Strong preferences for coding agents

### Prefer making the library more helpful
A small ergonomic improvement that saves every caller friction is usually a good trade.

### Prefer explicit recovery guidance
Do not just reject bad input.
Tell the user how to get back on track.

### Prefer local consistency
Follow patterns already used in the subsystem unless there is a clear upgrade path.

### Prefer `yggdrasil.pickle.json` (orjson) over the stdlib `json`
`orjson` is a hard dependency of `ygg`, and `yggdrasil.pickle.json` wraps it with the type coverage we already use across the codebase (datetime, date, time, UUID, Path, Enum, dataclasses, namedtuples, sets, Mapping subclasses, bytes, Decimal, …).

When you need to serialize or parse JSON in feature code, reach for `yggdrasil.pickle.json` (`from yggdrasil.pickle import json` or `from yggdrasil.pickle.json import dumps, loads, dump, load`) — not the stdlib `json` module. Same `loads / dumps / load / dump` surface, faster, and it already handles the rich types we round-trip through Arrow / Parquet / API responses. Drop down to `import orjson` directly only inside hot inner loops where you don't need any of the type coercions.

Stdlib `import json` is acceptable for: parsing third-party config files where stdlib semantics are part of the contract, debug `print` formatting, and anything inside `yggdrasil.pickle.json` itself (which uses stdlib only as a fallback for option combinations orjson can't express).

### Prefer extending existing abstractions
If the new behavior belongs naturally in an existing abstraction, put it there.
Do not create isolated side APIs without a strong reason.

Concrete checks before adding a new symbol:

- Search for existing helpers with the same intent. If one already exists, extend it or call it.
- If the new behavior is "like X but with one extra option", add the option to X — don't fork X.
- If the change touches conversion or schema, register it through `yggdrasil.data` so other callers benefit.
- If you add a public name, make sure something actually imports and uses it in this same change. Unused public surface is a maintenance tax.

### Keep the diff small and reuse-heavy
Default to the smallest change that solves the problem cleanly.

- Prefer a 5-line edit to an existing function over a new 50-line module.
- Don't add parameters, branches, or hooks "in case we need them later". Add them when a real caller needs them.
- Don't introduce abstractions before there are at least two real call sites that benefit from them. Three nearly-identical lines is fine; a premature base class is not.
- Don't add backwards-compat shims, deprecated re-exports, or `# removed` placeholder comments unless the user explicitly asked for a deprecation path.

### No dead or isolated code
Every new function, class, option, file, or test should be reachable from a real caller in the same change.

If you add it and nothing uses it yet, delete it or wait until the caller exists.

Specifically avoid:

- helpers defined but never called
- options/flags/parameters never read
- new classes with no instantiation site
- new files that only re-export already-public names
- speculative branches for inputs the public API does not actually accept
- TODO scaffolding without an issue or a follow-up call site

If a piece of code only exists to support a future hypothetical, it is dead weight today. Cut it.

### Prefer `...` (Ellipsis) as the unset / missing sentinel
When you need to distinguish "caller didn't pass this" from "caller passed `None`" — keyword-arg defaults, `dict.get(key, ...)` to tell missing from `None`, lazy-init cache slots — use the built-in `...` (`Ellipsis`) singleton instead of allocating a private `_UNSET = object()` / `_MISSING = object()` per module.

Why: it is a true singleton, has no other meaning in feature code, reads cleanly (`if cached is not ...:`), avoids per-module sentinel proliferation, and serializes / pickles deterministically across process boundaries. Reserve a private sentinel only when `...` itself is a legitimate value in the domain you're filtering.

### Prefer boring reliability
This library sits on integration boundaries.
Stable and obvious beats clever and fragile.

### Make objects picklable and hashable by default

This library ships into Spark workers, multiprocessing pools, joblib jobs, FastAPI worker forks, Power Query bridges — every long-lived "client" / "session" / "service" / "config" / "path" / "schema" / "field" / "request" / "response" object will sooner or later cross a pickle boundary. **Design every new long-lived object so that pickle (and hash, where it makes sense) Just Work, the first time, with the same semantics in every process.**

Concrete rules, in order of priority:

1. **Pure-data configs are dataclasses with `unsafe_hash=True`.** Anything that describes "what to connect to / what to do" — `*Config`, `*Options`, `*Spec`, `Field`, `Schema`, `URL`, credential records — should be a frozen-or-`unsafe_hash` dataclass with `compare=False, hash=False` on the unhashable members (callables, mutable buffers, live handles). Hashable configs are what makes per-config singleton caches (rule 3) possible. If a field is genuinely identity-bearing but unhashable (e.g. a `dict` of options), normalize it into a `tuple` of items in `__post_init__` or expose a `_cache_key()` method that returns a hashable projection — don't push the burden onto every caller.

2. **Live "client" / "session" / "service" objects are picklable too.** Pickling an object that owns a TCP connection pool, a boto session, or a thread pool is normal in this codebase — Spark workers do it on every partition. The contract is: pickling a live handle dehydrates it; unpickling rehydrates it lazily on first use. Use the generic state pattern from `yggdrasil.io.session.Session` / `yggdrasil.aws.AWSClient`:

   ```python
   _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({
       "_session", "_client_cache", "_lock", "_pool", ...
   })

   def __getstate__(self):
       return {k: v for k, v in self.__dict__.items()
               if k not in self._TRANSIENT_STATE_ATTRS}

   def __setstate__(self, state):
       if getattr(self, "_initialized", False):
           return
       self.__dict__.update(state)
       # re-init the transient slots to their fresh defaults
       self._session = None
       self._client_cache = {}
       ...
       self._initialized = True
   ```

   Subclasses extend `_TRANSIENT_STATE_ATTRS` with `Base._TRANSIENT_STATE_ATTRS | frozenset({...})` and override `__setstate__` to chain to super and reset their own transients.

3. **Singleton-cache by config in `__new__`.** When two callers build the same client / session / service with the same config, they must share the same in-process instance — otherwise the connection pool, cookie jar, boto-client cache, and per-instance lazy state get duplicated for nothing. Pattern:

   ```python
   from yggdrasil.dataclasses.expiring import ExpiringDict

   # ExpiringDict owns its own RLock and an atomic get_or_set, so
   # the singleton cache doesn't need an external _INSTANCES_LOCK.
   # default_ttl=None ⇒ entries live for the process lifetime; pass
   # seconds / timedelta / max_size if a config should age out or
   # be capped (rare for client singletons, common for query-result
   # caches).
   _INSTANCES: ClassVar[ExpiringDict[tuple[type, ConfigT], "Self"]] = ExpiringDict(default_ttl=None)

   def __new__(cls, config=None):
       if config is None:
           config = ConfigT()
       key = (cls, config)
       return cls._INSTANCES.get_or_set(key, lambda: object.__new__(cls))

   def __init__(self, config=None):
       if getattr(self, "_initialized", False):
           return  # idempotent — Python re-enters __init__ after cached __new__
       ...
       self._initialized = True

   def __getnewargs__(self):
       return (self.config,)  # collapse to live singleton on in-process unpickle
   ```

   `__getnewargs__` is what makes pickle round-trip preserve singleton identity within a process. The `_INSTANCES` `ExpiringDict` is also the seam tests use to clear cross-test bleed (`Cls._INSTANCES.clear()` in an autouse fixture). **Don't** reach for a bare `dict` + `threading.Lock` here — `ExpiringDict` is already the project-wide primitive for concurrent / TTL-bearing caches (see `databricks/sql/{schemas,tables,views,catalogs}.py`, `databricks/warehouse/service.py`, `MSALAuth._INSTANCES`). It gives you `get_or_set`, `set_many`, `pop`, `purge_expired`, an `on_evict` hook for values that own external resources, and pickling that preserves only live entries — all behaviors a hand-rolled `dict + Lock + sweep` triple has to grow over time.

4. **Hash and equality follow config, not identity.** Two clients built from equal configs must compare equal and hash equal so they collapse to the same `_INSTANCES` slot. Don't add `__hash__` based on `id(self)` — the singleton mechanism already gives you stable identity for free, and config-based equality lets configs flow into sets / dict keys / Spark `groupBy` keys without surprising the caller.

5. **Test the round-trip every time.** Any new long-lived class gets at least these tests (mirrored on `test_session_pickle.py` / `test_aws/test_client_singleton.py`):

   - same-config returns same instance (`a is b`),
   - `__init__` is idempotent (mutate live, re-construct, mutation survives),
   - in-process pickle collapses to live singleton (`pickle.loads(pickle.dumps(x)) is x`),
   - cross-process pickle (drop `_INSTANCES`, then unpickle) restores config + transient slots correctly,
   - transient attrs don't leak into `__getstate__`.

   If you can't write the cross-process test cleanly, the design is wrong — fix the seam, don't skip the test.

6. **Keep configs JSON-serializable when feasible.** A config that round-trips through `yggdrasil.pickle.json` (for HTTP payloads, FastAPI bodies, `to_url()` plumbing, debug dumps) is friendlier than one that only pickles. Limit non-JSON members (callables, live handles, file objects) to fields explicitly marked `compare=False, hash=False, repr=False` and document why they exist.

When you can't make something picklable (a real socket, a live Spark JVM handle, a memory-mapped buffer larger than a pickle frame), say so explicitly: raise a clear `TypeError("X is not picklable; pass Y instead")` from `__reduce_ex__` so the failure surfaces at the right line instead of inside a Spark worker stack trace 30 minutes later.

---

## Skills guidance for coding agents

Use normal repository workflows unless the task is specifically about skills.

Use:
- `skill-creator` only when authoring or revising reusable Codex skills/workflows
- `skill-installer` only when installing or listing skills into `$CODEX_HOME/skills`

Do not force skill usage for normal repository code changes.

---

## Final rule

When unsure, choose the implementation that would make a real user say:

> Nice. I passed the thing I already had, and the library either worked or told me exactly how to fix it.
