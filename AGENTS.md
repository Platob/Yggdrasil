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
- `DataField` / `Field` and `Schema` (names, nullability, metadata, tags, nested structure, engine dtype intent in one place). Use `Field.from_pandas`, `Field.from_polars`, `Field.from_arrow`, `Schema.from_any_fields` instead of re-building per-engine schemas by hand.
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

### Serialization and acceleration
- `pickle/` — custom serialization system
- `rs.py` — Rust bridge with Python fallback
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
- optional Rust fast paths

But correctness and user-visible behavior come first.
Python fallback behavior is canonical.

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

## Optional dependency policy

Optional dependencies must remain optional.

### Rules
- base `ygg` install must work without non-core engines
- base `ygg` install must work without Rust
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

## Rust acceleration policy

Rust is an optional fast path.
It is not the source of truth.

### Non-negotiable rules
- Python behavior is canonical
- Rust behavior must match Python behavior
- pure Python fallback must stay correct
- tests must pass with and without native support
- never import `yggrs` directly outside `yggdrasil/rs.py`

### When Rust is worth it
Add Rust only when:
- the path is actually hot
- semantics are stable
- Python fallback already exists and is correct
- the speedup is worth the added maintenance cost

Do not use Rust to hide unclear logic.
Fix the semantics first.
Then optimize.

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

### Prefer extending existing abstractions
If the new behavior belongs naturally in an existing abstraction, put it there.
Do not create isolated side APIs without a strong reason.

### Prefer boring reliability
This library sits on integration boundaries.
Stable and obvious beats clever and fragile.

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
