# Skill: encode / decode JSON and pickle with `yggdrasil.pickle`

## When to use

The user asks to "serialize / deserialize JSON", "encode a payload
with datetimes / UUIDs / dataclasses", "pickle this client / config
/ frame across workers", "save a config to disk", or pastes code
that imports stdlib `json` to dump a payload that contains rich
types. Also for "why is `json.dumps(...)` raising on my datetime?".

## JSON: prefer `yggdrasil.pickle.json`

```python
from yggdrasil.pickle import json as ygg_json
# or
from yggdrasil.pickle.json import loads, dumps, load, dump
```

Same `loads / dumps / load / dump` surface as stdlib, backed by
`orjson` (a hard dependency), with built-in coverage for:

- `datetime`, `date`, `time`, `timedelta` (ISO 8601)
- `UUID`
- `pathlib.Path`
- `Enum` (member value)
- `dataclasses` (`asdict`)
- `namedtuple` (`_asdict`)
- `set` / `frozenset` (encoded as list)
- `Mapping` / `MappingProxyType`
- `Decimal` (string, preserving precision)
- `bytes` (base64)

```python
from datetime import datetime
from uuid import uuid4
from pathlib import Path

payload = {
    "ts": datetime.utcnow(),
    "id": uuid4(),
    "path": Path("/Volumes/main/default/x.parquet"),
}
data = dumps(payload)        # bytes — orjson emits UTF-8 directly
loads(data)                  # round-trip back to dict
```

## When stdlib `json` is the right choice

Three narrow cases only:

1. Parsing third-party config where stdlib semantics are part of the
   contract (NaN handling, `parse_float=Decimal`, custom hooks).
2. Debug `print` formatting.
3. Inside `yggdrasil/pickle/json.py` itself (it keeps stdlib as a
   fallback for option combos orjson can't express).

Hot inner loop with no rich types? Drop straight to `import orjson`
— `yggdrasil.pickle.json` adds a thin coercion layer you don't need
there.

## Pickle: prefer `yggdrasil.pickle.serde` for rich objects

```python
from yggdrasil.pickle.serde import dumps, loads
```

Wraps `cloudpickle` / `dill` (when installed via `[pickle]` extra)
with `zstandard` compression and content hashing
(`xxhash` / `blake3`). Use it for:

- Sending closures to Spark workers / `multiprocessing` pools.
- Persisting a `DatabricksClient`, `HTTPSession`, or `Schema` to
  disk and reloading on a different process.
- Caching a function result keyed by argument hash.

```python
buf = dumps(my_closure)         # bytes
fn  = loads(buf)                # restored callable
```

Plain stdlib `pickle.dumps(...)` works for simple objects but fails
on lambdas, local functions, and live SDK clients — `serde` handles
those via cloudpickle / dill.

## Long-lived objects: built for singleton-by-config pickling

`DatabricksClient`, `HTTPSession`, `DatabricksPath`, `Schema`,
`URL`, etc. already pickle to their `(cls, config)` key and unpickle
back to the same in-process singleton. Don't write a parallel
`to_dict / from_dict` for them — just pickle them.

For frame-shaped data: pickle the Arrow `Table` (zero-copy IPC
under the hood), not the pandas / Polars wrapper.

## Don'ts

- Don't `import json` for a payload that contains a `datetime` /
  `UUID` / `Path` / `Enum` — `yggdrasil.pickle.json` already covers
  the type and saves the `default=str` workaround.
- Don't `json.dumps(..., default=str)` and lose round-trip
  precision (datetimes come back as strings) — use
  `yggdrasil.pickle.json`.
- Don't reinvent a "JSON-safe" dict transformer at every API
  boundary; route the payload through `dumps` and let the encoder
  handle it.
- Don't pickle a live `databricks.sdk.WorkspaceClient` — pickle the
  `DatabricksClient` wrapper, which rebuilds the workspace handle
  from config on the worker.
