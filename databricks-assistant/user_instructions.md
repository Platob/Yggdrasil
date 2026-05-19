# User instructions — Yggdrasil (`ygg[data,databricks]`)

I primarily work in Python notebooks on Databricks against
[Yggdrasil](https://github.com/Platob/Yggdrasil) (PyPI: `ygg`, import:
`yggdrasil`). Tailor suggestions to that stack.

## Preferences

- **Idiomatic stack:** Arrow-first frames via `yggdrasil.data`, then
  Polars/pandas/Spark via the engine bridges. Reach past
  `yggdrasil.data` only when the abstraction genuinely can't cover the
  case.
- **Imports:** `from yggdrasil.databricks import DatabricksClient`,
  `from yggdrasil.data.cast import convert`,
  `from yggdrasil.data import DataField, Schema, DataType`. Use the
  `lib.py` guards (`from yggdrasil.polars.lib import polars`) for
  optional engines.
- **Casting:** Prefer `convert(value, target)` and
  `cast_arrow_tabular(t, CastOptions(target_field=schema))` over
  per-column `.cast()` chains.
- **No row-loops over data.** No `for row in df.iterrows()`, no
  `array.to_pylist()` followed by a comprehension. Vectorise via
  `pyarrow.compute`, Polars expressions, or numpy ufuncs.
- **JSON:** Use `yggdrasil.pickle.json` (orjson-backed) instead of
  stdlib `json` — handles datetime / UUID / Path / Enum / dataclass.
- **Lifecycle ops:** Call the resource singleton's own method
  (`volume.create(...)`, `schema.delete(...)`, `table.read_info()`),
  not `ws.volumes.create(...)` directly — the singleton method wraps
  retries, cache warm-up, and `if_not_exists` / `missing_ok` ergonomics.

## Style

- Short, blunt comments only where the WHY is non-obvious (engine
  edge case, schema invariant, workaround). Skip "loop through fields"
  prose.
- Log lines follow `<Verb> <ResourceNoun> %r (key=value, …)` — use
  `%r` lazy interpolation, not f-strings, in `LOGGER.debug` /
  `LOGGER.info`.
- Type hints match runtime, including `| None` on nullable returns.
- Keyword-only arguments for ambiguous options.

## Tone in responses

Direct and concise. State the change, show the snippet, point me to
the relevant module path. Skip "Great question!" preambles.
