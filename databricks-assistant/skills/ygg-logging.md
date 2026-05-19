# Skill: log idiomatically inside Yggdrasil-flavored code

## When to use

The user is adding `LOGGER.debug(...)` / `LOGGER.info(...)` lines,
debugging "why don't I see anything in the logs?", standardising
log messages across a service, or reviewing generated code for log
hygiene. Also use when reviewing an f-string log call, a
`LOGGER.debug(f"...")`, a `if logger.isEnabledFor(...)` guard, or a
"Cache hit" debug line.

## Shape

`<Verb> <ResourceNoun> %r (key=value, key=value, …)`

- **Verb tense**: present-progressive at the start
  (`"Creating"`, `"Deleting"`, `"Starting"`, `"Listing"`,
  `"Fetching"`), past tense on completion (`"Created"`, `"Deleted"`,
  `"Started"`).
- **Resource noun**: always include it (`"job"`, `"job run"`,
  `"cluster"`, `"warehouse"`, `"catalog"`, `"schema"`, `"table"`,
  `"volume"`, `"secret"`, `"context"`, `"workspace directory"`,
  `"DBFS directory"`). `"Deleted job run %r"` reads cleanly in a
  wall of mixed lines; `"Deleted %r"` forces the reader to parse
  the `__repr__` to know what was deleted.
- **`%r`, lazy**: format the object via `__repr__` so disabled
  levels skip the call. Don't use f-strings.

```python
# good
LOGGER.debug("Creating volume %r (managed=True, comment=%r)", vol, comment)
LOGGER.info("Created volume %r", vol)

# bad
LOGGER.debug(f"Creating {vol} (managed=True, comment={comment!r})")
LOGGER.info("Created %s", vol)        # str(vol) often collapses to a path
LOGGER.info("Volume.create: done")    # no resource noun, no object
```

## Lifecycle pairs

Start the operation at `debug`, log success at `info`:

```python
LOGGER.debug("Deleting job run %r (run_id=%s)", run, run.id)
ws.jobs.delete_run(run_id=run.id)
LOGGER.info("Deleted job run %r", run)
```

No third log call on failure — the re-raised exception / stack
trace carries the diagnostic.

## `%r` vs `%s`

| Object kind | Format |
| --- | --- |
| Long-lived objects (`DatabricksClient`, `URL`, `Volume`, `Schema`, …) | `%r` |
| Stdlib formatter slots (`%(asctime)s`, `%(levelname)s`) | `%s` (forced by `logging`) |
| Primitives in message structure (`int`, `bool`, simple URL string) | `%s` reads more naturally |

The repr of long-lived objects is engineered to carry full
identity (`<VolumePath dbfs+volume://…>`,
`DatabricksClient(host='…', auth_type='pat')`); `str(obj)` often
collapses to an ambiguous path / hostname.

## Drop `if logger.isEnabledFor(...)` around a single lazy call

```python
# bad — the guard adds nothing
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("Deleting %r", self)

# good
logger.debug("Deleting %r", self)
```

`logger.debug("...", obj)` already skips formatting when the level
is disabled. The guard earns its keep only when there's real
pre-computation outside the call:

```python
# good — materialising a generator IS expensive
if logger.isEnabledFor(logging.DEBUG):
    entries = list(entries_iter)
    logger.debug("Listing %r -> %d entries", self, len(entries))
```

## Don't log cache hits at debug; log misses / expiries / invalidations

Hits are the steady-state success path. They drown the rare
interesting events.

```python
cached = self._cache.get(key)
if cached is not None:
    return cached                                          # silent

logger.debug("Cache expired for %r (age=%.0fs) — refreshing", key, age)
fresh = self._fetch(key)
```

Same rule for entity-tag caches, schema-info caches, warehouse
caches, singleton caches.

## Anti-patterns to flag

- Function-name prefixes: `"Catalog.schemas: listing …"`
- Bracketed call-site shorthand: `"Cache hit [Schemas.find] key=%s"`
- Pipe separators: `"Spark insert -> %s | mode=%s"`
- Platform-name redundancy: `"Creating Databricks job …"` — the
  logger name already says `databricks`.
- Bare-verb messages: `"Deleted %r"` — name the resource noun.
- Eager `logger.info(f"...")` — even when level is enabled, lazy
  `%r` keeps stack traces clean and avoids re-formatting on
  multiple handlers.

## Logger name

```python
import logging

LOGGER = logging.getLogger(__name__)
```

Always `__name__`, so the hierarchy reflects the module path
(`yggdrasil.databricks.table.async_write`) and the user can dial
verbosity per subsystem.
