# Working in `yggdrasil.io`

This file is the cheat sheet for LLM contributors touching anything under
`yggdrasil/io/`. Read it before editing buffers, paths, or IO leaves; the
project-level `AGENTS.md` and `CLAUDE.md` still apply on top.

## Mental model

The IO layer is three concentric rings. Pick the smallest one that fits.

1. **`yggdrasil.io.buffer.TabularIO`** — pure protocol over Apache Arrow
   record batches. No bytes, no codec, no path. Two abstract hooks
   (`_read_arrow_batches`, `_write_arrow_batches`) plus an options class.
   Reach for this when the source has a tabular surface but no useful
   byte representation (Spark catalog table, JDBC cursor, in-memory Arrow).
2. **`yggdrasil.io.buffer.BytesIO`** — spill-to-disk byte buffer that
   IS-A `TabularIO`. Three backings (memory `bytearray`, local-spilled
   fd, remote transactional). All the format leaves (`ParquetIO`,
   `CsvIO`, `JsonIO`, `NDJsonIO`, `ArrowIPCIO`, `XlsxIO`, `XmlIO`)
   inherit from it. Reach for `BytesIO` directly when you have raw
   bytes and want a tabular view.
3. **`yggdrasil.io.fs.Path`** — abstract filesystem path. `LocalPath`,
   `MemoryPath`, plus backend-specific subclasses (Databricks volumes,
   S3, …). Path is also a `TabularIO` — `path.read_arrow_table()`
   works; under the hood it dispatches to the format leaf via
   `path.media_type`.

## Picking entry points

| Goal | Use |
| --- | --- |
| Read/write any tabular file by URL | `Path.from_(url).read_arrow_table(...)` / `write_arrow_table(...)` |
| Stream Arrow batches | `TabularIO.read_arrow_batches(options)` / `write_arrow_batches(it, options)` |
| Open raw bytes with positional I/O | `BytesIO(path=..., mode="rb+")` |
| Build a format-typed IO directly | `ParquetIO(path=...)`, `CsvIO(path=...)`, etc. |
| Walk a folder / archive | `FolderIO(path=...)`, `ZipIO(path=...)` (NestedIO subclasses) |
| Convert between engines | `yggdrasil.data.cast.convert(value, target, options=...)` |

**Always prefer `yggdrasil.data` abstractions** (`DataField`, `Schema`,
`DataType`, `convert`) before reaching down to `polars` / `pandas` /
`pyarrow`. See the project `CLAUDE.md` "Reach for `yggdrasil.data` first"
section.

## Concurrency

The buffer layer is single-threaded by default. Opt in to concurrent
safety by passing `concurrent=True` to any `TabularIO`/`BytesIO`/
`NestedIO` constructor. With `concurrent=True`:

- **In-memory `BytesIO`** — every public read/write goes through a
  `threading.RLock`, so multiple threads in the same process can share
  the buffer without tearing.
- **Path-bound `BytesIO`** — a sidecar lock file is acquired for the
  IO's lifetime. Naming follows `<dir>/.<basename>.{r|w|rw}.lock`,
  selected from the open mode:
  - `rb` → `.r.lock` (shared, multiple readers OK)
  - `wb` / `ab` / `xb` → `.w.lock` (exclusive)
  - `rb+` / `wb+` / `ab+` → `.rw.lock` (exclusive)
- **`NestedIO` (folders, zips, Delta)** — folder-root `.rw.lock`
  exclusive for the whole IO lifetime.

```python
from yggdrasil.io.buffer.primitive.parquet_io import ParquetIO
from yggdrasil.dataclasses import WaitingConfig

with ParquetIO(
    path="s3://bucket/data.parquet",
    mode="wb+",
    concurrent=True,
    lock_wait=WaitingConfig(timeout=30, interval=0.1, backoff=1.5),
) as io:
    io.write_arrow_table(table)
```

`lock_wait` is a `WaitingConfig` argument:

- `None` → wait forever (default).
- A number → that many seconds with exponential backoff.
- `0` → fail-fast: raise `TimeoutError` immediately on contention.
- `dict` / `WaitingConfig` → full control over `timeout`, `interval`,
  `backoff`, `max_interval`, `retries`.

### Locking remote paths

`Path.lock(read=, write=, wait=, stale_after_seconds=)` is the
canonical entry point. It dispatches:

- `is_local=True` → `FileLock` (kernel `fcntl` / `msvcrt`).
- `is_local=False` → `AtomicLock` (atomic exclusive-create via
  `path.open_io("xb")` polling). Stale sidecars older than
  `stale_after_seconds` (default 15 min) are force-unlinked by the
  next acquirer — covers the crashed-remote-writer case where there's
  no kernel to release the lock.

```python
with path.lock(write=True, wait=60):
    path.write_bytes(payload)
```

### `YGGFolderIO` — checkpoints, metadata, stats sidecar

`FolderIO` is the generic folder-of-tabular-files class.
`YGGFolderIO` extends it with an out-of-band metadata sidecar at
`<root>/.ygg/`. The data layout is identical, so a YGG folder
reads back as a plain folder by any other tool that doesn't know
about the sidecar.

Auto-detection: `FolderIO(path=...)` probes `<path>/.ygg/` once on
construction; if the sidecar is present the result is a
`YGGFolderIO` automatically. Drop the constructor explicitly when
you need the sidecar on a fresh folder.

The sidecar:

- `.ygg/checkpoints.jsonl` — append-only JSON Lines log. Each
  `io.checkpoint(message=..., **extra)` call writes one record
  `{id, ts, pid, files, num_files, message, ...extra}` and takes a
  transient write lock on the log so concurrent checkpointers don't
  tear lines.
- `.ygg/metadata/<key>.json` — per-key value store via
  `write_metadata(key, value)` / `read_metadata(key, default)` /
  `list_metadata_keys()`. Keys are slugged (alphanumerics + `_-.`);
  writes are stage+rename atomic.
- `.ygg/stats.arrow` — Arrow-IPC encoded `Stats` (see below)
  written by `write_stats()` / read by `read_stats()`.

```python
from yggdrasil.io.buffer.nested import YGGFolderIO

with YGGFolderIO(path="s3://bucket/ingest/", concurrent=True) as io:
    for batch in producer():
        io.write_arrow_batches([batch], mode=Mode.APPEND)
        io.checkpoint(rows=batch.num_rows, source="kafka:ingest-7")
    io.write_stats()  # one-shot at end

# Downstream consumer
with FolderIO(path="s3://bucket/ingest/") as r:  # auto-upgrades
    assert isinstance(r, YGGFolderIO)
    last = r.latest_checkpoint()
    stats = r.read_stats()
    if stats and stats.column("id").max_value < cutoff:
        skip_partition(r)
```

Both `iter_children` and `read_arrow_batches` skip the `.ygg/`
folder — checkpoint, metadata, and stats records never contaminate
the data view. In-flight staging files
(`tmp-<start>-<end>-<seed>.<ext>` — time-sortable) are also
hidden, so a parallel reader never picks up a half-written child.

### Stats — format-agnostic columnar statistics

`yggdrasil.io.stats.Stats` is the canonical place for columnar
statistics across every IO leaf. Parquet has stats in its footer;
CSV / JSON / NDJSON / IPC / XLSX / XML do not. `Stats` levels the
playing field:

```python
from yggdrasil.io.stats import Stats

# From any TabularIO (every format works) or pa.Table / batches:
stats = Stats.compute(reader, name="ingest-2026-01-04.parquet",
                      distinct=True)

# Round-trip via Arrow IPC bytes or a Path.
blob = stats.to_ipc()
again = Stats.from_ipc(blob)

# Merge across files for partition-pruning style aggregates.
folder_stats = Stats.merge([per_file_a, per_file_b, per_file_c])
```

The on-disk encoding is **Arrow IPC** with a fixed schema
(`source`, `column`, `arrow_type`, `num_values`, `null_count`,
`distinct_count`, `min_value`, `max_value`, `byte_size`). Schema
metadata carries `yggdrasil.stats.version`; bumps are visible to
readers so old sidecars can be detected and recomputed.

### Spill-temp cleanup

`BytesIO` mints temp files under `tempfile.gettempdir()` with the TTL
encoded in the filename (`tmp-<seed>-<start>-<end>.<ext>`). Crashed
workers leave orphans. The cleaner runs lazily — every call to
`_mint_spill_path` triggers a throttled sweep that removes any expired
file under a directory-level lock.

You can run it manually:

```python
from yggdrasil.io.buffer._concurrency import cleanup_stale_spill_files
cleanup_stale_spill_files()  # uses tempfile.gettempdir() by default
```

## Style rules specific to this layer

### EAFP — don't pre-check existence

Calling `path.exists()` before `path.read_bytes()` / `pwrite()` /
`open_io()` is a wasted round-trip on remote backends. The actual
operation already raises the right exception — let it.

```python
# Bad — extra GET on S3, race against concurrent deletion.
if path.exists():
    data = path.read_bytes()
else:
    data = b""

# Good — single round-trip, race-free.
data = path.read_bytes(raise_error=False)  # returns b"" on missing
```

For exclusive create, prefer `xb` mode over an `exists()` check:

```python
try:
    with path.open_io("xb") as fh:
        fh.write(payload)
except FileExistsError:
    handle_already_present()
```

### Engine isolation

Every optional engine (polars, pandas, spark, blake3, xxhash, …) goes
through a `lib.py` guard:

```python
from yggdrasil.polars.lib import polars   # right
import polars                              # wrong — breaks base installs
```

Tests for engine-specific code subclass the matching `*TestCase` from
`yggdrasil.<engine>.tests`. See the project `CLAUDE.md` for the
canonical list.

### Schema-first writes

Don't hand-roll engine-specific schemas at write time. Build a
`yggdrasil.data.Schema` (or a list of `Field` instances) and pass it
through `CastOptions.target_schema`; the converter registry produces
the engine-native form (Arrow, Polars, pandas, Spark) on demand. Field
name, order, nullability, metadata, nested structure, and timezone
intent are part of the user contract — preserve them across boundaries.

### Path lifecycle

`Path` participates in the `Disposable` graph. Two open methods exist
on purpose:

- `Disposable.open(path)` (no args) — lifecycle "acquire". Idempotent.
- `path.open_io(mode, ...)` — returns a `BytesIO` for actual byte I/O.

A path constructed with default `auto_open=True` is already in the
"open" lifecycle state, so naive callers never have to think about
acquire/release. Set `temporary=True` to mark a path as cleanup-on-close.

## When NOT to extend the IO layer

- **Don't add a new optional dependency without a `lib.py` guard.**
  Base installs must keep working with only `pyarrow`.
- **Don't invent parallel options objects.** Extend `CastOptions` (or
  the appropriate options subclass) instead of inventing a new
  per-call config struct.
- **Don't bypass the converter registry.** Register a new converter
  via `@register_converter(from_hint, to_hint)` rather than calling
  `df.to_pandas()` / `pl.from_arrow(...)` from feature code.
- **Don't sprinkle `import polars` / `import pandas` at module level.**
  Optional imports belong inside functions or behind the `lib.py`
  guards so base installs stay green.

## Where to look when something breaks

| Symptom | First place to check |
| --- | --- |
| "No tabular IO registered for media_type X" | The format leaf isn't imported; `yggdrasil.io.buffer.primitive` package import, or `yggdrasil.io.buffer.nested` for archives/folders |
| Spilled file stays on disk after crash | `cleanup_stale_spill_files()` — the TTL is encoded in the filename |
| Concurrent writers tear each other's bytes | `concurrent=True` on the IO, plus a `lock_wait` budget |
| `FileExistsError` on retry | An `xb`-mode write succeeded then was retried; switch to `wb+` if overwrite is intended |
| `TimeoutError` on path lock | The holder is alive and busy, OR the sidecar is stale beyond `stale_after_seconds` |
| Slow remote `pwrite` for small writes | The default `_pwrite_via_rmw` is read-modify-write; the backend should override `pwrite` with a native positional API if it has one |
