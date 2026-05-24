# yggdrasil.concurrent

Bounded job execution primitives for large or unbounded streams of work — `Job`, `AsyncJob`, `ThreadJob`, and `JobPoolExecutor` with backpressure.

## One-liner

```python
from yggdrasil.concurrent import Job, JobPoolExecutor

with JobPoolExecutor(max_workers=4) as pool:
    for result in pool.as_completed(Job.make(fn, arg) for arg in items):
        print(result.value)
```

## Job types

```python
from yggdrasil.concurrent import Job, AsyncJob, ThreadJob

# Synchronous callable (runs in thread pool)
job = Job.make(lambda: 42)

# Async coroutine
import asyncio
async def fetch(url): ...
job = AsyncJob.make(fetch, "https://example.com")

# Explicit thread job
job = ThreadJob.make(heavy_io_fn, arg1, kwarg=val)
```

## JobPoolExecutor

```python
from yggdrasil.concurrent import Job, JobPoolExecutor

jobs = [Job.make(lambda x=x: x * x) for x in range(100)]

# Basic: collect results as they finish
with JobPoolExecutor(max_workers=4) as pool:
    for result in pool.as_completed(jobs):
        print(result.value)

# With backpressure: at most 10 jobs queued at once (protects memory)
with JobPoolExecutor(max_workers=4, max_in_flight=10) as pool:
    for result in pool.as_completed(iter(jobs)):
        print(result.value)
```

## JobResult

```python
from yggdrasil.concurrent import Job, JobPoolExecutor, JobResult

jobs = [Job.make(int, s) for s in ["1", "bad", "3"]]

with JobPoolExecutor(max_workers=2) as pool:
    for result in pool.as_completed(jobs):
        if result.exception is not None:
            print("Error:", result.exception)
        else:
            print("OK:", result.value)
```

## Parallelize with `yggdrasil.pyutils`

For simpler use cases (list of callables, collect all results) use `parallelize`:

```python
from yggdrasil.pyutils import parallelize

results = parallelize(
    [(fn, (arg,), {}) for arg in items],
    max_workers=8,
)
```

## Fan-out HTTP requests

```python
from yggdrasil.concurrent import Job, JobPoolExecutor
from yggdrasil.http_ import HTTPSession

http = HTTPSession()

def fetch_page(page: int) -> dict:
    return http.get("https://api.example.com/items", params={"page": page}).json()

jobs = [Job.make(fetch_page, p) for p in range(1, 21)]

pages = []
with JobPoolExecutor(max_workers=5, max_in_flight=10) as pool:
    for result in pool.as_completed(jobs):
        result.raise_for_exception()   # re-raises if the job failed
        pages.extend(result.value["items"])

print(f"Fetched {len(pages)} items total")
```

## Retry within a job

Combine with `yggdrasil.pyutils.retry` for resilient concurrent work:

```python
from yggdrasil.concurrent import Job, JobPoolExecutor
from yggdrasil.pyutils import retry

@retry(max_attempts=3, backoff=2.0)
def fetch_with_retry(url: str) -> bytes:
    from yggdrasil.http_ import HTTPSession
    return HTTPSession().get(url).content

jobs = [Job.make(fetch_with_retry, url) for url in urls]

with JobPoolExecutor(max_workers=8) as pool:
    for result in pool.as_completed(jobs):
        process(result.value)
```

---

## Ordered vs. completion-order

```python
from yggdrasil.concurrent import Job, JobPoolExecutor
import time, random

def slow(i: int) -> int:
    time.sleep(random.uniform(0.05, 0.3))
    return i * i

jobs = [Job.make(slow, i) for i in range(10)]

# Completion order (default) — faster, but output order varies
with JobPoolExecutor(max_workers=4) as pool:
    for res in pool.as_completed(jobs, ordered=False):
        print("finished:", res.value)

# Submission order — output always [0, 1, 4, 9, 16, ...]
with JobPoolExecutor(max_workers=4) as pool:
    for res in pool.as_completed(jobs, ordered=True):
        print("ordered:", res.value)
```

---

## Cancel-on-exit for infinite streams

When the job source is a live generator (e.g. reading from a message queue), use `cancel_on_exit=True` so in-flight futures are cancelled when you break out of the loop:

```python
from yggdrasil.concurrent import Job, JobPoolExecutor

def message_source():
    while True:
        msg = queue.poll(timeout=1.0)
        if msg:
            yield Job.make(process_message, msg)

with JobPoolExecutor(max_workers=8, max_in_flight=20) as pool:
    for result in pool.as_completed(
        message_source(),
        cancel_on_exit=True,      # cancel queued futures on break/exception
        shutdown_on_exit=True,
    ):
        if result.exception:
            log_error(result.exception)
        else:
            sink.write(result.value)
```

---

## Collect errors without short-circuit

```python
from yggdrasil.concurrent import Job, JobPoolExecutor

def risky(i: int) -> int:
    if i % 3 == 0:
        raise ValueError(f"bad input {i}")
    return i * 2

jobs = [Job.make(risky, i) for i in range(12)]

successes, failures = [], []
with JobPoolExecutor(max_workers=4) as pool:
    for res in pool.as_completed(jobs, raise_error=False):
        if res.exception:
            failures.append((res.job, res.exception))
        else:
            successes.append(res.value)

print(f"OK: {len(successes)}, Failed: {len(failures)}")
for job, exc in failures:
    print(f"  {job}: {exc}")
```

---

## Multi-stage pipeline: fetch → parse → write

```python
from yggdrasil.concurrent import Job, JobPoolExecutor
from yggdrasil.http_ import HTTPSession
import pyarrow as pa
from yggdrasil.io.primitive.parquet_file import ParquetFile

http = HTTPSession()

def fetch(page: int) -> bytes:
    return http.get("https://api.example.com/records", params={"page": page}).content

def parse(raw: bytes) -> pa.Table:
    import pyarrow.json as paj
    return paj.read_json(pa.py_buffer(raw))

# Stage 1: concurrent fetch
fetch_jobs = [Job.make(fetch, p) for p in range(1, 21)]
raw_pages = []
with JobPoolExecutor(max_workers=8) as pool:
    for res in pool.as_completed(fetch_jobs, ordered=True):
        raw_pages.append(res.value)

# Stage 2: concurrent parse
parse_jobs = [Job.make(parse, raw) for raw in raw_pages]
tables = []
with JobPoolExecutor(max_workers=4) as pool:
    for res in pool.as_completed(parse_jobs, ordered=True):
        tables.append(res.value)

# Stage 3: write
result = pa.concat_tables(tables)
ParquetFile("/tmp/records.parquet").write(result)
print(f"Wrote {result.num_rows} rows")
```
