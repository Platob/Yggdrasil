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
from yggdrasil.io.http_ import HTTPSession

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
    from yggdrasil.io.http_ import HTTPSession
    return HTTPSession().get(url).content

jobs = [Job.make(fetch_with_retry, url) for url in urls]

with JobPoolExecutor(max_workers=8) as pool:
    for result in pool.as_completed(jobs):
        process(result.value)
```
