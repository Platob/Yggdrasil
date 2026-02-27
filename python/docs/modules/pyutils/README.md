# yggdrasil.pyutils

Retry and parallel execution decorators for pipelines and services.

## Exports

```python
from yggdrasil.pyutils import retry, parallelize
```

---

## `retry` — decorator for flaky operations

```python
retry(
    exceptions=Exception,     # exception class or tuple to catch
    tries=3,                  # total attempts (including first)
    delay=0.5,                # initial sleep between retries (seconds)
    backoff=2.0,              # exponential backoff multiplier
    max_delay=None,           # cap on sleep duration
    jitter=None,              # callable(delay) → adjusted delay
    logger=None,              # logging.Logger for retry messages
    reraise=True,             # re-raise final exception
    timeout=None,             # wall-clock timeout across all attempts (seconds)
)
```

Works for both **sync** and **async** functions.

### Bootstrap: basic retry

```python
from yggdrasil.pyutils import retry

@retry(tries=4, delay=1.0, backoff=2.0)
def call_api(endpoint: str) -> dict:
    import requests
    return requests.get(endpoint, timeout=10).json()

data = call_api("https://api.example.com/data")
```

### Bootstrap: retry specific exceptions

```python
from yggdrasil.pyutils import retry
import requests

@retry(exceptions=(requests.Timeout, requests.ConnectionError), tries=5, delay=0.5)
def fetch(url: str) -> bytes:
    return requests.get(url, timeout=5).content
```

### Bootstrap: async retry

```python
from yggdrasil.pyutils import retry
import asyncio
import aiohttp

@retry(tries=3, delay=1.0)
async def fetch_async(url: str) -> str:
    async with aiohttp.ClientSession() as s:
        async with s.get(url) as r:
            return await r.text()

result = asyncio.run(fetch_async("https://api.example.com/ping"))
```

### Bootstrap: with total timeout

```python
from yggdrasil.pyutils import retry

@retry(tries=10, delay=0.5, backoff=1.5, timeout=30.0)
def poll_status(job_id: str) -> str:
    # retries up to 10 times but stops after 30 seconds total
    ...
```

---

## `parallelize` — decorator for fan-out

```python
parallelize(
    executor_cls=ThreadPoolExecutor,  # or ProcessPoolExecutor
    *,
    max_workers=None,         # thread/process pool size
    arg_index=0,              # which positional arg is the iterable
    timeout=None,             # per-item timeout
    return_exceptions=False,  # True: yield exceptions instead of raising
    show_progress=False,      # True: show tqdm progress bar
)
```

Wraps a function so it executes over each item in an iterable argument and returns a lazy **iterator**.

### Bootstrap: parallel map

```python
from yggdrasil.pyutils import parallelize

@parallelize(max_workers=8)
def enrich(item: str) -> dict:
    # called once per item, in parallel
    return {"item": item, "length": len(item)}

results = list(enrich(["apple", "banana", "cherry"]))
```

### Bootstrap: parallel with a method (arg_index)

```python
from yggdrasil.pyutils import parallelize

class Processor:
    @parallelize(max_workers=4, arg_index=1)  # arg_index=1 because self is arg 0
    def transform(self, records: list[dict]) -> dict:
        return {k: v.upper() if isinstance(v, str) else v for k, v in records.items()}

proc = Processor()
results = list(proc.transform([{"name": "alice"}, {"name": "bob"}]))
```

### Bootstrap: process pool for CPU-bound work

```python
from concurrent.futures import ProcessPoolExecutor
from yggdrasil.pyutils import parallelize

@parallelize(ProcessPoolExecutor, max_workers=4)
def compute(n: int) -> int:
    return sum(range(n))       # CPU-bound

results = list(compute([100_000, 200_000, 300_000]))
```

### Bootstrap: collect exceptions instead of raising

```python
from yggdrasil.pyutils import parallelize

@parallelize(max_workers=4, return_exceptions=True)
def safe_fetch(url: str):
    import requests
    return requests.get(url, timeout=5).json()

for item in safe_fetch(urls):
    if isinstance(item, Exception):
        print("failed:", item)
    else:
        process(item)
```
