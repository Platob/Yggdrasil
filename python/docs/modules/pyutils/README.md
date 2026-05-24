# yggdrasil.pyutils

General-purpose helpers — retry decorator with exponential backoff, parallelize decorator, and a few process utilities.

## One-liner

```python
from yggdrasil.pyutils import retry

@retry(tries=3, delay=1.0, backoff=2)
def read_remote() -> str:
    return fetch_from_api()
```

## Retry decorator

```python
from yggdrasil.pyutils import retry

# Simple: 3 tries, 200 ms initial delay, 2× backoff
@retry(tries=3, delay=0.2, backoff=2)
def flaky_call() -> str:
    return requests.get("https://api.example.com/data").text

# Retry only on specific exceptions
@retry(tries=5, delay=1.0, exceptions=(ConnectionError, TimeoutError))
def fetch_page(url: str) -> bytes:
    return http.get(url).content

# With a logger
import logging
@retry(tries=3, delay=0.5, logger=logging.getLogger(__name__))
def call_with_logging(): ...
```

## Parallelize decorator

Wraps a function so it accepts an iterable of inputs and runs them concurrently:

```python
from yggdrasil.pyutils import parallelize

@parallelize(max_workers=4)
def square(x: int) -> int:
    return x * x

results = list(square(range(10)))
print(results)   # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

## Functional parallelize (no decorator)

```python
from yggdrasil.pyutils import parallelize

def process(item: dict) -> dict:
    return {"id": item["id"], "result": heavy_fn(item)}

items = [{"id": i} for i in range(100)]
results = parallelize(process, items, max_workers=8)
```

## Retry + parallelize together

```python
from yggdrasil.pyutils import retry, parallelize

@retry(tries=3, delay=1.0, backoff=2)
def fetch(url: str) -> bytes:
    from yggdrasil.http_ import HTTPSession
    return HTTPSession().get(url).content

@parallelize(max_workers=10)
def fetch_all(url: str) -> bytes:
    return fetch(url)

pages = list(fetch_all(urls))
```

## HTTP fan-out pattern

```python
from yggdrasil.pyutils import parallelize
from yggdrasil.http_ import HTTPSession

http = HTTPSession()

def fetch_page(page: int) -> list:
    return http.get("https://api.example.com/items", params={"page": page}).json()["items"]

@parallelize(max_workers=5)
def fetch_pages(page: int) -> list:
    return fetch_page(page)

all_items = [item for page in fetch_pages(range(1, 21)) for item in page]
print(f"Total: {len(all_items)} items")
```

---

## Retry with backoff, jitter, and timeout

```python
import random
from yggdrasil.pyutils import retry

# Jitter avoids thundering-herd when many workers retry simultaneously
@retry(
    exceptions=(ConnectionError, TimeoutError),
    tries=5,
    delay=0.5,
    backoff=2.0,
    max_delay=30.0,
    jitter=lambda d: d + random.uniform(0, 0.5),
)
def call_api(endpoint: str) -> dict:
    from yggdrasil.http_ import HTTPSession
    return HTTPSession().get(endpoint).json()

# Total-timeout variant: give up after 60 seconds regardless of tries
@retry(tries=20, delay=1.0, backoff=1.5, timeout=60.0)
def poll_until_ready(resource_id: str) -> dict:
    from yggdrasil.http_ import HTTPSession
    resp = HTTPSession().get(f"https://api.example.com/resources/{resource_id}")
    if resp.status == 202:
        raise RuntimeError("Still pending")
    resp.raise_for_status()
    return resp.json()
```

---

## Async retry

`@retry` works transparently with `async def`:

```python
import asyncio
from yggdrasil.pyutils import retry

@retry(tries=3, delay=0.2, backoff=2)
async def async_fetch(url: str) -> bytes:
    import aiohttp
    async with aiohttp.ClientSession() as s:
        async with s.get(url) as r:
            return await r.read()

result = asyncio.run(async_fetch("https://example.com"))
```

---

## Parallelize with error collection

```python
from yggdrasil.pyutils import parallelize

def safe_process(item: dict) -> dict | None:
    try:
        return {"id": item["id"], "result": item["value"] * 2}
    except Exception as e:
        print(f"Failed {item['id']}: {e}")
        return None

items = [{"id": i, "value": i * 1.5} for i in range(100)]
results = [r for r in parallelize(safe_process, items, max_workers=8) if r is not None]
print(f"Processed {len(results)}/{len(items)} successfully")
```
