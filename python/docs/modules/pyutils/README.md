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
    from yggdrasil.io.http_ import HTTPSession
    return HTTPSession().get(url).content

@parallelize(max_workers=10)
def fetch_all(url: str) -> bytes:
    return fetch(url)

pages = list(fetch_all(urls))
```

## HTTP fan-out pattern

```python
from yggdrasil.pyutils import parallelize
from yggdrasil.io.http_ import HTTPSession

http = HTTPSession()

def fetch_page(page: int) -> list:
    return http.get("https://api.example.com/items", params={"page": page}).json()["items"]

@parallelize(max_workers=5)
def fetch_pages(page: int) -> list:
    return fetch_page(page)

all_items = [item for page in fetch_pages(range(1, 21)) for item in page]
print(f"Total: {len(all_items)} items")
```
