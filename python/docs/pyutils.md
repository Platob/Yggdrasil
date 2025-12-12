# yggdrasil.pyutils

Utility decorators for concurrency and retry logic that keep pipelines robust.

## `parallelize`
`parallelize(executor_cls=ThreadPoolExecutor, max_workers=None, arg_index=0, timeout=None, return_exceptions=False, show_progress=False)` turns a function into a generator that executes concurrently over one iterable argument.

```python
from yggdrasil.pyutils import parallelize

@parallelize(max_workers=8, show_progress=True)
def fetch(url: str) -> bytes:
    ...

for body in fetch(urls):
    handle(body)
```

Key behaviors:
- Supports thread or process executors; pass an existing executor via the `executor` keyword override.
- Preserves input order and can optionally yield exceptions instead of raising.
- Honors a per-call `timeout` for scheduling new work.

## `retry`
`retry(exceptions=Exception, tries=3, delay=0.5, backoff=2.0, max_delay=None, jitter=None, logger=None, reraise=True, timeout=None)` adds retry semantics to sync and async callables.

```python
from yggdrasil.pyutils import retry

@retry(tries=5, delay=0.2, backoff=2)
def unstable() -> str:
    ...
```

Key behaviors:
- Works with coroutine functions transparently.
- Accepts fixed or exponential backoff with optional jitter.
- `timeout` stops scheduling retries once elapsed time exceeds the limit.
- Logger hooks emit warnings/errors on retries and failures when provided.
