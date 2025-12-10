# yggdrasil.pyutils

Utility decorators for parallelism and retry logic.

## `parallelize(executor_cls=ThreadPoolExecutor, max_workers=None, arg_index=0, timeout=None, return_exceptions=False, show_progress=False)`
Decorator that turns a function into a generator producing results from concurrent execution over one iterable argument.

```python
from yggdrasil.pyutils import parallelize

@parallelize(max_workers=8, show_progress=True)
def fetch(url):
    ...

for body in fetch(urls):
    handle(body)
```

- Accepts a custom executor class (thread or process) and supports passing an existing executor via `executor` kwarg.
- Preserves input order and can optionally return exceptions instead of raising.

## `retry(exceptions=Exception, tries=3, delay=0.5, backoff=2.0, max_delay=None, jitter=None, logger=None, reraise=True, timeout=None)`
Retry decorator that works for both sync and async callables with fixed or exponential backoff and optional jitter/timeout controls.

```python
from yggdrasil.pyutils import retry

@retry(tries=5, delay=0.2, backoff=2)
def unstable():
    ...
```

- Supports coroutine functions transparently.
- Optional logger hooks emit warnings/errors on retries and failures.
- `timeout` stops scheduling new retries once the total elapsed time exceeds the limit.
