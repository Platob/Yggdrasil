# yggdrasil.pyutils

Utility decorators for concurrency and retry workflows.

## When to use
- Parallelize work over an iterable without writing executor boilerplate.
- Add retry/backoff behavior to fragile operations.
- Apply the same helpers to both sync and async functions.

## `parallelize`
`parallelize(executor_cls=ThreadPoolExecutor, max_workers=None, arg_index=0, timeout=None, return_exceptions=False, show_progress=False)` turns a function into a generator producing results from concurrent execution over one iterable argument.

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

## `retry`
`retry(exceptions=Exception, tries=3, delay=0.5, backoff=2.0, max_delay=None, jitter=None, logger=None, reraise=True, timeout=None)` adds retry semantics to sync and async callables.

```python
from yggdrasil.pyutils import retry

@retry(tries=5, delay=0.2, backoff=2)
def unstable():
    ...
```

- Supports coroutine functions transparently.
- Optional logger hooks emit warnings/errors on retries and failures.
- `timeout` stops scheduling new retries once the total elapsed time exceeds the limit.

## Related modules
- [yggdrasil.requests](../requests/README.md) for session-level retries.
- [yggdrasil.libs](../libs/README.md) for dependency guards that pair well with retries.
