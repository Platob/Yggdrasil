# yggdrasil.pyutils

Utility helpers for retries, parallel execution, environment management, and callable serialization.

## When to use
- Parallelize workloads with minimal executor boilerplate.
- Add retry/backoff behavior to fragile operations.
- Manage isolated Python environments (uv/pip) or serialize callables for remote execution.

## Core exports
### `parallelize`
Decorator that turns a function into a generator producing results from concurrent execution over one iterable argument.

```python
from yggdrasil.pyutils import parallelize

@parallelize(max_workers=8, show_progress=True)
def fetch(url):
    ...

for body in fetch(urls):
    handle(body)
```

### `retry`
Retry decorator that works for both sync and async functions.

```python
from yggdrasil.pyutils import retry

@retry(tries=5, delay=0.2, backoff=2)
def unstable():
    ...
```

### `PythonEnv`
Helper for managing isolated Python environments (create, install requirements, run commands) with uv/pip.
The `yggenv` CLI entry point is backed by `PythonEnv.cli`.

### `CallableSerde`
Serialize callables for cross-process or remote execution, with optional compression and import-by-reference behavior.

## Additional helpers
- `expiring_dict.ExpiringDict` – TTL-backed caching.
- `modules.PipIndexSettings` – helper for pip/uv index configuration.
- `equality` utilities for diffing dictionaries.

## Related modules
- [yggdrasil.requests](../requests/README.md) for request sessions that pair well with retries.
- [yggdrasil.databricks](../databricks/README.md) for remote execution workflows using `CallableSerde`.
