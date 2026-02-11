# yggdrasil.pyutils

`yggdrasil.pyutils` provides practical runtime utilities used in pipelines, batch jobs, and services.

It is most helpful when you need reliability controls and cross-environment consistency without heavyweight frameworks.

---

## What you get

- Retry strategies (`retry`)
- Parallel execution helpers (`parallelize`)
- Dynamic imports (`modules`)
- Serialization helpers (`serde`)
- Waiting/backoff controls (`waiting_config`)
- In-memory utilities (`expiring_dict`, `dynamic_buffer`)

---

## Bootstrap: retry a flaky operation

```python
from yggdrasil.pyutils import retry

@retry(max_retries=3)
def fetch_remote_state():
    # your transient network call here
    return {"ok": True}

state = fetch_remote_state()
```

---

## Bootstrap: parallel map across work items

```python
from yggdrasil.pyutils import parallelize

values = [1, 2, 3, 4, 5]
results = parallelize(lambda x: x * 10, values)
print(results)
```

---

## Bootstrap: controlled waiting configuration

```python
from yggdrasil.pyutils.waiting_config import WaitingConfig

wait_cfg = WaitingConfig(
    timeout_seconds=60,
    retry_interval_seconds=2,
)

print(wait_cfg)
```

---

## Bootstrap: safe dynamic import

```python
from yggdrasil.pyutils.modules import import_module

json_module = import_module("json")
print(json_module.dumps({"ok": True}))
```

---

## Recommended usage in jobs

- Wrap all external API and cloud operations with retry policies.
- Use parallel utilities for independent item-level transformations.
- Store operation-level timing and retry metadata for observability.
