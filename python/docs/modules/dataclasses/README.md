# yggdrasil.dataclasses

Dataclass helpers with Arrow awareness, TTL-expiring caches, and waiting/polling utilities used by the casting and execution layers.

## One-liner

```python
from yggdrasil.dataclasses import ExpiringDict

cache: ExpiringDict[str, int] = ExpiringDict(default_ttl=60)
cache["key"] = 42
print(cache["key"])   # 42
```

## Dataclass → Arrow field

```python
from dataclasses import dataclass
from typing import Optional
from yggdrasil.dataclasses import dataclass_to_arrow_field
import pyarrow as pa

@dataclass
class Order:
    id:     int
    amount: float
    tag:    Optional[str] = None

field = dataclass_to_arrow_field(Order)
print(field)    # Field("order", struct<id: int64, amount: double, tag: string>)

# Use as a schema component
schema = pa.schema([field])
```

## ExpiringDict — TTL cache

Thread-safe dict with per-key TTL expiry, capacity eviction, and optional `on_evict` callback.

```python
from yggdrasil.dataclasses import ExpiringDict
import datetime

# 5-minute TTL
cache: ExpiringDict[str, dict] = ExpiringDict(default_ttl=300)
cache["user:42"] = {"name": "Alice", "role": "admin"}
print(cache.get("user:42"))

# Timedelta TTL
cache2: ExpiringDict[str, bytes] = ExpiringDict(
    default_ttl=datetime.timedelta(minutes=10),
    max_size=1000,
)

# Atomic get-or-set (thread-safe)
def load_user(user_id: str) -> dict:
    return fetch_from_db(user_id)

value = cache.get_or_set("user:42", load_user, "user:42")

# Evict callback (e.g. close file handles)
def on_evict(key, value):
    value.close()

cache3 = ExpiringDict(default_ttl=60, on_evict=on_evict)

# Bulk operations
cache.set_many({"a": 1, "b": 2, "c": 3})
cache.update({"d": 4})

# Purge expired entries manually
cache.purge_expired()
cache.clear()
```

Process-lifetime singleton cache (no TTL):

```python
from yggdrasil.dataclasses import ExpiringDict

_INSTANCES: ExpiringDict = ExpiringDict(default_ttl=None)
```

## WaitingConfig — polling with backoff

Used throughout the Databricks layer to wait for async operations (cluster start, job run, warehouse start).

```python
from yggdrasil.dataclasses import WaitingConfig
import datetime

# From a bool (True = default wait, False = no wait)
wait = WaitingConfig.from_(True)

# From seconds
wait = WaitingConfig.from_(30.0)

# From timedelta
wait = WaitingConfig.from_(datetime.timedelta(minutes=5))

# Full configuration
wait = WaitingConfig(
    timeout=120.0,          # seconds total
    poll_interval=2.0,      # seconds between polls
    max_poll_interval=30.0, # cap on exponential backoff
)

# Use in a polling loop
import time

def poll_until_ready(resource, wait: WaitingConfig):
    deadline = time.monotonic() + wait.timeout
    interval = wait.poll_interval
    while time.monotonic() < deadline:
        if resource.is_ready:
            return resource
        time.sleep(interval)
        interval = min(interval * 1.5, wait.max_poll_interval)
    raise TimeoutError("Resource not ready in time")
```

## Expiring — single value with TTL

```python
from yggdrasil.dataclasses import Expiring
import datetime

# Wrap a value with a 60-second TTL
token = Expiring(value="my-access-token", ttl=60)

if not token.is_expired():
    use(token.value)
else:
    token = Expiring(value=refresh_token(), ttl=60)
```

## Singleton pattern

```python
from yggdrasil.dataclasses import Singleton

class MyService(Singleton):
    def _singleton_key(self, host: str, **kwargs):
        return (type(self), host)

    def __init__(self, host: str):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.host = host
        self._connection = connect(host)

svc1 = MyService("example.com")
svc2 = MyService("example.com")
assert svc1 is svc2   # same instance
```

## Safe function wrapper

```python
from yggdrasil.dataclasses import SafeFunction

# Wrap a callable — catches and returns exceptions instead of raising
safe_fn = SafeFunction(int)

result = safe_fn("42")    # 42
result = safe_fn("bad")   # ValueError stored, not raised

if safe_fn.last_error is not None:
    print("Failed:", safe_fn.last_error)
```
