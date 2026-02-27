# yggdrasil.requests

Retry-enabled HTTP sessions for pipelines and services.

## Key export

```python
from yggdrasil.requests import YGGSession
```

`YGGSession` extends `requests.Session` with a pre-mounted retry adapter on both `http://` and `https://`.

```python
YGGSession(
    num_retry=4,      # retries for total, read, and connect errors
    headers=None,     # optional default headers dict
)
```

---

## Bootstrap: simple GET with retries

```python
from yggdrasil.requests import YGGSession

session = YGGSession(num_retry=4)
response = session.get("https://api.example.com/data", timeout=30)
response.raise_for_status()
data = response.json()
```

---

## Bootstrap: default headers

```python
import os
from yggdrasil.requests import YGGSession

session = YGGSession(
    num_retry=4,
    headers={
        "Authorization": f"Bearer {os.environ['API_TOKEN']}",
        "Accept": "application/json",
    },
)

resp = session.get("https://api.example.com/v1/items", timeout=30)
items = resp.json()
```

---

## Bootstrap: reuse session across calls

```python
from yggdrasil.requests import YGGSession

def make_session(token: str) -> YGGSession:
    return YGGSession(
        num_retry=5,
        headers={"Authorization": f"Bearer {token}"},
    )

session = make_session(token="...")

# all calls share the same retry adapter and headers
users = session.get("https://api.example.com/users", timeout=20).json()
orgs  = session.get("https://api.example.com/orgs",  timeout=20).json()
```

---

## Bootstrap: POST with JSON body

```python
from yggdrasil.requests import YGGSession

session = YGGSession(num_retry=3)

resp = session.post(
    "https://api.example.com/events",
    json={"event": "page_view", "user_id": 42},
    timeout=15,
)
resp.raise_for_status()
```

---

## Bootstrap: Azure MSAL authentication

```python
from yggdrasil.requests import MSALSession

session = MSALSession(
    tenant_id="<tenant-id>",
    client_id="<client-id>",
    client_secret="<secret>",
    scope=["https://management.azure.com/.default"],
)

resp = session.get("https://management.azure.com/subscriptions", timeout=30)
```

---

## Retry behavior

| Parameter | Default | Retries on |
|---|---|---|
| `total` | `num_retry` | any failure |
| `read` | `num_retry` | read timeouts |
| `connect` | `num_retry` | connection errors |
| `backoff_factor` | `0.1` | sleep = 0.1 × 2^(attempt-1) |
