# yggdrasil.requests

This module wraps HTTP request workflows with Yggdrasil-friendly defaults, including authentication support modules.

Use it to standardize service-to-service communication in ETL jobs, orchestration scripts, and backend services.

---

## Core ideas

- Session-style request handling
- Integration points for token-based authentication (including MSAL-oriented flows)
- Reusable request setup for enterprise APIs

---

## Bootstrap: create a reusable session

```python
from yggdrasil.requests import requests

session = requests.Session()
response = session.get("https://httpbin.org/get", timeout=30)
print(response.status_code)
```

---

## Bootstrap: shared headers + retries pattern

```python
from yggdrasil.requests import requests

session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "User-Agent": "yggdrasil-client/1.0",
})

response = session.get("https://httpbin.org/headers", timeout=30)
print(response.json())
```

---

## Bootstrap: authenticated API pattern

```python
import os
from yggdrasil.requests import requests

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {os.environ['ACCESS_TOKEN']}",
})

response = session.get("https://api.example.com/v1/me", timeout=30)
```

---

## Best practices

- Reuse one session per service endpoint group.
- Put auth/token acquisition in a single bootstrap layer.
- Pair with `yggdrasil.pyutils.retry` for transient failures.
