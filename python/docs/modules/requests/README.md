# yggdrasil.requests

Legacy/simple retry-enabled requests session plus MSAL-enabled variant.

## YGGSession

```python
from yggdrasil.requests import YGGSession

session = YGGSession(num_retry=4)
resp = session.get("https://example.com", timeout=10)
print(resp.status_code)
```

## MSALSession (Azure scenarios)

```python
from yggdrasil.requests import MSALSession

msal_session = MSALSession()
```

For new HTTP features (URL resources, caching, pagination helpers), prefer `yggdrasil.io.http_.HTTPSession`.
