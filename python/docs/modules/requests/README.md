# yggdrasil.requests

HTTP helpers with retries and Azure MSAL authentication support.

## `YGGSession(num_retry=4, headers=None)`
Subclass of `requests.Session` that mounts an `HTTPAdapter` with retry support for both HTTP and HTTPS schemes. Default retry count is 4; custom headers are merged into session defaults.

```python
from yggdrasil.requests import YGGSession

with YGGSession(num_retry=5) as sess:
    resp = sess.get("https://example.com/api")
    resp.raise_for_status()
```

## `MSALAuth`
Dataclass carrying Azure authentication details (`tenant_id`, `client_id`, `client_secret`, `authority`, `scopes`).
- Automatically loads values from `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_SCOPES`, and optionally `AZURE_AUTHORITY`.
- `find_in_env` discovers credentials in a mapping.
- `export_to` writes populated values back to an environment mapping.

## `MSALSession`
Session wrapper that uses `MSALAuth` to acquire and cache access tokens via `msal.ConfidentialClientApplication`. Falls back to plain `YGGSession` if `msal` is not installed.

```python
from yggdrasil.requests import MSALAuth, MSALSession

auth = MSALAuth(tenant_id="...", client_id="...", client_secret="...", scopes=[".default"])
sess = MSALSession(auth)
response = sess.get("https://resource")
```

## Navigation
- [Module overview](../../modules.md)
- [Dataclasses](../dataclasses/README.md)
- [Libs](../libs/README.md)
- [Requests](./README.md)
- [Types](../types/README.md)
- [Databricks](../databricks/README.md)
