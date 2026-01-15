# yggdrasil.requests

HTTP session helpers focused on Azure AD client-credential authentication.

## When to use
- You need a requests `Session` that automatically attaches MSAL access tokens.
- You want a small wrapper around `requests.Session` with retry defaults.

## `MSALAuth`
Dataclass that holds tenant/client credentials and scopes. It automatically populates missing values from the environment and validates required fields.

Key helpers:
- `find_in_env(env=None, prefix=None)` – build an auth config from environment mappings.
- `export_to(mapping)` – write populated auth values back to a mapping.

## `MSALSession`
Session wrapper that acquires tokens via `msal.ConfidentialClientApplication` and injects `Authorization` headers on each request.

```python
from yggdrasil.requests import MSALAuth, MSALSession

auth = MSALAuth(
    tenant_id="...",
    client_id="...",
    client_secret="...",
    scopes=[".default"],
)

session = MSALSession(auth)
response = session.get("https://resource")
```

## `YGGSession`
Located in `yggdrasil.requests.session`, this is a retry-configured `requests.Session` for standard HTTP usage.

## Notes
- `MSALSession` requires the optional `msal` dependency.
- Tokens are cached in memory per session instance.
