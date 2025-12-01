# HTTP sessions and Azure AD auth

Resilient HTTP client helpers plus optional MSAL integration for Azure Active
Directory.

## `YGGSession`
- Subclass of `requests.Session` that mounts retry-enabled adapters for HTTP and
  HTTPS with configurable retry counts.
- Accepts optional default headers merged into each request.
- Use as a context manager or drop-in replacement wherever `requests.Session`
  is expected.

## MSAL-based authentication
- `MSALAuth` dataclass reads credentials from arguments or `AZURE_*`
  environment variables (`TENANT_ID`, `CLIENT_ID`, `CLIENT_SECRET`, `SCOPES`,
  optional `AUTHORITY`).
- Provides `refresh()`/`access_token`/`authorization` helpers and
  `export_to()` for populating environment variables.
- `MSALSession` wraps `YGGSession`, automatically injecting a Bearer token into
  outbound requests via `prepare_request`.
- `MSALAuth.find_in_env` offers discovery from arbitrary environment mappings
  (with an overridable prefix) to simplify configuration loading.

## Example
```python
from yggdrasil.requests import MSALAuth

auth = MSALAuth.find_in_env()
session = auth.requests_session()
response = session.get("https://graph.microsoft.com/v1.0/me")
response.raise_for_status()
```
