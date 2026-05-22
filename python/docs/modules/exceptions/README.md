# yggdrasil.exceptions

Library-wide exception hierarchy. Every error yggdrasil raises derives from `YGGException`, so a single `except YGGException` catches everything the library generates.

## One-liner

```python
from yggdrasil.exceptions import YGGException, NotFoundError, TooManyRequests
```

## Hierarchy

```
YGGException
├── CastError                    # type / schema conversion failures
└── HTTPError
    ├── RequestError             # pre-send failures
    │   ├── AuthRequiredError
    │   ├── ConnectionError
    │   ├── TimeoutError
    │   │   ├── ConnectTimeoutError
    │   │   └── ReadTimeoutError
    │   └── ProxyError
    ├── ResponseError            # HTTP response errors
    │   ├── HTTPStatusError      # base for all status codes
    │   │   ├── ClientError (4xx)
    │   │   │   ├── BadRequest         (400)
    │   │   │   ├── UnauthorizedError  (401)
    │   │   │   ├── ForbiddenError     (403)
    │   │   │   ├── NotFoundError      (404)
    │   │   │   ├── MethodNotAllowed   (405)
    │   │   │   ├── ConflictError      (409)
    │   │   │   ├── GoneError          (410)
    │   │   │   ├── UnprocessableEntity (422)
    │   │   │   └── TooManyRequests    (429)
    │   │   └── ServerError (5xx)
    │   │       ├── InternalServerError (500)
    │   │       ├── BadGatewayError    (502)
    │   │       ├── ServiceUnavailable (503)
    │   │       └── GatewayTimeout     (504)
    │   ├── DecodeError
    │   ├── InvalidChunkLength
    │   └── IncompleteRead
    ├── PoolError
    │   ├── ClosedPoolError
    │   ├── EmptyPoolError
    │   └── HostChangedError
    ├── LocationError
    │   ├── LocationValueError
    │   └── LocationParseError
    ├── SSLError
    └── CacheError
```

## Catch-all pattern

```python
from yggdrasil.exceptions import YGGException

try:
    result = do_yggdrasil_things()
except YGGException as exc:
    # Every deliberate error the library raises lands here
    print("yggdrasil error:", exc)
```

## Narrowing to HTTP status errors

```python
from yggdrasil.exceptions import (
    NotFoundError,
    TooManyRequests,
    ForbiddenError,
    ServerError,
    HTTPStatusError,
)

try:
    response = session.get(url)
    response.raise_for_status()
except NotFoundError:
    print("Resource not found — skip or create")
except TooManyRequests as exc:
    retry_after = exc.response.headers.get("Retry-After")
    print(f"Rate limited — retry after {retry_after}s")
except ForbiddenError:
    print("No permission")
except ServerError:
    print("Server-side failure — safe to retry")
except HTTPStatusError as exc:
    print(f"HTTP {exc.response.status} — {exc}")
```

## Cast errors

```python
from yggdrasil.exceptions import CastError
from yggdrasil.data.cast.registry import convert

try:
    convert("not-a-date", "date")
except CastError as exc:
    print("Conversion failed:", exc)
```

## Connection and timeout errors

```python
from yggdrasil.exceptions import ConnectionError, TimeoutError, ConnectTimeoutError

try:
    session.get("https://unreachable.example.com")
except ConnectTimeoutError:
    print("Could not connect in time")
except TimeoutError:
    print("Request timed out")
except ConnectionError:
    print("Network error")
```

## Factory helpers

```python
from yggdrasil.exceptions import make_for_status, from_urllib3

# Raise the right subclass for an HTTP status code
exc = make_for_status(404, message="item not found")   # NotFoundError
exc = make_for_status(429, message="rate limited")     # TooManyRequests

# Translate a transport-layer exception into a yggdrasil exception.
# The HTTP transport ships in :mod:`yggdrasil._http_pool` (stdlib-backed,
# urllib3-shaped); :func:`from_urllib3` accepts any exception in that
# hierarchy and returns the matching :class:`HTTPError` subclass.
from yggdrasil._http_pool import exceptions as pool_exc
try:
    pass  # HTTP call
except pool_exc.MaxRetryError as raw:
    raise from_urllib3(raw) from raw
```

## Defining a new exception

New types go in `yggdrasil/exceptions/<area>.py` and are re-exported from `yggdrasil/exceptions/__init__.py`. **Never** define ad-hoc `class FooError(Exception)` in feature modules.

```python
# yggdrasil/exceptions/myarea.py
from .base import YGGException
from .http import NotFoundError

class WidgetNotFoundError(NotFoundError):
    """Raised when a widget cannot be found by ID."""
```
