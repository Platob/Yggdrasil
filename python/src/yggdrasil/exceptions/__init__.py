"""Library-wide exception types — the single canonical surface.

Every exception yggdrasil raises on its own derives from
:class:`YGGException` so callers can write::

    try:
        do_yggdrasil_things()
    except YGGException:
        ...

and catch every error this library deliberately raises in one branch.
Specialised types live in peer modules and are re-exported here so
``from yggdrasil.exceptions import <Anything>`` always works:

- :mod:`yggdrasil.exceptions.cast` → :class:`CastError`
- :mod:`yggdrasil.exceptions.http` → :class:`HTTPError` and the full
  HTTP status / connection / pool / location hierarchy.

When you need a new exception type, add it here (or in a peer file
under :mod:`yggdrasil.exceptions`) — don't define ad-hoc local
exception classes in feature modules. See ``AGENTS.md`` →
"Centralise exceptions in :mod:`yggdrasil.exceptions`" for the rule
and the worked example.
"""
from __future__ import annotations

from .api import (
    APIError,
    BadRequestError,
    ConflictError as APIConflictError,
    ForbiddenError as APIForbiddenError,
    MethodNotAllowedError,
    NotFoundError as APINotFoundError,
    TimeoutError as APITimeoutError,
    TooManyRequestsError,
    UnauthorizedError as APIUnauthorizedError,
    UnprocessableError,
    register_api_exception_handlers,
)
from .base import YGGException
from .cast import CastError
from .loki import LokiError, TokenBudgetExceeded
from .node import NodeBadRequestError, NodeError, NodeNotFoundError
from .http import (
    AuthRequiredError,
    BadGatewayError,
    BadRequest,
    CacheError,
    ClientError,
    ClosedPoolError,
    ConflictError,
    ConnectionError,
    ConnectTimeoutError,
    DecodeError,
    EmptyPoolError,
    ForbiddenError,
    GatewayTimeout,
    GoneError,
    HostChangedError,
    HTTPError,
    HTTPStatusError,
    IncompleteRead,
    InsecureRequestWarning,
    InternalServerError,
    InvalidChunkLength,
    LocationError,
    LocationParseError,
    LocationValueError,
    MethodNotAllowed,
    NotFoundError,
    PoolError,
    ProxyError,
    ReadTimeoutError,
    RequestError,
    ResponseError,
    SecurityWarning,
    ServerError,
    ServiceUnavailable,
    SSLError,
    TimeoutError,
    TooManyRequests,
    UnauthorizedError,
    UnprocessableEntity,
    from_urllib3,
    make_for_status,
)


__all__ = [
    # Root
    "YGGException",
    # API server errors
    "APIError",
    "BadRequestError",
    "APIConflictError",
    "APIForbiddenError",
    "APINotFoundError",
    "APITimeoutError",
    "APIUnauthorizedError",
    "MethodNotAllowedError",
    "TooManyRequestsError",
    "UnprocessableError",
    "register_api_exception_handlers",
    # Cast
    "CastError",
    "LokiError",
    "TokenBudgetExceeded",
    # Node
    "NodeError",
    "NodeNotFoundError",
    "NodeBadRequestError",
    # HTTP — base
    "HTTPError",
    # HTTP — request-bound
    "RequestError",
    "AuthRequiredError",
    "ConnectionError",
    "TimeoutError",
    "ConnectTimeoutError",
    "ReadTimeoutError",
    "ProxyError",
    # HTTP — response-bound
    "ResponseError",
    "HTTPStatusError",
    "ClientError",
    "BadRequest",
    "UnauthorizedError",
    "ForbiddenError",
    "NotFoundError",
    "MethodNotAllowed",
    "ConflictError",
    "GoneError",
    "UnprocessableEntity",
    "TooManyRequests",
    "ServerError",
    "InternalServerError",
    "BadGatewayError",
    "ServiceUnavailable",
    "GatewayTimeout",
    "DecodeError",
    "InvalidChunkLength",
    "IncompleteRead",
    # HTTP — pool
    "PoolError",
    "ClosedPoolError",
    "EmptyPoolError",
    "HostChangedError",
    # HTTP — location
    "LocationError",
    "LocationValueError",
    "LocationParseError",
    # HTTP — security / SSL
    "SecurityWarning",
    "InsecureRequestWarning",
    "SSLError",
    # HTTP — cache
    "CacheError",
    # HTTP — factory helpers
    "make_for_status",
    "from_urllib3",
]
