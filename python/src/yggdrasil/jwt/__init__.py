"""yggdrasil JWT integration.

Parsing-only primitives for JSON Web Tokens (RFC 7519). Signature
verification is intentionally out of scope — that needs an algorithm
plug-in (PyJWT, ``cryptography``) and a key-resolution policy, and
both belong in the caller, not in a base-install primitive.

Quick start
-----------

    >>> from yggdrasil.jwt import JWTToken
    >>>
    >>> tok = JWTToken.parse("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.sig")
    >>> tok.alg, tok.sub
    ('HS256', '123')
    >>>
    >>> # Strip the Bearer prefix from an Authorization header in one call:
    >>> JWTToken.from_authorization("Bearer " + tok.raw).sub
    '123'
"""
from __future__ import annotations

from .token import JWTParseError, JWTToken


__all__ = [
    "JWTToken",
    "JWTParseError",
]
