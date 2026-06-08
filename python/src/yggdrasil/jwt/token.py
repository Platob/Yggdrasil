"""JWTToken — parse a JSON Web Token from a base-install dependency footprint.

This module covers the **parsing** half of the JWT contract (RFC 7519):
splitting the three base64url segments, decoding the header / payload
JSON, exposing the standard claim accessors. Signature verification is
left out on purpose — it needs algorithm-specific crypto and a
key-resolution policy, which belong in the caller, not in a
base-install primitive. Implementation uses only :mod:`base64` from
the stdlib and :mod:`yggdrasil.pickle.json` (``orjson``-backed) — no
third-party JWT library is involved.

Quick start
-----------

    >>> from yggdrasil.jwt import JWTToken
    >>>
    >>> tok = JWTToken.from_("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.sig")
    >>> tok.alg
    'HS256'
    >>> tok.sub
    '123'
    >>> tok.is_expired()
    False

Authorization header convenience::

    >>> tok = JWTToken.from_authorization("Bearer eyJhbGciOiJIUzI1NiJ9...")
"""
from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from yggdrasil.pickle.json import loads as _json_loads


__all__ = ["JWTToken", "JWTParseError"]


# Three base64url segments joined by dots; signature may be empty for
# the unsecured ``alg=none`` shape (the trailing dot is still required).
_JWT_RE = re.compile(r"^([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]*)$")

_BEARER_RE = re.compile(r"^\s*Bearer\s+(.+?)\s*$", re.IGNORECASE)


class JWTParseError(ValueError):
    """Raised when a value can't be parsed as a JWT.

    Subclasses :class:`ValueError` so callers that already catch the
    generic shape (``except ValueError:``) keep working.
    """


def _b64url_decode(segment: str) -> bytes:
    # RFC 7515 §2: base64url, no padding on the wire. Python's decoder
    # wants the padding, so pad to a multiple of 4 here.
    padding = (-len(segment)) % 4
    try:
        return base64.urlsafe_b64decode(segment + ("=" * padding))
    except (binascii.Error, ValueError) as exc:
        raise JWTParseError(
            f"Invalid base64url in JWT segment: {segment!r}. "
            "Segments must use the RFC 4648 base64url alphabet "
            "(A-Z, a-z, 0-9, '-', '_') with no padding."
        ) from exc


def _decode_json_segment(segment: str, *, kind: str) -> Mapping[str, Any]:
    raw = _b64url_decode(segment)
    try:
        decoded = _json_loads(raw)
    except Exception as exc:  # noqa: BLE001 — narrow next line
        raise JWTParseError(
            f"JWT {kind} is not valid JSON. Got {raw!r}. "
            "The base64url payload must decode to a JSON object."
        ) from exc
    if not isinstance(decoded, dict):
        raise JWTParseError(
            f"JWT {kind} must be a JSON object, got {type(decoded).__name__}. "
            "RFC 7519 requires both the header and payload to be JSON objects."
        )
    return decoded


def _strip_bearer(value: str) -> str:
    match = _BEARER_RE.match(value)
    return match.group(1) if match else value


def _epoch_to_aware(value: Any) -> datetime | None:
    """Turn a numeric ``NumericDate`` claim into a UTC ``datetime``.

    Returns ``None`` when the value is missing or non-numeric — JWT
    libs in the wild emit string seconds sometimes, so be forgiving on
    input (rule: forgiving on shape, strict on meaning).
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class JWTToken:
    """A parsed JSON Web Token (RFC 7519).

    Holds both the **decoded** view (``header`` / ``payload`` /
    ``signature``) and the **raw** view (``header_segment`` /
    ``payload_segment`` / ``signature_segment``, plus ``raw``). The raw
    segments are kept so a caller that wants to verify the signature
    later has the exact bytes that were signed (``header.payload``)
    without re-encoding — base64url round-trips aren't byte-stable
    when the original token used no padding (the wire shape).

    The class doesn't verify the signature — that needs an algorithm
    plug-in and a key resolver, which the caller is in a better
    position to wire up. Parsing alone is the common use case
    (peek at ``exp`` / ``sub`` / ``aud`` before forwarding the token).
    """

    raw: str
    header_segment: str
    payload_segment: str
    signature_segment: str
    header: Mapping[str, Any] = field(repr=False)
    payload: Mapping[str, Any] = field(repr=False)
    signature: bytes = field(repr=False)

    # --- constructors --------------------------------------------------

    @classmethod
    def from_(cls, value: Any) -> "JWTToken":
        """Generic dispatch — parse a JWT from whatever the caller has.

        Accepts:

        * an existing :class:`JWTToken` (returned as-is, identity short-circuit),
        * a :class:`str` — delegated to :meth:`from_str`,
        * :class:`bytes` / :class:`bytearray` / :class:`memoryview` —
          delegated to :meth:`from_bytes`.

        Other shapes raise :class:`JWTParseError` with the value's
        type so the caller sees what they passed.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return cls.from_bytes(bytes(value))
        raise JWTParseError(
            f"JWTToken.from_ expects str, bytes, or JWTToken; "
            f"got {type(value).__name__}."
        )

    @classmethod
    def from_str(cls, value: str) -> "JWTToken":
        """Parse a JWT from a string.

        Accepts the bare ``a.b.c`` form and the ``Bearer a.b.c`` form
        from an ``Authorization`` header — the prefix is stripped
        transparently. Surrounding whitespace is tolerated.
        """
        if not isinstance(value, str):
            raise JWTParseError(
                f"JWTToken.from_str expects str, got {type(value).__name__}. "
                "Use JWTToken.from_(value) for generic dispatch."
            )

        candidate = _strip_bearer(value.strip())
        match = _JWT_RE.match(candidate)
        if not match:
            # Show a truncated preview so the message stays log-safe
            # for long tokens; full token would leak credentials anyway.
            preview = candidate[:24] + "…" if len(candidate) > 24 else candidate
            raise JWTParseError(
                f"Not a JWT: {preview!r}. "
                "Expected three base64url segments separated by '.' "
                "(header.payload.signature)."
            )

        header_seg, payload_seg, signature_seg = match.group(1, 2, 3)
        header = _decode_json_segment(header_seg, kind="header")
        payload = _decode_json_segment(payload_seg, kind="payload")
        signature = _b64url_decode(signature_seg) if signature_seg else b""

        return cls(
            raw=candidate,
            header_segment=header_seg,
            payload_segment=payload_seg,
            signature_segment=signature_seg,
            header=header,
            payload=payload,
            signature=signature,
        )

    @classmethod
    def from_bytes(cls, value: bytes) -> "JWTToken":
        """Parse a JWT from raw bytes — must be ASCII (base64url + dots)."""
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise JWTParseError(
                f"JWTToken.from_bytes expects bytes, got {type(value).__name__}."
            )
        try:
            text = bytes(value).decode("ascii")
        except UnicodeDecodeError as exc:
            raise JWTParseError(
                "JWT bytes must be ASCII (base64url + dots). "
                "Got non-ASCII bytes."
            ) from exc
        return cls.from_str(text)

    @classmethod
    def from_authorization(cls, header_value: str | None) -> "JWTToken | None":
        """Parse the JWT out of an ``Authorization`` header value.

        Returns ``None`` when the value is empty, missing, or doesn't
        carry a Bearer-style token — callers usually want a no-token
        path instead of a try/except. Use :meth:`from_str` directly if a
        missing token should raise.
        """
        if not header_value:
            return None
        candidate = _strip_bearer(header_value.strip())
        if not _JWT_RE.match(candidate):
            return None
        return cls.from_str(candidate)

    # --- header claims -------------------------------------------------

    @property
    def alg(self) -> str | None:
        """``alg`` header claim — the signing algorithm (e.g. ``HS256``)."""
        value = self.header.get("alg")
        return value if isinstance(value, str) else None

    @property
    def typ(self) -> str | None:
        """``typ`` header claim — usually ``JWT``."""
        value = self.header.get("typ")
        return value if isinstance(value, str) else None

    @property
    def kid(self) -> str | None:
        """``kid`` header claim — key identifier used for verification."""
        value = self.header.get("kid")
        return value if isinstance(value, str) else None

    # --- payload claims (RFC 7519 §4.1) --------------------------------

    @property
    def iss(self) -> str | None:
        """``iss`` claim — issuer."""
        value = self.payload.get("iss")
        return value if isinstance(value, str) else None

    @property
    def sub(self) -> str | None:
        """``sub`` claim — subject."""
        value = self.payload.get("sub")
        return value if isinstance(value, str) else None

    @property
    def aud(self) -> str | tuple[str, ...] | None:
        """``aud`` claim — audience.

        RFC 7519 allows either a single string or an array of strings.
        Lists are normalized to a tuple so the value stays hashable
        and the token instance stays effectively immutable.
        """
        value = self.payload.get("aud")
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return tuple(v for v in value if isinstance(v, str))
        return None

    @property
    def jti(self) -> str | None:
        """``jti`` claim — JWT ID (unique per token)."""
        value = self.payload.get("jti")
        return value if isinstance(value, str) else None

    @property
    def exp(self) -> int | float | None:
        """``exp`` claim — raw NumericDate (seconds since epoch)."""
        value = self.payload.get("exp")
        if isinstance(value, bool):
            return None
        return value if isinstance(value, (int, float)) else None

    @property
    def iat(self) -> int | float | None:
        """``iat`` claim — raw NumericDate (issued-at seconds)."""
        value = self.payload.get("iat")
        if isinstance(value, bool):
            return None
        return value if isinstance(value, (int, float)) else None

    @property
    def nbf(self) -> int | float | None:
        """``nbf`` claim — raw NumericDate (not-before seconds)."""
        value = self.payload.get("nbf")
        if isinstance(value, bool):
            return None
        return value if isinstance(value, (int, float)) else None

    @property
    def expires_at(self) -> datetime | None:
        """``exp`` as a UTC :class:`~datetime.datetime`, or ``None``."""
        return _epoch_to_aware(self.payload.get("exp"))

    @property
    def issued_at(self) -> datetime | None:
        """``iat`` as a UTC :class:`~datetime.datetime`, or ``None``."""
        return _epoch_to_aware(self.payload.get("iat"))

    @property
    def not_before_at(self) -> datetime | None:
        """``nbf`` as a UTC :class:`~datetime.datetime`, or ``None``."""
        return _epoch_to_aware(self.payload.get("nbf"))

    # --- temporal checks ----------------------------------------------

    def is_expired(self, *, now: datetime | None = None, leeway: float = 0.0) -> bool:
        """Return whether the token's ``exp`` has passed.

        Tokens without an ``exp`` claim are treated as **non-expiring**
        (returns ``False``) — the general "absence of a claim is
        permissive" reading of RFC 7519. If your verifier requires
        ``exp``, gate on ``token.exp is None`` before calling this.

        *leeway* (seconds) widens the window to absorb clock skew
        between issuer and verifier.
        """
        expires = self.expires_at
        if expires is None:
            return False
        reference = now or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        return reference.timestamp() > (expires.timestamp() + leeway)

    def is_not_yet_valid(
        self, *, now: datetime | None = None, leeway: float = 0.0,
    ) -> bool:
        """Return whether ``nbf`` is still in the future.

        Mirrors :meth:`is_expired`'s shape — *leeway* widens the
        window in the caller's favor. Tokens without ``nbf`` are
        always valid by this check.
        """
        nbf = self.not_before_at
        if nbf is None:
            return False
        reference = now or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        return reference.timestamp() < (nbf.timestamp() - leeway)

    # --- signing-input ------------------------------------------------

    @property
    def signing_input(self) -> bytes:
        """The exact bytes a verifier feeds into the signing algorithm.

        Defined by RFC 7515 §5.2 as
        ``ASCII(BASE64URL(header)) || '.' || ASCII(BASE64URL(payload))``
        — the two raw segments with the dot between them, no
        re-encoding. Surfaced so a future signature-verification
        helper doesn't have to reconstruct it from the decoded views.
        """
        return f"{self.header_segment}.{self.payload_segment}".encode("ascii")

    # --- dunder -------------------------------------------------------

    def __str__(self) -> str:
        return self.raw

    def __repr__(self) -> str:
        # Keep the signature out of the repr — tokens leak credentials
        # by design, and logs are the most common leak path.
        alg = self.alg or "?"
        sub = self.sub or self.iss or "?"
        return f"<JWTToken alg={alg!r} sub={sub!r}>"
