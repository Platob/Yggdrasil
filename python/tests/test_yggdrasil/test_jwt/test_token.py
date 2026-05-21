"""Tests for :class:`yggdrasil.jwt.JWTToken`."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone

import pytest

from yggdrasil.jwt import JWTParseError, JWTToken
from yggdrasil.pickle.json import dumps as _json_dumps


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _build_token(
    header: dict,
    payload: dict,
    *,
    signature: bytes = b"signature-bytes",
) -> str:
    """Encode a JWT-shaped string for tests.

    Uses base64url without padding (the on-the-wire shape RFC 7515 §2
    mandates) so the parser exercises the padding-recovery branch.
    """
    h = _b64url(_json_dumps(header))
    p = _b64url(_json_dumps(payload))
    s = _b64url(signature) if signature else ""
    return f"{h}.{p}.{s}"


# ---------------------------------------------------------------------------
# happy-path parsing
# ---------------------------------------------------------------------------


def test_from_str_decodes_header_payload_signature():
    raw = _build_token(
        {"alg": "HS256", "typ": "JWT", "kid": "abc"},
        {"sub": "user-1", "iss": "issuer-x", "aud": "audience-y"},
        signature=b"\x00\x01\x02",
    )

    tok = JWTToken.from_str(raw)

    assert tok.alg == "HS256"
    assert tok.typ == "JWT"
    assert tok.kid == "abc"
    assert tok.sub == "user-1"
    assert tok.iss == "issuer-x"
    assert tok.aud == "audience-y"
    assert tok.signature == b"\x00\x01\x02"
    assert tok.raw == raw


def test_from_str_audience_list_normalized_to_tuple():
    raw = _build_token(
        {"alg": "RS256"},
        {"aud": ["api://a", "api://b"]},
    )
    assert JWTToken.from_str(raw).aud == ("api://a", "api://b")


def test_from_bytes_decodes_ascii_token():
    raw = _build_token({"alg": "HS256"}, {"sub": "bob"})
    tok = JWTToken.from_bytes(raw.encode("ascii"))
    assert tok.sub == "bob"


def test_from_dispatches_by_input_type():
    raw = _build_token({"alg": "HS256"}, {"sub": "dispatch"})
    # str → from_str
    assert JWTToken.from_(raw).sub == "dispatch"
    # bytes → from_bytes
    assert JWTToken.from_(raw.encode("ascii")).sub == "dispatch"
    # bytearray + memoryview also accepted
    assert JWTToken.from_(bytearray(raw, "ascii")).sub == "dispatch"
    assert JWTToken.from_(memoryview(raw.encode("ascii"))).sub == "dispatch"
    # JWTToken passthrough (identity short-circuit)
    tok = JWTToken.from_str(raw)
    assert JWTToken.from_(tok) is tok


def test_from_rejects_unsupported_type():
    with pytest.raises(JWTParseError, match="expects str, bytes, or JWTToken"):
        JWTToken.from_(12345)  # type: ignore[arg-type]


def test_from_str_strips_bearer_prefix_and_whitespace():
    raw = _build_token({"alg": "HS256"}, {"sub": "alice"})
    tok = JWTToken.from_str(f"  Bearer  {raw}  ")
    assert tok.sub == "alice"
    # The Bearer prefix and outer whitespace are stripped, but the
    # token bytes themselves are unchanged.
    assert tok.raw == raw


def test_from_str_unsecured_token_has_empty_signature():
    # alg=none style — trailing dot present, signature segment empty.
    raw = _build_token({"alg": "none"}, {"sub": "x"}, signature=b"")
    tok = JWTToken.from_str(raw)
    assert tok.signature == b""
    assert tok.signature_segment == ""


def test_from_authorization_strips_bearer():
    raw = _build_token({"alg": "HS256"}, {"sub": "carol"})
    tok = JWTToken.from_authorization(f"Bearer {raw}")
    assert tok is not None
    assert tok.sub == "carol"


def test_from_authorization_returns_none_for_missing_or_unrecognized():
    assert JWTToken.from_authorization(None) is None
    assert JWTToken.from_authorization("") is None
    assert JWTToken.from_authorization("Basic dXNlcjpwYXNz") is None
    assert JWTToken.from_authorization("Bearer not-a-jwt") is None


# ---------------------------------------------------------------------------
# expiry / nbf
# ---------------------------------------------------------------------------


def test_expires_at_returns_aware_utc_datetime():
    raw = _build_token({"alg": "HS256"}, {"exp": 1_700_000_000})
    tok = JWTToken.from_str(raw)
    expected = datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)
    assert tok.expires_at == expected
    assert tok.expires_at.tzinfo is timezone.utc


def test_is_expired_uses_now_and_leeway():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fresh = _build_token(
        {"alg": "HS256"},
        {"exp": (now + timedelta(seconds=60)).timestamp()},
    )
    stale = _build_token(
        {"alg": "HS256"},
        {"exp": (now - timedelta(seconds=60)).timestamp()},
    )

    assert JWTToken.from_str(fresh).is_expired(now=now) is False
    assert JWTToken.from_str(stale).is_expired(now=now) is True
    # Leeway of 120s widens the window enough to accept the stale token.
    assert JWTToken.from_str(stale).is_expired(now=now, leeway=120) is False


def test_is_expired_false_when_exp_missing():
    raw = _build_token({"alg": "HS256"}, {"sub": "no-exp"})
    assert JWTToken.from_str(raw).is_expired() is False


def test_is_not_yet_valid_uses_nbf():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    future = _build_token(
        {"alg": "HS256"},
        {"nbf": (now + timedelta(seconds=60)).timestamp()},
    )
    past = _build_token(
        {"alg": "HS256"},
        {"nbf": (now - timedelta(seconds=60)).timestamp()},
    )

    assert JWTToken.from_str(future).is_not_yet_valid(now=now) is True
    assert JWTToken.from_str(past).is_not_yet_valid(now=now) is False


# ---------------------------------------------------------------------------
# signing_input + repr
# ---------------------------------------------------------------------------


def test_signing_input_matches_header_dot_payload_segments():
    raw = _build_token({"alg": "HS256"}, {"sub": "x"})
    tok = JWTToken.from_str(raw)
    expected = f"{tok.header_segment}.{tok.payload_segment}".encode("ascii")
    assert tok.signing_input == expected


def test_repr_hides_signature():
    raw = _build_token({"alg": "HS256"}, {"sub": "secret-leak-test"})
    tok = JWTToken.from_str(raw)
    text = repr(tok)
    assert "HS256" in text
    assert "secret-leak-test" in text
    # Signature bytes / segment must not leak into repr.
    assert tok.signature_segment not in text


def test_str_returns_raw_token():
    raw = _build_token({"alg": "HS256"}, {"sub": "x"})
    tok = JWTToken.from_str(raw)
    assert str(tok) == raw


# ---------------------------------------------------------------------------
# error shapes
# ---------------------------------------------------------------------------


def test_from_str_rejects_wrong_segment_count():
    with pytest.raises(JWTParseError, match="three base64url segments"):
        JWTToken.from_str("only.two")


def test_from_str_rejects_bad_base64():
    # Valid shape regex-wise but the header segment is not valid JSON
    # after base64url decode.
    payload_seg = _b64url(_json_dumps({"sub": "x"}))
    raw = f"!!!!.{payload_seg}.sig"
    with pytest.raises(JWTParseError):
        JWTToken.from_str(raw)


def test_from_str_rejects_non_object_payload():
    header_seg = _b64url(_json_dumps({"alg": "HS256"}))
    # Payload decodes to a JSON array — RFC 7519 requires an object.
    payload_seg = _b64url(b"[1, 2, 3]")
    raw = f"{header_seg}.{payload_seg}.sig"
    with pytest.raises(JWTParseError, match="must be a JSON object"):
        JWTToken.from_str(raw)


def test_from_str_rejects_non_str():
    with pytest.raises(JWTParseError, match="expects str"):
        JWTToken.from_str(12345)  # type: ignore[arg-type]


def test_from_bytes_rejects_non_bytes():
    with pytest.raises(JWTParseError, match="expects bytes"):
        JWTToken.from_bytes("not-bytes")  # type: ignore[arg-type]


def test_from_bytes_rejects_non_ascii():
    with pytest.raises(JWTParseError, match="ASCII"):
        JWTToken.from_bytes("café.payload.sig".encode("utf-8"))


# ---------------------------------------------------------------------------
# dataclass surface
# ---------------------------------------------------------------------------


def test_token_is_frozen():
    raw = _build_token({"alg": "HS256"}, {"sub": "x"})
    tok = JWTToken.from_str(raw)
    with pytest.raises(Exception):  # FrozenInstanceError subclass of AttributeError
        tok.raw = "different"  # type: ignore[misc]


def test_equal_tokens_compare_equal():
    raw = _build_token({"alg": "HS256"}, {"sub": "x"})
    assert JWTToken.from_str(raw) == JWTToken.from_str(raw)
