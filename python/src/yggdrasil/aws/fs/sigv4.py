"""AWS Signature Version 4 signing for S3 over pure HTTP.

A small, dependency-free SigV4 signer (S3 flavor) — enough to sign the
``GET`` / ``HEAD`` / ``PUT`` / ``POST`` / ``DELETE`` requests
:class:`yggdrasil.aws.fs.path.S3Bucket` issues over
:class:`yggdrasil.http_.session.HTTPSession`, without dragging botocore's S3
client onto the data path. botocore is still the *credential* source (env /
profile / SSO / STS / instance-metadata resolution lives there); this module
only turns a frozen ``(access_key, secret_key, token)`` triple plus a request
into the ``Authorization`` + ``x-amz-*`` headers AWS expects.

S3-specific rules (vs the generic SigV4):

* the canonical URI is **not** path-normalized (``a/./b`` and ``a//b`` are
  meaningful S3 keys) and ``/`` is left unescaped;
* the payload hash may be a real SHA-256 or the literal ``UNSIGNED-PAYLOAD``
  (used for large streamed bodies over TLS, where re-hashing the whole object
  just to sign it would defeat streaming).

Verified byte-for-byte against ``botocore.auth.S3SigV4Auth`` — see
``tests/test_yggdrasil/test_aws/test_sigv4.py``.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
from typing import Mapping, Optional

from yggdrasil.url import URL

__all__ = ["SigV4Signer", "EMPTY_PAYLOAD_SHA256", "UNSIGNED_PAYLOAD", "sha256_hex"]

_ALGORITHM = "AWS4-HMAC-SHA256"
_AMZ_DATE_FMT = "%Y%m%dT%H%M%SZ"
_DATESTAMP_FMT = "%Y%m%d"

#: SHA-256 of the empty string — the payload hash for body-less requests.
EMPTY_PAYLOAD_SHA256 = hashlib.sha256(b"").hexdigest()
#: Sentinel payload hash for TLS-streamed bodies (no whole-object re-hash).
UNSIGNED_PAYLOAD = "UNSIGNED-PAYLOAD"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SigV4Signer:
    """Signs S3 requests with AWS Signature V4.

    Stateless w.r.t. the request — credentials are pulled fresh from
    ``credentials_provider`` on every :meth:`sign` so a rotating STS token is
    always current. ``region`` / ``service`` are fixed per bucket.
    """

    def __init__(
        self,
        *,
        region: str,
        credentials_provider,
        service: str = "s3",
    ) -> None:
        #: ``() -> (access_key, secret_key, session_token | None)``.
        self._credentials = credentials_provider
        self.region = region
        self.service = service

    def sign(
        self,
        method: str,
        url: URL,
        *,
        headers: Optional[Mapping[str, str]] = None,
        payload_sha256: str = EMPTY_PAYLOAD_SHA256,
        now: Optional[_dt.datetime] = None,
    ) -> "dict[str, str]":
        """Return the headers to add to *method url* so it's SigV4-authorized.

        The returned dict always carries ``Host``, ``x-amz-date``,
        ``x-amz-content-sha256`` and ``Authorization`` (plus
        ``x-amz-security-token`` when the credentials are temporary). Merge it
        into the request headers and send as-is.
        """
        access_key, secret_key, token = self._credentials()
        now = now or _dt.datetime.now(_dt.timezone.utc)
        amz_date = now.strftime(_AMZ_DATE_FMT)
        datestamp = now.strftime(_DATESTAMP_FMT)

        host = url.host or ""
        if url.port:
            host = f"{host}:{url.port}"

        # Headers that participate in the signature. Start from the caller's
        # (lower-cased) then stamp the required x-amz-* set.
        signed: dict[str, str] = {}
        for k, v in (headers or {}).items():
            signed[k.lower()] = str(v).strip()
        signed["host"] = host
        signed["x-amz-date"] = amz_date
        signed["x-amz-content-sha256"] = payload_sha256
        if token:
            signed["x-amz-security-token"] = token

        signed_header_names = sorted(signed)
        canonical_headers = "".join(f"{n}:{signed[n]}\n" for n in signed_header_names)
        signed_headers = ";".join(signed_header_names)

        canonical_request = "\n".join([
            method.upper(),
            url.path or "/",
            self._canonical_query(url),
            canonical_headers,
            signed_headers,
            payload_sha256,
        ])

        scope = f"{datestamp}/{self.region}/{self.service}/aws4_request"
        string_to_sign = "\n".join([
            _ALGORITHM,
            amz_date,
            scope,
            sha256_hex(canonical_request.encode("utf-8")),
        ])

        signing_key = self._signing_key(secret_key, datestamp)
        signature = hmac.new(
            signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        authorization = (
            f"{_ALGORITHM} Credential={access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        out = {
            "Host": host,
            "x-amz-date": amz_date,
            "x-amz-content-sha256": payload_sha256,
            "Authorization": authorization,
        }
        if token:
            out["x-amz-security-token"] = token
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_query(url: URL) -> str:
        # S3 canonicalizes the query exactly as it appears on the wire: split
        # the (already percent-encoded) query string into ``key=value`` pairs,
        # sort lexicographically, and join — no re-encoding. The signer and
        # the request share one URL, so whatever ``url.query`` holds is both
        # signed and sent (S3 validates against what it receives).
        query = getattr(url, "query", None)
        if not query:
            return ""
        pairs = []
        for pair in query.split("&"):
            key, _, value = pair.partition("=")
            pairs.append((key, value))
        return "&".join(f"{k}={v}" for k, v in sorted(pairs))

    def _signing_key(self, secret_key: str, datestamp: str) -> bytes:
        def _hmac(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        k_date = _hmac(("AWS4" + secret_key).encode("utf-8"), datestamp)
        k_region = _hmac(k_date, self.region)
        k_service = _hmac(k_region, self.service)
        return _hmac(k_service, "aws4_request")
