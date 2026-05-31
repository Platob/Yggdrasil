"""SigV4 signer parity tests — verified byte-for-byte against botocore.

The pure-HTTP S3 path signs its own requests (no boto3 on the data plane). To
prove the signer is correct without a live bucket, we sign the *exact same wire
URL* with both :class:`yggdrasil.aws.fs.sigv4.SigV4Signer` and
``botocore.auth.S3SigV4Auth`` (the reference implementation) under a frozen
clock and assert the ``Authorization`` headers are identical.
"""
from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.url import URL
from yggdrasil.aws.fs.sigv4 import SigV4Signer, EMPTY_PAYLOAD_SHA256, sha256_hex

botocore_auth = pytest.importorskip("botocore.auth")
from botocore.auth import S3SigV4Auth  # noqa: E402
from botocore.credentials import Credentials  # noqa: E402
from botocore.awsrequest import AWSRequest  # noqa: E402


_AK = "AKIDEXAMPLE"
_SK = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
_REGION = "us-east-1"
_FIXED = dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)


@pytest.fixture(autouse=True)
def _freeze_botocore_clock(monkeypatch):
    # botocore stamps the signing time from ``get_current_datetime()``; pin it
    # so both signers use the same instant.
    monkeypatch.setattr(botocore_auth, "get_current_datetime", lambda: _FIXED)


def _both(method, url_str, headers, payload, token=None):
    url = URL.from_(url_str)
    wire = str(url)  # the exact URL that goes on the wire — sign what you send

    signer = SigV4Signer(region=_REGION, credentials_provider=lambda: (_AK, _SK, token))
    mine = signer.sign(method, url, headers=headers, payload_sha256=payload, now=_FIXED)

    req = AWSRequest(method=method, url=wire, headers=dict(headers or {}))
    req.context["payload_signing_enabled"] = False
    auth = S3SigV4Auth(Credentials(_AK, _SK, token), "s3", _REGION)
    auth.payload = lambda r: payload  # sign the same body hash both ways
    auth.add_auth(req)
    return mine, dict(req.headers)


CASES = [
    ("GET", "https://b.s3.us-east-1.amazonaws.com/path/to/key.parquet", {}, EMPTY_PAYLOAD_SHA256, None),
    ("GET", "https://b.s3.us-east-1.amazonaws.com/a/b%20c/d.txt", {}, EMPTY_PAYLOAD_SHA256, None),
    ("HEAD", "https://b.s3.us-east-1.amazonaws.com/obj", {}, EMPTY_PAYLOAD_SHA256, None),
    ("GET", "https://b.s3.us-east-1.amazonaws.com/?list-type=2&prefix=foo/bar&delimiter=/", {}, EMPTY_PAYLOAD_SHA256, None),
    ("PUT", "https://b.s3.us-east-1.amazonaws.com/obj", {"content-type": "text/plain"}, sha256_hex(b"hi"), None),
    ("GET", "https://b.s3.us-east-1.amazonaws.com/big.bin", {"range": "bytes=0-1023"}, EMPTY_PAYLOAD_SHA256, None),
    ("DELETE", "https://b.s3.us-east-1.amazonaws.com/k", {}, EMPTY_PAYLOAD_SHA256, "FwoTOKEN=="),
    ("POST", "https://b.s3.us-east-1.amazonaws.com/k?uploads=", {}, sha256_hex(b""), "TMPTOKEN"),
    ("POST", "https://b.s3.us-east-1.amazonaws.com/k?uploadId=abc&partNumber=2", {}, sha256_hex(b"part"), None),
]


@pytest.mark.parametrize("method,url,headers,payload,token", CASES)
def test_authorization_matches_botocore(method, url, headers, payload, token):
    mine, boto = _both(method, url, headers, payload, token)
    assert mine["Authorization"] == boto["Authorization"]


def test_required_headers_present():
    signer = SigV4Signer(region=_REGION, credentials_provider=lambda: (_AK, _SK, "TOK"))
    out = signer.sign("GET", URL.from_("https://b.s3.us-east-1.amazonaws.com/k"), now=_FIXED)
    assert out["Host"] == "b.s3.us-east-1.amazonaws.com"
    assert out["x-amz-date"] == "20240102T030405Z"
    assert out["x-amz-content-sha256"] == EMPTY_PAYLOAD_SHA256
    assert out["x-amz-security-token"] == "TOK"
    assert out["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20240102/us-east-1/s3/aws4_request")


def test_no_token_omits_security_header():
    signer = SigV4Signer(region=_REGION, credentials_provider=lambda: (_AK, _SK, None))
    out = signer.sign("GET", URL.from_("https://b.s3.us-east-1.amazonaws.com/k"), now=_FIXED)
    assert "x-amz-security-token" not in out
