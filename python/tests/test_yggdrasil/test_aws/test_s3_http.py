"""S3HttpClient ops against an in-memory S3 transport.

Signing is proven against botocore in ``test_sigv4``; here we drive the REST
verbs (head / ranged get / put / delete / list / multi-delete / multipart)
through a dict-backed fake transport and assert the request shaping + XML
round-trips. The fake doubles as a reusable in-memory S3 for path-layer tests.
"""
from __future__ import annotations

import pytest

from yggdrasil.aws.fs.s3_http import S3HttpClient, S3NotFound, S3Response
from yggdrasil.aws.fs.sigv4 import SigV4Signer
from yggdrasil.url import URL

from tests.test_yggdrasil.test_aws._fake_s3 import FakeS3


@pytest.fixture
def client():
    fake = FakeS3()
    signer = SigV4Signer(region="us-east-1", credentials_provider=lambda: ("AK", "SK", None))
    c = S3HttpClient(
        bucket="bkt",
        endpoint=URL.from_("https://bkt.s3.us-east-1.amazonaws.com"),
        signer=signer,
        transport=fake,
    )
    return c, fake


def test_put_get_head_roundtrip(client):
    c, _ = client
    c.put("dir/obj.bin", b"hello world", content_type="application/octet-stream")
    assert bytes(c.get("dir/obj.bin").content) == b"hello world"
    head = c.head("dir/obj.bin")
    assert head is not None and head.headers["Content-Length"] == "11"
    assert c.head("missing") is None


def test_ranged_get(client):
    c, _ = client
    c.put("f", b"0123456789")
    assert bytes(c.get("f", start=2, length=4).content) == b"2345"
    assert bytes(c.get("f", start=5).content) == b"56789"


def test_get_missing_raises_notfound(client):
    c, _ = client
    with pytest.raises(S3NotFound):
        c.get("nope")


def test_delete(client):
    c, fake = client
    c.put("x", b"1")
    c.delete("x")
    assert "x" not in fake.objects


def test_list_with_delimiter(client):
    c, _ = client
    for k in ["a/1", "a/2", "a/sub/3", "b/4"]:
        c.put(k, b"z")
    page = c.list_page("a/", delimiter="/")
    assert sorted(o.key for o in page.objects) == ["a/1", "a/2"]
    assert page.prefixes == ["a/sub/"]


def test_list_recursive(client):
    c, _ = client
    for k in ["p/1", "p/q/2", "p/q/r/3"]:
        c.put(k, b"z")
    keys = [o.key for page in c.list("p/") for o in page.objects]
    assert sorted(keys) == ["p/1", "p/q/2", "p/q/r/3"]


def test_delete_batch(client):
    c, fake = client
    for k in ["d1", "d2", "d3"]:
        c.put(k, b"z")
    c.delete_batch(["d1", "d3"])
    assert set(fake.objects) == {"d2"}


def test_multipart_upload(client):
    c, fake = client
    chunks = [b"A" * 8, b"B" * 8, b"C" * 4]
    total = c.put_streamed("big.bin", iter(chunks), content_type="application/octet-stream")
    assert total == 20
    assert fake.objects["big.bin"] == b"A" * 8 + b"B" * 8 + b"C" * 4


def test_region_redirect_auto_discovers(client):
    """A wrong-region bucket answers 301 + x-amz-bucket-region; the client
    re-points to the real region and retries once (boto3 parity)."""
    c, _ = client
    calls = {"n": 0}

    def redirecting(*, method, url, headers, body):
        calls["n"] += 1
        if calls["n"] == 1:
            return S3Response(301, {"x-amz-bucket-region": "eu-west-1"}, b"<Error><Code>PermanentRedirect</Code><Message>wrong region</Message></Error>")
        # After re-region the host + signer should have moved to eu-west-1.
        assert "eu-west-1" in (url.host or "")
        return S3Response(200, {"Content-Length": "2"}, b"ok")

    c._transport = redirecting
    resp = c.get("obj")
    assert bytes(resp.content) == b"ok"
    assert calls["n"] == 2
    assert c.region == "eu-west-1" and c.signer.region == "eu-west-1"


def test_path_style_url_shape():
    from yggdrasil.aws.fs.path import _can_virtual_host

    signer = SigV4Signer(region="us-east-1", credentials_provider=lambda: ("AK", "SK", None))
    c = S3HttpClient(
        bucket="my.dotted.bucket",
        endpoint=URL.from_("https://s3.us-east-1.amazonaws.com"),
        signer=signer, path_style=True, transport=lambda **k: S3Response(200, {}, b""),
    )
    assert c._url("a/b.txt").path == "/my.dotted.bucket/a/b.txt"
    # Dotted buckets can't virtual-host (TLS cert), DNS-safe ones can.
    assert _can_virtual_host("my.dotted.bucket") is False
    assert _can_virtual_host("normal-bucket") is True


def test_multipart_aborts_on_failure(client):
    c, fake = client

    def boom():
        yield b"A" * 8
        raise RuntimeError("mid-stream failure")

    with pytest.raises(RuntimeError):
        c.put_streamed("x", boom())
    assert fake.uploads == {}  # upload aborted, no orphan parts
    assert "x" not in fake.objects
