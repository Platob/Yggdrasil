"""S3HttpClient ops against an in-memory S3 transport.

Signing is proven against botocore in ``test_sigv4``; here we drive the REST
verbs (head / ranged get / put / delete / list / multi-delete / multipart)
through a dict-backed fake transport and assert the request shaping + XML
round-trips. The fake doubles as a reusable in-memory S3 for path-layer tests.
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from yggdrasil.aws.fs.s3_http import S3HttpClient, S3Response, S3NotFound
from yggdrasil.aws.fs.sigv4 import SigV4Signer
from yggdrasil.url import URL


class FakeS3:
    """Routes signed requests against an in-memory ``{key: bytes}`` store."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.uploads: dict[str, dict[int, bytes]] = {}
        self._next_upload = 0

    def __call__(self, *, method, url: URL, headers, body):
        key = (url.path or "/").lstrip("/")
        q = dict(url.query_items())
        # --- multipart ---
        if "uploads" in q:
            self._next_upload += 1
            uid = f"u{self._next_upload}"
            self.uploads[uid] = {}
            return self._xml(200, f"<InitiateMultipartUploadResult><UploadId>{uid}</UploadId></InitiateMultipartUploadResult>")
        if "partNumber" in q and "uploadId" in q:
            self.uploads[q["uploadId"]][int(q["partNumber"])] = body
            return S3Response(200, {"ETag": f'"etag-{q["partNumber"]}"'}, b"")
        if "uploadId" in q and method == "POST":
            parts = self.uploads.pop(q["uploadId"])
            self.objects[key] = b"".join(parts[n] for n in sorted(parts))
            return self._xml(200, "<CompleteMultipartUploadResult/>")
        if "uploadId" in q and method == "DELETE":
            self.uploads.pop(q["uploadId"], None)
            return S3Response(204, {}, b"")
        # --- multi-delete ---
        if "delete" in q and method == "POST":
            root = ET.fromstring(body)
            for obj in root.findall("Object"):
                self.objects.pop((obj.findtext("Key") or "").strip(), None)
            return self._xml(200, "<DeleteResult/>")
        # --- list ---
        if q.get("list-type") == "2":
            return self._list(q)
        # --- single object ---
        if method == "HEAD":
            data = self.objects.get(key)
            if data is None:
                return S3Response(404, {}, b"")
            return S3Response(200, {"Content-Length": str(len(data))}, b"")
        if method == "GET":
            data = self.objects.get(key)
            if data is None:
                return self._xml(404, "<Error><Code>NoSuchKey</Code><Message>missing</Message></Error>")
            rng = headers.get("Range") or headers.get("range")
            if rng:
                spec = rng.split("=", 1)[1]
                lo, _, hi = spec.partition("-")
                lo = int(lo)
                data = data[lo : int(hi) + 1] if hi else data[lo:]
            return S3Response(206 if rng else 200, {"Content-Length": str(len(data))}, data)
        if method == "PUT":
            self.objects[key] = body or b""
            return S3Response(200, {"ETag": '"put"'}, b"")
        if method == "DELETE":
            self.objects.pop(key, None)
            return S3Response(204, {}, b"")
        return S3Response(400, {}, b"<Error><Code>Bad</Code><Message>?</Message></Error>")

    def _list(self, q):
        prefix = q.get("prefix", "")
        delim = q.get("delimiter")
        contents, prefixes = [], set()
        for k in sorted(self.objects):
            if not k.startswith(prefix):
                continue
            if delim:
                rest = k[len(prefix):]
                if delim in rest:
                    prefixes.add(prefix + rest.split(delim, 1)[0] + delim)
                    continue
            contents.append(k)
        body = "".join(
            f"<Contents><Key>{k}</Key><Size>{len(self.objects[k])}</Size></Contents>" for k in contents
        ) + "".join(f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in sorted(prefixes))
        return self._xml(200, f"<ListBucketResult>{body}<IsTruncated>false</IsTruncated></ListBucketResult>")

    @staticmethod
    def _xml(status, inner):
        return S3Response(status, {"Content-Type": "application/xml"}, inner.encode("utf-8"))


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


def test_multipart_aborts_on_failure(client):
    c, fake = client

    def boom():
        yield b"A" * 8
        raise RuntimeError("mid-stream failure")

    with pytest.raises(RuntimeError):
        c.put_streamed("x", boom())
    assert fake.uploads == {}  # upload aborted, no orphan parts
    assert "x" not in fake.objects
