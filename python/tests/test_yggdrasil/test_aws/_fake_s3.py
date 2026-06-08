"""Shared in-memory S3 for AWS tests.

:class:`FakeS3` is a dict-backed ``S3HttpClient`` transport — it answers the
signed REST requests :class:`yggdrasil.aws.fs.s3_http.S3HttpClient` issues
without a network. :func:`wire_s3_path` builds a real
:class:`yggdrasil.aws.fs.path.S3Path` whose :class:`S3Bucket` is primed with an
``S3HttpClient`` riding the fake, so path-layer tests exercise the genuine
pure-HTTP code path (stat / range-read / put / multipart / list / delete)
end-to-end against the fake store.
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

from yggdrasil.aws.fs.s3_http import S3HttpClient, S3Response
from yggdrasil.aws.fs.sigv4 import SigV4Signer
from yggdrasil.url import URL

__all__ = ["FakeS3", "wire_s3_path", "reset_s3_singletons"]


class FakeS3:
    """Routes signed S3 requests against an in-memory ``{key: bytes}`` store."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.uploads: dict[str, dict[int, bytes]] = {}
        self.calls: dict[str, int] = {}
        self._next_upload = 0

    def __call__(self, *, method, url: URL, headers, body):
        key = (url.path or "/").lstrip("/")
        q = dict(url.query_items())
        if "uploads" in q:
            self._bump("create_multipart")
            self._next_upload += 1
            uid = f"u{self._next_upload}"
            self.uploads[uid] = {}
            return self._xml(200, f"<InitiateMultipartUploadResult><UploadId>{uid}</UploadId></InitiateMultipartUploadResult>")
        if "partNumber" in q and "uploadId" in q:
            self._bump("upload_part")
            self.uploads[q["uploadId"]][int(q["partNumber"])] = body
            return S3Response(200, {"ETag": f'"etag-{q["partNumber"]}"'}, b"")
        if "uploadId" in q and method == "POST":
            self._bump("complete_multipart")
            parts = self.uploads.pop(q["uploadId"])
            self.objects[key] = b"".join(parts[n] for n in sorted(parts))
            return self._xml(200, "<CompleteMultipartUploadResult/>")
        if "uploadId" in q and method == "DELETE":
            self.uploads.pop(q["uploadId"], None)
            return S3Response(204, {}, b"")
        if "delete" in q and method == "POST":
            self._bump("delete_batch")
            for obj in ET.fromstring(body).findall("Object"):
                self.objects.pop((obj.findtext("Key") or "").strip(), None)
            return self._xml(200, "<DeleteResult/>")
        if q.get("list-type") == "2":
            self._bump("list")
            return self._list(q)
        if method == "HEAD":
            self._bump("head")
            data = self.objects.get(key)
            return S3Response(404, {}, b"") if data is None else S3Response(200, {"Content-Length": str(len(data))}, b"")
        if method == "GET":
            self._bump("get")
            data = self.objects.get(key)
            if data is None:
                return self._xml(404, "<Error><Code>NoSuchKey</Code><Message>missing</Message></Error>")
            rng = headers.get("Range") or headers.get("range")
            if rng:
                lo, _, hi = rng.split("=", 1)[1].partition("-")
                lo = int(lo)
                data = data[lo : int(hi) + 1] if hi else data[lo:]
            return S3Response(206 if rng else 200, {"Content-Length": str(len(data))}, data)
        if method == "PUT":
            self._bump("put")
            # ``If-None-Match: *`` is S3's atomic create-if-absent — a PUT that
            # only succeeds when the key is absent. A losing writer gets 412
            # PreconditionFailed. This is what makes a Delta commit JSON race
            # genuinely atomic on object storage, so model it: without it the
            # conditional-commit path can't be tested.
            if (headers.get("If-None-Match") or headers.get("if-none-match")) == "*" \
                    and key in self.objects:
                self._bump("put_precondition_failed")
                return self._xml(412, "<Error><Code>PreconditionFailed</Code>"
                                      "<Message>At least one of the pre-conditions "
                                      "you specified did not hold</Message></Error>")
            self.objects[key] = body or b""
            return S3Response(200, {"ETag": '"put"'}, b"")
        if method == "DELETE":
            self._bump("delete")
            self.objects.pop(key, None)
            return S3Response(204, {}, b"")
        return S3Response(400, {}, b"<Error><Code>Bad</Code><Message>?</Message></Error>")

    def _bump(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def _list(self, q):
        prefix, delim = q.get("prefix", ""), q.get("delimiter")
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
        body = "".join(f"<Contents><Key>{k}</Key><Size>{len(self.objects[k])}</Size></Contents>" for k in contents)
        body += "".join(f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in sorted(prefixes))
        return self._xml(200, f"<ListBucketResult>{body}<IsTruncated>false</IsTruncated></ListBucketResult>")

    @staticmethod
    def _xml(status, inner):
        return S3Response(status, {"Content-Type": "application/xml"}, inner.encode("utf-8"))


def reset_s3_singletons() -> None:
    from yggdrasil.aws.fs.path import S3Bucket, S3Path

    S3Bucket._INSTANCES.clear()
    S3Path._INSTANCES.clear()


def wire_s3_path(fake: FakeS3, url: str, *, bucket: str = "bkt"):
    """Build a real S3Path whose S3Bucket is primed with a fake-backed client."""
    from yggdrasil.aws.fs.path import S3Bucket, S3Path

    signer = SigV4Signer(region="us-east-1", credentials_provider=lambda: ("AK", "SK", None))
    client = S3HttpClient(
        bucket=bucket,
        endpoint=URL.from_(f"https://{bucket}.s3.us-east-1.amazonaws.com"),
        signer=signer,
        transport=fake,
    )
    S3Bucket(bucket=bucket, service=None, http=client)  # prime the singleton
    return S3Path(url, singleton_ttl=False)
