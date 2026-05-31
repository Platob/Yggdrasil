"""Pure-HTTP S3 REST client — the boto3-free S3 data plane.

Speaks the S3 REST API directly over :class:`yggdrasil.http_.session.HTTPSession`
(connection pooling, retries, streaming) with requests signed by
:class:`yggdrasil.aws.fs.sigv4.SigV4Signer`. Covers exactly what
:class:`yggdrasil.aws.fs.path.S3Path` needs:

* ``HEAD`` / ranged ``GET`` / ``PUT`` / ``DELETE`` single objects,
* ``ListObjectsV2`` (paginated, ``CommonPrefixes`` + ``Contents``),
* multi-object ``DeleteObjects``,
* multipart upload (create / upload-part / complete / abort) so a multi-GB
  write streams to S3 in bounded-memory parts.

The wire is reached through an injectable ``transport`` — ``(method, url,
headers, body) -> S3Response`` — defaulting to an :class:`HTTPSession`. Tests
swap in an in-memory transport; production rides the pooled session. XML is
parsed with the stdlib (S3 list/multipart payloads are small + flat).
"""
from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Optional
from xml.etree import ElementTree as ET

from yggdrasil.aws.fs.sigv4 import (
    EMPTY_PAYLOAD_SHA256,
    UNSIGNED_PAYLOAD,
    SigV4Signer,
    sha256_hex,
)
from yggdrasil.url import URL

__all__ = ["S3HttpClient", "S3Response", "S3Error", "S3NotFound", "ListPage", "S3Object"]

# S3 ListObjectsV2 / multipart responses are in this namespace.
_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


class S3Error(OSError):
    """Non-success S3 response (4xx/5xx that isn't a benign 404/416)."""

    def __init__(self, status: int, code: str, message: str, key: str = "") -> None:
        self.status = status
        self.code = code
        super().__init__(f"S3 {status} {code}: {message} ({key})" if key else f"S3 {status} {code}: {message}")


class S3NotFound(S3Error):
    """404 / NoSuchKey."""


@dataclass(slots=True)
class S3Response:
    """Minimal wire response the transport hands back."""

    status: int
    headers: Mapping[str, str]
    content: bytes


@dataclass(slots=True)
class S3Object:
    key: str
    size: int
    last_modified: float = 0.0


@dataclass(slots=True)
class ListPage:
    objects: "list[S3Object]" = field(default_factory=list)
    prefixes: "list[str]" = field(default_factory=list)
    next_token: Optional[str] = None


def _strip(text: Optional[str]) -> str:
    return (text or "").strip()


class S3HttpClient:
    """Signed S3 REST client for one bucket.

    ``endpoint`` is the scheme+host the bucket lives behind (virtual-hosted
    ``https://<bucket>.s3.<region>.amazonaws.com`` by default; a path-style
    override host is honored for S3-compatible stores). One instance per
    :class:`yggdrasil.aws.fs.path.S3Bucket`, so the pooled session and signer
    are shared by every key under it.
    """

    #: Single-PUT cap; at/above this an upload runs as multipart.
    MULTIPART_THRESHOLD: int = 100 * 1024 * 1024
    MULTIPART_CHUNKSIZE: int = 16 * 1024 * 1024

    def __init__(
        self,
        *,
        bucket: str,
        endpoint: URL,
        signer: SigV4Signer,
        transport: Optional[Callable[..., S3Response]] = None,
        path_style: bool = False,
        region: Optional[str] = None,
        managed: bool = True,
    ) -> None:
        self.bucket = bucket
        self.endpoint = endpoint
        self.signer = signer
        self.path_style = path_style
        # ``managed`` AWS endpoints can self-correct their region on a redirect
        # (boto3 does the same); a custom ``endpoint_url`` (MinIO/localstack) is
        # left alone.
        self.region = region or signer.region
        self._managed = managed
        self._transport = transport or self._default_transport
        self._session = None

    # ------------------------------------------------------------------
    # URL + transport
    # ------------------------------------------------------------------
    def _url(self, key: str = "", *, query: "Mapping[str, str] | None" = None) -> URL:
        # Virtual-hosted: bucket is already in ``endpoint`` host. Path-style:
        # ``/<bucket>/<key>``. Either way ``URL`` percent-encodes the key once
        # and the signer signs that exact path.
        path = f"/{self.bucket}/{key}" if self.path_style else f"/{key}"
        url = self.endpoint._replace_path(path)
        if query:
            url = url.with_query_items(query)
        return url

    def _send(
        self,
        method: str,
        key: str = "",
        *,
        query: "Mapping[str, str] | None" = None,
        headers: "Mapping[str, str] | None" = None,
        body: Optional[bytes] = None,
        payload_sha256: Optional[str] = None,
        stream_body: Any = None,
    ) -> S3Response:
        if payload_sha256 is None:
            payload_sha256 = sha256_hex(body) if body is not None else EMPTY_PAYLOAD_SHA256
        wire = stream_body if stream_body is not None else body
        for attempt in range(2):
            url = self._url(key, query=query)
            sign_headers = self.signer.sign(method, url, headers=headers, payload_sha256=payload_sha256)
            merged = dict(headers or {})
            merged.update(sign_headers)
            resp = self._transport(method=method, url=url, headers=merged, body=wire)
            # Wrong-region buckets answer 301/400 with the real region in a
            # header; re-point once and retry (matches boto3's auto-discovery).
            if attempt == 0 and resp.status in (301, 400) and self._managed:
                region = resp.headers.get("x-amz-bucket-region") or resp.headers.get("X-Amz-Bucket-Region")
                if region and region != self.region:
                    self._reregion(region)
                    continue
            return resp
        return resp

    def _reregion(self, region: str) -> None:
        self.region = region
        self.signer.region = region
        if not self.path_style:
            self.endpoint = URL.from_(f"https://{self.bucket}.s3.{region}.amazonaws.com")
        else:
            self.endpoint = URL.from_(f"https://s3.{region}.amazonaws.com")

    def _default_transport(self, *, method: str, url: URL, headers: Mapping[str, str], body: Any) -> S3Response:
        if self._session is None:
            from yggdrasil.http_.session import HTTPSession

            self._session = HTTPSession()
        resp = self._session.request(
            method,
            str(url),
            headers=dict(headers),
            body=body,
            raise_error=False,
            remote_cache=None,
            local_cache=None,
        )
        return S3Response(status=resp.status_code, headers=dict(resp.headers or {}), content=resp.content or b"")

    @staticmethod
    def _check(resp: S3Response, key: str = "", *, ok=(200, 204, 206)) -> S3Response:
        if resp.status in ok:
            return resp
        code, message = _parse_error(resp.content)
        if resp.status in (404,) or code in ("NoSuchKey", "NoSuchBucket", "NotFound"):
            raise S3NotFound(resp.status, code or "NotFound", message, key)
        raise S3Error(resp.status, code or str(resp.status), message, key)

    # ------------------------------------------------------------------
    # Single-object ops
    # ------------------------------------------------------------------
    def head(self, key: str) -> "S3Response | None":
        """``HEAD`` — returns the response (Content-Length / Last-Modified /
        Content-Type in headers) or ``None`` when the object is absent."""
        resp = self._send("HEAD", key)
        if resp.status == 404:
            return None
        return self._check(resp, key, ok=(200,))

    def get(self, key: str, *, start: int = 0, length: int = -1) -> S3Response:
        """Ranged ``GET``. ``length < 0`` from ``start`` reads to EOF; the
        whole-object fast path omits the Range header entirely."""
        headers: dict[str, str] = {}
        if length > 0:
            headers["Range"] = f"bytes={start}-{start + length - 1}"
        elif start > 0:
            headers["Range"] = f"bytes={start}-"
        return self._check(self._send("GET", key, headers=headers), key)

    def put(self, key: str, body: bytes, *, content_type: "str | None" = None) -> S3Response:
        headers = {"Content-Type": content_type} if content_type else {}
        return self._check(self._send("PUT", key, headers=headers, body=body), key, ok=(200, 201))

    def delete(self, key: str) -> None:
        self._check(self._send("DELETE", key), key, ok=(200, 204))

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------
    def list_page(
        self,
        prefix: str,
        *,
        delimiter: "str | None" = None,
        token: "str | None" = None,
        max_keys: int = 1000,
    ) -> ListPage:
        query = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
        if delimiter:
            query["delimiter"] = delimiter
        if token:
            query["continuation-token"] = token
        resp = self._check(self._send("GET", "", query=query))
        return _parse_list(resp.content)

    def list(
        self,
        prefix: str,
        *,
        delimiter: "str | None" = None,
        max_keys: int = 1000,
    ) -> Iterator[ListPage]:
        token: "str | None" = None
        while True:
            page = self.list_page(prefix, delimiter=delimiter, token=token, max_keys=max_keys)
            yield page
            if not page.next_token:
                return
            token = page.next_token

    def delete_batch(self, keys: "list[str]") -> None:
        """Multi-object delete (``POST /?delete``). Content-MD5 is required by
        S3 for this op; ``DeleteObjects`` errors are surfaced as one OSError."""
        if not keys:
            return
        body = _delete_xml(keys).encode("utf-8")
        headers = {
            "Content-MD5": base64.b64encode(hashlib.md5(body).digest()).decode("ascii"),
            "Content-Type": "application/xml",
        }
        resp = self._check(self._send("POST", "", query={"delete": ""}, headers=headers, body=body))
        errors = _parse_delete_errors(resp.content)
        if errors:
            sample = ", ".join(f"{k!r}={c}" for k, c in errors[:3])
            more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
            raise OSError(f"S3 DeleteObjects reported {len(errors)} error(s): {sample}{more}")

    # ------------------------------------------------------------------
    # Multipart upload
    # ------------------------------------------------------------------
    def create_multipart(self, key: str, *, content_type: "str | None" = None) -> str:
        headers = {"Content-Type": content_type} if content_type else {}
        resp = self._check(self._send("POST", key, query={"uploads": ""}, headers=headers))
        upload_id = _find_text(resp.content, "UploadId")
        if not upload_id:
            raise S3Error(resp.status, "MalformedResponse", "CreateMultipartUpload had no UploadId", key)
        return upload_id

    def upload_part(self, key: str, upload_id: str, part_number: int, body: bytes) -> str:
        resp = self._check(
            self._send(
                "PUT", key,
                query={"partNumber": str(part_number), "uploadId": upload_id},
                body=body,
            ),
            key, ok=(200,),
        )
        etag = resp.headers.get("ETag") or resp.headers.get("etag")
        if not etag:
            raise S3Error(resp.status, "MalformedResponse", "UploadPart returned no ETag", key)
        return etag

    def complete_multipart(self, key: str, upload_id: str, parts: "list[tuple[int, str]]") -> None:
        body = _complete_xml(parts).encode("utf-8")
        self._check(
            self._send(
                "POST", key,
                query={"uploadId": upload_id},
                headers={"Content-Type": "application/xml"},
                body=body,
            ),
            key, ok=(200,),
        )

    def abort_multipart(self, key: str, upload_id: str) -> None:
        try:
            self._check(self._send("DELETE", key, query={"uploadId": upload_id}), key, ok=(204, 200))
        except S3Error:
            pass  # best-effort cleanup

    def put_streamed(self, key: str, parts: "Iterator[bytes]", *, content_type: "str | None" = None) -> int:
        """Multipart-upload an iterator of byte chunks; returns total bytes.

        Each chunk should be ``>= 5 MiB`` (the S3 part minimum) except the
        last. Aborts the upload on any failure so no orphaned parts linger.
        """
        upload_id = self.create_multipart(key, content_type=content_type)
        etags: "list[tuple[int, str]]" = []
        total = 0
        try:
            for i, chunk in enumerate(parts, start=1):
                if not chunk:
                    continue
                etags.append((i, self.upload_part(key, upload_id, i, chunk)))
                total += len(chunk)
            self.complete_multipart(key, upload_id, etags)
        except BaseException:
            self.abort_multipart(key, upload_id)
            raise
        return total


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------
def _tag(elem: ET.Element) -> str:
    return elem.tag.split("}", 1)[-1]


def _find_text(content: bytes, local_name: str) -> "str | None":
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return None
    for elem in root.iter():
        if _tag(elem) == local_name:
            return _strip(elem.text)
    return None


def _parse_error(content: bytes) -> "tuple[str, str]":
    code = _find_text(content, "Code") or ""
    message = _find_text(content, "Message") or ""
    return code, message


def _parse_list(content: bytes) -> ListPage:
    page = ListPage()
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return page
    for elem in root:
        name = _tag(elem)
        if name == "Contents":
            key = _strip(_child_text(elem, "Key"))
            if not key:
                continue
            size = int(_child_text(elem, "Size") or 0)
            page.objects.append(S3Object(key=key, size=size, last_modified=_parse_iso(_child_text(elem, "LastModified"))))
        elif name == "CommonPrefixes":
            p = _strip(_child_text(elem, "Prefix"))
            if p:
                page.prefixes.append(p)
        elif name == "NextContinuationToken":
            page.next_token = _strip(elem.text) or None
        elif name == "IsTruncated" and _strip(elem.text).lower() != "true":
            page.next_token = page.next_token  # keep explicit None unless token seen
    return page


def _child_text(elem: ET.Element, local_name: str) -> "str | None":
    for child in elem:
        if _tag(child) == local_name:
            return child.text
    return None


def _parse_delete_errors(content: bytes) -> "list[tuple[str, str]]":
    out: "list[tuple[str, str]]" = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return out
    for elem in root:
        if _tag(elem) == "Error":
            out.append((_strip(_child_text(elem, "Key")), _strip(_child_text(elem, "Code"))))
    return out


def _delete_xml(keys: "list[str]") -> str:
    objs = "".join(f"<Object><Key>{_xml_escape(k)}</Key></Object>" for k in keys)
    return f'<?xml version="1.0" encoding="UTF-8"?><Delete>{objs}<Quiet>true</Quiet></Delete>'


def _complete_xml(parts: "list[tuple[int, str]]") -> str:
    body = "".join(
        f"<Part><PartNumber>{n}</PartNumber><ETag>{_xml_escape(etag)}</ETag></Part>"
        for n, etag in parts
    )
    return f'<?xml version="1.0" encoding="UTF-8"?><CompleteMultipartUpload>{body}</CompleteMultipartUpload>'


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _parse_iso(value: "str | None") -> float:
    if not value:
        return 0.0
    import datetime as _dt

    try:
        return _dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
