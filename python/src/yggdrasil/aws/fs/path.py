"""S3-backed :class:`Path` implementation — pure-HTTP data plane.

Two cooperating :class:`RemotePath` types:

* :class:`S3Bucket` — the ``s3://<bucket>/`` root. A **long-lived singleton**
  (one per ``(bucket, service)``) that owns the boto3-free S3 data plane: a
  :class:`~yggdrasil.aws.fs.s3_http.S3HttpClient` (signed with
  :class:`~yggdrasil.aws.fs.sigv4.SigV4Signer`, riding a pooled
  :class:`~yggdrasil.http_.session.HTTPSession`) plus the bucket-wide listing
  cache. Credentials still come from the AWS *client* (env / profile / SSO /
  STS resolution lives in :class:`~yggdrasil.aws.client.AWSClient`); only the
  S3 wire moved off boto3.

* :class:`S3Path` — a key under a bucket. Lightweight: it resolves its
  :class:`S3Bucket` (``self.s3_bucket``) and **redirects** every backend call
  there — ``head`` / ranged ``get`` / ``put`` / multipart ``put`` / ``delete``
  / ``list`` / batch-delete. Each path only carries its own per-key concerns
  (range math, stat-cache seeding); bucket-scoped state lives once on the
  bucket. Reads range-GET (Parquet footer probes cost one request); writes
  single-PUT under the threshold and stream as multipart above it.

``arrow_filesystem`` / ``arrow_uri`` still hand out a ``pyarrow.fs.S3FileSystem``
(credential snapshot) for the Arrow/Parquet fast path — orthogonal to the REST
data plane here.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterator, Optional

from yggdrasil.aws.fs.s3_http import S3HttpClient, S3NotFound
from yggdrasil.aws.fs.sigv4 import SigV4Signer
from yggdrasil.dataclasses import ExpiringDict, WaitingConfig
from yggdrasil.enums import Scheme
from yggdrasil.enums.media_type import MediaType
from yggdrasil.path import RemotePath
from yggdrasil.path.remote_path import _STAT_CACHE_TTL
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.url import URL

if TYPE_CHECKING:
    from yggdrasil.aws.fs.service import S3Service


__all__ = ["S3Path", "S3Bucket"]

LOGGER = logging.getLogger(__name__)

#: Buckets are stable identities (region/creds rotate underneath, the bucket
#: doesn't), and each one anchors a pooled HTTPSession — so the bucket singleton
#: lives much longer than a per-key path's stat-cache horizon.
_BUCKET_SINGLETON_TTL: float = 3600.0

_ACCEPTED_SCHEMES: frozenset[str] = frozenset({"s3", "s3a", "s3n"})

#: Part size for multipart streaming uploads (>= the 5 MiB S3 minimum).
_MULTIPART_THRESHOLD: int = 100 * 1024 * 1024
_MULTIPART_CHUNKSIZE: int = 16 * 1024 * 1024


def _bucket_of(data: Any, url: "URL | None") -> "str | None":
    if url is not None:
        return url.host
    if isinstance(data, URL):
        return data.host
    if isinstance(data, str):
        return URL.from_(data).host
    return None


def _normalize_scheme(url: URL) -> URL:
    """Coerce ``s3a://`` / ``s3n://`` (common in Spark) to canonical ``s3://``."""
    if url.scheme in _ACCEPTED_SCHEMES and url.scheme != Scheme.S3:
        return url.with_scheme(Scheme.S3)
    return url


def _can_virtual_host(bucket: str) -> bool:
    """True when *bucket* is a DNS-safe label that can ride a virtual-host TLS
    cert (``<bucket>.s3...``). Dotted names break the wildcard cert, so boto3 —
    and we — drop those to path-style."""
    return "." not in bucket and 3 <= len(bucket) <= 63


# ---------------------------------------------------------------------------
# S3Bucket — long-lived bucket singleton owning the HTTP data plane
# ---------------------------------------------------------------------------
class S3Bucket(RemotePath):
    """``s3://<bucket>/`` root — owns the signed pure-HTTP S3 client.

    One instance per ``(bucket, service)``, cached for an hour. Build it
    implicitly via any :class:`S3Path` (``path.s3_bucket``) or directly with a
    test transport: ``S3Bucket("b", http=fake_client)``.
    """

    # NB: no ``scheme`` in the class body — S3Path owns the ``s3`` registry
    # slot (for ``Path.from_("s3://…")`` dispatch). ``S3Bucket.scheme`` is set
    # post-class so the bucket still carries the typed scheme without a
    # duplicate registration.
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=_BUCKET_SINGLETON_TTL, max_size=4096
    )
    _SINGLETON_TTL: ClassVar[Any] = _BUCKET_SINGLETON_TTL
    _ACCEPTED_SCHEMES: ClassVar[frozenset[str]] = _ACCEPTED_SCHEMES

    @classmethod
    def _singleton_key(
        cls, data: Any = None, *, bucket: "str | None" = None,
        service: Any = None, url: "URL | None" = None, **kwargs: Any,
    ) -> Any:
        return (cls, bucket or _bucket_of(data, url), service)

    def __init__(
        self,
        data: Any = None,
        *,
        bucket: "str | None" = None,
        service: Optional["S3Service"] = None,
        http: Optional[S3HttpClient] = None,
        singleton_ttl: Any = ...,
        **kwargs: Any,
    ) -> None:
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return
        bucket = bucket or _bucket_of(data, None)
        if not bucket:
            raise ValueError(f"S3Bucket needs a bucket name (got {data!r})")
        super().__init__(url=URL.from_(f"s3://{bucket}/"), **kwargs)
        self._bucket = bucket
        self._service = service
        self._http = http
        self._ls_cache: Optional[ExpiringDict] = None
        self._initialized = True

    # -- service / client plumbing ------------------------------------
    @property
    def service(self) -> "S3Service":
        if self._service is None:
            from yggdrasil.aws.fs.service import S3Service

            self._service = S3Service.current()
        return self._service

    @property
    def http(self) -> S3HttpClient:
        """The signed pure-HTTP S3 client, built once from the service creds."""
        if self._http is None:
            client = self.service.client
            region = getattr(client, "region", None) or "us-east-1"
            endpoint_url = getattr(client, "endpoint_url", None)
            managed = not endpoint_url
            if endpoint_url:  # S3-compatible store (MinIO / localstack) → path-style
                endpoint, path_style = URL.from_(endpoint_url), True
            elif _can_virtual_host(self._bucket):
                endpoint, path_style = URL.from_(f"https://{self._bucket}.s3.{region}.amazonaws.com"), False
            else:
                # Dotted / non-DNS-safe bucket names can't ride a virtual-host
                # TLS cert — fall back to path-style, exactly as boto3 does.
                endpoint, path_style = URL.from_(f"https://s3.{region}.amazonaws.com"), True
            signer = SigV4Signer(region=region, credentials_provider=self._credentials)
            self._http = S3HttpClient(
                bucket=self._bucket, endpoint=endpoint, signer=signer,
                path_style=path_style, region=region, managed=managed,
            )
        return self._http

    def _credentials(self) -> "tuple[str, str, Optional[str]]":
        creds = self.service.client.session.get_credentials()
        if creds is None:
            raise PermissionError(
                f"No AWS credentials resolved for {self!r}; pure-HTTP S3 requires "
                "signed requests (configure env / profile / role / SSO)."
            )
        frozen = creds.get_frozen_credentials()
        return frozen.access_key, frozen.secret_key, frozen.token

    # -- bucket-level ops (S3Path redirects here) ---------------------
    @property
    def name(self) -> str:
        return self._bucket

    def iter_keys(self, prefix: str, *, delimiter: "str | None" = None) -> Iterator[str]:
        """Yield child keys / common-prefixes under *prefix*, one page at a time."""
        for page in self.http.list(prefix, delimiter=delimiter):
            yield from page.prefixes
            for obj in page.objects:
                # Skip zero-byte ``foo/`` placeholders on a shallow walk — they
                # resurface as CommonPrefixes.
                if delimiter and obj.key.endswith("/") and obj.size == 0:
                    continue
                yield obj.key

    def delete_prefix(self, prefix: str) -> int:
        """Bulk-delete every object under *prefix*; returns the count."""
        batch: "list[str]" = []
        deleted = 0
        for page in self.http.list(prefix):
            for obj in page.objects:
                batch.append(obj.key)
                if len(batch) >= 1000:
                    self.http.delete_batch(batch)
                    deleted += len(batch)
                    batch = []
        if batch:
            self.http.delete_batch(batch)
            deleted += len(batch)
        return deleted

    # -- listing cache (shared by every key under the bucket) ---------
    @property
    def ls_cache(self) -> ExpiringDict:
        if self._ls_cache is None:
            self._ls_cache = ExpiringDict(default_ttl=_STAT_CACHE_TTL, max_size=10_000)
        return self._ls_cache

    def invalidate_ls_cache(self, prefix: "str | None" = None) -> None:
        if self._ls_cache is None:
            return
        if prefix is None:
            self._ls_cache.clear()
            return
        for k in [k for k in self._ls_cache if str(k).startswith(prefix)]:
            self._ls_cache.pop(k, None)

    # -- RemotePath surface for the bucket root -----------------------
    def _stat_uncached(self) -> IOStats:
        return IOStats(size=0, mtime=0.0, kind=IOKind.DIRECTORY, mode=0)

    def _ls(self, recursive: bool = False, *, singleton_ttl: Any = False) -> Iterator["S3Path"]:
        for key in self.iter_keys("", delimiter=None if recursive else "/"):
            yield self._child(key, singleton_ttl=singleton_ttl)

    def _child(self, key: str, *, singleton_ttl: Any = False) -> "S3Path":
        url = self.url._replace_path("/" + key.lstrip("/"))
        return S3Path(url=url, service=self._service, s3_bucket=self, singleton_ttl=singleton_ttl)

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        del parents, exist_ok

    def full_path(self) -> str:
        return f"s3://{self._bucket}/"

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        del missing_ok, wait
        raise IsADirectoryError(self.full_path())

    def _remove_dir(self, recursive: bool, missing_ok: bool, wait: WaitingConfig) -> None:
        del missing_ok, wait
        if recursive:
            self.delete_prefix("")
        self.invalidate_ls_cache()

    def _from_url(self, url: URL) -> "RemotePath":
        return S3Path(url=url, service=self._service, s3_bucket=self)

    def arrow_filesystem(self, *, region: Optional[str] = None, **overrides: Any) -> Any:
        return self.service.arrow_filesystem(region=region, **overrides)

    @property
    def explore_url(self) -> URL:
        """AWS Console deep-link to this bucket — clickable from code."""
        from yggdrasil.aws.console import s3_bucket_url

        return s3_bucket_url(self._bucket, getattr(self.service.client, "region", None))

    def _repr_html_(self) -> str:
        return f'<a href="{self.explore_url}" target="_blank">S3Bucket: {self._bucket}</a>'

    def __repr__(self) -> str:
        return f"S3Bucket({self._bucket!r})"


# ---------------------------------------------------------------------------
# S3Path — a key under a bucket; redirects backend ops to its S3Bucket
# ---------------------------------------------------------------------------
class S3Path(RemotePath):
    """:class:`Path` over an S3 object key. Backend ops redirect to
    :attr:`s3_bucket` (the long-lived :class:`S3Bucket` singleton)."""

    scheme: ClassVar[Scheme] = Scheme.S3
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=_STAT_CACHE_TTL, max_size=10_000
    )
    _ACCEPTED_SCHEMES: ClassVar[frozenset[str]] = _ACCEPTED_SCHEMES

    # Range-GET random access (Parquet footer/column projection) + multipart
    # streaming uploads (multi-GB writes never materialise whole in memory).
    SUPPORTS_RANGED_RANDOM_ACCESS: ClassVar[bool] = True
    SUPPORTS_STREAMING_UPLOAD: ClassVar[bool] = True

    MULTIPART_THRESHOLD: ClassVar[int] = _MULTIPART_THRESHOLD
    MULTIPART_CHUNKSIZE: ClassVar[int] = _MULTIPART_CHUNKSIZE

    @classmethod
    def _singleton_key(
        cls, data: Any = None, *, url: URL | None = None,
        service: Any = None, **kwargs: Any,
    ) -> Any:
        if url is None:
            if isinstance(data, URL):
                url = data
            elif isinstance(data, str):
                url = URL.from_(data)
            else:
                return (cls, object())
        return (cls, _normalize_scheme(url), service)

    def __init__(
        self,
        data: Any = None,
        *,
        url: URL | None = None,
        service: Optional["S3Service"] = None,
        s3_bucket: Optional[S3Bucket] = None,
        temporary: bool = False,
        singleton_ttl: Any = ...,
        **kwargs: Any,
    ) -> None:
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return
        if url is None and isinstance(data, str):
            url, data = URL.from_(data), None
        if url is None and isinstance(data, URL):
            url, data = data, None
        if url is not None:
            url = _normalize_scheme(URL.from_(url))
        super().__init__(data=data, url=url, temporary=temporary, **kwargs)
        self._service = service
        self._s3_bucket = s3_bucket
        self._initialized = True

    # ==================================================================
    # Bucket / key + the S3Bucket redirect
    # ==================================================================
    @property
    def bucket(self) -> str:
        host = self.url.host
        if not host:
            raise ValueError(f"S3 path has no bucket: {self.url!r}")
        return host

    @property
    def key(self) -> str:
        return (self.url.path or "").lstrip("/")

    @property
    def s3_bucket(self) -> S3Bucket:
        """The long-lived :class:`S3Bucket` that owns this key's data plane."""
        if self._s3_bucket is None:
            self._s3_bucket = S3Bucket(bucket=self.bucket, service=self._service)
        return self._s3_bucket

    @property
    def service(self) -> "S3Service":
        return self.s3_bucket.service

    def full_path(self) -> str:
        key = self.key
        return f"s3://{self.bucket}/{key}" if key else f"s3://{self.bucket}/"

    @property
    def size(self) -> int:
        return int(self._stat().size)

    def _from_url(self, url: URL) -> "S3Path":
        # Same bucket → reuse the resolved S3Bucket so siblings share the client.
        same_bucket = self._s3_bucket if url.host == self.url.host else None
        return S3Path(url=url, service=self._service, s3_bucket=same_bucket)

    # ==================================================================
    # PyArrow filesystem fast path (orthogonal to the REST plane)
    # ==================================================================
    def arrow_filesystem(self, *, region: Optional[str] = None, **overrides: Any) -> Any:
        return self.service.arrow_filesystem(region=region, **overrides)

    @property
    def arrow_uri(self) -> str:
        return f"{self.bucket}/{self.key}"

    @property
    def explore_url(self) -> URL:
        """AWS Console deep-link to this object/prefix — clickable from code."""
        from yggdrasil.aws.console import s3_object_url

        return s3_object_url(self.bucket, self.key, getattr(self.service.client, "region", None))

    def _repr_html_(self) -> str:
        return f'<a href="{self.explore_url}" target="_blank">S3Path: {self.full_path()}</a>'

    # ==================================================================
    # Stat
    # ==================================================================
    def _stat_uncached(self) -> IOStats:
        if not self.key:
            return IOStats(size=0, mtime=0.0, kind=IOKind.DIRECTORY, mode=0)
        resp = self.s3_bucket.http.head(self.key)
        if resp is not None:
            return IOStats(
                size=int(resp.headers.get("Content-Length", 0) or 0),
                mtime=_mtime_from_headers(resp.headers),
                kind=IOKind.FILE,
                mode=0,
                media_type=_media_type_from_headers(resp.headers),
            )
        # No object at the exact key — is it a prefix?
        prefix = self.key if self.key.endswith("/") else self.key + "/"
        page = self.s3_bucket.http.list_page(prefix, delimiter="/", max_keys=1)
        if page.objects or page.prefixes:
            return IOStats(size=0, mtime=0.0, kind=IOKind.DIRECTORY, mode=0)
        return IOStats(size=0, mtime=0.0, kind=IOKind.MISSING, mode=0)

    # ==================================================================
    # Listing (redirects to the bucket)
    # ==================================================================
    def _ls(self, recursive: bool = False, *, singleton_ttl: Any = False) -> Iterator["S3Path"]:
        prefix = self.key if (not self.key or self.key.endswith("/")) else self.key + "/"
        cache = self.s3_bucket.ls_cache
        cache_key = (prefix, recursive)
        cached = cache.get(cache_key)
        if cached is not None:
            for key in cached:
                yield self._make_child(key, singleton_ttl=singleton_ttl)
            return
        collected: "list[str] | None" = []
        for key in self.s3_bucket.iter_keys(prefix, delimiter=None if recursive else "/"):
            if collected is not None:
                if len(collected) < 10_000:
                    collected.append(key)
                else:
                    collected = None  # too big to cache — keep streaming
            yield self._make_child(key, singleton_ttl=singleton_ttl)
        if collected is not None:
            cache[cache_key] = collected

    def _make_child(self, key: str, *, singleton_ttl: Any = False) -> "S3Path":
        cleaned = key.lstrip("/")
        url = self.url._replace_path("/" + cleaned if cleaned else "/")
        return S3Path(url=url, service=self._service, s3_bucket=self._s3_bucket, singleton_ttl=singleton_ttl)

    def _invalidate_parent_ls(self) -> None:
        if self._s3_bucket is not None:
            self._s3_bucket.invalidate_ls_cache()

    def invalidate_singleton(self, remove_global: bool = True) -> None:
        super().invalidate_singleton(remove_global=remove_global)

    # ==================================================================
    # mkdir / remove
    # ==================================================================
    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        del parents, exist_ok  # S3 prefixes materialise when a child lands.

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        del wait
        try:
            self.s3_bucket.http.delete(self.key)
        except Exception:
            if not missing_ok:
                raise
            return
        self.invalidate_singleton()
        self._invalidate_parent_ls()

    def _remove_dir(self, recursive: bool, missing_ok: bool, wait: WaitingConfig) -> None:
        del wait
        if not recursive:
            placeholder = self.key if self.key.endswith("/") else self.key + "/"
            try:
                self.s3_bucket.http.delete(placeholder)
            except Exception:
                if not missing_ok:
                    raise
            self.invalidate_singleton()
            self._invalidate_parent_ls()
            return
        prefix = self.key if self.key.endswith("/") else self.key + "/"
        try:
            self.s3_bucket.delete_prefix(prefix)
        except Exception:
            if not missing_ok:
                raise
        self.invalidate_singleton()
        self._invalidate_parent_ls()

    # ==================================================================
    # Holder I/O — read / upload
    # ==================================================================
    def read_mv(self, size: int = -1, offset: int = 0, *, cursor: bool = False) -> memoryview:
        # Whole-file fast path skips the size probe: the no-Range GET returns
        # the object + canonical Content-Length, folded into the stat cache.
        if cursor:
            offset = self._pos
        if size < 0 and offset == 0:
            out = self._read_mv(-1, 0)
            if cursor:
                self._pos = len(out)
            return out
        return super().read_mv(size, offset, cursor=cursor)

    def _read_mv(self, n: int, pos: int) -> memoryview:
        if n == 0:
            return memoryview(b"")
        try:
            resp = self.s3_bucket.http.get(self.key, start=pos, length=n)
        except S3NotFound as exc:
            raise FileNotFoundError(self.full_path()) from exc
        data = resp.content or b""
        # Seed the stat cache from a whole-object GET's Content-Length.
        if n < 0 and pos == 0 and self._stat_cached is None:
            self._persist_stat_cache(
                IOStats(size=len(data), kind=IOKind.FILE, mtime=time.time(),
                        media_type=_media_type_from_headers(resp.headers))
            )
        return memoryview(data)

    def _upload(self, content: bytes) -> int:
        size = len(content)
        ct = self.media_type.mime_type.value if self.media_type else None
        if size >= self.MULTIPART_THRESHOLD:
            self.s3_bucket.http.put_streamed(self.key, _chunked(content, self.MULTIPART_CHUNKSIZE), content_type=ct)
        else:
            self.s3_bucket.http.put(self.key, content, content_type=ct)
        self._persist_stat_cache(
            IOStats(size=size, kind=IOKind.FILE, mtime=time.time(), media_type=self.media_type)
        )
        self._cache_after_upload(bytes(content), size)
        self._invalidate_parent_ls()
        return size

    def _upload_stream(self, source: Any) -> int:
        """Stream a local spill file to S3 as multipart parts (bounded memory)."""
        os_path = getattr(source, "os_path", None) if getattr(source, "is_local_path", False) else None
        if os_path is None:
            return self._upload(source.read_bytes())
        ct = self.media_type.mime_type.value if self.media_type else None
        size = int(source.size)

        def _parts() -> Iterator[bytes]:
            with open(os_path, "rb") as fh:
                while True:
                    chunk = fh.read(self.MULTIPART_CHUNKSIZE)
                    if not chunk:
                        return
                    yield chunk

        if size >= self.MULTIPART_THRESHOLD:
            self.s3_bucket.http.put_streamed(self.key, _parts(), content_type=ct)
        else:
            self.s3_bucket.http.put(self.key, source.read_bytes(), content_type=ct)
        self._persist_stat_cache(
            IOStats(size=size, kind=IOKind.FILE, mtime=time.time(), media_type=self.media_type)
        )
        self._note_streamed_upload(size)
        self._invalidate_parent_ls()
        return size

    def reserve(self, n: int) -> None:
        if n < 0:
            raise ValueError(f"reserve size must be >= 0, got {n!r}")

    def _clear(self) -> None:
        self._remove_file(missing_ok=True, wait=WaitingConfig.from_(True))

    def __repr__(self) -> str:
        marker = ", temporary=True" if self.temporary else ""
        return f"S3Path({self.full_path()!r}{marker})"


# S3Bucket carries the typed scheme but defers the ``s3`` registry slot to
# S3Path (set after both classes so __init_subclass__ doesn't double-register).
S3Bucket.scheme = Scheme.S3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _chunked(data: bytes, size: int) -> Iterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


def _media_type_from_headers(headers: Any) -> "MediaType | None":
    ct = headers.get("Content-Type") or headers.get("content-type") if headers else None
    if not ct:
        return None
    return MediaType.from_(ct, default=None)


def _mtime_from_headers(headers: Any) -> float:
    lm = (headers.get("Last-Modified") or headers.get("last-modified")) if headers else None
    if not lm:
        return 0.0
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(lm).timestamp()
    except Exception:
        return 0.0
