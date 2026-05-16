"""S3-backed :class:`Path` implementation.

:class:`S3Path` is a :class:`RemotePath` over the ``s3://`` URL
scheme, talking to a boto3-shaped S3 client. The client is injected
at construction (``client=`` kwarg) — concrete production code uses
the :class:`yggdrasil.aws.fs.service.S3Service` factory, tests pass
a :class:`unittest.mock.Mock` that stubs the boto methods.

Holder primitives
-----------------

The new :class:`Path` substrate exposes positional reads / writes
through five hooks; :class:`S3Path` overrides three of them with
S3-native equivalents:

* :meth:`_read_mv` — range-based ``GetObject`` with
  ``Range: bytes={pos}-{pos+n-1}``. ``n=0`` short-circuits to an
  empty :class:`memoryview`. A miss surfaces as
  :class:`FileNotFoundError`; an EOF range yields an empty buffer
  to match file semantics.
* :meth:`_write_mv` — read existing object (if any), splice the
  incoming bytes at ``pos``, and ``PutObject`` the result. S3 has
  no native partial-write API, so this is a read-modify-write at
  the object granularity.
* :meth:`truncate` — ``PutObject`` of the head N bytes.
* :meth:`_clear` — ``DeleteObject`` (idempotent).

Reads of the whole object pass through the inherited ``read_bytes``
which calls ``_read_mv(-1, 0)`` once, so a parquet footer probe
costs one S3 request.

Filesystem surface
------------------

* :meth:`_stat_uncached` — ``HeadObject`` for the file shape;
  ``ListObjectsV2(MaxKeys=1, Delimiter='/')`` to disambiguate
  directory prefixes from missing keys.
* :meth:`_ls` — ``ListObjectsV2`` paginator, treats common-prefixes
  as sub-directories. Children inherit the same client.
* :meth:`_mkdir` — no-op (S3 has no directory concept; prefixes
  come into existence the moment a child object lands).
* :meth:`_remove_file` — ``DeleteObject``.
* :meth:`_remove_dir` — paginated bulk ``DeleteObjects`` (1000 keys
  per call). Idempotent under ``missing_ok=True``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterator, Optional

from yggdrasil.data.enums import Scheme
from yggdrasil.data.enums.media_type import MediaType
from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.io.path import RemotePath
from yggdrasil.io.path._retry import retry_sdk_call
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL

if TYPE_CHECKING:
    from botocore.client import BaseClient  # type: ignore[import-untyped]


__all__ = ["S3Path"]


# ---------------------------------------------------------------------------
# S3Path
# ---------------------------------------------------------------------------


class S3Path(RemotePath):
    """:class:`Path` over an S3 bucket via a boto3-shaped client.

    Construction shapes::

        S3Path("s3://bucket/key.parquet")                # uses default service
        S3Path("s3://bucket/", client=my_boto_client)    # explicit client
        S3Path(url=URL("s3://bucket/key"), client=mock)  # tests inject mocks

    The ``client`` kwarg accepts any object that quacks like the
    boto3 S3 client surface — ``head_object``, ``get_object``,
    ``put_object``, ``delete_object``, ``delete_objects``,
    ``list_objects_v2``, ``get_paginator``. Tests use
    :class:`unittest.mock.Mock`; production code passes the boto
    client owned by :class:`S3Service`.

    Singleton identity caching, the 5-minute default TTL, and the
    bounded ``_INSTANCES`` dict are inherited from
    :class:`RemotePath`. The matching ``stat_cache_ttl`` (also 300s)
    means a hot read after a hot write rides one ``HeadObject`` no
    matter how many call sites request the same key.

    Mutating ops (``put_object`` / ``DeleteObject`` /
    ``DeleteObjects``) call :meth:`_invalidate_stat_cache` before
    returning so two consumers sharing the singleton see consistent
    metadata after a write.
    """

    scheme: ClassVar[Scheme] = Scheme.S3

    #: URL schemes accepted on input; always normalized to ``s3``.
    _ACCEPTED_SCHEMES: ClassVar[frozenset[str]] = frozenset({"s3", "s3a", "s3n"})

    # ==================================================================
    # Singleton key
    # ==================================================================

    @classmethod
    def _singleton_key(
        cls,
        data: Any = None,
        *,
        url: URL | None = None,
        client: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Identity = ``(cls, canonical s3 URL, client)``.

        ``data`` collapses into ``url`` before keying so the string
        ``"s3://b/k"`` and the equivalent :class:`URL` map to the
        same singleton. ``client`` is part of the key because two
        callers binding the same URL to different boto clients
        (cross-account access, separate test fixtures) must NOT
        collide — passing ``client=None`` collapses to the lazy-
        resolved :class:`S3Service` singleton, so production callers
        share one instance per URL without paying the singleton key
        for explicit clients.
        """
        if url is None and isinstance(data, URL):
            url = data
        elif url is None and isinstance(data, str):
            url = URL.from_(data)
        if url is None:
            # No URL → no canonical identity. Fall back to a unique
            # sentinel so the Singleton machinery doesn't collide
            # unrelated paths.
            return (cls, object())
        normalized = cls._normalize_scheme(URL.from_(url))
        return (cls, str(normalized), client)

    # ==================================================================
    # Construction
    # ==================================================================

    def __init__(
        self,
        data: Any = None,
        *,
        url: URL | None = None,
        client: "BaseClient | Any | None" = None,
        temporary: bool = False,
        retry_sleep: Optional[Callable[[float], None]] = None,
        singleton_ttl: Any = ...,
        **kwargs: Any,
    ) -> None:
        # ``singleton_ttl`` is consumed by :meth:`Singleton.__new__`;
        # accept it here so the constructor signature stays open. The
        # singleton-cached re-init guard mirrors ``DatabricksPath`` —
        # the second constructor call collapses onto the live
        # instance, preserving the bound boto client + warm stat cache.
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return

        if url is None and isinstance(data, str):
            url = URL.from_(data)
            data = None
        if url is None and isinstance(data, URL):
            url = data
            data = None
        if url is not None:
            url = self._normalize_scheme(URL.from_(url))

        super().__init__(data=data, url=url, temporary=temporary, **kwargs)

        self._client = client
        # Injection point for tests — replace ``time.sleep`` with a
        # spy / no-op so retry behavior is observable without burning
        # wall-clock seconds.
        self._retry_sleep: Optional[Callable[[float], None]] = retry_sleep
        self._initialized = True

    @classmethod
    def _normalize_scheme(cls, url: URL) -> URL:
        """Coerce ``s3a://`` / ``s3n://`` to ``s3://``.

        Both Hadoop variants are common in Spark contexts; the
        canonical form on disk and in logs is ``s3://``, so we
        normalize at construction time and round-trip clean.
        """
        if url.scheme in cls._ACCEPTED_SCHEMES and url.scheme != cls.scheme:
            return url.with_scheme(cls.scheme)
        return url

    # ==================================================================
    # Client access
    # ==================================================================

    @property
    def client(self) -> Any:
        """The boto-shaped S3 client.

        Lazily builds a real boto3 client via :class:`S3Service`
        when none was passed. Tests that inject a :class:`Mock` at
        construction never trigger this branch.
        """
        if self._client is None:
            from yggdrasil.aws.fs.service import S3Service
            self._client = S3Service.current().boto_client
        return self._client

    def with_client(self, client: Any) -> "S3Path":
        """Return *self* with *client* installed.

        Mutating in place — same identity, new transport. Useful
        for swapping a stubbed client onto an existing path tree
        (e.g. wiring a test mock onto paths that came from a real
        URL parse).
        """
        self._client = client
        return self

    def _call(self, func, *args, **kwargs):
        """Invoke *func(*args, **kwargs)* under the standard retry policy.

        Transient errors (5xx, 429, BadRequest, throttling, transport
        timeouts) get up to 4 retries with 1 / 2 / 4 / 8 s sleeps.
        Permission errors (403, AccessDenied, expired token) get
        exactly one retry — usually enough to dodge a credential
        refresh race. Anything else (404 / NoSuchKey / InvalidRange)
        is deterministic and propagates.
        """
        if self._retry_sleep is not None:
            return retry_sdk_call(func, *args, sleep=self._retry_sleep, **kwargs)
        return retry_sdk_call(func, *args, **kwargs)

    # ==================================================================
    # URL parts → bucket / key
    # ==================================================================

    @property
    def bucket(self) -> str:
        host = self.url.host
        if not host:
            raise ValueError(f"S3 path has no bucket: {self.url!r}")
        return host

    @property
    def key(self) -> str:
        path = self.url.path or ""
        return path.lstrip("/")

    def full_path(self) -> str:
        """Render as ``s3://bucket/key``."""
        key = self.key
        return f"s3://{self.bucket}/{key}" if key else f"s3://{self.bucket}/"

    @property
    def size(self) -> int:
        """Object size from a (cached) ``HeadObject``. ``0`` when missing."""
        return int(self._stat().size)

    def _from_url(self, url: URL) -> "S3Path":
        """Sibling :class:`S3Path` with the same client."""
        return S3Path(url=url, client=self._client)

    # ==================================================================
    # PyArrow filesystem fast path
    # ==================================================================

    def arrow_filesystem(
        self,
        *,
        region: Optional[str] = None,
        **overrides: Any,
    ) -> Any:
        """Return a :class:`pyarrow.fs.S3FileSystem` for this path's bucket.

        Delegates to :meth:`S3Service.arrow_filesystem` so the
        credential snapshot logic lives in one place — every call
        site (raw ``S3Path``, ``VolumePath.arrow_filesystem``, the
        tabular fast path below) sees the same auth surface.

        For the pyarrow filesystem to know which bucket it's
        talking to you also need :attr:`arrow_uri` — pyarrow's
        S3FileSystem takes ``"bucket/key"`` strings, not full
        ``s3://`` URLs.
        """
        from yggdrasil.aws.fs.service import S3Service
        if isinstance(self._client, S3Service):
            service: S3Service = self._client
        else:
            service = S3Service.current()
        return service.arrow_filesystem(region=region, **overrides)

    @property
    def arrow_uri(self) -> str:
        """``bucket/key`` form expected by ``pyarrow.fs.S3FileSystem``.

        pyarrow's filesystem-aware readers (``pq.ParquetFile``,
        ``pa.ipc.open_file``, …) take a ``bucket/key`` path string
        paired with the filesystem object — not a full ``s3://`` URL.
        This property renders that form once so the fast-path call
        sites stay readable.
        """
        return f"{self.bucket}/{self.key}"

    # ==================================================================
    # Stat — uncached probe
    # ==================================================================

    def _stat_uncached(self) -> IOStats:
        """One ``HeadObject``; falls through to a list probe for prefixes."""
        if not self.key:
            return IOStats(size=0, mtime=0.0, kind=IOKind.DIRECTORY, mode=0)

        try:
            response = self._call(
                self.client.head_object,
                Bucket=self.bucket, Key=self.key,
            )
        except Exception as exc:
            if not _is_not_found(exc):
                raise
            response = None

        if response is not None:
            return IOStats(
                size=int(response.get("ContentLength", 0)),
                mtime=_mtime_from_response(response),
                kind=IOKind.FILE,
                mode=0,
                media_type=_media_type_from_response(response),
            )

        # No object at the exact key — probe whether it's a prefix.
        prefix = self.key
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        try:
            response = self._call(
                self.client.list_objects_v2,
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=1,
                Delimiter="/",
            )
        except Exception:
            return IOStats(size=0, mtime=0.0, kind=IOKind.MISSING, mode=0)

        if response.get("KeyCount", 0) > 0 or response.get("CommonPrefixes"):
            return IOStats(size=0, mtime=0.0, kind=IOKind.DIRECTORY, mode=0)
        return IOStats(size=0, mtime=0.0, kind=IOKind.MISSING, mode=0)

    # ==================================================================
    # Listing
    # ==================================================================

    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator["S3Path"]:
        """List direct (or recursive) children under this prefix."""
        prefix = self.key
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
        if not recursive:
            kwargs["Delimiter"] = "/"

        paginator = self.client.get_paginator("list_objects_v2")
        try:
            pages = paginator.paginate(**kwargs)
        except Exception:
            return

        for page in pages:
            for cp in page.get("CommonPrefixes") or ():
                sub_prefix = cp.get("Prefix")
                if sub_prefix:
                    yield self._make_child(sub_prefix, singleton_ttl=singleton_ttl)
            for obj in page.get("Contents") or ():
                key = obj.get("Key")
                if not key:
                    continue
                if not recursive and key.endswith("/") and obj.get("Size", 0) == 0:
                    # Listing leaks placeholder objects (zero-byte
                    # keys ending in '/') under non-recursive mode;
                    # treat them as directories that will surface as
                    # CommonPrefixes on a directory walk.
                    continue
                yield self._make_child(key, singleton_ttl=singleton_ttl)

    def _make_child(self, key: str, *, singleton_ttl: Any = False) -> "S3Path":
        # Skip the ``URL.from_(f"s3://...")`` parse — the bucket/host /
        # scheme are already canonical on ``self.url`` and only the
        # path changes. The ``_replace_path`` clone preserves the
        # invariants without going through :func:`urlsplit` for
        # every child a listing page yields.
        #
        # ``singleton_ttl`` defaults to ``False`` so listing children
        # stay out of the bounded ``S3Path._INSTANCES`` cache; callers
        # that want cached children thread it through ``ls``.
        cleaned = key.lstrip("/")
        url = self.url._replace_path("/" + cleaned if cleaned else "/")
        return S3Path(url=url, client=self._client, singleton_ttl=singleton_ttl)

    # ==================================================================
    # Mutators — mkdir / remove
    # ==================================================================

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        """No-op — S3 has no directory concept.

        Pure-prefix directories materialize when a child object
        lands under them. Writing a placeholder zero-byte key would
        just create cleanup work later.
        """
        del parents, exist_ok

    def _remove_file(self, missing_ok: bool = True, wait: WaitingConfig = True) -> None:
        try:
            self._call(
                self.client.delete_object,
                Bucket=self.bucket, Key=self.key,
            )
        except Exception:
            if not missing_ok:
                raise
            return
        self._invalidate_stat_cache()

    def _remove_dir(
        self, recursive: bool = True, missing_ok: bool = True, wait: WaitingConfig = True
    ) -> None:
        """Bulk-delete every object under the prefix.

        Pages through ``ListObjectsV2`` and batches up to 1000 keys
        per ``DeleteObjects`` call. Errors per-key are aggregated by
        boto into the response; we surface a single :class:`OSError`
        when any keys fail unless ``missing_ok=True`` swallows them.
        """
        if not recursive:
            placeholder = self.key
            if placeholder and not placeholder.endswith("/"):
                placeholder = placeholder + "/"
            try:
                self._call(
                    self.client.delete_object,
                    Bucket=self.bucket, Key=placeholder,
                )
            except Exception:
                if missing_ok:
                    return
                raise
            self._invalidate_stat_cache()
            return

        prefix = self.key
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        paginator = self.client.get_paginator("list_objects_v2")
        batch: list[dict[str, str]] = []
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents") or ():
                    key = obj.get("Key")
                    if not key:
                        continue
                    batch.append({"Key": key})
                    if len(batch) >= 1000:
                        self._delete_batch(batch)
                        batch = []
            if batch:
                self._delete_batch(batch)
        except Exception:
            if missing_ok:
                return
            raise
        self._invalidate_stat_cache()

    def _delete_batch(self, batch: list[dict]) -> None:
        if not batch:
            return
        response = self._call(
            self.client.delete_objects,
            Bucket=self.bucket,
            Delete={"Objects": batch, "Quiet": True},
        )
        errors = response.get("Errors") or []
        if errors:
            sample = ", ".join(
                f"{e.get('Key')!r}={e.get('Code')}" for e in errors[:3]
            )
            more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
            raise OSError(
                f"S3 delete_objects reported {len(errors)} error(s): "
                f"{sample}{more}"
            )

    # ==================================================================
    # Holder I/O — _read_mv / _write_mv / truncate / _clear
    # ==================================================================

    def read_mv(self, n: int, pos: int) -> memoryview:
        """Range read with a whole-file fast path that skips the size probe.

        The base :meth:`Holder.read_mv` resolves a ``n < 0`` "read to
        EOF" request by calling ``self.size`` first, which is one
        ``HeadObject`` round trip every time the stat cache is cold.
        For the dominant shape — ``read_bytes()`` /
        ``read_arrow_table()`` against a fresh path — that probe is
        wasted: the ``GetObject`` we're about to issue without a
        ``Range`` header returns the whole object *and* carries the
        canonical size in ``Content-Length``, which :meth:`_read_mv`
        folds back into the cache. Partial / positional reads keep
        the base bounds check so out-of-range windows still raise.
        """
        if n < 0 and pos == 0:
            return self._read_mv(-1, 0)
        return super().read_mv(n, pos)

    def _read_mv(self, n: int, pos: int) -> memoryview:
        """Range-based ``GetObject`` → :class:`memoryview` over bytes.

        :class:`Holder.read_mv` already normalized ``(n, pos)`` to a
        non-negative range that fits within :attr:`size`, so the
        ``Range`` header always covers a valid window — except for
        the ``(-1, 0)`` whole-file fast path above, which feeds
        ``n == -1`` straight through. Omitting the ``Range`` header
        in that case asks S3 for the whole object in one call.
        """
        if n == 0:
            return memoryview(b"")

        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": self.key}
        if n > 0:
            # Closed range — boto wants inclusive end byte.
            kwargs["Range"] = f"bytes={pos}-{pos + n - 1}"
        elif pos > 0:
            # ``bytes={pos}-`` — open-ended range from ``pos`` to EOF.
            kwargs["Range"] = f"bytes={pos}-"
        # else: full-object GET — no ``Range`` header.

        try:
            response = self._call(self.client.get_object, **kwargs)
        except Exception as exc:
            if _is_not_found(exc):
                raise FileNotFoundError(self.full_path()) from exc
            if _is_invalid_range(exc):
                return memoryview(b"")
            raise

        body = response.get("Body")
        try:
            data = body.read()
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        # Seed the stat cache from the response when S3 hands us the
        # canonical total. ``Content-Range: bytes <a>-<b>/<total>`` is
        # present on every successful ranged GET; whole-object GETs
        # (no ``Range`` header) instead surface the size as
        # ``Content-Length``. Either way the next ``size`` / ``exists``
        # lookup on this singleton path becomes a local hit — no
        # separate ``HeadObject`` round trip.
        total = _content_range_total(response)
        if total is None and "Range" not in kwargs:
            cl = response.get("ContentLength") if isinstance(response, dict) else None
            if isinstance(cl, int):
                total = cl
        if total is not None:
            mtime = _mtime_from_response(response)
            media_type = _media_type_from_response(response)
            if self._stat_cached is None:
                self._seed_stat_cache(IOStats(
                    size=total,
                    kind=IOKind.FILE,
                    mtime=mtime,
                    media_type=media_type,
                ))
            else:
                self._stat_cached.size = total
                if mtime:
                    self._stat_cached.mtime = mtime
                if media_type is not None and self._stat_cached.media_type is None:
                    self._stat_cached.media_type = media_type
                # Reset the TTL window so the fresh data we just folded
                # in lives a full ``stat_cache_ttl`` before re-probing.
                self._seed_stat_cache(self._stat_cached)

        return memoryview(data or b"")

    def _write_mv(self, data: memoryview, pos: int) -> int:
        """Splice *data* at *pos* via read-modify-write ``PutObject``.

        S3 has no positional write; the only honest answer at the
        object level is to download the existing payload, splice
        the new bytes in at *pos*, and re-upload. Most callers don't
        actually hit this hot path because :class:`BytesIO`'s
        scratch transaction absorbs positional writes locally and
        commits the whole object once on close.
        """
        n = len(data)
        if n == 0:
            return 0

        try:
            existing = bytes(self._read_mv(self._existing_size_or_zero(), 0))
        except FileNotFoundError:
            existing = b""

        # Pad up to pos with zeros if the splice lands past EOF —
        # mirrors local-fd behavior under O_RDWR.
        if pos > len(existing):
            existing = existing + b"\x00" * (pos - len(existing))

        head = existing[:pos]
        tail = existing[pos + n:]
        payload = head + bytes(data) + tail

        self._call(
            self.client.put_object,
            Bucket=self.bucket, Key=self.key, Body=payload,
        )
        self._seed_stat_cache(IOStats(
            size=len(payload),
            kind=IOKind.FILE,
            mtime=time.time(),
            media_type=self.media_type,
        ))
        return n

    def _existing_size_or_zero(self) -> int:
        """Read-side size probe used by RMW; tolerates a missing object."""
        try:
            return int(self._stat().size)
        except Exception:
            return 0

    def truncate(self, n: int) -> int:
        """Re-upload the head *n* bytes (zero-padded if extending)."""
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")

        current = 0
        try:
            existing = bytes(self._read_mv(self._existing_size_or_zero(), 0))
            current = len(existing)
        except FileNotFoundError:
            existing = b""

        if n == current and current > 0:
            return n

        if n == 0:
            payload = b""
        elif n <= len(existing):
            payload = existing[:n]
        else:
            payload = existing + b"\x00" * (n - len(existing))

        self._call(
            self.client.put_object,
            Bucket=self.bucket, Key=self.key, Body=payload,
        )
        self._seed_stat_cache(IOStats(
            size=len(payload),
            kind=IOKind.FILE,
            mtime=time.time(),
            media_type=self.media_type,
        ))
        return n

    def reserve(self, n: int) -> None:
        """No-op — S3 has no capacity-vs-size distinction."""
        if n < 0:
            raise ValueError(f"reserve size must be >= 0, got {n!r}")

    def _clear(self) -> None:
        """``DeleteObject``. Idempotent."""
        self._remove_file(missing_ok=True)

    # ==================================================================
    # _bread / _bwrite — fallbacks the base class declares abstract
    # ==================================================================

    def _bread(self, n: int, pos: int, mode):  # noqa: D401
        """Fallback whole-file read into a fresh :class:`BytesIO`.

        Used by callers that explicitly want a buffer instead of a
        :class:`memoryview`; the hot path goes through :meth:`_read_mv`
        directly.
        """
        from yggdrasil.io.bytes_io import BytesIO
        del mode
        size = n if n >= 0 else max(0, self._existing_size_or_zero() - pos)
        if size <= 0:
            return BytesIO()
        try:
            data = bytes(self._read_mv(size, pos))
        except FileNotFoundError:
            data = b""
        return BytesIO(data)

    def _bwrite(self, data, pos: int, mode) -> int:
        """Fallback whole-file write from a :class:`BytesIO`."""
        del mode
        if hasattr(data, "to_bytes"):
            payload = data.to_bytes()
        elif hasattr(data, "read"):
            payload = data.read()
        else:
            payload = bytes(data)
        if pos == 0:
            self.client.put_object(
                Bucket=self.bucket, Key=self.key, Body=payload,
            )
            self._seed_stat_cache(IOStats(
                size=len(payload),
                kind=IOKind.FILE,
                mtime=time.time(),
                media_type=self.media_type,
            ))
            return len(payload)
        return self._write_mv(memoryview(payload), pos)

    # ==================================================================
    # Repr
    # ==================================================================

    def __repr__(self) -> str:
        marker = ", temporary=True" if self.temporary else ""
        return f"S3Path({self.full_path()!r}{marker})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_not_found(exc: BaseException) -> bool:
    """True when *exc* looks like a ``404`` / ``NoSuchKey``.

    Doesn't import botocore — checks duck-typed attributes so the
    helper works with both real boto exceptions and the simpler
    shapes a test mock may raise.
    """
    code = ""
    status = None
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "") or ""
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if isinstance(exc, FileNotFoundError):
        return True
    if status == 404:
        return True
    if code in ("404", "NoSuchKey", "NotFound"):
        return True
    name = type(exc).__name__
    return name in ("NoSuchKey", "NotFound", "404")


def _is_invalid_range(exc: BaseException) -> bool:
    """True when *exc* looks like a ``416`` Range Not Satisfiable."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "")
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code == "InvalidRange" or status == 416:
            return True
    return False


def _content_range_total(response: Any) -> "int | None":
    """Extract the full-object size from an S3 ranged-GET response.

    Boto surfaces the ``Content-Range`` response header verbatim as
    ``ContentRange`` on the response dict — format ``bytes 0-99/200``
    where ``200`` is the canonical object size. Returns ``None`` when
    the header is absent (whole-object GETs leave it off) or when the
    total slot is ``*`` (rare; means the size was unknown to S3).
    """
    content_range = response.get("ContentRange") if isinstance(response, dict) else None
    if not content_range or "/" not in content_range:
        return None
    suffix = content_range.split("/", 1)[1].strip()
    if not suffix.isdigit():
        return None
    return int(suffix)


def _media_type_from_response(response: Any) -> "MediaType | None":
    """Resolve a :class:`MediaType` from an S3 GET/HEAD response.

    S3 surfaces the object's MIME type in the ``ContentType`` field
    (when the upload set one, or when ``inferred-content-type`` is
    enabled on the bucket). Returns ``None`` when the field is absent
    or unparseable — the caller falls back to URL-extension inference.
    """
    if not isinstance(response, dict):
        return None
    content_type = response.get("ContentType")
    if not content_type:
        return None
    return MediaType.from_(content_type, default=None)


def _mtime_from_response(response: Any) -> float:
    last_modified = response.get("LastModified")
    if last_modified is None:
        return 0.0
    try:
        return float(last_modified.timestamp())
    except Exception:
        return 0.0
