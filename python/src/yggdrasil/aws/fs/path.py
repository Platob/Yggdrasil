"""S3-backed :class:`Path` implementation.

Reads and writes go through an :class:`S3Service` (one per path
instance, shared across joinpath / parent / etc.) so every path
under one bucket reuses the same boto3 session and credential
refresh. The service holds a reference to the :class:`AWSClient`,
which owns the actual session.

Path mechanics
--------------

S3 has no real directories, only key prefixes. We model:

- ``s3://bucket/`` → the bucket root. ``ls`` lists objects with no
  prefix.
- ``s3://bucket/key/with/slashes`` → either an object (``is_file``)
  or a directory marker (``is_dir`` — empty key with trailing slash,
  or any key prefix that's a parent of other keys). The
  classification is done on demand via :meth:`_stat`.
- Reading non-existent keys raises :class:`FileNotFoundError` from
  the underlying boto error (``NoSuchKey``).

We accept ``s3://``, ``s3a://``, and ``s3n://`` as inputs (Hadoop
variants are common in Spark contexts) but always normalize the
URL's scheme to ``s3``. Output paths render as ``s3://``.

I/O patterns
------------

:meth:`pread` issues ``GetObject`` with a ``Range`` header. This is
the right primitive for the BytesIO transaction-buffer pattern
(one ``pread(n=-1, pos=0)`` per open) AND for parquet footer reads
(``pread(n=8, pos=size-8)``). Both are one S3 round-trip.

:meth:`write_stream` is overridden to use boto3's
``upload_fileobj``, which auto-multiparts above 8 MiB. This is the
flush path used by every BytesIO bound to an S3Path.

:meth:`pwrite` falls back to the inherited read-modify-write
helper from :class:`Path`. S3 has no positional write; this is the
honest answer. Most callers don't hit it because the BytesIO
transaction-buffer absorbs positional writes locally, then flushes
the whole object via ``write_stream``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Iterator, List, Optional, Tuple, Union

from yggdrasil.aws.fs.service import S3Service
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.fs import Path
from yggdrasil.io.path_stat import PathKind, PathStats
from yggdrasil.io.url import URL
from yggdrasil.lazy_imports import botocore_module

if TYPE_CHECKING:
    from botocore.client import BaseClient  # type: ignore[import-untyped]


__all__ = ["S3Path"]


#: Above this size, write_stream uses upload_fileobj (which internally
#: multiparts). Below, we use a single PutObject. boto3's default
#: multipart threshold is 8 MiB; matching it keeps behaviour predictable.
_MULTIPART_THRESHOLD: int = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# S3Path
# ---------------------------------------------------------------------------


class S3Path(Path):
    """Path subclass over an S3 bucket via boto3.

    Construction:

        >>> p = S3Path("s3://my-bucket/data/file.parquet")
        >>> p.read_bytes()  # via pread → one GetObject

        >>> # With explicit creds:
        >>> from yggdrasil.aws import AWSClient, AWSConfig
        >>> client = AWSClient(AWSConfig(role_arn="arn:aws:iam::1234:role/Reader"))
        >>> p = S3Path("s3://my-bucket/data/", service=client.s3)

    The ``service`` kwarg holds the :class:`S3Service`. Derived
    paths (``parent``, ``joinpath``) inherit the same service so a
    single set of credentials covers a whole tree.
    """

    __slots__ = ("_service",)

    scheme: ClassVar[str] = "s3"

    #: URL schemes we accept on input. Always normalized to ``s3``.
    _ACCEPTED_SCHEMES: ClassVar[frozenset[str]] = frozenset({"s3", "s3a", "s3n"})

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        obj: Any = None,
        *,
        url: URL | None = None,
        temporary: bool = False,
        auto_open: bool = True,
        service: Optional[S3Service] = None,
    ) -> None:
        # Normalize the URL scheme so ``s3a://...`` round-trips as
        # ``s3://...``. We do this before super().__init__ stamps
        # the URL on self.
        if url is not None:
            url = self._normalize_scheme(URL.from_(url))
        elif isinstance(obj, S3Path):
            url = obj.url
            if service is None:
                service = obj._service
        elif obj is not None and not isinstance(obj, Path):
            url = self._normalize_scheme(URL.from_(obj))

        super().__init__(
            url=url,
            temporary=temporary,
            auto_open=auto_open,
        )
        self._service: S3Service = (
            service if service is not None else S3Service.current()
        )

    @classmethod
    def _normalize_scheme(cls, url: URL) -> URL:
        """Coerce ``s3a://`` / ``s3n://`` to ``s3://``."""
        if url.scheme in cls._ACCEPTED_SCHEMES and url.scheme != cls.scheme:
            return url.with_scheme(cls.scheme)
        return url

    @classmethod
    def handles(cls, obj: Any) -> bool:
        """Accept any of the three S3 URI schemes."""
        if isinstance(obj, URL):
            return obj.scheme in cls._ACCEPTED_SCHEMES
        if isinstance(obj, str):
            return any(
                obj.startswith(f"{s}:/") for s in cls._ACCEPTED_SCHEMES
            )
        try:
            return URL.from_(obj).scheme in cls._ACCEPTED_SCHEMES
        except (ValueError, TypeError):
            return False

    # ------------------------------------------------------------------
    # Service / client access
    # ------------------------------------------------------------------

    @property
    def service(self) -> S3Service:
        return self._service

    @property
    def client(self) -> "BaseClient":
        return self._service.boto_client

    # ------------------------------------------------------------------
    # URL parts → bucket / key
    # ------------------------------------------------------------------

    @property
    def bucket(self) -> str:
        host = self.url.host
        if not host:
            raise ValueError(f"S3 path has no bucket: {self.url!r}")
        return host

    @property
    def key(self) -> str:
        """The S3 key — path part with leading slash stripped."""
        path = self.url.path or ""
        return path.lstrip("/")

    def full_path(self) -> str:
        """Render as ``s3://bucket/key``. Always uses ``s3://``."""
        key = self.key
        if key:
            return f"s3://{self.bucket}/{key}"
        return f"s3://{self.bucket}/"

    @property
    def is_local(self) -> bool:
        return False

    def _from_url(self, url: URL) -> "S3Path":
        """Override base — preserve the service across URL-derived paths."""
        return type(self)(url=url, service=self._service)

    # ==================================================================
    # Stat
    # ==================================================================

    def _stat(self) -> PathStats:
        """One HeadObject call; falls through to a list-objects probe
        for prefixes that are pure directories (no object at the key).
        """
        if not self.key:
            # Bucket root. HeadBucket would tell us the bucket exists
            # but not whether anything's in it. Treat the root as a
            # directory always — concrete contents come from _ls.
            return PathStats(size=0, mtime=0.0, kind=PathKind.DIRECTORY, mode=0)

        # Try HeadObject for an exact-key match.
        botocore = botocore_module()
        try:
            response = self.client.head_object(
                Bucket=self.bucket, Key=self.key,
            )
        except botocore.exceptions.ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            # 404 / NoSuchKey → fall through to directory probe below.
            # All other errors propagate (e.g. 403 = caller's problem).
            if code not in ("404", "NoSuchKey", "NotFound") and status != 404:
                raise
        else:
            return PathStats(
                size=int(response.get("ContentLength", 0)),
                mtime=_mtime_from_response(response),
                kind=PathKind.FILE,
                mode=0,
            )

        # No object at that key — could be a directory prefix.
        # ListObjectsV2 with prefix=key+"/" max-keys=1 tells us
        # whether anything's there.
        prefix = self.key
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=1,
                Delimiter="/",
            )
        except Exception:
            return PathStats(size=0, mtime=0.0, kind=PathKind.MISSING, mode=0)

        if response.get("KeyCount", 0) > 0 or response.get("CommonPrefixes"):
            return PathStats(size=0, mtime=0.0, kind=PathKind.DIRECTORY, mode=0)

        return PathStats(size=0, mtime=0.0, kind=PathKind.MISSING, mode=0)

    # ==================================================================
    # Listing
    # ==================================================================

    def _ls(
        self,
        recursive: bool = False,
        allow_not_found: bool = True,
    ) -> Iterator["S3Path"]:
        """List children. Non-recursive uses ``Delimiter="/"`` to get
        common-prefixes (directory analogues) plus immediate keys.
        """
        prefix = self.key
        # The bucket root has no prefix; otherwise ensure trailing "/"
        # so we don't accidentally match siblings of a key (e.g.
        # listing "data" shouldn't match "data2.txt").
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
        if not recursive:
            kwargs["Delimiter"] = "/"

        paginator = self.client.get_paginator("list_objects_v2")

        try:
            pages = paginator.paginate(**kwargs)
            for page in pages:
                # Common prefixes = "subdirectories" in the non-recursive case.
                for cp in page.get("CommonPrefixes") or ():
                    sub_prefix = cp.get("Prefix")
                    if not sub_prefix:
                        continue
                    yield self._make_child(sub_prefix)
                for obj in page.get("Contents") or ():
                    key = obj.get("Key")
                    if not key:
                        continue
                    # Skip directory placeholder objects (key ends in
                    # "/" with size 0) — they're listed as common
                    # prefixes when non-recursive; including them
                    # again would duplicate.
                    if not recursive and key.endswith("/") and obj.get("Size", 0) == 0:
                        continue
                    yield self._make_child(key)
        except Exception:
            if allow_not_found:
                return
            raise

    def _make_child(self, key: str) -> "S3Path":
        """Build a child :class:`S3Path` from an absolute key string."""
        # key might end in "/" for prefixes; URL.from_str preserves
        # that and our `key` property strips the leading slash.
        url = URL.from_str(f"s3://{self.bucket}/{key.lstrip('/')}")
        return self._from_url(url)

    # ==================================================================
    # Mkdir
    # ==================================================================

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        """Directory creation in S3 is mostly a no-op.

        Pure-prefix directories (``s3://bucket/data/`` with no
        ``data/`` object) come into existence the moment any child
        is written — listing the prefix returns them as common
        prefixes. We don't write a placeholder object; that would
        create cleanup work later.

        ``parents=True`` and ``exist_ok=True`` are vacuously
        satisfied. ``exist_ok=False`` against an existing prefix
        would normally raise — we don't enforce that for S3 because
        "exists" of a prefix is racy by nature.
        """
        del parents, exist_ok

    # ==================================================================
    # Remove
    # ==================================================================

    def _remove_file(self, allow_not_found: bool = True) -> None:
        """Single DeleteObject call. Idempotent."""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=self.key)
        except Exception:
            if allow_not_found:
                return
            raise

    def _remove_dir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> None:
        """Bulk-delete every object under the prefix.

        S3's batch DeleteObjects accepts up to 1000 keys per call.
        We page through the prefix listing, batch-deleting up to
        1000 at a time. ``with_root`` is informational on S3 (no
        directory placeholder unless one was explicitly written;
        the "root" of the prefix has no object identity).
        """
        del with_root  # No directory inode to remove on S3.

        if not recursive:
            # Non-recursive remove of an "empty directory" — the
            # placeholder object case. Try delete_object on the
            # prefix-with-trailing-slash key; succeeds whether or
            # not it exists.
            placeholder = self.key
            if placeholder and not placeholder.endswith("/"):
                placeholder = placeholder + "/"
            try:
                self.client.delete_object(Bucket=self.bucket, Key=placeholder)
            except Exception:
                if allow_not_found:
                    return
                raise
            return

        prefix = self.key
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        paginator = self.client.get_paginator("list_objects_v2")
        try:
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            batch: list[dict[str, str]] = []
            for page in pages:
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
            if allow_not_found:
                return
            raise

    def _delete_batch(self, batch: List[dict]) -> None:
        """One DeleteObjects call. Errors per-key are aggregated by
        boto into the response; we surface as a single error if any
        keys failed.
        """
        if not batch:
            return
        response = self.client.delete_objects(
            Bucket=self.bucket,
            Delete={"Objects": batch, "Quiet": True},
        )
        errors = response.get("Errors") or []
        if errors:
            # Build a compact summary; full per-key detail is in errors.
            sample = ", ".join(
                f"{e.get('Key')!r}={e.get('Code')}"
                for e in errors[:3]
            )
            more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
            raise OSError(
                f"S3 delete_objects reported {len(errors)} error(s): "
                f"{sample}{more}"
            )

    # ==================================================================
    # Open — return a BytesIO bound to this path
    # ==================================================================

    def _open(
        self,
        mode: str = "rb",
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        auto_open: bool = True,
        touch: bool = False,
    ) -> BytesIO:
        """Return a :class:`BytesIO` bound to this S3 path.

        The :class:`BytesIO`'s acquire flow:

        1. Recognize ``self`` as non-local (``is_local=False``).
        2. Build an internal transaction buffer (a regular BytesIO
           with local-spill).
        3. Call ``self.pread(n=-1, pos=0)`` once to download the
           object into the transaction buffer.

        On flush:

        4. ``self.write_stream(transaction_buffer)`` uploads via
           ``upload_fileobj``.

        ``touch`` is honored for ``"r"`` modes (verifies presence)
        and ignored for ``"w"`` / ``"a"`` / ``"x"`` (the file is
        about to be created anyway).
        """
        del encoding, errors, newline  # Binary I/O only at this layer.

        if touch and "r" in mode and "+" not in mode:
            if not self.exists():
                raise FileNotFoundError(self.full_path())

        return BytesIO(
            path=self,
            mode=mode,
            auto_open=auto_open,
        )

    # ==================================================================
    # pread — Range-based GetObject
    # ==================================================================

    def pread(
        self,
        n: int,
        pos: int,
        *,
        default: Any = ...,
    ) -> bytes:
        """Read *n* bytes at offset *pos* via GetObject Range header.

        ``n=-1`` reads from *pos* to end of object — a single
        GetObject with ``Range: bytes={pos}-`` (or no Range header
        if pos=0, which is slightly cheaper on some endpoints).

        Failure handling: 404 / NoSuchKey returns *default* if
        provided, else raises :class:`FileNotFoundError`. Any other
        boto error propagates.
        """
        if pos < 0:
            raise ValueError("pread position must be >= 0")
        if n == 0:
            return b""

        botocore = botocore_module()

        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": self.key}
        if n > 0:
            kwargs["Range"] = f"bytes={pos}-{pos + n - 1}"
        elif pos > 0:
            kwargs["Range"] = f"bytes={pos}-"
        # n=-1 and pos=0 → no Range; full-object download.

        try:
            response = self.client.get_object(**kwargs)
        except botocore.exceptions.ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("NoSuchKey", "404", "NotFound") or status == 404:
                if default is ...:
                    raise FileNotFoundError(self.full_path()) from exc
                return default
            # 416 = Range not satisfiable (pos >= object size).
            # Mirror file semantics: return empty bytes rather than raise.
            if code == "InvalidRange" or status == 416:
                return b""
            raise

        body = response.get("Body")
        try:
            data = body.read()
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        return data

    def pwrite(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
        *,
        parents: bool = True,
    ) -> int:
        """Positional write via read-modify-write.

        S3 has no positional write primitive. We delegate to the
        inherited helper which reads the full object, splices, and
        writes back — slow but correct.

        Most callers don't hit this: a :class:`BytesIO` bound to
        an S3 path absorbs positional writes into its local
        transaction buffer and flushes once on close via
        :meth:`write_stream`.
        """
        return self._pwrite_via_rmw(data, pos, parents=parents)

    # ==================================================================
    # write_stream — multipart-aware upload
    # ==================================================================

    def write_stream(
        self,
        src,
        *,
        batch_size: int = 4 * 1024 * 1024,
        parents: bool = True,
    ) -> int:
        """Upload *src* to this S3 key.

        Above :data:`_MULTIPART_THRESHOLD`, we use boto3's
        ``upload_fileobj`` which auto-multiparts and parallelizes.
        Below, single PutObject — one round-trip, lower overhead.

        The size probe is best-effort: we look at ``src.size`` for
        yggdrasil :class:`BytesIO`, or ``len()`` for bytes-likes,
        and fall back to multipart for anything else (safer
        default; multipart handles arbitrary streams).
        """
        del parents, batch_size  # S3 has no parents; multipart owns chunking.

        size = self._probe_src_size(src)
        client = self.client

        if size is not None and size <= _MULTIPART_THRESHOLD:
            # Single PutObject. Materialize bytes if we have to.
            payload = self._materialize_small(src, size)
            client.put_object(
                Bucket=self.bucket,
                Key=self.key,
                Body=payload,
            )
            return len(payload)

        # Multipart via upload_fileobj. boto3 needs a file-like with
        # .read(). yggdrasil BytesIO satisfies that; raw bytes need a
        # quick BytesIO wrap.
        fileobj, total = self._adapt_for_upload(src)
        try:
            client.upload_fileobj(
                Fileobj=fileobj,
                Bucket=self.bucket,
                Key=self.key,
            )
        finally:
            close = getattr(fileobj, "close", None)
            # Only close adapter wrappers we created ourselves —
            # don't close the caller's file-like.
            if total is not None and callable(close) and fileobj is not src:
                close()

        # ``total`` is None when we couldn't size the source up front;
        # post-upload we don't have a cheap byte count without a HEAD.
        # Best-effort: stat the result.
        if total is not None:
            return total
        try:
            return int(self.size)
        except Exception:
            return 0

    @staticmethod
    def _probe_src_size(src: Any) -> Optional[int]:
        """Return *src*'s byte size if cheaply known, else None."""
        if isinstance(src, (bytes, bytearray, memoryview)):
            return len(src)
        if isinstance(src, BytesIO):
            try:
                return src.size
            except Exception:
                return None
        # Stdlib io.BytesIO has getbuffer(); use that for the size.
        getbuffer = getattr(src, "getbuffer", None)
        if callable(getbuffer):
            try:
                return len(getbuffer())
            except Exception:
                return None
        return None

    @staticmethod
    def _materialize_small(src: Any, size: int) -> bytes:
        """Coerce *src* to a bytes payload for a single PutObject."""
        if isinstance(src, bytes):
            return src
        if isinstance(src, (bytearray, memoryview)):
            return bytes(src)
        if isinstance(src, BytesIO):
            # Use to_bytes — for memory-mode this is a single copy
            # of the bytearray; for spilled it's an mmap copy.
            return src.to_bytes()
        # File-like fallback: read everything.
        return src.read(size)

    @staticmethod
    def _adapt_for_upload(src: Any) -> Tuple[Any, Optional[int]]:
        """Return a (file-like, total_or_None) pair suitable for
        :meth:`upload_fileobj`.

        - bytes-like → wrap in stdlib io.BytesIO; total = len.
        - yggdrasil BytesIO → already a file-like; total = src.size
          if cheaply known.
        - anything with .read() → pass through; total = None.
        """
        import io as _io

        if isinstance(src, (bytes, bytearray, memoryview)):
            data = bytes(src)
            return _io.BytesIO(data), len(data)

        if isinstance(src, BytesIO):
            # Make sure we're at offset 0 so upload_fileobj reads
            # the whole thing.
            try:
                src.seek(0)
            except Exception:
                pass
            try:
                return src, src.size
            except Exception:
                return src, None

        if hasattr(src, "read"):
            return src, None

        raise TypeError(
            f"S3Path.write_stream: cannot adapt source of type "
            f"{type(src).__name__} for upload."
        )

    # ==================================================================
    # Repr
    # ==================================================================

    def __repr__(self) -> str:
        if self.temporary:
            return f"S3Path({self.full_path()!r}, temporary=True)"
        return f"S3Path({self.full_path()!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mtime_from_response(response: Any) -> float:
    """Extract LastModified as epoch-seconds from a HeadObject response."""
    last_modified = response.get("LastModified")
    if last_modified is None:
        return 0.0
    try:
        return float(last_modified.timestamp())
    except Exception:
        return 0.0