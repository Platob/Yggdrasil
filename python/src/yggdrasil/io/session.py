"""HTTP session abstraction with transparent local and remote cache support."""

from __future__ import annotations

import datetime as dt
import itertools
import logging
import os
import pathlib
import pickle
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Iterator,
    Mapping,
    Optional,
    Sequence,
)

import pyarrow as pa

from yggdrasil.arrow.cast import rechunk_arrow_batches
from yggdrasil.concurrent.threading import Job, JobPoolExecutor
from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.dataclasses.waiting import (
    DEFAULT_WAITING_CONFIG,
    WaitingConfig,
    WaitingConfigArg,
)
from yggdrasil.data.enums import Codec, Codecs, MediaType, MimeType, MimeTypes, Mode
from .authorization.base import Authorization
from .bytes_io import BytesIO
from .headers import Headers
from .memory import Memory
from .request import PreparedRequest
from .response import RESPONSE_ARROW_SCHEMA, Response, RESPONSE_SCHEMA
from .response_batch import ResponseBatch
from .send_config import CacheConfig, SendConfig, SendManyConfig, _request_column_sql_name
from .url import URL

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame

__all__ = [
    "Session",
    "CacheConfig",
    "SendConfig",
    "SendManyConfig",
    "ResponseBatch",
]


LOGGER = logging.getLogger(__name__)


# Cap on per-batch byte size when emitting responses from a Spark
# `mapInArrow` worker. 128 MiB matches Spark's default Arrow batch
# preference and keeps a single oversized response from inflating the
# whole partition's output. A response that is itself larger than the
# cap is sliced row-wise by the shared rechunker, which never splits a
# single row across batches.
_SPARK_RESPONSE_BATCH_BYTE_LIMIT: int = 128 * 1024 * 1024


# Local cache is a flat URL-mirrored tree of Arrow IPC files —
# ``<root>/<METHOD>/<host>/<seg>/.../<public_hash>.arrow`` (see
# :func:`_local_fast_path_relative`). One file per request identity,
# resolved by ``stat`` + IPC decode; no partitioned reader, no
# schema sidecar, no Hive layout. Stale-file sweep is a TTL walker
# (see :func:`_cleanup_local_fast_path`) the writer fires after a
# successful insert; throttled by a sentinel file so a hot batch
# only walks the tree once per cleanup window.


# Per-segment cap when mirroring URL paths into the on-disk fast-path
# tree. Most filesystems allow 255 bytes per filename, but URL segments
# can carry tokens (signed query payloads, opaque ids, base64 blobs)
# that blow past that limit on their own. 80 bytes is a conservative
# upper bound that still leaves plenty of headroom for the ``<hex>``
# suffix appended when a segment has to be folded into its xxh3 digest.
_FAST_PATH_SEGMENT_MAX_BYTES: int = 80

# Filesystem-hostile chars in a single segment: NUL + control chars,
# the path separators on every OS we ship to, and the Windows-reserved
# set. Everything else (incl. unicode, ``.``, ``-``, ``_``, ``%``) is
# kept as-is so an operator browsing the cache can read the URL back.
_FAST_PATH_UNSAFE_RE = re.compile(r'[\x00-\x1f/\\:*?"<>|]+')


def _safe_fast_path_segment(
    seg: str, *, max_bytes: int = _FAST_PATH_SEGMENT_MAX_BYTES,
) -> str:
    """Sanitize one URL segment for use as a cache directory name.

    Replaces NUL / control chars and the filesystem-reserved set with
    a single ``_`` so the result is a legal directory entry on every
    supported OS. When the sanitized segment's UTF-8 length exceeds
    *max_bytes* the segment is folded to a short readable prefix plus
    the xxh3_64 hex of the *original* token — the prefix stays human
    grep-able while the digest restores uniqueness for the directory.
    The leaf ``<public_hash>.arrow`` filename still carries the full
    request identity, so collisions on a sanitized segment only group
    sibling entries under the same parent, never overwrite them.
    """
    import xxhash

    if not seg:
        return "_"
    safe = _FAST_PATH_UNSAFE_RE.sub("_", seg).strip(" .")
    if not safe:
        safe = "_"
    encoded = safe.encode("utf-8")
    if len(encoded) <= max_bytes:
        return safe
    digest = xxhash.xxh3_64(seg.encode("utf-8")).hexdigest()  # 16 hex chars
    # Reserve room for ``-<digest>``; clip on a byte boundary so we
    # never split a multi-byte UTF-8 character mid-encoding.
    head_budget = max_bytes - len(digest) - 1
    if head_budget <= 0:
        return digest
    head = encoded[:head_budget].decode("utf-8", errors="ignore").rstrip(" .")
    return f"{head}-{digest}" if head else digest


def _local_fast_path_relative(
    method: "str | None",
    url: "URL | None",
    public_hash: int,
) -> str:
    """Build the per-request fast-path file location under the cache root.

    Mirrors the request's URL structure on disk — ``<METHOD>/<host>/<seg>/.../<hex>.arrow``
    — so an operator can ``ls`` the cache tree and see which endpoints are
    populated, while still keying the leaf file by the full request
    ``public_hash`` (xxh3_64 over anonymized method+url+headers+body) so
    requests that share a path but differ on query / body / headers
    don't overwrite each other. Each directory segment is sanitized via
    :func:`_safe_fast_path_segment`, which folds segments longer than
    ``_FAST_PATH_SEGMENT_MAX_BYTES`` to ``<prefix>-<xxh3_64hex>`` so a
    rogue token can't bust the 255-byte filename limit on any FS.
    Returns a relative ``os.sep``-joined string so the fire-and-forget
    :class:`Job` payload stays a plain string across worker boundaries.
    """
    parts: list[str] = [_safe_fast_path_segment((method or "GET").upper())]
    if url is not None:
        host = (url.host or "").lower()
        if host:
            parts.append(_safe_fast_path_segment(host))
        for raw in (url.path or "").strip("/").split("/"):
            if raw:
                parts.append(_safe_fast_path_segment(raw))
    parts.append(f"{public_hash & 0xFFFFFFFFFFFFFFFF:016x}.arrow")
    return os.path.join(*parts)


# Minimum body size (bytes) before the cache-persist auto-compress
# helper bothers to gzip. Below ~1 MiB the gzip header overhead +
# per-call CPU + the small storage win is mostly noise; the Arrow IPC
# writer on the local fast-path already zstd-compresses the row.
_BODY_AUTOCOMPRESS_MIN_SIZE: int = 1 * 1024 * 1024

# Plaintext MIME types worth gzipping before cache persistence. Routed
# through the :class:`MimeType` enum so the alias / case / prefix
# normalization is consistent with the rest of the codebase — if a
# new plaintext format gets added to :class:`MimeTypes`, drop it into
# this set in one place rather than maintaining a string list. Binary
# entropy-dense formats (parquet, arrow IPC, mp4, zip, jpeg, …) stay
# out — gzipping them costs CPU for near-zero savings.
_AUTOCOMPRESS_MIMES: frozenset[MimeType] = frozenset({
    MimeTypes.JSON,
    MimeTypes.NDJSON,
    MimeTypes.XML,
    MimeTypes.HTML,
    MimeTypes.PLAIN,
    MimeTypes.CSV,
    MimeTypes.TSV,
    MimeTypes.YAML,
    MimeTypes.TOML,
})


def _maybe_autocompress_body_for_cache(
    response: Response,
    *,
    min_size: int = _BODY_AUTOCOMPRESS_MIN_SIZE,
    codec: Codec = Codecs.GZIP,
) -> None:
    """Gzip-compress *response.body* in place before cache persistence.

    Applies only when ALL of:

    * the response has no existing ``Content-Encoding`` header (we
      don't recompress brotli / gzip / zstd payloads — that's a no-win
      for storage and breaks ``.content`` round-tripping),
    * the response's resolved :class:`MimeType` is in
      :data:`_AUTOCOMPRESS_MIMES`,
    * the body is at least *min_size* bytes,
    * gzip actually shrinks it by >10% — the storage win has to
      outweigh the extra CPU on both the persist and the eventual
      read.

    On compress: swaps a fresh :class:`Memory` over the gzipped bytes
    into ``response.buffer`` and stamps ``Content-Encoding: gzip`` +
    new ``Content-Length`` via :meth:`Response.set_media_type`, which
    already invalidates the response's deterministic-projection cache
    so the row that lands in the Arrow batch carries the compressed
    body + correct headers in one consistent shape. The cache reader
    side picks the codec back off ``Content-Encoding`` and routes
    ``.content`` / ``.text`` / ``.json`` through the existing
    :class:`Codec` path — no special-case decompression on the read
    path.
    """
    if response.headers.get("Content-Encoding"):
        return
    buffer = response.buffer
    if buffer is None or buffer.size < min_size:
        return
    mime = response.media_type.mime_type if response.media_type is not None else None
    if mime is None or mime not in _AUTOCOMPRESS_MIMES:
        return

    raw = buffer.to_bytes()
    compressed = codec.compress_bytes(raw)
    if len(compressed) >= int(len(raw) * 0.9):
        # < 10% savings — gzip overhead + read-side decompress isn't
        # worth it. Skip and persist the original bytes.
        return

    new_buffer = Memory()
    with new_buffer.open(mode="wb", owns_holder=False) as bio:
        bio.write(compressed)
    response.buffer = new_buffer
    response.set_media_type(MediaType.from_mime(mime_type=mime, codec=codec))


def _store_fast_path_arrow_batch(
    cache_root_str: str,
    rel_path: str,
    batch: pa.RecordBatch,
) -> None:
    """Persist *batch* to the per-request fast-path file under *cache_root_str*.

    *rel_path* is the precomputed URL-mirrored location (see
    :func:`_local_fast_path_relative`) — strings only, so the
    fire-and-forget :class:`Job` pickles cheaply. Bypasses
    :class:`FolderIO` entirely: one Arrow IPC stream per request
    ``public_hash``, parent dirs created on demand. Tmp + atomic
    ``os.replace`` keeps a partial write from being read mid-flush by
    a concurrent fast-path reader.
    """
    final = pathlib.Path(cache_root_str) / rel_path
    try:
        final.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    tmp = final.parent / f".{final.name}.{os.urandom(4).hex()}.tmp"
    try:
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, batch.schema) as writer:
            writer.write_batch(batch)
        tmp.write_bytes(sink.getvalue().to_pybytes())
        os.replace(tmp, final)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _read_fast_path_arrow_batch(
    cache_root: Any, rel_path: str,
) -> "pa.RecordBatch | None":
    """Read the per-request fast-path file at *rel_path* under *cache_root*.

    Returns ``None`` when the file is missing, empty, or otherwise
    unreadable so the caller can fall through to the bulk-lookup
    path without a special error case (corrupt entry, race with a
    concurrent rewrite, schema drift, …).
    """
    file_path = pathlib.Path(os.fspath(cache_root)) / rel_path
    try:
        payload = file_path.read_bytes()
    except OSError:
        return None
    if not payload:
        return None
    try:
        reader = pa.ipc.open_stream(pa.BufferReader(payload))
        batches = [b for b in reader if b.num_rows > 0]
    except Exception:
        return None
    if not batches:
        return None
    if len(batches) == 1:
        return batches[0]
    chunks = pa.Table.from_batches(batches).combine_chunks().to_batches()
    return chunks[0] if chunks else None


# Sentinel filename written under the cache root after each cleanup
# pass. Its mtime drives the throttle so concurrent writers and
# repeat batches don't fan out into a full-tree walk every time.
_FAST_PATH_CLEANUP_SENTINEL: str = ".last_cleanup"


def _cleanup_local_fast_path(
    cache_root_str: str,
    *,
    ttl_seconds: float,
    throttle_seconds: float = 60.0,
) -> int:
    """Walk the URL-mirrored fast-path tree and unlink ``.arrow`` files older than *ttl_seconds*.

    Throttled by the ``.last_cleanup`` sentinel under *cache_root_str*
    — a write within ``throttle_seconds`` of the last successful
    pass short-circuits, so a hot batch of 10k requests only walks
    the tree once. Tmp / hidden files (leading ``.``) are skipped so
    a partial fast-path write in flight isn't yanked out from under
    the writer. Best-effort: any per-file ``OSError`` is logged and
    the walk continues. Returns the number of files unlinked.
    """
    if ttl_seconds <= 0:
        return 0
    root = pathlib.Path(cache_root_str)
    if not root.is_dir():
        return 0

    sentinel = root / _FAST_PATH_CLEANUP_SENTINEL
    now = time.time()
    try:
        last = sentinel.stat().st_mtime
    except OSError:
        last = 0.0
    if last and now - last < throttle_seconds:
        return 0

    cutoff = now - ttl_seconds
    removed = 0
    try:
        for entry in root.rglob("*.arrow"):
            if entry.name.startswith("."):
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if mtime >= cutoff:
                continue
            try:
                entry.unlink()
                removed += 1
            except OSError as exc:
                LOGGER.debug("Cache cleanup: failed to unlink %s: %s", entry, exc)
    except OSError as exc:
        LOGGER.debug("Cache cleanup: walk under %s aborted: %s", root, exc)
        return removed

    # Refresh the sentinel even when nothing was unlinked — that's
    # what the throttle is checking for, not the unlink count.
    try:
        sentinel.touch()
    except OSError:
        pass
    return removed


def _encode_request_data(
    data: Any,
) -> tuple[bytes, "str | None"]:
    """Encode a ``requests``-style ``data=`` payload into request bytes.

    * ``bytes`` / ``bytearray`` / ``memoryview`` → raw bytes, no header
      change (caller's ``Content-Type`` wins).
    * ``str`` → UTF-8 bytes, no header change.
    * ``Mapping`` / ``Iterable[tuple]`` → ``urlencode(doseq=True)`` with
      ``application/x-www-form-urlencoded`` as the suggested header,
      matching :meth:`requests.Session.post`.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data), None
    if isinstance(data, str):
        return data.encode("utf-8"), None
    from urllib.parse import urlencode

    return urlencode(data, doseq=True).encode("utf-8"), "application/x-www-form-urlencoded"


def _hashable_identity_value(value: Any) -> Any:
    """Coerce an ``__init__`` argument into a hashable form for the singleton key.

    Most argument shapes (frozen :class:`WaitingConfig`,
    :class:`Authorization`, :class:`URL`, primitives, ``None``) are
    already hashable and pass through unchanged. A few common
    constructor-input shapes that aren't get canonicalised here so
    two callers that spell the same identity slightly differently
    still collapse onto one singleton:

    * a ``str`` that parses as a URL is fed through :meth:`URL.from_`
      and stringified, so ``"https://x.com"`` and ``URL("https://x.com")``
      key the same way;
    * :class:`Headers` / generic :class:`Mapping` collapse to a tuple
      of sorted items;
    * sets become sorted tuples, lists become tuples.
    """
    if value is None:
        return None
    if isinstance(value, URL):
        return value.to_string()
    if isinstance(value, Mapping):
        return tuple(sorted(value.items()))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(value))
    if isinstance(value, list):
        return tuple(value)
    return value


def _format_cookie_header(cookies: Any) -> str:
    """Serialize a ``requests``-style ``cookies=`` arg into one ``Cookie`` header value."""
    if hasattr(cookies, "to_header"):
        return cookies.to_header()
    if isinstance(cookies, Mapping):
        return "; ".join(f"{k}={v}" for k, v in cookies.items())
    raise TypeError(
        f"cookies must be a Mapping or expose to_header(); got "
        f"{type(cookies).__name__}."
    )


class Session(Singleton, ABC):
    """HTTP session base — singleton-keyed by its post-init ``__dict__``.

    Inherits the standard :class:`Singleton` plumbing:

    - same-config constructor calls collapse to one process-lifetime
      instance (``_SINGLETON_TTL = None``), so connection pool,
      cookie jar, and per-host state survive across every call site
      that re-spells the same configuration;
    - the singleton key is built by running ``__init__`` on a probe
      and projecting the resulting ``self.__dict__`` minus the
      attributes named in :attr:`_TRANSIENT_STATE_ATTRS` /
      :attr:`_IDENTITY_BOOKKEEPING_ATTRS`. Every attribute the
      subclass writes during init participates in identity, so a
      subclass that adds new constructor knobs (catalog name, auth
      handler, cache mode, …) gets them in the key automatically —
      no parallel ``_singleton_key`` listing to keep in sync;
    - the only way to *exclude* something from identity is to keep
      it out of ``__init__``'s normalisation output, or to add the
      attribute name to :attr:`_TRANSIENT_STATE_ATTRS` (which also
      drops it from the pickle payload). There is no other knob.

    Pickling is handled by the :class:`Singleton` base
    (``__getstate__`` filters :attr:`_TRANSIENT_STATE_ATTRS`,
    ``__setstate__`` short-circuits on a live singleton). The
    only Session-specific addition is that the receiver rebuilds
    a fresh :class:`threading.RLock` instead of carrying the
    sender's lock state.
    """

    # Process-lifetime singleton cache — every constructor call lands
    # in :attr:`_INSTANCES` keyed on the full argument tuple so two
    # callers building the same session shape share the live pool /
    # cookies / cache state.
    _SINGLETON_TTL: ClassVar[Any] = None

    # Instance attributes that don't survive pickling — excluded by
    # ``__getstate__`` and rebuilt by ``__setstate__``. Subclasses extend
    # this with their own non-picklable handles (e.g. connection pools).
    # This is also the sole exclusion list for the singleton key: a
    # subclass that wants a constructor arg out of the identity should
    # not add it to ``__init__`` in the first place; a subclass that
    # has a *derived* attribute it doesn't want shared across pickles
    # adds the attribute name here.
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({
        "_lock", "_job_pool",
    })

    # Prepared-payload / response / batch types the prepare → send
    # pipeline emits. HTTP sessions inherit these defaults; SQL
    # :class:`StatementExecutor` subclasses pin
    # :class:`PreparedStatement` / :class:`StatementResult` /
    # :class:`StatementBatch` instead so the same prepare / send
    # vocabulary covers both transports.
    _PREPARED_CLASS: ClassVar[type] = PreparedRequest
    _RESPONSE_CLASS: ClassVar[type] = Response
    _BATCH_CLASS: ClassVar[type] = ResponseBatch

    # Bookkeeping attributes that ``__init__`` / the Singleton plumbing
    # write into ``__dict__`` but that aren't part of the user-facing
    # identity. Excluded from ``__getnewargs_ex__`` alongside
    # :attr:`_TRANSIENT_STATE_ATTRS`.
    _IDENTITY_BOOKKEEPING_ATTRS: ClassVar[frozenset[str]] = frozenset({
        "_initialized", "_singleton_key_",
    })

    @classmethod
    def _identity_excluded_attrs(cls) -> frozenset[str]:
        """Attribute names omitted from ``__getnewargs_ex__`` / pickle args."""
        return cls._TRANSIENT_STATE_ATTRS | cls._IDENTITY_BOOKKEEPING_ATTRS

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        # Run the subclass's ``__init__`` on a throwaway probe and
        # project the post-init ``__dict__`` into the key, keeping
        # only the attributes whose names appear in some ``__init__``
        # along the MRO. Reusing the real init path means every
        # normalisation the constructor applies (URL parsing, header
        # coercion, pool-size clamping, schema lookup, …) lands in
        # the key for free; the parameter-name filter keeps derived
        # caches, lazy handles, and other non-identity state out by
        # construction — exactly the same set ``__getnewargs_ex__``
        # ships back, so the receiver re-derives the same key.
        kwargs.pop("singleton_ttl", None)
        probe = object.__new__(cls)
        object.__setattr__(probe, "_initialized", False)
        # ``_in_probe`` flags this object as a throwaway used purely
        # for key derivation. ``__init__`` paths that have user-visible
        # side-effects on constructor arguments (e.g. reading
        # ``auth.authorization`` on a rotating handler, opening a
        # connection pool, hitting a credentials cache) gate on this
        # flag so the probe stays observation-free — the real instance
        # init runs the side-effects exactly once.
        object.__setattr__(probe, "_in_probe", True)
        cls.__init__(probe, *args, **kwargs)
        param_names = cls._init_param_names()
        excluded = cls._identity_excluded_attrs()
        items = tuple(
            (k, _hashable_identity_value(v))
            for k, v in sorted(probe.__dict__.items())
            if k in param_names and k not in excluded
        )
        return (cls, items)

    def __init__(
        self,
        base_url: Optional[URL | str] = None,
        verify: bool = True,
        pool_maxsize: int = 10,
        headers: "Headers | Mapping[str, str] | None" = None,
        waiting: WaitingConfig = DEFAULT_WAITING_CONFIG,
        *,
        auth: Optional[Authorization] = None,
    ) -> None:
        # Singleton-cached instances are re-entered on every constructor call
        # (Python always invokes ``__init__`` after ``__new__``); skip the
        # second pass so we don't drop the live connection pool / cookies.
        if getattr(self, "_initialized", False):
            return
        if auth is not None and not isinstance(auth, Authorization):
            raise TypeError(
                f"auth must be an Authorization instance or None; got "
                f"{type(auth).__name__}."
            )
        self.base_url = URL.from_(base_url) if base_url else None
        self.verify = verify
        self.pool_maxsize = pool_maxsize if pool_maxsize and pool_maxsize > 0 else 8
        self.headers: Headers = Headers.from_(headers)
        self.waiting = waiting
        self.auth: Authorization | None = auth
        # When a session-wide auth handler is bound at construction
        # time, pre-stamp ``self.headers["Authorization"]`` so anyone
        # inspecting the session sees the current credential without
        # going through a request first. :meth:`refresh_auth` keeps
        # the session header in sync on subsequent refreshes.
        # Skipped on the singleton-key probe (see ``_singleton_key``)
        # so rotating handlers don't tick a counter twice per
        # constructor call.
        if auth is not None and not getattr(self, "_in_probe", False):
            self.headers["Authorization"] = auth.authorization
        self._lock = threading.RLock()
        self._job_pool: Optional[JobPoolExecutor] = None
        self._initialized = True

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._job_pool:
            self._job_pool.shutdown(wait=True)
            self._job_pool = None

    def __getnewargs_ex__(self):
        # Pull every constructor knob the subclass stashes on ``self``
        # back out of ``__dict__``, but only the attributes whose names
        # actually appear in some ``__init__`` along the MRO — that's
        # what the receiver's ``__new__`` knows how to feed back into
        # the probe-driven :meth:`_singleton_key`. Transient handles,
        # singleton bookkeeping, and subclass-private slots that don't
        # match a constructor parameter (test queues, lazy caches the
        # subclass stashes outside ``__init__``'s surface) stay out by
        # construction; the rest of ``__dict__`` rides into
        # :meth:`Singleton.__setstate__`'s payload instead.
        param_names = self._init_param_names()
        excluded = self._identity_excluded_attrs()
        state = {
            k: v for k, v in self.__dict__.items()
            if k in param_names and k not in excluded and k != "base_url"
        }
        return (self.base_url,), state

    @classmethod
    def _init_param_names(cls) -> frozenset[str]:
        """Union of named ``__init__`` parameters across the MRO.

        ``*args`` / ``**kwargs`` capture parameters are excluded so a
        subclass with a bare ``def __init__(self, *args, **kwargs)``
        passthrough (the test stubs do this) still inherits the
        parent's named knobs without leaking attribute slots back
        through ``__getnewargs_ex__``.
        """
        import inspect

        names: set[str] = set()
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None or init is object.__init__:
                continue
            try:
                sig = inspect.signature(init)
            except (TypeError, ValueError):
                continue
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                names.add(name)
        return frozenset(names)

    def __setstate__(self, state):
        # Defer to :meth:`Singleton.__setstate__` for the live-singleton
        # short-circuit and the transient-attr defaulting; then promote
        # ``_lock`` from ``None`` to a fresh ``RLock`` so the receiver
        # has a real lock instead of a sentinel. ``_job_pool`` stays
        # ``None`` — it's lazy-built on first :attr:`job_pool` access.
        if getattr(self, "_initialized", False):
            return
        super().__setstate__(state)
        self._lock = threading.RLock()
        self._initialized = True

    @property
    def job_pool(self) -> JobPoolExecutor:
        if self._job_pool is None:
            with self._lock:
                if self._job_pool is None:
                    self._job_pool = JobPoolExecutor(max_workers=self.pool_maxsize)
                    LOGGER.debug("Created job pool with max_workers=%s", self.pool_maxsize)
        return self._job_pool

    @property
    def x_api_key(self) -> Optional[str]:
        if self.headers:
            return self.headers.get("X-API-Key")
        return None

    @x_api_key.setter
    def x_api_key(self, value: Optional[str]) -> None:
        if self.headers is None:
            self.headers = Headers()
        if value is None:
            self.headers.pop("X-API-Key", None)
        else:
            self.headers["X-API-Key"] = value

    def _request_log_id(self, request: PreparedRequest) -> str:
        try:
            return request.xxh3_b64(url_safe=True)
        except Exception:
            return request.url.to_string()

    def _load_local_cached_response(
        self,
        request: PreparedRequest,
        cache_cfg: CacheConfig,
    ) -> Optional[Response]:
        """Resolve a single request against the URL-mirrored fast-path cache.

        One ``stat`` + Arrow IPC decode keyed off the request's
        ``public_hash`` (xxh3_64 over anonymized
        method+url+headers+body). Returns ``None`` when the file is
        missing, the row's identity tuple disagrees with the request,
        or the row falls outside the configured ``received_*`` window.
        """
        try:
            public_hash = request.public_hash
        except Exception:
            return None

        root = cache_cfg.local_cache_path(session=self)
        rel_path = _local_fast_path_relative(
            request.method, request.url, public_hash,
        )
        batch = _read_fast_path_arrow_batch(root, rel_path)
        if batch is None:
            return None

        received_from = cache_cfg.received_from
        received_to = cache_cfg.received_to
        for resp in Response.from_arrow_tabular(batch):
            ts = resp.received_at
            if received_from is not None and ts is not None and ts < received_from:
                continue
            if received_to is not None and ts is not None and ts >= received_to:
                continue
            if not cache_cfg.filter_response(resp, request=request):
                continue
            LOGGER.debug(
                "Found local %s %s under %s (fast path)",
                request.method, request.url, root,
            )
            # Stamp the origin so downstream consumers (and the next
            # arrow projection) see "this came from the local cache".
            # ``_state_token`` folds both cache flags in, so the
            # projection cache invalidates without an explicit reset.
            resp.local_cached = True
            resp.remote_cached = False
            return resp
        return None

    def _store_local_cached_response(
        self,
        response: Response,
        cache_cfg: CacheConfig,
        *,
        root: "str | None" = None,
    ) -> None:
        """Persist one response to the URL-mirrored fast-path cache.

        Builds the Arrow batch synchronously on the caller's thread
        (the response buffer is still live here) and fires the
        actual write through the job pool so the caller doesn't
        block on disk IO. The original response is persisted as-is
        — userinfo and sensitive headers stay in the row; cache
        matching on the read side keys off the ``public_*`` hash
        columns which already use the anonymize='remove' projection.

        ``root`` lets a hot-loop caller pass the resolved cache root
        once instead of re-resolving it per response.
        """
        if not response.ok:
            return
        req = response.request
        if req is None:
            return
        try:
            public_hash = req.public_hash
        except Exception:
            return

        root = root or cache_cfg.local_cache_path(session=self)
        rel_path = _local_fast_path_relative(req.method, req.url, public_hash)
        _maybe_autocompress_body_for_cache(response)
        batch = response.to_arrow_batch(parse=False)
        Job.make(
            _store_fast_path_arrow_batch,
            root, rel_path, batch,
        ).fire_and_forget()

    def _load_remote_cached_response(
        self,
        request: PreparedRequest,
        cache_cfg: CacheConfig,
        *,
        spark_session: Optional["SparkSession"] = None,
    ) -> Optional[Response]:
        if not cache_cfg.remote_cache_enabled:
            return None

        # Skip the per-request ``anonymize()`` when the match keys are
        # all ``public_*`` — the SQL clause and the response-side
        # join key both come out identical without it.
        lookup_request = (
            request
            if cache_cfg.request_by_is_public
            else request.anonymize(mode=cache_cfg.anonymize)
        )
        query = cache_cfg.make_batch_lookup_sql(
            table_name=cache_cfg.tabular.full_name(safe=True),
            requests=[lookup_request],
        )

        try:
            cache_result = cache_cfg.tabular.sql.execute(
                query,
                spark_session=spark_session,
            )
        except Exception as exc:
            if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                cache_cfg.tabular.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                cache_result = cache_cfg.tabular.sql.execute(
                    query,
                    spark_session=spark_session,
                )
            else:
                raise

        for response in Response.from_arrow_tabular(cache_result.read_arrow_batches()):
            if cache_cfg.filter_response(response, request=request):
                LOGGER.debug(
                    "Found remote %s %s in %s",
                    request.method,
                    request.url,
                    cache_cfg.tabular,
                )
                response.remote_cached = True
                response.local_cached = False
                return response

        return None

    def _store_remote_cached_response(
        self,
        response: Response,
        cache_cfg: CacheConfig,
        *,
        spark_session: Optional["SparkSession"] = None,
        mode: Optional[Mode] = None,
    ) -> None:
        if not response.ok:
            return

        # Persist the response as-is; cache lookups match on the
        # ``public_*`` hash columns which already collapse to the
        # anonymize='remove' projection, so writing the original
        # row keeps the userinfo/headers available for replay
        # without breaking deduplication.
        _maybe_autocompress_body_for_cache(response)
        batch = response.to_arrow_batch(parse=False)

        cache_cfg.tabular.insert(
            batch,
            mode=mode if mode is not None else cache_cfg.mode,
            match_by=cache_cfg.sql_match_by or None,
            wait=cache_cfg.wait,
            # Two-level prune: ``partition_key`` triggers Delta file
            # pruning (partition column on RESPONSE_SCHEMA);
            # ``public_hash`` narrows the merge's target side to the
            # exact row identities being upserted, so the MERGE join
            # can short-circuit on the int64 equality before
            # touching anything else. Both keys are int64 so the IN
            # literals stay compact.
            prune_values={
                "partition_key": batch["partition_key"],
                "public_hash":   batch["public_hash"],
            },
            spark_session=spark_session,
        )

    @classmethod
    def from_url(
        cls,
        url: URL | str,
        *,
        verify: bool = True,
        normalize: bool = True,
        waiting: WaitingConfigArg = True,
    ) -> "Session":
        parsed = URL.from_(url, normalize=normalize)

        if parsed.scheme.startswith("http"):
            from .http_ import HTTPSession

            return HTTPSession(
                base_url=parsed,
                verify=verify,
                waiting=WaitingConfig.from_(waiting) if waiting is not None else None,
            )

        raise ValueError(f"Cannot build session from scheme: {parsed.scheme!r}")

    def send(
        self,
        request: PreparedRequest,
        config: SendConfig | Mapping[str, Any] | None = None,
        *,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        cache_only: bool = False,
        spark_session: Optional["SparkSession"] = None,
        start: bool = True,
        **options,
    ) -> Response:
        """Prepare, dispatch, and (optionally) await the response.

        ``start=True`` (default) fires the wire call. ``start=False``
        builds the prepared request + response shell without crossing
        the wire — the :class:`StatementExecutor` override uses the
        same knob to return an idled :class:`StatementResult` whose
        backend submission is deferred until
        :meth:`StatementResult.start` fires. Plain HTTP sessions don't
        need an idle :class:`Response` (the network call is
        synchronous), so the base raises a clean
        ``NotImplementedError`` via :meth:`_build_idle_response`.
        """
        cfg = SendConfig.check_arg(
            config,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
            cache_only=cache_only,
            spark_session=spark_session,
            **options,
        )
        if not start:
            return self._build_idle_response(request, cfg)
        return self._send(request, cfg)

    def _build_idle_response(
        self,
        request: PreparedRequest,
        config: SendConfig,
    ) -> Response:
        """Return a not-yet-sent response shell for *request*.

        Concrete HTTP sessions don't need an idle :class:`Response`
        (the network call is synchronous), so the default raises.
        :class:`StatementExecutor` overrides this to return the
        standard ``start=False`` :class:`StatementResult` so callers
        get a uniform "build now, dispatch later" knob across HTTP
        and SQL.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.send(start=False) is not implemented; "
            "use the synchronous send path (start=True) for HTTP sessions, "
            "or override _build_idle_response on a custom subclass."
        )

    def refresh_auth(
        self,
        request: PreparedRequest,
        force: bool = True,
    ) -> "tuple[Session, bool]":
        """Resolve the auth handler and stamp the Authorization header.

        Per-request ``request.auth`` wins over the session-wide
        ``self.auth``. When a handler is bound, ``refresh_auth`` calls
        its ``refresh(force=force)`` if it exposes one — MSAL-style
        handlers use ``force`` to bypass the in-memory token cache
        and mint a fresh credential — then reads its ``authorization``
        property, writes the value to ``request.headers["Authorization"]``,
        and (when the resolved handler is the session-wide one) keeps
        ``self.headers["Authorization"]`` in sync so the session-level
        view stays current too.

        Returns ``(self, refreshed)`` where ``refreshed`` is ``True``
        when a handler ran and the header was stamped, ``False`` when
        the silent no-op branch was taken (force=False + no handler).
        Returning ``self`` lets the caller chain
        (``session.refresh_auth(req)[0].send(req)``); the bool surfaces
        whether the request now carries an Authorization header so a
        retry loop can short-circuit when nothing changed.

        When **no** handler is bound:

        - ``force=True`` (default) → raises
          :class:`~yggdrasil.exceptions.AuthRequiredError`. The caller
          explicitly asked to force-refresh credentials and there is
          nothing to refresh — failing fast catches misconfigured
          integrations at the right line instead of letting an
          un-authenticated send go to the wire.
        - ``force=False`` → returns ``(self, False)``. This is the
          steady-state path used by
          :meth:`prepare_request_before_send`: a request to a public
          endpoint shouldn't fail just because the session doesn't
          have a token to refresh.

        Two regular call sites:

        - :meth:`prepare_request_before_send` calls this with
          ``force=False`` so steady-state requests reuse the cached
          token (the handler's own ``is_expired`` / refresh-skew
          logic still mints a new one when the cache is empty or
          stale).
        - The HTTP send path calls this with the default ``force=True``
          after a 403, to mint a fresh token before the single retry —
          some vendors (Salesforce, M365, …) return 403 instead of
          401 when a previously-valid token has been silently rotated
          upstream.

        Subclasses with vendor-specific auth (HMAC signing, SigV4,
        challenge-response) override this to do whatever their API
        needs while keeping the same contract.
        """
        handler = request.auth or self.auth
        if handler is None:
            if force:
                from yggdrasil.exceptions import AuthRequiredError
                raise AuthRequiredError(
                    f"refresh_auth(force=True) requested but no Authorization "
                    f"handler is bound to the request or to {type(self).__name__}. "
                    "Bind one via Session(auth=handler), request.auth=handler, "
                    "or call refresh_auth(request, force=False) if a missing "
                    "handler should be tolerated.",
                    request=request,
                )
            return self, False
        refresh = getattr(handler, "refresh", None)
        if callable(refresh):
            try:
                refresh(force=force)
            except TypeError:
                # Handler's refresh() doesn't accept ``force`` — fall
                # back to a no-arg call so legacy handlers still work.
                refresh()
        authorization = handler.authorization
        if request.headers is None:
            request.headers = Headers()
        request.headers["Authorization"] = authorization
        # Mirror the refresh on the session-level header when the
        # session-wide handler is what we just ran — a per-request
        # override (request.auth) deliberately doesn't pollute the
        # session view.
        if handler is self.auth:
            self.headers["Authorization"] = authorization
        return self, True

    def prepare_request_before_send(self, request: PreparedRequest) -> PreparedRequest:
        """Session-wide request hook fired once per outbound request.

        Default returns *request* unchanged. Subclasses override to inject
        session-level concerns — auth, signing, correlation IDs, mandatory
        headers — that should apply to every request leaving this session.
        Runs in :meth:`_send` just before :meth:`_local_send`, so cache hits
        bypass it. Travels with the session into Spark workers via
        ``__getstate__`` / ``__setstate__``.
        """
        request.sent_at = dt.datetime.now(dt.timezone.utc)
        if self.headers:
            if request.headers is None:
                request.headers = {}
            request.headers.update(self.headers)
        # Steady-state requests reuse the handler's cached token; the
        # 403 retry path in HTTPSession._local_send re-calls
        # ``refresh_auth`` with ``force=True`` to bypass that cache
        # when the upstream rotated the credential.
        self.refresh_auth(request, force=False)
        return request

    def prepare_response_after_received(self, response: Response) -> Response:
        """Session-wide response hook fired once per completed network send.

        Default returns *response* unchanged. Subclasses override to log,
        redact, enrich, or wrap responses returned from the wire. Runs in
        :meth:`_send` after :meth:`_local_send` and before cache writeback,
        so the persisted response reflects any post-processing. Cache hits
        bypass it. Travels with the session into Spark workers via
        ``__getstate__`` / ``__setstate__``.
        """
        return response

    def _send(
        self,
        request: PreparedRequest,
        config: SendConfig,
    ) -> Response:
        """Core send pipeline: local cache → remote cache → network → writeback.

        Assumes `config` is already a fully-resolved `SendConfig` (no kwargs
        merging, no `check_arg`). Intended to be called by `send`, `_send_many`,
        and any other path that has already built its effective config.
        """
        remote_cfg = config.remote_cache
        local_cfg = config.local_cache

        # Per-request configs take precedence over the session-level ones.
        effective_local_cfg = request.local_cache_config or local_cfg
        effective_remote_cfg = request.remote_cache_config or remote_cfg

        # --- 1. Check local cache first (fast, disk-based) ---
        # UPSERT mode skips the lookup outright — the fresh fetch
        # below will overwrite the on-disk entry on its way through
        # ``_store_fast_path_arrow_batch``'s atomic ``os.replace``.
        local_cache_root: "str | None" = None
        if effective_local_cfg.local_cache_enabled:
            local_cache_root = effective_local_cfg.local_cache_path(session=self)
            if effective_local_cfg.mode != Mode.UPSERT:
                local_response = self._load_local_cached_response(
                    request, effective_local_cfg
                )
                if local_response is not None:
                    if config.raise_error:
                        local_response.raise_for_status()
                    return local_response

        # --- 2. Check remote cache (slower, SQL-based) ---
        # Skip when the effective config demands a forced refresh (UPSERT).
        if (
            effective_remote_cfg.remote_cache_enabled
            and effective_remote_cfg.mode != Mode.UPSERT
        ):
            remote_response = self._load_remote_cached_response(
                request,
                effective_remote_cfg,
                spark_session=config.spark_session,
            )
            if remote_response is not None:
                # Backfill local cache with the remote hit
                if local_cache_root is not None:
                    self._store_local_cached_response(
                        remote_response,
                        effective_local_cfg,
                        root=local_cache_root,
                    )
                if config.raise_error:
                    remote_response.raise_for_status()
                return remote_response

        # --- 3. No cache hit — perform actual request ---
        # ``cache_only`` callers opted out of the network fallback. The
        # cache lookups above already ran; reaching here means both
        # missed, so raise instead of crossing the wire.
        if config.cache_only:
            raise LookupError(
                f"cache_only=True but no cached response for {request.method} "
                f"{request.url} (local_cache_enabled="
                f"{effective_local_cfg.local_cache_enabled}, "
                f"remote_cache_enabled="
                f"{effective_remote_cfg.remote_cache_enabled})."
            )

        request = self.prepare_request_before_send(request)
        LOGGER.debug("Sending %s %s", request.method, request.url)
        response = self._local_send(request, config=config)
        response = self.prepare_response_after_received(response)
        LOGGER.info("Sent %s %s", request.method, request.url)

        if local_cache_root is not None:
            self._store_local_cached_response(
                response,
                effective_local_cfg,
                root=local_cache_root,
            )

        if effective_remote_cfg.remote_cache_enabled:
            # Pass the effective config so that its mode (UPSERT or APPEND)
            # is used directly by _store_remote_cached_response.
            self._store_remote_cached_response(
                response,
                effective_remote_cfg,
                spark_session=config.spark_session,
            )

        if config.raise_error:
            response.raise_for_status()

        return response

    @abstractmethod
    def _local_send(
        self,
        request: PreparedRequest,
        config: SendConfig,
    ) -> Response:
        raise NotImplementedError

    def send_many(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig | SendConfig | Mapping[str, Any] | None = None,
        *,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: bool | None = None,
        stream: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        cache_only: bool = False,
        batch_size: int | None = None,
        ordered: bool = False,
        max_in_flight: int | None = None,
        max_batch_ttl: float | None = None,
        spark_session: Optional["SparkSession"] = None,
        **options,
    ) -> Iterator[Response]:
        """Stream responses one at a time, in both Python and Spark modes.

        Spark-backed buckets are drained via the holder's
        :meth:`Tabular.read_records`, which for :class:`Dataset`
        uses ``df.toLocalIterator()`` — rows stream from the executors
        one at a time, so the driver memory footprint stays bounded
        even for large network-fetch batches. Callers that want a
        :class:`SparkDataFrame` (or the per-bucket origin breakdown)
        should consume :meth:`send_many_batches` and call
        ``ResponseBatch.to_dataframe()`` themselves.

        ``max_batch_ttl`` (default :data:`DEFAULT_MAX_BATCH_TTL`,
        300 s) caps how long the batcher will wait for ``requests`` to
        produce a full chunk before flushing what it has — bounds tail
        latency when the upstream iterator is slow. ``None`` disables
        the time cap; the batch only closes when ``batch_size`` is
        reached or the iterator is exhausted.
        """
        cfg = SendManyConfig.check_arg(
            config,
            wait=wait,
            raise_error=raise_error,
            normalize=normalize,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
            cache_only=cache_only,
            batch_size=batch_size,
            ordered=ordered,
            max_in_flight=max_in_flight,
            max_batch_ttl=max_batch_ttl,
            spark_session=spark_session,
            **options,
        )
        return self._send_many(requests, config=cfg)

    # ------------------------------------------------------------------ #
    # send_many — staged pipeline                                         #
    #                                                                    #
    # The flow per batch is:                                             #
    #   1. Local cache: yield hits, evict UPSERT entries, collect misses #
    #   2. Remote cache: group misses by effective table, run one SQL    #
    #      lookup per table, yield hits, collect misses                  #
    #   3. Network: fan out misses through the job pool                  #
    #   4. Bulk remote writeback: group successful responses by          #
    #      (table, mode, match_by, wait, anonymize) so per-request       #
    #      cache configs are honoured exactly                            #
    # ------------------------------------------------------------------ #

    def _effective_local_cfg(
        self,
        request: PreparedRequest,
        session_cfg: CacheConfig,
    ) -> CacheConfig:
        # Prebuild the per-request override the same way
        # :meth:`_send_many_batches` prebuilds the session-level
        # config — so the downstream code can reach ``eff.path``
        # uniformly without a ``local_cache_path(session=...)``
        # dance. ``prebuild`` is idempotent and skips remote-only /
        # disabled configs.
        cfg = request.local_cache_config or session_cfg
        return cfg.prebuild(session=self)

    def _effective_remote_cfg(
        self,
        request: PreparedRequest,
        session_cfg: CacheConfig,
    ) -> CacheConfig:
        return request.remote_cache_config or session_cfg

    @staticmethod
    def _remote_write_group_key(cfg: CacheConfig) -> tuple:
        """Identity used to group responses for a single bulk remote insert.

        Two responses can share an insert iff every config dimension that
        affects the write call is identical: target table, mode, match-by
        columns, wait flag, and anonymize mode. Without all five, distinct
        per-request configs get silently collapsed onto whichever config
        landed in the bucket first.
        """
        return (
            cfg.tabular.full_name(safe=True),
            cfg.mode,
            tuple(cfg.match_by) if cfg.match_by else (),
            bool(cfg.wait),
            cfg.anonymize,
        )

    def _split_local_cache(
        self,
        batch: list[PreparedRequest],
        session_local_cfg: CacheConfig,
        *,
        key_to_local_cfg: Optional[Mapping[int, CacheConfig]] = None,
    ) -> tuple[dict[str, list[Response]], list[PreparedRequest]]:
        """Stage 1: scan the local cache.

        Returns ``(hits_by_path, misses)``. UPSERT entries are evicted
        on the way through so the eventual fresh response can be
        written in their place. Each request is evaluated against its
        own effective local cache config (per-request override or
        session-level fallback) via the per-``public_hash`` fast-path
        file — one ``stat`` + IPC decode per request, no folder walk.

        Hits are grouped by the effective config's resolved cache root
        — :attr:`CacheConfig.path` after :meth:`prebuild` filled in
        the default ``~/.yggdrasil/cache/response`` suffix — so the
        per-config split survives all the way to
        :class:`ResponseBatch.local_hits`.

        ``key_to_local_cfg`` is the per-request effective-config map
        :meth:`_send_many_batches` builds once at the top of a chunk.
        When passed, the per-request resolution short-circuits to a
        dict lookup; otherwise we resolve on the fly (used by callers
        that don't precompute, e.g. unit tests).
        """
        hits: dict[str, list[Response]] = {}
        misses: list[PreparedRequest] = []

        if not session_local_cfg.local_cache_enabled and not any(
            r.local_cache_config for r in batch
        ):
            # Cheap path: no local cache anywhere in this batch.
            return hits, list(batch)

        for req in batch:
            eff = (
                key_to_local_cfg.get(req.public_url_hash)
                if key_to_local_cfg is not None
                else None
            ) or self._effective_local_cfg(req, session_local_cfg)
            if not eff.local_cache_enabled or eff.mode == Mode.UPSERT:
                misses.append(req)
                continue
            cached = self._load_local_cached_response(req, eff)
            if cached is None:
                misses.append(req)
                continue
            hits.setdefault(eff.local_cache_path(session=self), []).append(cached)

        if hits:
            total = sum(len(v) for v in hits.values())
            LOGGER.debug(
                "Batch local cache: %s/%s hits across %s path(s)",
                total, len(batch), len(hits),
            )
        return hits, misses

    def _split_remote_cache(
        self,
        requests: list[PreparedRequest],
        session_remote_cfg: CacheConfig,
        *,
        spark_session: Optional["SparkSession"] = None,
        key_to_remote_cfg: Optional[Mapping[int, CacheConfig]] = None,
    ) -> tuple[dict[str, list[Response]], list[PreparedRequest]]:
        """Stage 2: scan the remote cache.

        UPSERT requests bypass the read entirely (always misses, refetch).
        Non-UPSERT requests are grouped by their effective cache table so we
        execute exactly one batch SQL lookup per table.

        Returns hits as a per-table mapping keyed by
        ``CacheConfig.table.full_name(safe=True)`` so the downstream
        :class:`ResponseBatch` can preserve which table answered which
        subset of the batch — collapsing them back into one bucket
        would lose that provenance.

        ``key_to_remote_cfg`` is the per-request effective-config map
        precomputed by :meth:`_send_many_batches` so we don't pay a
        per-request override resolution twice (snapshot + this stage);
        absent, we resolve on the fly.
        """
        # Single-pass classify: UPSERT → miss, APPEND with remote cache
        # → bucket by table, anything else → miss. Replaces a previous
        # O(N^2) shape that built ``upsert_reqs`` and then re-walked
        # ``requests`` with ``if req in upsert_reqs`` per element.
        hits: dict[str, list[Response]] = {}
        misses: list[PreparedRequest] = []
        table_to_cfg: dict[str, CacheConfig] = {}
        table_to_reqs: dict[str, list[PreparedRequest]] = {}

        for req in requests:
            t_cfg = (
                key_to_remote_cfg.get(req.public_url_hash)
                if key_to_remote_cfg is not None
                else None
            ) or self._effective_remote_cfg(req, session_remote_cfg)
            if t_cfg.mode == Mode.UPSERT:
                misses.append(req)
                continue
            if not t_cfg.remote_cache_enabled or t_cfg.mode != Mode.APPEND:
                misses.append(req)
                continue
            tkey = t_cfg.tabular.full_name(safe=True)
            bucket = table_to_reqs.get(tkey)
            if bucket is None:
                table_to_cfg[tkey] = t_cfg
                table_to_reqs[tkey] = bucket = []
            bucket.append(req)

        total_hits = 0
        for tkey, t_reqs in table_to_reqs.items():
            t_cfg = table_to_cfg[tkey]
            t_hits, t_misses = self._lookup_remote_table(
                t_cfg, t_reqs, spark_session=spark_session,
            )
            if t_hits:
                hits[tkey] = t_hits
                total_hits += len(t_hits)
            misses.extend(t_misses)

        if total_hits:
            LOGGER.debug(
                "Batch remote cache: %s/%s hits across %s table(s)",
                total_hits, len(requests), len(table_to_cfg),
            )
        return hits, misses

    def _lookup_remote_table(
        self,
        cfg: CacheConfig,
        requests: list[PreparedRequest],
        *,
        spark_session: Optional["SparkSession"] = None,
    ) -> tuple[list[Response], list[PreparedRequest]]:
        """Execute one batch SQL lookup against a single cache table.

        When ``cfg.request_by_is_public`` holds, the per-request
        ``anonymize()`` pass is skipped — ``public_*`` match keys hash
        to the same value on the original and the anonymized request,
        so the lookup tuple and SQL clause both come out identical
        without paying for one URL parse + header normalize per
        request.
        """
        if cfg.request_by_is_public:
            lookup_batch: list[PreparedRequest] = list(requests)
        else:
            lookup_batch = [r.anonymize(mode=cfg.anonymize) for r in requests]

        query = cfg.make_batch_lookup_sql(
            table_name=cfg.tabular.full_name(safe=True),
            requests=lookup_batch,
        )
        try:
            cache_result = cfg.tabular.sql.execute(query, spark_session=spark_session)
        except Exception as exc:
            if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                cfg.tabular.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                cache_result = cfg.tabular.sql.execute(query, spark_session=spark_session)
            else:
                raise

        result_map: dict[tuple, Response] = {}
        for response in Response.from_arrow_tabular(cache_result.read_arrow_batches()):
            result_map[cfg.request_tuple(response.request)] = response

        hits: list[Response] = []
        misses: list[PreparedRequest] = []
        for req, lookup in zip(requests, lookup_batch):
            candidate = result_map.get(cfg.request_tuple(lookup))
            if candidate is not None and cfg.filter_response(candidate, request=req):
                candidate.remote_cached = True
                candidate.local_cached = False
                hits.append(candidate)
            else:
                misses.append(req)
        return hits, misses

    # Per-SparkSession cache of the empty :class:`SparkDataFrame` keyed
    # to :data:`RESPONSE_SCHEMA`. The bare ``createDataFrame([],
    # schema=...)`` path costs ~30 ms per call (most of which is JVM
    # round-trip overhead on the empty payload), and the cache layer
    # hits it once per bucket pass + once per empty-table branch in
    # :meth:`_lookup_remote_table_spark`. Caching by ``id(spark)`` keeps
    # this safe across multiple concurrent SparkSessions in the same
    # process; ``WeakValueDictionary`` would be cleaner but Spark
    # DataFrames hold a strong reference to their session so the
    # entries don't outlive the session anyway.
    _EMPTY_SPARK_FRAMES: "ClassVar[dict[int, SparkDataFrame]]" = {}

    @classmethod
    def _cached_empty_spark_frame(
        cls, spark: "SparkSession",
    ) -> "SparkDataFrame":
        """Return a process-cached empty :class:`SparkDataFrame` with
        :data:`RESPONSE_SCHEMA`.

        One DataFrame per :class:`SparkSession` — same identity for
        repeat callers, so downstream ``unionByName`` paths can keep
        their plan cached. The cache key is ``id(spark)``: each
        SparkSession owns its own DataFrames anyway, and the entry
        is dropped when the session is.
        """
        key = id(spark)
        cached = cls._EMPTY_SPARK_FRAMES.get(key)
        if cached is not None:
            return cached
        df = spark.createDataFrame(
            [], schema=RESPONSE_SCHEMA.to_spark_schema(),
        )
        cls._EMPTY_SPARK_FRAMES[key] = df
        return df

    @classmethod
    def _responses_to_spark(
        cls,
        responses: list[Response],
        spark: "SparkSession",
    ) -> "SparkDataFrame":
        """Lift a list of :class:`Response` to a schema-bearing Spark frame.

        Used on the spark path to keep every bucket frame-resident.
        Empty input yields a cached empty DataFrame keyed to
        :data:`RESPONSE_SCHEMA` (one per :class:`SparkSession`) so the
        ``createDataFrame([], schema=...)`` JVM round-trip — about 30 ms
        on a warm cluster — only pays once per session instead of per
        bucket pass.
        """
        if not responses:
            return cls._cached_empty_spark_frame(spark)
        # Single bulk struct-array build — same ~30x speedup over the
        # per-row ``[r.to_arrow_batch(...) for r]`` loop documented on
        # :meth:`Response.values_to_arrow_batch`.
        table = pa.Table.from_batches(
            [Response.values_to_arrow_batch(responses)]
        )
        return spark.createDataFrame(table)

    @staticmethod
    def _flatten_remote_hits(
        remote_hits_by_table: (
            "dict[str, list[Response]] | dict[str, SparkDataFrame]"
        ),
    ) -> "list[Response] | SparkDataFrame | None":
        """Collapse a per-table remote-hits dict into one bucket.

        Empty dict → ``None``. Python lists chain into a single list
        (or ``None`` when every list is empty). Spark frames union via
        ``unionByName(allowMissingColumns=True)`` so the response
        schema is enforced across tables; a missing column would mean
        a real schema drift, and silent column-fill is worse than a
        loud failure.
        """
        if not remote_hits_by_table:
            return None
        values = list(remote_hits_by_table.values())
        first = values[0]
        if isinstance(first, list):
            flat: list[Response] = [
                r for v in remote_hits_by_table.values() for r in v  # type: ignore[union-attr]
            ]
            return flat or None
        result = first
        for part in values[1:]:
            result = result.unionByName(part, allowMissingColumns=True)
        return result

    def _split_remote_cache_spark(
        self,
        requests: list[PreparedRequest],
        session_remote_cfg: CacheConfig,
        *,
        key_to_remote_cfg: Optional[Mapping[int, CacheConfig]] = None,
        spark: "SparkSession",
    ) -> tuple[dict[str, "SparkDataFrame"], list[PreparedRequest]]:
        """Spark variant of :meth:`_split_remote_cache`.

        Returns ``(hits_by_table, misses)`` — hits stay as Spark
        DataFrames keyed by ``CacheConfig.table.full_name(safe=True)``
        so the caller can hand them straight to :class:`ResponseBatch`
        without ever materialising rows on the driver and without
        losing the per-table provenance to a premature
        ``unionByName``. Misses still come back as a Python list
        because the driver needs concrete request objects to scatter
        through stage 3.
        """
        # Single-pass classify mirrors :meth:`_split_remote_cache`.
        misses: list[PreparedRequest] = []
        table_to_cfg: dict[str, CacheConfig] = {}
        table_to_reqs: dict[str, list[PreparedRequest]] = {}
        for req in requests:
            t_cfg = (
                key_to_remote_cfg.get(req.public_url_hash)
                if key_to_remote_cfg is not None
                else None
            ) or self._effective_remote_cfg(req, session_remote_cfg)
            if t_cfg.mode == Mode.UPSERT:
                misses.append(req)
                continue
            if not t_cfg.remote_cache_enabled or t_cfg.mode != Mode.APPEND:
                misses.append(req)
                continue
            tkey = t_cfg.tabular.full_name(safe=True)
            bucket = table_to_reqs.get(tkey)
            if bucket is None:
                table_to_cfg[tkey] = t_cfg
                table_to_reqs[tkey] = bucket = []
            bucket.append(req)

        hits_by_table: dict[str, "SparkDataFrame"] = {}
        for tkey, t_reqs in table_to_reqs.items():
            t_cfg = table_to_cfg[tkey]
            t_hits_df, t_misses = self._lookup_remote_table_spark(
                t_cfg, t_reqs, spark=spark,
            )
            if t_hits_df is not None:
                hits_by_table[tkey] = t_hits_df
            misses.extend(t_misses)

        if any(table_to_reqs.values()):
            LOGGER.debug(
                "Batch remote cache (spark): scanned %s table(s) for %s request(s)",
                len(table_to_cfg), len(requests),
            )
        return hits_by_table, misses

    def _lookup_remote_table_spark(
        self,
        cfg: CacheConfig,
        requests: list[PreparedRequest],
        *,
        spark: "SparkSession",
    ) -> tuple[Optional["SparkDataFrame"], list[PreparedRequest]]:
        """Spark variant of :meth:`_lookup_remote_table`.

        Runs the same batch lookup SQL, but keeps the result as a Spark
        DataFrame instead of materialising :class:`Response` objects on
        the driver. Misses are computed by collecting the distinct
        ``request_by`` key tuples back to the driver — bounded by the
        number of cached rows that match this batch, not by total cache
        size — and diffing against the input requests.

        :meth:`CacheConfig.filter_response`'s per-row branch is skipped
        on the spark path: ``received_from`` / ``received_to`` are
        already encoded in :meth:`CacheConfig.make_batch_lookup_sql`'s
        ``WHERE`` clause, and the request-key check is what the
        ``request_tuple`` diff already enforces.
        """
        if cfg.request_by_is_public:
            lookup_batch: list[PreparedRequest] = list(requests)
        else:
            lookup_batch = [r.anonymize(mode=cfg.anonymize) for r in requests]
        query = cfg.make_batch_lookup_sql(
            table_name=cfg.tabular.full_name(safe=True),
            requests=lookup_batch,
        )
        try:
            cache_result = cfg.tabular.sql.execute(query, spark_session=spark)
        except Exception as exc:
            if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                cfg.tabular.create(RESPONSE_ARROW_SCHEMA, if_not_exists=True)
                cache_result = cfg.tabular.sql.execute(query, spark_session=spark)
            else:
                raise

        hits_df = cache_result.read_spark_frame()
        # Stamp the origin flags on the read side. Stored values may be
        # stale (``mirror_local_to_remote`` pushes local hits into the
        # remote table with ``local_cached=True``) — overwrite both so
        # the downstream consumer always sees the layer that answered.
        from pyspark.sql import functions as F
        hits_df = (
            hits_df
            .withColumn("local_cached", F.lit(False))
            .withColumn("remote_cached", F.lit(True))
        )

        key_cols = list(cfg.request_by or [])
        if not key_cols:
            # No request-key columns means the SQL can't disambiguate
            # rows per request; mirror the Python path's behaviour by
            # treating every input request as a hit when any row came
            # back, otherwise everything is a miss. Pin the snapshot
            # (see the keyed branch below for the rationale) so the
            # caller's later ``.count()`` doesn't re-read the cache
            # table after stage 4 inserted the freshly-fetched misses.
            try:
                any_row = hits_df.head(1)
            except Exception:
                any_row = None
            if any_row:
                return self._pin_spark_snapshot(hits_df), []
            return None, list(requests)

        # Request-side ``request_by`` keys (``public_url_hash``,
        # ``method`` …) are stored on the response cache table under
        # the flattened ``request_<col>`` form, so the bare keys can't
        # be referenced as Spark column names — select via
        # ``_request_column_sql_name`` and read rows back through the
        # same prefixed names.
        sql_cols = [_request_column_sql_name(c) for c in key_cols]
        matched_rows = hits_df.select(*sql_cols).distinct().toLocalIterator()
        matched: set[tuple] = {
            tuple(row[c] for c in sql_cols) for row in matched_rows
        }

        misses: list[PreparedRequest] = []
        for req, lookup in zip(requests, lookup_batch):
            if cfg.request_tuple(lookup) not in matched:
                misses.append(req)
        if not matched:
            # Cold-cache short-circuit: returning a still-bound
            # SparkDataFrame for the empty match would let any later
            # action — e.g. :attr:`ResponseBatch.counts` — re-execute
            # the SELECT after stage 4 has inserted ``misses`` into the
            # same cache table, double-counting those rows as remote
            # hits. The bucket really is empty; let the consumer drop it.
            return None, misses
        # Pin the matched-row snapshot. The lazy ``hits_df`` reads the
        # cache table, which stage 4 mutates in place — without an
        # eagerly-materialised cache snapshot a later ``.count()`` would
        # re-issue the SELECT and pick up the freshly-inserted miss rows.
        return self._pin_spark_snapshot(hits_df), misses

    @staticmethod
    def _pin_spark_snapshot(df: "SparkDataFrame") -> "SparkDataFrame":
        """Cache ``df`` and force one action so the snapshot is stable.

        Spark's ``DataFrame.cache`` is lazy — the partitions only
        materialise on the first action against the frame. When the
        downstream caller's first action runs after a sibling write to
        the same source table, the cached snapshot ends up containing
        rows that landed *after* the logical read. Forcing a single
        ``.count()`` here pins the partitions to the pre-mutation state.
        ``.cache`` itself can fail on Spark Connect logical plans the
        backend won't materialise — log and return the original frame
        so a best-effort pin doesn't crash the caller.
        """
        try:
            df = df.cache()
            df.count()
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "Failed to pin Spark snapshot for %r; downstream counts "
                "may re-execute the plan",
                df, exc_info=True,
            )
        return df

    def _fetch_misses(
        self,
        misses: list[PreparedRequest],
        config: SendManyConfig,
    ) -> Iterator[Response]:
        """Stage 3: send misses through the job pool.

        Returns the raw `Response` stream — caller decides what to do with
        ok/error responses (yield them, persist them, raise).
        """
        # Both local and remote writes are mutualised — workers only run
        # the network call. Local writes are bulk-upserted by
        # `_backfill_local_cache` and remote writes by `_persist_remote`,
        # so per-request cache configs are stripped from the copies handed
        # to the workers to avoid the per-response fan-out.
        miss_send_config = config.to_send_config(
            with_remote_cache=False,
            with_local_cache=False,
            with_spark=False,
            raise_error=False,
        )

        pool = self.job_pool
        LOGGER.debug(
            "Fetching %d send_many miss(es) through job pool "
            "(max_in_flight=%d, ordered=%s)",
            len(misses),
            config.max_in_flight or pool.max_workers,
            config.ordered,
        )
        for result in pool.as_completed(
            (
                Job.make(
                    self._send,
                    r.copy(remote_cache_config=None, local_cache_config=None),
                    miss_send_config,
                )
                for r in misses
            ),
            ordered=config.ordered,
            max_in_flight=config.max_in_flight or self.pool_maxsize,
            cancel_on_exit=False,
            shutdown_on_exit=False,
            raise_error=True,
        ):
            yield result.result

    @staticmethod
    def _enable_fair_spark_scheduler(spark: "SparkSession") -> None:
        """Best-effort switch the Spark session into FAIR scheduling.

        Stage 4 in Spark mode submits one Spark job per remote-cache
        insert; concurrent inserts (e.g. when several requests carry
        per-table remote configs, or when the local-cache writeback
        runs alongside the remote insert) all want share-the-cluster
        semantics rather than the default FIFO queue, where a slow
        insert blocks every other job behind it.

        ``spark.scheduler.mode`` is normally a SparkContext-level
        setting; some Spark builds will accept the runtime
        ``conf.set`` and apply it to subsequent jobs, others reject
        it. Either way this is non-fatal — we don't want a managed
        cluster's scheduler policy to break the request flow.
        """
        if spark is None:
            return
        try:
            spark.conf.set("spark.scheduler.mode", "FAIR")
        except Exception as exc:
            LOGGER.debug(
                "Could not switch Spark scheduler to FAIR: %s", exc,
            )

    @staticmethod
    def _run_concurrently(
        tasks: "Sequence[Callable[[], Any]]",
        *,
        max_workers: "int | None" = None,
        thread_name_prefix: str = "ygg-session",
    ) -> None:
        """Run *tasks* in parallel, re-raising the first failure.

        Used by the cache-write path (``_backfill_local_cache``,
        ``_persist_remote``, stage 4) to fan out independent inserts
        across threads so a batch that targets several remote tables
        or local cache roots doesn't pay for a head-to-tail
        serialization. ``ThreadPoolExecutor`` is used unconditionally
        because the work is I/O-bound (SQL statements, disk writes,
        Spark job submissions) — GIL contention is not the
        bottleneck.

        - 0 tasks → no-op.
        - 1 task  → run inline (skip pool / thread-spawn overhead).
        - N tasks → spawn ``min(N, max_workers or os.cpu_count())``
          workers, wait for all, then propagate the first captured
          exception so the caller sees the same failure shape it
          would have under the old sequential loop.
        """
        if not tasks:
            return
        if len(tasks) == 1:
            tasks[0]()
            return

        workers = min(len(tasks), max_workers or (os.cpu_count() or 4))
        first_exc: "BaseException | None" = None
        with ThreadPoolExecutor(
            max_workers=max(workers, 1),
            thread_name_prefix=thread_name_prefix,
        ) as pool:
            futures = [pool.submit(task) for task in tasks]
            for fut in futures:
                try:
                    fut.result()
                except BaseException as exc:
                    if first_exc is None:
                        first_exc = exc
        if first_exc is not None:
            raise first_exc

    def _backfill_local_cache(
        self,
        responses: list[Response],
        key_to_local_cfg: Mapping[int, CacheConfig],
        session_local_cfg: CacheConfig,
    ) -> None:
        """Write remote-cache hits back to the local fast-path cache.

        Each response is stored against its originating request's
        effective local config (looked up by ``public_url_hash``) —
        using the session-level config for every response would be
        wrong whenever a request carries a custom per-request local
        cache. Keying by ``public_url_hash`` (already cached on
        :class:`PreparedRequest`) instead of the anonymized URL
        string costs one cached attribute read per response in
        place of a full ``request.anonymize(mode="remove")`` rebuild.

        Each response becomes one fire-and-forget fast-path write
        keyed by ``public_hash``; concurrent writes against distinct
        ``rel_path``s never contend, so no per-root serialization is
        needed.

        After the writes are queued, the per-root cleanup walker
        runs once to unlink stale ``.arrow`` entries — throttled by
        the in-tree sentinel so a hot batch only walks the tree once
        per cleanup window.
        """
        cleanup_roots: dict[str, CacheConfig] = {}
        for response in responses:
            req = response.request
            cfg_key = req.public_url_hash if req is not None else None
            eff = key_to_local_cfg.get(cfg_key) if cfg_key is not None else None
            if eff is None:
                eff = session_local_cfg
            if not eff.local_cache_enabled:
                continue
            root = eff.local_cache_path(session=self)
            self._store_local_cached_response(response, eff, root=root)
            cleanup_roots.setdefault(root, eff)

        for root, eff in cleanup_roots.items():
            ttl = eff.cleanup_ttl
            if ttl is None:
                continue
            ttl_seconds = ttl.total_seconds()
            if ttl_seconds <= 0:
                continue
            if eff.wait:
                _cleanup_local_fast_path(root, ttl_seconds=ttl_seconds)
            else:
                Job.make(
                    _cleanup_local_fast_path,
                    root, ttl_seconds=ttl_seconds,
                ).fire_and_forget()

    def _persist_remote(
        self,
        responses: list[Response],
        key_to_remote_cfg: Mapping[int, CacheConfig],
        session_remote_cfg: CacheConfig,
    ) -> None:
        """Stage 4: bulk-insert successful responses into the remote cache.

        Responses are bucketed by the full write-group key
        (table, mode, match_by, wait, anonymize) so that distinct per-request
        configs targeting the same table never get collapsed onto a single
        insert with the wrong parameters. Distinct write groups (i.e.
        distinct remote tables / modes) run their inserts concurrently
        — the underlying SQL clients are thread-safe per-statement, so
        a batch fanning out to several remote tables doesn't have to
        serialize the network round trips head-to-tail.
        """
        groups: dict[tuple, tuple[CacheConfig, list[Response]]] = {}
        for response in responses:
            # Per-request lookup keys ride on ``public_url_hash`` —
            # already cached on :class:`PreparedRequest` and stable
            # across the ``request.copy(...)`` worker scatter — so we
            # don't pay an ``anonymize()`` rebuild per response just to
            # turn the URL into a dict key. Match-by/identity all hash
            # through the ``public_*`` columns so anonymizing the row
            # before insert isn't required to keep cache lookups
            # consistent.
            req = response.request
            cfg_key = req.public_url_hash if req is not None else None
            eff = key_to_remote_cfg.get(cfg_key) if cfg_key is not None else None
            if eff is None:
                eff = session_remote_cfg
            if not eff.remote_cache_enabled:
                continue
            gkey = self._remote_write_group_key(eff)
            if gkey not in groups:
                groups[gkey] = (eff, [])
            groups[gkey][1].append(response)

        def _insert_one(
            mode: "Mode",
            cfg: "CacheConfig",
            group_responses: "list[Response]",
        ) -> None:
            LOGGER.debug(
                "%s %s response(s) in remote cache %s",
                "Upserting" if mode == Mode.UPSERT else "Persisting",
                len(group_responses),
                cfg.tabular,
            )
            # One C++ struct walk over the whole group beats N
            # per-row builds + an outer ``combine_chunks`` concat —
            # see :meth:`Response.values_to_arrow_batch`. The result
            # is a single chunked-on-construction batch wrapped in a
            # one-element table so the downstream
            # ``batches["partition_key"]`` slot lookup keeps working.
            batches = pa.Table.from_batches(
                [Response.values_to_arrow_batch(group_responses)]
            )

            cfg.tabular.insert(
                batches,
                mode=mode,
                match_by=cfg.sql_match_by or None,
                wait=cfg.wait,
                prune_values={
                    "partition_key": batches["partition_key"],
                    "public_hash":   batches["public_hash"],
                },
            )

        self._run_concurrently(
            [
                lambda m=gkey[1], c=cfg, r=group_responses: _insert_one(m, c, r)
                for gkey, (cfg, group_responses) in groups.items()
            ],
            thread_name_prefix="ygg-remote-cache-insert",
        )

    def _mirror_local_hits_to_remote(
        self,
        local_hits_by_path: Mapping[str, "list[Response]"],
        key_to_remote_cfg: Mapping[int, CacheConfig],
        session_remote_cfg: CacheConfig,
    ) -> None:
        """Bulk-upsert local-cache hits into the remote cache.

        Runs between stage 2 (remote cache lookup) and stage 3
        (network fetch) when ``CacheConfig.mirror_local_to_remote``
        is set on the active remote config. The "diff" is implicit:
        the remote MERGE handles deduplication on
        ``(partition_key, public_hash)`` so hits the remote already
        knows about become idempotent no-ops, and only genuinely new
        rows land. Coupled with stage 4's network-driven persist,
        this keeps the remote cache eventually-consistent with the
        local one without forcing a network call to repopulate.

        Activation is gated on the *remote* config — local-only
        sessions skip it entirely. Per-request remote config
        overrides are honoured the same way :meth:`_persist_remote`
        honours them: the URL-keyed map fed in by
        :meth:`_send_many_batches` is the source of truth, with the
        session-level config as the fallback.
        """
        if not local_hits_by_path:
            return

        flat: list[Response] = [
            r for hits in local_hits_by_path.values() for r in hits
        ]
        if not flat:
            return

        # Filter down to responses whose effective remote cache is
        # both enabled AND opted into the mirror — keeps a session
        # that toggles the flag for one cache from accidentally
        # syncing into another. Keyed by ``public_url_hash`` so the
        # per-response lookup is a cached attribute read instead of a
        # full ``request.anonymize(mode="remove")`` rebuild.
        keep: list[Response] = []
        for response in flat:
            req = response.request
            cfg_key = req.public_url_hash if req is not None else None
            eff = key_to_remote_cfg.get(cfg_key) if cfg_key is not None else None
            if eff is None:
                eff = session_remote_cfg
            if not eff.remote_cache_enabled:
                continue
            if not eff.mirror_local_to_remote:
                continue
            keep.append(response)

        if not keep:
            return

        LOGGER.debug(
            "Mirroring %s local-cache hit(s) to remote cache",
            len(keep),
        )
        self._persist_remote(keep, key_to_remote_cfg, session_remote_cfg)

    def _send_many(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig,
    ) -> Iterator[Response]:
        """Stream responses, flattening the per-chunk :class:`ResponseBatch`.

        Iteration order matches :class:`ResponseBatch.parts`: local hits
        first, then remote hits, then network fetches. Callers that need
        the origin breakdown should use :meth:`send_many_batches`
        instead.

        Works in both Python and Spark modes. Spark-backed buckets are
        drained via the holder's :meth:`Tabular.read_records`, which
        for :class:`Dataset` uses ``df.toLocalIterator()`` — rows
        stream from the executors one at a time, so the driver memory
        footprint stays bounded even for large network-fetch batches.
        :class:`ResponseBatch.__iter__` rejects Spark mode (it would
        force a ``df.toArrow()`` collect); going through the holders
        sidesteps that guard.
        """
        for batch in self._send_many_batches(requests, config):
            for holder in batch.parts():
                yield from Response.from_records(holder.read_records())

    def send_many_batches(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig | SendConfig | Mapping[str, Any] | None = None,
        *,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: bool | None = None,
        stream: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        cache_only: bool = False,
        batch_size: int | None = None,
        ordered: bool = False,
        max_in_flight: int | None = None,
        max_batch_ttl: float | None = None,
        spark_session: Optional["SparkSession"] = None,
        **options,
    ) -> Iterator[ResponseBatch]:
        """Yield one :class:`ResponseBatch` per processed chunk.

        Public entry point: both Python and Spark modes yield the same
        ``Iterator[ResponseBatch]`` shape, chunked the same way, so
        downstream consumers can stream partial results uniformly. Each
        yielded batch carries schema-bearing holders even when a stage
        produced no rows — the schema is preserved for empty results.

        ``max_batch_ttl`` (default :data:`DEFAULT_MAX_BATCH_TTL`,
        300 s) caps how long the batcher waits for ``requests`` to
        fill one chunk before flushing what's accumulated — keeps
        downstream stages moving when the upstream iterator is slow.
        ``None`` disables the time cap.
        """
        cfg = SendManyConfig.check_arg(
            config,
            wait=wait,
            raise_error=raise_error,
            normalize=normalize,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
            cache_only=cache_only,
            batch_size=batch_size,
            ordered=ordered,
            max_in_flight=max_in_flight,
            max_batch_ttl=max_batch_ttl,
            spark_session=spark_session,
            **options,
        )
        yield from self._send_many_batches(requests, cfg)

    def spark_send(
        self,
        request: PreparedRequest,
        config: SendConfig | Mapping[str, Any] | None = None,
        *,
        spark_session: "SparkSession",
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        cache_only: bool = False,
        **options,
    ) -> "SparkDataFrame":
        """Send one request via Spark and return a lazy ``DataFrame[Response]``.

        Thin wrapper over :meth:`spark_send_many` for the single-request
        case. The returned Spark DataFrame carries :data:`RESPONSE_SCHEMA`
        and stays unmaterialised on the driver — no executor job fires
        until the caller triggers an action (``.collect()``,
        ``.write...``, ``.count()``, …) — so callers can chain additional
        lazy Spark transforms on top before pulling rows.
        """
        return self.spark_send_many(
            iter([request]),
            config,
            spark_session=spark_session,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
            cache_only=cache_only,
            **options,
        )

    def spark_send_many(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig | SendConfig | Mapping[str, Any] | None = None,
        *,
        spark_session: "SparkSession",
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: bool | None = None,
        stream: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        cache_only: bool = False,
        batch_size: int | None = None,
        ordered: bool = False,
        max_in_flight: int | None = None,
        max_batch_ttl: float | None = None,
        **options,
    ) -> "SparkDataFrame":
        """Send many requests via Spark and return a lazy ``DataFrame[Response]``.

        Drives the same staged ``send_many`` pipeline as :meth:`send_many`
        but in Spark mode (cf. :meth:`_send_many_batches`): every bucket
        — local hits, remote hits, network responses — stays
        frame-resident on the executors, the network fetch fans out via
        ``mapInArrow``, and the per-chunk :class:`ResponseBatch` frames
        are stitched into a single union via
        ``unionByName(allowMissingColumns=True)``.
        Schema matches :data:`RESPONSE_SCHEMA`.
        The returned DataFrame is lazy — driver-side cache lookups and
        the ``mapInArrow`` plan are built eagerly, but no executor job
        fires until the caller triggers a Spark action.
        Pass an explicit ``batch_size`` larger than the request count to
        collapse the pipeline into a single ``mapInArrow`` scatter (one
        big chunk, one frame); the default chunks at
        ``min(SendManyConfig.max_batch_size, 1024)`` so an unbounded
        upstream iterator can't pin every request on the driver before
        any work scatters.
        """
        cfg = SendManyConfig.check_arg(
            config,
            wait=wait,
            raise_error=raise_error,
            normalize=normalize,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
            cache_only=cache_only,
            batch_size=batch_size,
            ordered=ordered,
            max_in_flight=max_in_flight,
            max_batch_ttl=max_batch_ttl,
            spark_session=spark_session,
            **options,
        )
        if cfg.spark_session is None or cfg.spark_session is ...:
            raise ValueError(
                "spark_send_many requires a live SparkSession — got "
                f"{cfg.spark_session!r}. Pass `spark_session=...` (or set it "
                "on the SendManyConfig) so the staged pipeline can keep its "
                "buckets frame-resident."
            )
        spark = cfg.spark_session

        # Drive the staged pipeline; ``_send_many_batches`` already
        # picks the Spark path from ``cfg.spark_session`` and yields
        # one Spark-backed :class:`ResponseBatch` per chunk. Union the
        # per-chunk frames lazily — ``unionByName`` builds a plan
        # without firing an action — so the caller gets a single
        # ``DataFrame[Response]`` to compose with.
        frames: list["SparkDataFrame"] = []
        for batch in self._send_many_batches(requests, cfg):
            frames.append(batch.to_dataframe(spark))

        if not frames:
            return self._cached_empty_spark_frame(spark)
        result = frames[0]
        for part in frames[1:]:
            result = result.unionByName(part, allowMissingColumns=True)
        return result

    def _send_many_batches(
        self,
        requests: Iterator[PreparedRequest],
        config: SendManyConfig,
    ) -> Iterator[ResponseBatch]:
        """Yield one :class:`ResponseBatch` per processed chunk.

        Single pipeline for both Python and Spark modes — the only
        differences are stage 3 (fetch misses through the local job
        pool vs. ``mapInArrow`` over executors) and stage 4 (per-row
        Arrow insert vs. lazy Spark insert). Mode is picked from
        ``config.spark_session``.

        Both modes chunk requests by ``batch_size`` and yield one
        :class:`ResponseBatch` per chunk so callers see the same
        streaming shape regardless of engine. In Spark mode each chunk
        produces its own ``mapInArrow`` job — pass a larger
        ``batch_size`` (or ``max_batch_size``) when you'd rather
        amortise scheduler overhead across a single bulk fetch. Empty
        buckets are returned as schema-bearing holders so a chunk that
        fully short-circuited on local cache still advertises the
        response schema for ``remote_hits`` / ``new_hits``.
        """
        is_spark = config.spark_session is not None and config.spark_session is not ...
        spark = config.spark_session if is_spark else None

        session_remote_cfg = config.remote_cache
        # Build the session-level local cache's :class:`Tabular` once
        # at entry so the rest of the pipeline can reach for
        # ``cfg.tabular`` symmetrically with the remote side. The
        # session-aware path (``base_url`` host/path → cache root)
        # only resolves correctly when we have ``self`` in scope, so
        # the prebuild has to happen here rather than at config-build
        # time. ``prebuild`` is idempotent and a no-op on
        # already-built / disabled / remote configs.
        session_local_cfg = config.local_cache.prebuild(session=self)

        if is_spark:
            # FAIR scheduling lets the concurrent stage-4 inserts
            # (and any upstream Spark jobs the cache flow kicks off)
            # share executor slots instead of queueing strictly FIFO
            # behind whichever job won the race. ``spark.conf.set``
            # is best-effort: some Spark builds reject runtime
            # changes to scheduler-mode and we don't want a managed
            # cluster's policy to fail the request flow.
            self._enable_fair_spark_scheduler(spark)
            # Spark mode has no driver-side thread pool to scale the
            # default against — fall back to ``max_batch_size`` (or
            # 1024) so each chunk maps to one ``mapInArrow`` scatter
            # of bounded width. Callers who want a single mega-chunk
            # (preserving the original bulk-fetch optimisation) can
            # pass an explicit ``batch_size`` larger than their
            # request count.
            batch_size = config.batch_size or config.max_batch_size or 1024
        else:
            pool = self.job_pool
            batch_size = config.batch_size or min(
                config.max_batch_size or 1024, pool.max_workers * 10
            )

        ttl = config.max_batch_ttl

        LOGGER.debug(
            "Starting send_many pipeline (mode=%s, batch_size=%d, "
            "max_in_flight=%s, ttl=%s, ordered=%s, "
            "local_cache=%s, remote_cache=%s)",
            "spark" if is_spark else "python",
            batch_size,
            config.max_in_flight,
            ttl,
            config.ordered,
            session_local_cfg.local_cache_enabled,
            session_remote_cfg.remote_cache_enabled,
        )

        chunk_index = 0
        total_local = 0
        total_remote = 0
        total_network = 0
        total_failed = 0

        def _batched(
            it: Iterator[PreparedRequest],
            n: int,
            ttl_seconds: float | None,
        ) -> Iterator[list[PreparedRequest]]:
            # When no TTL is set, fall back to the cheap islice path —
            # avoids the per-request monotonic() probe.
            iterator = iter(it)
            if ttl_seconds is None or ttl_seconds <= 0:
                while True:
                    b = list(itertools.islice(iterator, n))
                    if not b:
                        break
                    yield b
                return

            # Time-bounded path: pull one item at a time and flush
            # when either the size cap or the wall-clock deadline is
            # reached. The deadline is reset per chunk so a slow
            # upstream gets a fresh window after each flush.
            buf: list[PreparedRequest] = []
            deadline: float | None = None
            for item in iterator:
                if not buf:
                    deadline = time.monotonic() + ttl_seconds
                buf.append(item)
                if len(buf) >= n or (
                    deadline is not None and time.monotonic() >= deadline
                ):
                    yield buf
                    buf = []
                    deadline = None
            if buf:
                yield buf

        chunks = _batched(requests, batch_size, ttl)

        for chunk in chunks:
            if not chunk:
                continue

            chunk_index += 1
            LOGGER.debug(
                "Processing send_many chunk #%d (requests=%d, mode=%s)",
                chunk_index, len(chunk), "spark" if is_spark else "python",
            )

            # Snapshot per-request effective configs BEFORE stage 1 so
            # every downstream stage (split_local_cache, split_remote_cache,
            # _persist_remote, _backfill_local_cache,
            # _mirror_local_hits_to_remote) reads from the same map
            # instead of calling :meth:`_effective_local_cfg` /
            # :meth:`_effective_remote_cfg` per request per stage.
            # Keyed by ``PreparedRequest.public_url_hash`` — a cached
            # xxh3_64 of ``(method, url.anonymize('remove'))`` — so
            # the snapshot survives the ``request.copy(...)`` scatter
            # in :meth:`_fetch_misses`. Building it once up front is
            # also what kills the old O(N^2) ``if req in upsert_reqs``
            # shape in stage 2 — every request resolves exactly once.
            key_to_remote_cfg: dict[int, CacheConfig] = {}
            key_to_local_cfg: dict[int, CacheConfig] = {}
            for r in chunk:
                k = r.public_url_hash
                key_to_remote_cfg[k] = self._effective_remote_cfg(r, session_remote_cfg)
                key_to_local_cfg[k] = self._effective_local_cfg(r, session_local_cfg)

            # --- Stage 1: local cache ---
            local_hits_by_path, after_local = self._split_local_cache(
                chunk, session_local_cfg, key_to_local_cfg=key_to_local_cfg,
            )
            # Flatten across cache-folder paths into a single bucket
            # for :class:`ResponseBatch`. On the spark path, lift the
            # flat list to a Spark frame once so every bucket
            # downstream is frame-resident — matches stage 2/3 and
            # lets the caller union holders without a per-bucket type
            # switch. ``local_hits_by_path`` stays a per-path dict
            # internally because :meth:`_mirror_local_hits_to_remote`
            # walks it before stage 3.
            local_flat: list[Response] = [
                r for hits in local_hits_by_path.values() for r in hits
            ]
            local_hits: "list[Response] | SparkDataFrame | None"
            if not local_flat:
                local_hits = None
            elif is_spark:
                local_hits = self._responses_to_spark(local_flat, spark)
            else:
                local_hits = local_flat
            # Remote hits: dict keyed by ``CacheConfig.table.full_name``
            # internally so :meth:`_backfill_local_cache` knows which
            # rows came from which table; collapsed into one bucket at
            # the :class:`ResponseBatch` boundary.
            remote_hits_by_table: (
                "dict[str, list[Response]] | dict[str, SparkDataFrame]"
            ) = {}
            new_hits: "list[Response] | SparkDataFrame | None" = None

            if not after_local:
                # Even when every request is a local hit we may still
                # owe the remote a diff — call the mirror before the
                # early return so an all-cache batch keeps the remote
                # in sync without ever hitting the network.
                if not is_spark:
                    self._mirror_local_hits_to_remote(
                        local_hits_by_path,
                        key_to_remote_cfg,
                        session_remote_cfg,
                    )
                local_count = len(local_flat)
                total_local += local_count
                LOGGER.debug(
                    "Completed send_many chunk #%d (requests=%d, "
                    "local_hits=%d, remote_hits=0, network=0, failed=0) "
                    "— fully short-circuited on local cache",
                    chunk_index, len(chunk), local_count,
                )
                yield ResponseBatch(
                    local_hits=local_hits,
                    remote_hits=None,
                    new_hits=new_hits,
                    spark=spark,
                )
                continue

            # --- Stage 2: remote cache ---
            # Python path drains the StatementResult into Response
            # objects. Spark path keeps the result as a Spark DataFrame
            # (via :meth:`_split_remote_cache_spark`) so ``remote_hits``
            # never collects to the driver and downstream callers can
            # union it with stage 3's Spark output.
            if is_spark:
                remote_hits_by_table, after_remote = self._split_remote_cache_spark(
                    after_local,
                    session_remote_cfg,
                    spark=spark,
                    key_to_remote_cfg=key_to_remote_cfg,
                )
                # Local-cache backfill from a Spark frame would force a
                # toLocalIterator on the driver — skip it on the spark
                # path, matching how stage 3/4 keep network results
                # frame-resident. Drivers that want a hot local cache
                # should use the Python path explicitly.
            else:
                remote_hits_by_table, after_remote = self._split_remote_cache(
                    after_local,
                    session_remote_cfg,
                    spark_session=spark,
                    key_to_remote_cfg=key_to_remote_cfg,
                )
                # Backfill local cache with remote hits using each request's
                # effective local config — not the session-level fallback.
                # Flatten the per-table split here: the local cache
                # doesn't care which remote table sourced a row, only
                # the response-to-request mapping by URL.
                self._backfill_local_cache(
                    [
                        r
                        for table_hits in remote_hits_by_table.values()
                        for r in table_hits
                    ],
                    key_to_local_cfg,
                    session_local_cfg,
                )
                # Mirror local-cache hits to remote in bulk before any
                # network fetch fires, when the remote config opts in.
                # The remote MERGE deduplicates on the partition / public
                # hash so this is idempotent for rows the remote already
                # knows about — net effect is "diff sync" of any local
                # entries the remote was missing.
                self._mirror_local_hits_to_remote(
                    local_hits_by_path,
                    key_to_remote_cfg,
                    session_remote_cfg,
                )

            # Collapse the per-table remote split into one bucket for
            # :class:`ResponseBatch`. Python lists chain; Spark frames
            # union via ``unionByName(allowMissingColumns=True)``.
            remote_hits = self._flatten_remote_hits(remote_hits_by_table)

            if not after_remote:
                local_count = len(local_flat)
                if is_spark:
                    remote_count = -1  # Spark frame — count not materialised
                else:
                    remote_count = sum(
                        len(v) for v in remote_hits_by_table.values()  # type: ignore[union-attr]
                    )
                    total_remote += remote_count
                total_local += local_count
                LOGGER.debug(
                    "Completed send_many chunk #%d (requests=%d, "
                    "local_hits=%d, remote_hits=%s, network=0, failed=0) "
                    "— short-circuited on remote cache",
                    chunk_index, len(chunk), local_count,
                    "<spark>" if remote_count < 0 else remote_count,
                )
                yield ResponseBatch(
                    local_hits=local_hits,
                    remote_hits=remote_hits,
                    new_hits=new_hits,
                    spark=spark,
                )
                continue

            # --- Stage 3: fetch misses ---
            # ``cache_only`` skips the network fan-out entirely: drop
            # the remaining misses from the stream and emit just what
            # the caches answered. No writeback either — there are no
            # new hits to persist.
            failed: list[Response] = []
            if config.cache_only:
                LOGGER.debug(
                    "send_many chunk #%d: cache_only=True, dropping "
                    "%d miss(es) without fetching",
                    chunk_index, len(after_remote),
                )
                yield ResponseBatch(
                    local_hits=local_hits,
                    remote_hits=remote_hits,
                    new_hits=new_hits,
                    spark=spark,
                )
                continue
            if is_spark:
                # Network results stay in Spark — never collected to
                # the driver. raise_error doesn't short-circuit a
                # partial mapInArrow batch; callers filter on
                # response_status_code if they care.
                new_hits = self._spark_fetch_misses(after_remote, config, spark)
            else:
                new_list: list[Response] = []
                for response in self._fetch_misses(after_remote, config):
                    if response.ok:
                        new_list.append(response)
                    elif config.raise_error:
                        failed.append(response)
                new_hits = new_list

            # --- Stage 4: bulk remote writeback ---
            if is_spark:
                # `cfg.tabular.insert` accepts the Spark DataFrame
                # directly, so we hand off the lazy DF without
                # materialising on the driver. Per-request overrides
                # ride through ``key_to_remote_cfg`` — mirrors the
                # non-Spark ``_persist_remote`` so a chunk targeting
                # multiple remote tables fans out instead of collapsing
                # onto the session-level cfg.
                if new_hits is not None:
                    self._spark_persist_remote(
                        new_hits,
                        key_to_remote_cfg,
                        session_remote_cfg,
                        spark=spark,
                    )
            else:
                # Remote and local writebacks touch independent
                # backends — different tables / different folders —
                # so run them concurrently rather than head-to-tail.
                # Each side is itself parallel-per-table /
                # parallel-per-cache-root; this top-level fan-out
                # just lets the local disk writes overlap with the
                # remote network round trips.
                stage4: "list[Callable[[], Any]]" = []
                if new_hits:
                    stage4.append(
                        lambda: self._persist_remote(
                            new_hits,
                            key_to_remote_cfg,
                            session_remote_cfg,
                        )
                    )
                # Stage 3 workers no longer write the local cache per
                # response; bulk-upsert the new hits in one write per
                # effective cache root. ``_backfill_local_cache``
                # groups responses by their per-request local config,
                # writes the table in UPSERT mode (or APPEND when the
                # config asks for it), and triggers a pruned optimize
                # bounded by the touched partitions — same end state
                # as the prior per-worker write + post-batch optimize,
                # but with one bulk write per cache root.
                if isinstance(new_hits, list) and new_hits:
                    stage4.append(
                        lambda: self._backfill_local_cache(
                            new_hits,
                            key_to_local_cfg,
                            session_local_cfg,
                        )
                    )
                self._run_concurrently(
                    stage4, thread_name_prefix="ygg-stage4",
                )

            local_count = len(local_flat)
            total_local += local_count
            if is_spark:
                remote_count_log: "int | str" = "<spark>"
                network_count_log: "int | str" = "<spark>"
            else:
                remote_count = sum(
                    len(v) for v in remote_hits_by_table.values()  # type: ignore[union-attr]
                )
                total_remote += remote_count
                remote_count_log = remote_count
                net_count = len(new_hits) if isinstance(new_hits, list) else 0
                total_network += net_count
                network_count_log = net_count
            failed_count = len(failed)
            total_failed += failed_count
            LOGGER.debug(
                "Completed send_many chunk #%d (requests=%d, "
                "local_hits=%d, remote_hits=%s, network=%s, failed=%d)",
                chunk_index, len(chunk), local_count,
                remote_count_log, network_count_log, failed_count,
            )

            yield ResponseBatch(
                local_hits=local_hits,
                remote_hits=remote_hits,
                new_hits=new_hits,
                spark=spark,
            )

            if not is_spark and config.raise_error and failed:
                failed[-1].raise_for_status()

        if is_spark:
            LOGGER.debug(
                "Finished send_many pipeline (chunks=%d, mode=spark) "
                "— per-bucket counts deferred to Spark action",
                chunk_index,
            )
        else:
            LOGGER.debug(
                "Finished send_many pipeline (chunks=%d, local_hits=%d, "
                "remote_hits=%d, network=%d, failed=%d)",
                chunk_index, total_local, total_remote,
                total_network, total_failed,
            )

    # ------------------------------------------------------------------ #
    # Spark stage 3 / 4 helpers                                           #
    # ------------------------------------------------------------------ #

    def _spark_persist_remote(
        self,
        new_responses_df: "SparkDataFrame",
        key_to_remote_cfg: Mapping[int, CacheConfig],
        session_remote_cfg: CacheConfig,
        *,
        spark: "SparkSession",
    ) -> None:
        """Stage 4 on Spark: per-request bulk-insert into the remote cache.

        Mirrors :meth:`_persist_remote`: each ``public_url_hash`` resolves
        to its effective :class:`CacheConfig` via ``key_to_remote_cfg``
        (falling back to ``session_remote_cfg``), groups bucket by
        :meth:`_remote_write_group_key`, and each group's insert runs
        concurrently. The Spark frame is persisted once when more than
        one group fires so the network fetch behind it doesn't re-execute
        per group; single-group inserts keep the legacy zero-persist
        plan.

        Before inserting, APPEND-mode writes are de-duplicated against the
        existing remote rows via a ``left_anti`` join on the response
        ``(partition_key, public_hash)`` keys — the remote table stores
        anonymized requests (cf. ``_persist_remote``), so a row whose
        hash already lives in the cache is suppressed rather than
        re-inserted. UPSERT mode keeps its read-free fast path and
        relies on ``match_by`` to collapse duplicates server-side.
        """
        from pyspark.sql import functions as F

        # Bucket request hashes by their effective remote-cache config's
        # write group. Disabled configs drop out here so the persist
        # path never fires for them.
        groups: dict[tuple, tuple[CacheConfig, list[int]]] = {}
        for cfg_key, eff in key_to_remote_cfg.items():
            if eff is None:
                eff = session_remote_cfg
            if not eff.remote_cache_enabled:
                continue
            gkey = self._remote_write_group_key(eff)
            if gkey not in groups:
                groups[gkey] = (eff, [])
            groups[gkey][1].append(cfg_key)

        if not groups:
            return

        ok_df = new_responses_df.where(
            (F.col("status_code") >= 200)
            & (F.col("status_code") < 300)
        )

        # Persist only when more than one group will read ``ok_df`` —
        # otherwise the single insert action is the one and only
        # evaluation and we save the storage round trip. Route the
        # cache through :class:`yggdrasil.io.tabular.spark.Dataset`
        # so backends that reject ``persist`` (Databricks Connect
        # serverless raises ``[NOT_SUPPORTED_WITH_SERVERLESS] PERSIST
        # TABLE``) fall through to the un-cached frame instead of
        # crashing stage 4 — pass two then runs twice, but the
        # alternative is a hard failure on serverless compute.
        multi_group = len(groups) > 1
        ok_dataset: "Dataset | None" = None
        if multi_group:
            from yggdrasil.io.tabular.spark import Dataset

            ok_dataset = Dataset(frame=ok_df).cache()
            ok_df = ok_dataset.frame

        # When every key in the chunk's cfg map collapses onto one
        # group, the inserted frame IS the whole ``ok_df``. With any
        # disabled / split-out keys present we have to filter by
        # ``request_public_url_hash`` so dropped requests don't leak
        # into the surviving group's insert.
        covers_chunk = (
            len(groups) == 1
            and sum(len(hs) for _, (_, hs) in groups.items())
            == len(key_to_remote_cfg)
        )

        def _insert_one(cfg: CacheConfig, hashes: list[int]) -> None:
            if covers_chunk:
                df = ok_df
            else:
                df = ok_df.where(
                    F.col("request_public_url_hash").isin(hashes)
                )

            if cfg.mode != Mode.UPSERT:
                table_name = cfg.tabular.full_name(safe=True)
                try:
                    wanted_partitions = [
                        row["partition_key"]
                        for row in df.select("partition_key").distinct().collect()
                    ]
                except Exception:
                    wanted_partitions = []
                try:
                    if wanted_partitions:
                        literals = ", ".join(str(int(v)) for v in wanted_partitions)
                        existing_df = spark.sql(
                            "SELECT DISTINCT partition_key, public_hash "
                            f"FROM {table_name} WHERE partition_key IN ({literals})"
                        )
                    else:
                        existing_df = spark.sql(
                            "SELECT DISTINCT partition_key, public_hash "
                            f"FROM {table_name}"
                        )
                except Exception as exc:
                    if "TABLE_OR_VIEW_NOT_FOUND" not in str(exc):
                        raise
                    existing_df = None

                if existing_df is not None:
                    df = df.join(
                        existing_df,
                        on=["partition_key", "public_hash"],
                        how="left_anti",
                    )

            LOGGER.debug(
                "%s ok response(s) into remote cache %s (spark insert)",
                "Upserting" if cfg.mode == Mode.UPSERT else "Persisting",
                cfg.tabular,
            )
            cfg.tabular.insert(
                df,
                mode=cfg.mode,
                match_by=cfg.sql_match_by or None,
                wait=cfg.wait,
                spark_session=spark,
            )

        try:
            self._run_concurrently(
                [
                    lambda c=cfg, hs=hashes: _insert_one(c, hs)
                    for (_gkey, (cfg, hashes)) in groups.items()
                ],
                thread_name_prefix="ygg-spark-remote-cache-insert",
            )
        finally:
            if ok_dataset is not None:
                ok_dataset.unpersist()

    def _spark_fetch_misses(
        self,
        misses: list[PreparedRequest],
        config: SendManyConfig,
        spark: "SparkSession",
    ) -> "SparkDataFrame":
        """Stage 3 on Spark: scatter misses to workers via mapInArrow.

        Requests cross the wire as an Arrow table with the canonical
        :data:`~yggdrasil.io.request.REQUEST_SCHEMA` — deterministic
        columns, no pickled Python objects in the row payload.

        The driver-side :class:`Session` itself is pickled once with
        :func:`pickle.dumps` and that bytes blob travels with the
        ``mapInArrow`` closure; each partition rehydrates the same
        subclass via :func:`pickle.loads`, so user overrides (custom
        auth, header injection, request hooks) execute on the worker.
        ``Session.__getstate__`` / ``__setstate__`` strip the
        threading.RLock and JobPoolExecutor before serialising and
        re-init them on the way back in, and ``Singleton.__new__``
        collapses the loads onto the executor's per-``(cls, config)``
        cached instance so all partitions share one connection pool.
        Spark Connect / Databricks Connect callers need the user's
        subclass module on the cluster's ``sys.path`` for the
        cloudpickle round-trip to resolve — that's the price of running
        a driver-side subclass on the worker.

        Each Spark partition becomes one :meth:`Session.send_many` call,
        fanning out via the rehydrated session's thread pool. Both local
        and remote cache configs are forwarded: workers consult the same
        caches as the driver, so a request the driver fan-out missed but
        a peer worker has already cached can still short-circuit before
        hitting the network.
        """
        if not misses:
            return self._cached_empty_spark_frame(spark)

        # One C++-side struct walk turns the request list into a single
        # Arrow table that matches REQUEST_SCHEMA column-for-column.
        # No per-row pickle, no closure capture of ``self``.
        request_table = pa.Table.from_batches(
            [PreparedRequest.values_to_arrow_batch(misses)]
        )

        # Spread requests across many partitions so mapInArrow scatters
        # across the whole cluster instead of piling them onto a handful
        # of executors. ``createDataFrame`` defaults to a single partition
        # for small Python lists, which serialises stage 3. Target one
        # request per partition, capped at ``defaultParallelism * 8`` so
        # huge request lists don't explode into thousands of micro-tasks
        # whose scheduler overhead dominates the actual fetch.
        #
        # ``sparkContext`` isn't reachable from a Spark Connect proxy
        # (``PySparkAttributeError(JVM_ATTRIBUTE_NOT_SUPPORTED)``); fall
        # back to a sensible default for the partition fan-out.
        try:
            default_par = max(spark.sparkContext.defaultParallelism, 1)
        except Exception:
            default_par = 8
        n_parts = max(1, min(len(misses), default_par * 8))
        LOGGER.debug(
            "Scattering %d send_many miss(es) across %d Spark partition(s) "
            "via mapInArrow (default_parallelism=%d)",
            len(misses), n_parts, default_par,
        )
        request_df = spark.createDataFrame(request_table).repartition(n_parts)

        # Per-executor send config: keep both local and remote cache
        # configs so the worker's send_many can short-circuit on cache
        # hits the driver-side fan-out didn't already catch. No spark
        # session — the worker runs the Python path. ``raise_error=False``
        # so individual failures don't blow up the whole partition.
        send_config = config.to_send_config(
            with_remote_cache=True,
            with_local_cache=True,
            with_spark=False,
            raise_error=False,
        )

        # Pickle once on the driver; the bytes ride along with the
        # ``mapInArrow`` closure. ``Singleton.__new__`` collapses the
        # per-partition ``pickle.loads`` onto the executor's cached
        # ``(cls, config)`` instance, so all partitions on one executor
        # share a single connection pool.
        self_serialized = pickle.dumps(self)
        response_spark_schema = RESPONSE_SCHEMA.to_spark_schema()

        def _send_partition(
            batches: Iterator[pa.RecordBatch],
        ) -> Iterator[pa.RecordBatch]:
            session = pickle.loads(self_serialized)
            for batch in batches:
                partition_requests = list(PreparedRequest.from_arrow(batch))
                if not partition_requests:
                    continue

                def _row_batches() -> Iterator[pa.RecordBatch]:
                    for resp in session.send_many(
                        iter(partition_requests), send_config,
                    ):
                        yield resp.to_arrow_batch(parse=False)

                yield from rechunk_arrow_batches(
                    _row_batches(),
                    byte_size=_SPARK_RESPONSE_BATCH_BYTE_LIMIT,
                )

        result_df = request_df.mapInArrow(
            _send_partition, schema=response_spark_schema,
        )
        # Cache so stage 4's insert (the first action on this frame)
        # both materialises and caches it. Without the cache, every
        # later action — including :attr:`ResponseBatch.counts` — would
        # re-execute the ``mapInArrow``, re-issuing per-partition
        # network calls AND letting workers' ``send_many`` short-circuit
        # on the very rows stage 4 has just persisted to the remote
        # cache table (double-counting them as fresh hits).
        try:
            result_df = result_df.cache()
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "Failed to cache stage-3 mapInArrow result; downstream "
                "counts may re-execute the per-partition fetch",
                exc_info=True,
            )
        return result_df
    
    def get(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
    ) -> Response:
        return self.request(
            "GET",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
        )

    def post(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
    ) -> Response:
        return self.request(
            "POST",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            json=json,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
        )

    def put(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
    ) -> Response:
        return self.request(
            "PUT",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            json=json,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
        )

    def patch(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
    ) -> Response:
        return self.request(
            "PATCH",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            json=json,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
        )

    def delete(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
    ) -> Response:
        return self.request(
            "DELETE",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            json=json,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
        )

    def head(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = False,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
    ) -> Response:
        return self.request(
            "HEAD",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
        )

    def options(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
    ) -> Response:
        return self.request(
            "OPTIONS",
            url,
            config=config,
            params=params,
            headers=headers,
            body=body,
            data=data,
            tags=tags,
            json=json,
            cookies=cookies,
            wait=wait,
            timeout=timeout,
            raise_error=raise_error,
            stream=stream,
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
        )

    def request(
        self,
        method: str,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
    ) -> Response:
        # ``requests``-style aliases: ``data=`` becomes ``body=`` (with
        # form-urlencoding for mappings/sequences), ``timeout=`` becomes
        # ``wait=``, and ``cookies=`` joins ``headers={'Cookie': ...}``.
        # Conflicting pairs raise — picking a side silently would mask a
        # caller bug.
        if data is not None:
            if body is not None:
                raise ValueError(
                    "Pass only one of body= or data= (got both). Use data= "
                    "for requests-style form/raw bodies, body= for the "
                    "native BytesIO/bytes path."
                )
            body, form_content_type = _encode_request_data(data)
            if form_content_type is not None:
                merged = Headers(headers) if headers else Headers()
                if "Content-Type" not in merged:
                    merged["Content-Type"] = form_content_type
                headers = merged

        if timeout is not None:
            if wait is not None:
                raise ValueError(
                    "Pass only one of wait= or timeout= (got both). They "
                    "map to the same underlying WaitingConfig; pick one."
                )
            wait = timeout

        if cookies:
            cookie_header = _format_cookie_header(cookies)
            if cookie_header:
                merged = Headers(headers) if headers else Headers()
                if "Cookie" not in merged:
                    merged["Cookie"] = cookie_header
                headers = merged

        prepared = self.prepare_request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
        )

        return self.send(
            prepared,
            config=config,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
        )

    def prepare_request(
        self,
        method: str,
        url: URL | str | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        local_cache_config: Optional[CacheConfig] = None,
        remote_cache_config: Optional[CacheConfig] = None,
        *,
        json: Any | None = None,
        normalize: bool = True,
    ) -> PreparedRequest:
        full_url: URL | str | None = url

        if self.base_url:
            full_url = self.base_url.join(url) if url else self.base_url
        elif url is None:
            raise ValueError("url is required when base_url is not set on the session.")

        if params:
            parsed = URL.from_(full_url, normalize=normalize)
            full_url = parsed.with_query_items(params)

        return PreparedRequest.prepare(
            method=method,
            url=full_url,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            local_cache_config=local_cache_config,
            remote_cache_config=remote_cache_config
        )
