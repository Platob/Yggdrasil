"""HTTP session abstraction with transparent local and remote cache support."""

from __future__ import annotations

import datetime as dt
import itertools
import logging
import os
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
from yggdrasil.dataclasses.waiting import (
    DEFAULT_WAITING_CONFIG,
    WaitingConfig,
    WaitingConfigArg,
)
from yggdrasil.data.enums import Mode
from .authorization.base import Authorization
from .bytes_io import BytesIO
from .headers import Headers
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
    import pathlib

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
    import pathlib

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
    import pathlib

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


class Session(ABC):
    # Singleton cache keyed by ``(class, normalized base_url string, key)``.
    # A ``Session`` constructed with a ``base_url`` is intentionally shared
    # so the connection pool, cookie jar, and any other per-host state
    # survive across the call sites that re-spell the same URL. The ``key``
    # tag splits same-URL singletons when callers need parallel sessions
    # against one host (e.g. different credentials / tenants) — default ``""``
    # keeps the historical "same URL → same instance" behavior.
    # ``base_url=None`` callers always get a fresh instance — there is no
    # canonical key.
    _singleton_cache: ClassVar[dict[tuple[type, str, str], "Session"]] = {}
    _singleton_lock: ClassVar[threading.Lock] = threading.Lock()

    # Instance attributes that don't survive pickling — excluded by
    # ``__getstate__`` and rebuilt by ``__setstate__``. Subclasses extend
    # this with their own non-picklable handles (e.g. connection pools).
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({
        "_lock", "_job_pool",
    })

    def __new__(
        cls,
        base_url: Optional[URL | str] = None,
        *args: Any,
        key: str = "",
        **kwargs: Any,
    ) -> "Session":
        if not base_url:
            return super().__new__(cls)
        key_url = base_url if isinstance(base_url, URL) else URL.from_(base_url)
        cache_key = (cls, key_url.to_string(), key)
        with cls._singleton_lock:
            cached = cls._singleton_cache.get(cache_key)
            if cached is not None:
                return cached
            instance = super().__new__(cls)
            cls._singleton_cache[cache_key] = instance
            return instance

    def __init__(
        self,
        base_url: Optional[URL | str] = None,
        verify: bool = True,
        pool_maxsize: int = 10,
        headers: "Headers | Mapping[str, str] | None" = None,
        waiting: WaitingConfig = DEFAULT_WAITING_CONFIG,
        *,
        key: str = "",
        auth: Optional[Authorization] = None,
    ) -> None:
        # Singleton-cached instances are re-entered on every constructor call
        # (Python always invokes ``__init__`` after ``__new__``); skip the
        # second pass so we don't drop the live connection pool / cookies.
        if getattr(self, "_initialized", False):
            return
        self.base_url = URL.from_(base_url) if base_url else None
        self.key = key
        self.verify = verify
        self.pool_maxsize = pool_maxsize if pool_maxsize and pool_maxsize > 0 else 8
        self.headers: Headers = Headers.from_(headers)
        self.waiting = waiting
        self._auth: Authorization | None = auth
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
        # Route unpickling through ``__new__`` so a session reconstructed
        # with the same ``(base_url, key)`` collapses to the in-process
        # singleton instead of cloning the connection pool / cookie jar.
        # Use the ``_ex__`` variant so ``key`` reaches ``__new__`` as a
        # keyword (it's keyword-only on the constructor).
        return ((self.base_url,), {"key": self.key})

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in self._TRANSIENT_STATE_ATTRS
        }

    def __setstate__(self, state):
        # ``__new__`` may have returned a live singleton — keep its
        # in-flight state (pool, cookies, init-time config) untouched.
        if getattr(self, "_initialized", False):
            return
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._job_pool = None
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
    def auth(self) -> Optional[Authorization]:
        """Session-wide :class:`Authorization` handler.

        When set, :meth:`prepare_request_before_send` resolves it
        lazily into each outbound request's ``Authorization`` header,
        unless the request carries its own :attr:`PreparedRequest.auth`
        (per-request wins). Travels into Spark workers via the standard
        ``__getstate__`` / ``__setstate__`` round-trip.
        """
        return self._auth

    @auth.setter
    def auth(self, value: Optional[Authorization]) -> None:
        if value is not None and not isinstance(value, Authorization):
            raise TypeError(
                f"auth must be an Authorization instance or None; got "
                f"{type(value).__name__}."
            )
        self._auth = value

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
        spark_session: Optional["SparkSession"] = None,
        **options,
    ) -> Response:
        cfg = SendConfig.check_arg(
            config,
            wait=wait,
            raise_error=raise_error,
            stream=stream,
            remote_cache=remote_cache,
            local_cache=local_cache,
            spark_session=spark_session,
            **options,
        )
        return self._send(request, cfg)

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
        # Lazy auth resolution: per-request handler wins, else fall
        # back to the session-wide handler. Each call hits the
        # handler's ``authorization`` property so refresh-on-expiry
        # handlers (e.g. MSAL) emit the current token per send. We
        # write the resolved header directly instead of mutating
        # ``request.auth`` so a request reused across sessions doesn't
        # carry a stale session-level binding.
        handler = request.auth or self._auth
        if handler is not None:
            if request.headers is None:
                request.headers = Headers()
            request.headers["Authorization"] = handler.authorization
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
        batch_size: int | None = None,
        ordered: bool = False,
        max_in_flight: int | None = None,
        max_batch_ttl: float | None = None,
        spark_session: Optional["SparkSession"] = None,
        **options,
    ) -> Iterator[Response]:
        """Stream responses one at a time, in both Python and Spark modes.

        Spark-backed buckets are drained via the holder's
        :meth:`Tabular.read_records`, which for :class:`SparkTabular`
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
        """
        hits: dict[str, list[Response]] = {}
        misses: list[PreparedRequest] = []

        if not session_local_cfg.local_cache_enabled and not any(
            r.local_cache_config for r in batch
        ):
            # Cheap path: no local cache anywhere in this batch.
            return hits, list(batch)

        for req in batch:
            eff = self._effective_local_cfg(req, session_local_cfg)
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
        """
        hits: dict[str, list[Response]] = {}
        # UPSERT is unconditional miss.
        upsert_reqs = [
            r for r in requests
            if self._effective_remote_cfg(r, session_remote_cfg).mode == Mode.UPSERT
        ]
        misses: list[PreparedRequest] = list(upsert_reqs)

        # Group APPEND-mode requests by effective table.
        table_to_cfg: dict[str, CacheConfig] = {}
        table_to_reqs: dict[str, list[PreparedRequest]] = {}
        for req in requests:
            if req in upsert_reqs:
                continue
            t_cfg = self._effective_remote_cfg(req, session_remote_cfg)
            if not t_cfg.remote_cache_enabled or t_cfg.mode != Mode.APPEND:
                misses.append(req)
                continue
            tkey = t_cfg.tabular.full_name(safe=True)
            if tkey not in table_to_cfg:
                table_to_cfg[tkey] = t_cfg
                table_to_reqs[tkey] = []
            table_to_reqs[tkey].append(req)

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
                hits.append(candidate)
            else:
                misses.append(req)
        return hits, misses

    @staticmethod
    def _responses_to_spark(
        responses: list[Response],
        spark: "SparkSession",
    ) -> "SparkDataFrame":
        """Lift a list of :class:`Response` to a schema-bearing Spark frame.

        Used on the spark path to keep every bucket frame-resident.
        Empty input yields an empty DataFrame keyed to
        :data:`RESPONSE_SCHEMA` so downstream ``unionByName`` calls never
        trip on a column-list mismatch.
        """
        if not responses:
            return spark.createDataFrame(
                [], schema=RESPONSE_SCHEMA.to_spark_schema(),
            )
        table = pa.Table.from_batches(
            [r.to_arrow_batch(parse=False) for r in responses]
        )
        return spark.createDataFrame(table)

    def _split_remote_cache_spark(
        self,
        requests: list[PreparedRequest],
        session_remote_cfg: CacheConfig,
        *,
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
        upsert_reqs = [
            r for r in requests
            if self._effective_remote_cfg(r, session_remote_cfg).mode == Mode.UPSERT
        ]
        misses: list[PreparedRequest] = list(upsert_reqs)

        table_to_cfg: dict[str, CacheConfig] = {}
        table_to_reqs: dict[str, list[PreparedRequest]] = {}
        for req in requests:
            if req in upsert_reqs:
                continue
            t_cfg = self._effective_remote_cfg(req, session_remote_cfg)
            if not t_cfg.remote_cache_enabled or t_cfg.mode != Mode.APPEND:
                misses.append(req)
                continue
            tkey = t_cfg.tabular.full_name(safe=True)
            if tkey not in table_to_cfg:
                table_to_cfg[tkey] = t_cfg
                table_to_reqs[tkey] = []
            table_to_reqs[tkey].append(req)

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

        key_cols = list(cfg.request_by or [])
        if not key_cols:
            # No request-key columns means the SQL can't disambiguate
            # rows per request; mirror the Python path's behaviour by
            # treating every input request as a hit when any row came
            # back, otherwise everything is a miss.
            try:
                any_row = hits_df.head(1)
            except Exception:
                any_row = None
            if any_row:
                return hits_df, []
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
        return hits_df, misses

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
        url_to_local_cfg: Mapping[str, CacheConfig],
        session_local_cfg: CacheConfig,
    ) -> None:
        """Write remote-cache hits back to the local fast-path cache.

        Each response is stored against its originating request's
        effective local config (looked up by URL) — using the
        session-level config for every response would be wrong
        whenever a request carries a custom per-request local
        cache. Each response becomes one fire-and-forget fast-path
        write keyed by ``public_hash``; concurrent writes against
        distinct ``rel_path``s never contend, so no per-root
        serialization is needed.

        After the writes are queued, the per-root cleanup walker
        runs once to unlink stale ``.arrow`` entries — throttled by
        the in-tree sentinel so a hot batch only walks the tree once
        per cleanup window.
        """
        cleanup_roots: dict[str, CacheConfig] = {}
        for response in responses:
            url_key = str(response.request.url) if response.request else None
            eff = url_to_local_cfg.get(url_key) if url_key else None
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
        url_to_remote_cfg: Mapping[str, CacheConfig],
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
            # Per-request lookup keys still use the anonymized URL
            # because that's how :meth:`_send_many_batches` keyed the
            # ``url_to_remote_cfg`` map; the *stored* response is the
            # original. Match-by/identity all hash through the
            # ``public_*`` columns so anonymizing the row before
            # insert isn't required to keep cache lookups consistent.
            url_key = (
                str(response.request.anonymize(mode="remove").url)
                if response.request else None
            )
            eff = url_to_remote_cfg.get(url_key) if url_key else None
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
            batches = pa.Table.from_batches([
                r.to_arrow_batch(parse=False)
                for r in group_responses
            ]).combine_chunks()

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
        url_to_remote_cfg: Mapping[str, CacheConfig],
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
        # syncing into another.
        keep: list[Response] = []
        for response in flat:
            url_key = (
                str(response.request.anonymize(mode="remove").url)
                if response.request else None
            )
            eff = url_to_remote_cfg.get(url_key) if url_key else None
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
        self._persist_remote(keep, url_to_remote_cfg, session_remote_cfg)

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
        for :class:`SparkTabular` uses ``df.toLocalIterator()`` — rows
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
            batch_size=batch_size,
            ordered=ordered,
            max_in_flight=max_in_flight,
            max_batch_ttl=max_batch_ttl,
            spark_session=spark_session,
            **options,
        )
        yield from self._send_many_batches(requests, cfg)

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

            # --- Stage 1: local cache ---
            local_hits_by_path, after_local = self._split_local_cache(
                chunk, session_local_cfg,
            )
            # On the spark path, lift each path's responses to its own
            # Spark frame so every bucket downstream is frame-resident
            # — matches stage 2/3 and lets the caller union holders
            # without a per-bucket type switch. Empty dict (no local
            # hits) is left as-is; ResponseBatch installs a
            # schema-bearing default placeholder, Spark or Arrow as
            # appropriate.
            local_hits: "dict[str, list[Response]] | dict[str, SparkDataFrame]"
            if is_spark:
                local_hits = {
                    pkey: self._responses_to_spark(rs, spark)
                    for pkey, rs in local_hits_by_path.items()
                }
            else:
                local_hits = local_hits_by_path
            # Remote hits are split per cache table — keyed by
            # ``CacheConfig.table.full_name(safe=True)`` — so the
            # downstream :class:`ResponseBatch` preserves which table
            # answered which subset. An empty dict tells
            # ``ResponseBatch`` to install a schema-bearing default.
            remote_hits: "dict[str, list[Response]] | dict[str, SparkDataFrame]" = {}
            # Default new_hits to None so ResponseBatch coerces it to a
            # schema-bearing empty holder (Spark or Arrow depending on
            # mode) — no special-case for "stage skipped".
            new_hits: "list[Response] | SparkDataFrame | None" = None

            # Snapshot per-request effective configs BEFORE we mutate copies
            # for the worker pool. Keyed by URL string — the natural identity
            # for matching a response back to its originating request. We
            # cover the *whole* chunk (not just ``after_local``) so the
            # local-hit mirror below can resolve per-request remote configs
            # for responses that never reached stage 2.
            url_to_remote_cfg: dict[str, CacheConfig] = {
                str(r.anonymize(mode="remove").url): self._effective_remote_cfg(r, session_remote_cfg)
                for r in chunk
            }
            url_to_local_cfg: dict[str, CacheConfig] = {
                str(r.anonymize(mode="remove").url): self._effective_local_cfg(r, session_local_cfg)
                for r in chunk
            }

            if not after_local:
                # Even when every request is a local hit we may still
                # owe the remote a diff — call the mirror before the
                # early return so an all-cache batch keeps the remote
                # in sync without ever hitting the network.
                if not is_spark:
                    self._mirror_local_hits_to_remote(
                        local_hits_by_path,
                        url_to_remote_cfg,
                        session_remote_cfg,
                    )
                yield ResponseBatch(
                    local_hits=local_hits,
                    remote_hits=remote_hits,
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
                remote_hits, after_remote = self._split_remote_cache_spark(
                    after_local,
                    session_remote_cfg,
                    spark=spark,
                )
                # Local-cache backfill from a Spark frame would force a
                # toLocalIterator on the driver — skip it on the spark
                # path, matching how stage 3/4 keep network results
                # frame-resident. Drivers that want a hot local cache
                # should use the Python path explicitly.
            else:
                remote_hits, after_remote = self._split_remote_cache(
                    after_local,
                    session_remote_cfg,
                    spark_session=spark,
                )
                # Backfill local cache with remote hits using each request's
                # effective local config — not the session-level fallback.
                # Flatten the per-table split here: the local cache
                # doesn't care which remote table sourced a row, only
                # the response-to-request mapping by URL.
                self._backfill_local_cache(
                    [r for table_hits in remote_hits.values() for r in table_hits],
                    url_to_local_cfg,
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
                    url_to_remote_cfg,
                    session_remote_cfg,
                )

            if not after_remote:
                yield ResponseBatch(
                    local_hits=local_hits,
                    remote_hits=remote_hits,
                    new_hits=new_hits,
                    spark=spark,
                )
                continue

            # --- Stage 3: fetch misses ---
            failed: list[Response] = []
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
                # materialising on the driver.
                if (
                    new_hits is not None
                    and session_remote_cfg.remote_cache_enabled
                ):
                    self._spark_persist_remote(
                        new_hits, session_remote_cfg, spark=spark,
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
                            url_to_remote_cfg,
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
                            url_to_local_cfg,
                            session_local_cfg,
                        )
                    )
                self._run_concurrently(
                    stage4, thread_name_prefix="ygg-stage4",
                )

            yield ResponseBatch(
                local_hits=local_hits,
                remote_hits=remote_hits,
                new_hits=new_hits,
                spark=spark,
            )

            if not is_spark and config.raise_error and failed:
                failed[-1].raise_for_status()

    # ------------------------------------------------------------------ #
    # Spark stage 3 / 4 helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _spark_persist_remote(
        new_responses_df: "SparkDataFrame",
        cfg: CacheConfig,
        *,
        spark: "SparkSession",
    ) -> None:
        """Stage 4 on Spark: bulk-insert successful responses into the remote cache.

        Honours the session-level remote config only — per-request overrides
        collapse onto it on the spark path, mirroring stage 3 where workers
        see only the session-level local cache config. ``cfg.tabular.insert``
        accepts the Spark DataFrame directly via ``spark_insert_into``, so
        no driver-side collect is needed.

        Before inserting, APPEND-mode writes are de-duplicated against the
        existing remote rows via a ``left_anti`` join on the response
        ``hash`` column — the remote table stores anonymized requests
        (cf. ``_persist_remote``), so a row whose hash already lives in
        the cache is suppressed rather than re-inserted. UPSERT mode
        keeps its read-free fast path and relies on ``match_by`` to
        collapse duplicates server-side.
        """
        from pyspark.sql import functions as F

        ok_df = new_responses_df.where(
            (F.col("status_code") >= 200)
            & (F.col("status_code") < 300)
        )

        if cfg.mode != Mode.UPSERT:
            table_name = cfg.tabular.full_name(safe=True)
            # Restrict the SELECT DISTINCT to the partitions actually
            # in play — ``partition_key`` is the table's partition
            # column so the engine prunes the existing-rows scan
            # before reading any data files.
            try:
                wanted_partitions = [
                    row["partition_key"]
                    for row in ok_df.select("partition_key").distinct().collect()
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
                # Table doesn't exist yet — nothing to dedup against; the
                # downstream `cfg.tabular.insert` handles creation. Match the
                # error-string sniff used by `_lookup_remote_table`.
                if "TABLE_OR_VIEW_NOT_FOUND" not in str(exc):
                    raise
                existing_df = None

            if existing_df is not None:
                ok_df = ok_df.join(
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
            ok_df,
            mode=cfg.mode,
            match_by=cfg.sql_match_by or None,
            wait=cfg.wait,
            prune_by=["partition_key", "public_hash"],
            spark_session=spark,
        )

    def _spark_fetch_misses(
        self,
        misses: list[PreparedRequest],
        config: SendManyConfig,
        spark: "SparkSession",
    ) -> "SparkDataFrame":
        """Stage 3 on Spark: scatter misses to workers via mapInArrow.

        Each Spark partition becomes one `send_many` call on the executor,
        fanning out via the session's local thread pool. Per-request remote
        cache configs are stripped (driver concern); per-request local cache
        configs are dropped on workers (they see only the session config).

        The session is shipped to executors via ``sparkContext.broadcast``
        — once per executor instead of once per task closure — and
        re-attached to each request on the worker via
        :meth:`PreparedRequest.attach_session`. Requests cross the wire
        as a single pickled-bytes column; pickle preserves the full
        :class:`PreparedRequest` (closures, buffer, cache configs)
        without the per-engine schema dance the Arrow round-trip needed.
        """
        import pickle

        from pyspark.sql.types import BinaryType, StructField, StructType

        req_schema = StructType([StructField("request", BinaryType(), nullable=False)])
        req_rows = [
            (pickle.dumps(r.copy(remote_cache_config=None), protocol=pickle.HIGHEST_PROTOCOL),)
            for r in misses
        ]

        # Spread requests across many partitions so mapInArrow scatters
        # across the whole cluster instead of piling them onto a handful
        # of executors. ``createDataFrame`` defaults to a single partition
        # for small Python lists, which serialises stage 3. Target one
        # request per partition, capped at ``defaultParallelism * 8`` so
        # huge request lists don't explode into thousands of micro-tasks
        # whose scheduler overhead dominates the actual fetch.
        default_par = max(spark.sparkContext.defaultParallelism, 1)
        n_parts = max(1, min(len(req_rows), default_par * 8))
        request_df = spark.createDataFrame(req_rows, req_schema).repartition(n_parts)

        # Per-executor send config: remote cache disabled (driver-only),
        # local cache passthrough, no spark session, raise_error=False so
        # individual failures don't blow up the whole partition.
        send_config = config.to_send_config(
            with_remote_cache=False,
            with_local_cache=True,
            with_spark=False,
            raise_error=False,
        )

        # Broadcast the session so every executor receives the
        # (pickle-safe) session state once and reuses it across tasks,
        # rather than re-shipping a closure-captured copy per partition.
        # Session.__getstate__ / __setstate__ make this pickle-safe by
        # dropping the threading.RLock and JobPoolExecutor.
        session_bc = spark.sparkContext.broadcast(self)
        response_spark_schema = RESPONSE_SCHEMA.to_spark_schema()

        def _send_partition(
            batches: Iterator[pa.RecordBatch],
        ) -> Iterator[pa.RecordBatch]:
            import pickle as _pickle

            session = session_bc.value
            for batch in batches:
                partition_requests = [
                    _pickle.loads(buf).attach_session(session)
                    for buf in batch.column("request").to_pylist()
                ]
                if not partition_requests:
                    continue

                def _row_batches() -> Iterator[pa.RecordBatch]:
                    for resp in session.send_many(iter(partition_requests), send_config):
                        yield resp.to_arrow_batch(parse=False)

                yield from rechunk_arrow_batches(
                    _row_batches(),
                    byte_size=_SPARK_RESPONSE_BATCH_BYTE_LIMIT,
                )

        return request_df.mapInArrow(_send_partition, schema=response_spark_schema)
    
    def get(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: BytesIO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        wait: WaitingConfigArg = None,
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
            tags=tags,
            wait=wait,
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
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        wait: WaitingConfigArg = None,
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
            tags=tags,
            json=json,
            wait=wait,
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
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        wait: WaitingConfigArg = None,
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
            tags=tags,
            json=json,
            wait=wait,
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
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        wait: WaitingConfigArg = None,
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
            tags=tags,
            json=json,
            wait=wait,
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
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        wait: WaitingConfigArg = None,
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
            tags=tags,
            json=json,
            wait=wait,
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
        tags: Mapping[str, str] | None = None,
        wait: WaitingConfigArg = None,
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
            tags=tags,
            wait=wait,
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
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        wait: WaitingConfigArg = None,
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
            tags=tags,
            json=json,
            wait=wait,
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
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
        stream: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
    ) -> Response:
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
