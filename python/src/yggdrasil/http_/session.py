"""Concrete HTTP/HTTPS session — the single public entry point of :mod:`yggdrasil.http_`.

Construct one :class:`HTTPSession` per host (singleton-cached by config), drive
verb methods (``get`` / ``post`` / ``put`` / ``patch`` / ``delete`` / ``head``
/ ``options`` / ``request``), and read the returned :class:`HTTPResponse`. All
the HTTP machinery lives on this class:

* connection cache + per-host keep-alive sockets,
* status-aware retry (:class:`_TieredRetry`) + redirect handling,
* the prepare → cache-lookup → wire-send → cache-writeback pipeline,
* :meth:`send_many` with Spark fan-out,
* verb sugar and cookie-header coercion.

Wire calls go straight to :mod:`http.client` — no intermediate transport
class wraps the send. The supporting types (:class:`Retry`,
:class:`Timeout`, :class:`HTTPResponse`, :class:`HTTPHeaderDict`,
:mod:`exceptions`) live in the small side modules (:mod:`yggdrasil.http_.retry`,
:mod:`yggdrasil.http_.timeout`, :mod:`yggdrasil.http_.exceptions`,
:mod:`yggdrasil.http_.headers`); feature code
should not import them directly.

This module also hosts the abstract :class:`Session` base class — singleton +
pickle + concurrency-pool plumbing that every concrete session needs.
Transport-specific surface — URL handling, headers, auth, verb methods,
``send`` / ``send_many``, local/remote cache pipeline, Spark integration —
lives on the concrete subclass:

* :class:`HTTPSession` for HTTP/HTTPS;
* :class:`yggdrasil.data.executor.StatementExecutor` for SQL backends.

Two non-obvious rules every subclass must follow to participate in the
singleton cache cleanly:

1. ``__init__`` guards on ``getattr(self, "_initialized", False)`` so the
   re-entry Python performs after a singleton cache hit doesn't clobber
   live state.
2. Identity-bearing constructor arguments are written to ``self.__dict__``
   under names that appear in ``__init__``'s parameter list.
   :meth:`_singleton_key` runs the subclass's own ``__init__`` on a
   throwaway probe and projects matching attributes into the key — every
   normalisation the constructor applies (URL parsing, header coercion,
   pool-size clamping, …) lands in the key for free; derived caches /
   lazy handles stay out by construction because their attribute names
   don't match parameter names.
"""

from __future__ import annotations

import collections
import datetime as dt
import http.client
import itertools
import logging
import os
import pathlib
import socket
import ssl
import threading
import time
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from itertools import takewhile
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
from urllib.parse import urlsplit, urlunsplit

from yggdrasil.concurrent.threading import Job, JobPoolExecutor
from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.dataclasses.waiting import (
    DEFAULT_WAITING_CONFIG,
    WaitingConfig,
    WaitingConfigArg,
)
from yggdrasil.enums import MediaTypes
from yggdrasil.http_.authorization.base import Authorization
from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse
from yggdrasil.http_.response_batch import HTTPResponseBatch
from yggdrasil.io.holder import IO
from yggdrasil.http_.headers import HTTPHeaders
from yggdrasil.path.memory import Memory
from yggdrasil.io.primitive import ArrowIPCFile
from yggdrasil.http_.cache_config import CacheConfig
from yggdrasil.http_.send_config import DEFAULT_MAX_BATCH_TTL, SendConfig
from yggdrasil.url import URL

from .exceptions import (
    InsecureRequestWarning,
    LocationParseError,
    LocationValueError,
    MaxRetryError,
    NewConnectionError,
    ProxyError,
    ReadTimeoutError,
    SSLError,
)
from .response import HTTPResponse
from .retry import Retry
from .timeout import _resolve_timeout

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from yggdrasil.path.folder import Folder

__all__ = ["Session", "HTTPSession"]


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------


def _resolve_proxy_url(
    proxy: "URL | str | None",
    target_scheme: str | None = None,
) -> "URL | None":
    """Resolve a proxy URL from the explicit argument or environment.

    Lookup order (first non-empty wins):

    1. ``proxy`` if not ``None``.
    2. ``HTTPS_PROXY`` / ``https_proxy`` when *target_scheme* is ``"https"``.
    3. ``HTTP_PROXY`` / ``http_proxy`` when *target_scheme* is ``"http"``.
    4. ``ALL_PROXY`` / ``all_proxy`` as a catch-all.
    """
    if proxy is not None:
        return URL.from_(proxy) if not isinstance(proxy, URL) else proxy

    if target_scheme == "https":
        env = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if env:
            return URL.from_(env)

    if target_scheme == "http":
        env = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        if env:
            return URL.from_(env)

    env = os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
    if env:
        return URL.from_(env)

    return None


def _should_bypass_proxy(host: str, no_proxy: str | None = None) -> bool:
    """Return ``True`` when *host* matches a ``no_proxy`` pattern.

    Reads ``NO_PROXY`` / ``no_proxy`` from the environment when
    *no_proxy* is ``None``.  The wildcard ``"*"`` bypasses everything.
    Entries are comma-separated; leading dots match any subdomain
    (``".example.com"`` matches ``"foo.example.com"``).
    """
    if no_proxy is None:
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""

    if not no_proxy:
        return False

    host = host.lower().strip(".")

    for entry in no_proxy.split(","):
        entry = entry.strip().lower().strip(".")
        if not entry:
            continue
        if entry == "*":
            return True
        if host == entry:
            return True
        if host.endswith("." + entry):
            return True

    return False


# ---------------------------------------------------------------------------
# SSL context helpers
# ---------------------------------------------------------------------------

# Accepted shapes for the ``verify`` parameter:
#   True          — default CA bundle (ssl.create_default_context())
#   False         — no verification (InsecureRequestWarning emitted)
#   str / Path    — path to a custom CA bundle or directory
VerifyArg = "bool | str | pathlib.Path"


_DEFAULT_CONNECT_TIMEOUT: float = 30.0


def _floor_connect_timeout(timeout: Optional[float]) -> float:
    """Ensure a connect timeout is never ``None`` (infinite).

    ``http.client.HTTPConnection(timeout=None)`` means "block until the
    OS gives up" — which on some platforms is 2+ minutes and on others
    is infinite.  This helper applies a floor so callers that forget to
    set a timeout (``WaitingConfig(timeout=0)`` → ``Timeout(connect=None)``)
    don't hang the process.
    """
    if timeout is None or timeout <= 0:
        return _DEFAULT_CONNECT_TIMEOUT
    return timeout


def _make_ssl_context(verify: "bool | str | pathlib.Path") -> ssl.SSLContext:
    """Build an :class:`ssl.SSLContext` from a ``verify`` argument.

    - ``True`` — system default CA bundle.
    - ``False`` — no certificate verification, no hostname check.
    - ``str`` / ``pathlib.Path`` — custom CA bundle file or directory.
    """
    if verify is False:
        ctx = ssl._create_unverified_context()  # type: ignore[attr-defined]
        ctx.check_hostname = False
        return ctx

    ctx = ssl.create_default_context()
    if verify is not True:
        ca_path = str(verify)
        if os.path.isdir(ca_path):
            ctx.load_verify_locations(capath=ca_path)
        else:
            ctx.load_verify_locations(cafile=ca_path)
    return ctx


def _warn_if_insecure(verify: "bool | str | pathlib.Path") -> None:
    """Emit :class:`InsecureRequestWarning` when verification is off."""
    if verify is False:
        import warnings
        warnings.warn(
            "SSL certificate verification is disabled. "
            "Connections to HTTPS endpoints will not validate the server's identity.",
            InsecureRequestWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Session base — singleton + pickle + concurrency-pool plumbing
# ---------------------------------------------------------------------------


def _hashable_identity_value(value: Any) -> Any:
    """Coerce an ``__init__`` argument into a hashable form for the singleton key.

    Most argument shapes (frozen :class:`WaitingConfig`,
    :class:`Authorization`, :class:`URL`, primitives, ``None``) are
    already hashable and pass through unchanged. A few common
    constructor-input shapes that aren't get canonicalised here so two
    callers that spell the same identity slightly differently still
    collapse onto one singleton:

    * :class:`URL` is stringified, so ``"https://x.com"`` and
      ``URL("https://x.com")`` key the same way once the subclass's
      ``__init__`` has normalised both into a :class:`URL`;
    * :class:`HTTPHeaders` / generic :class:`Mapping` collapse to a tuple
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


class Session(Singleton, ABC):
    """Abstract per-transport session base — singleton-keyed by post-init ``__dict__``.

    Inherits the standard :class:`Singleton` plumbing:

    - same-config constructor calls collapse to one process-lifetime
      instance (``_SINGLETON_TTL = None``), so transport pool, cookie
      jar, per-host state etc. survive across every call site that
      re-spells the same configuration;
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
    ``__setstate__`` short-circuits on a live singleton). The only
    Session-specific addition is that the receiver rebuilds a fresh
    :class:`threading.RLock` instead of carrying the sender's lock
    state.
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
        "_lock", "_job_pool", "_local_cache",
    })

    # Prepared / response / batch types the prepare → send pipeline
    # emits. Each transport pins these to its own concrete types —
    # :class:`HTTPSession` to :class:`HTTPRequest` / :class:`Response`
    # / :class:`HTTPResponseBatch`; :class:`StatementExecutor` subclasses
    # to :class:`PreparedStatement` / :class:`StatementResult` /
    # :class:`StatementBatch` — so the same vocabulary covers every
    # transport.
    _PREPARED_CLASS: ClassVar[type]
    _RESPONSE_CLASS: ClassVar[type]
    _BATCH_CLASS: ClassVar[type]

    # Default concurrency-pool sizing used when ``__init__`` is called
    # without an explicit ``pool_maxsize``. Subclasses (HTTPSession)
    # may clamp tighter for transport-specific reasons (per-host
    # connection limits) by overriding ``__init__``.
    DEFAULT_POOL_MAXSIZE: ClassVar[int] = 8

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

    def __init__(self, *, pool_maxsize: Optional[int] = None) -> None:
        # Singleton-cached instances are re-entered on every constructor call
        # (Python always invokes ``__init__`` after ``__new__``); skip the
        # second pass so we don't drop live state.
        if getattr(self, "_initialized", False):
            return
        self.pool_maxsize = (
            int(pool_maxsize)
            if pool_maxsize and pool_maxsize > 0
            else self.DEFAULT_POOL_MAXSIZE
        )
        self._lock = threading.RLock()
        self._job_pool: Optional[JobPoolExecutor] = None
        self._local_cache: Optional["Folder"] = None
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
            if k in param_names and k not in excluded
        }
        return (), state

    @classmethod
    def _init_param_names(cls) -> frozenset[str]:
        """Union of named ``__init__`` parameters across the MRO.

        ``*args`` / ``**kwargs`` capture parameters are excluded so a
        subclass with a bare ``def __init__(self, *args, **kwargs)``
        passthrough (the test stubs do this) still inherits the
        parent's named knobs without leaking attribute slots back
        through ``__getnewargs_ex__``.

        Memoised on the class itself (``__dict__`` slot, not
        ``setattr``) so a subclass doesn't pick up the parent's
        cached set. The :class:`Singleton` probe path consults this
        on *every* ``HTTPSession(…)`` construction — without the
        cache, the :mod:`inspect`-driven MRO walk dominated the
        singleton-hit cost (~24 us / 60 us total).
        """
        cached = cls.__dict__.get("_INIT_PARAM_NAMES_CACHE")
        if cached is not None:
            return cached

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
        result = frozenset(names)
        type.__setattr__(cls, "_INIT_PARAM_NAMES_CACHE", result)
        return result

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
        self._local_cache = None
        self._initialized = True

    @property
    def job_pool(self) -> JobPoolExecutor:
        if self._job_pool is None:
            with self._lock:
                if self._job_pool is None:
                    self._job_pool = JobPoolExecutor(max_workers=self.pool_maxsize)
                    LOGGER.debug("Created job pool with max_workers=%s", self.pool_maxsize)
        return self._job_pool

    def local_cache(self) -> "Folder":
        """Return the session-scoped local cache folder, creating the directory on first access.

        The folder lives under ``~/.cache/http/<host>/<path>``
        when :attr:`base_url` is set (so different APIs on the same machine
        don't collide), or ``~/.cache/http/default`` otherwise.

        Thread-safe: the directory is created under the session lock and
        the resulting :class:`Folder` is cached for the lifetime of
        the singleton.
        """
        cached = getattr(self, "_local_cache", None)
        if cached is not None:
            return cached
        with self._lock:
            cached = getattr(self, "_local_cache", None)
            if cached is not None:
                return cached
            from yggdrasil.path.folder import Folder
            from yggdrasil.path import Path

            root = pathlib.Path.home() / ".cache" / "http"
            base_url = getattr(self, "base_url", None)
            host = getattr(base_url, "host", None) if base_url is not None else None
            if not host:
                folder = root / "default"
            else:
                url_path = (getattr(base_url, "path", "") or "").strip("/")
                folder = root / host / url_path if url_path else root / host
            path = Path.from_(folder)
            self._local_cache: "Folder" = Folder(path=path)
            return self._local_cache


# ---------------------------------------------------------------------------
# HTTPSession — concrete HTTP/HTTPS transport
# ---------------------------------------------------------------------------


# Cap on per-batch byte size when emitting responses from a Spark
# `mapInArrow` worker. 128 MiB matches Spark's default Arrow batch
# preference and keeps a single oversized response from inflating the
# whole partition's output. A response that is itself larger than the
# cap is sliced row-wise by the shared rechunker, which never splits a
# single row across batches.

# Rechunk byte target for paginated responses assembled by
# ``_combine_paginated_pages``.  Targets the IPC page content size
# (``RecordBatch.serialize().size``), not the in-memory buffer size.
_PAGINATED_RECHUNK_BYTE_SIZE: int = 128 * 1024 * 1024


# Local cache is a partitioned tabular tree backed by
# :class:`yggdrasil.path.folder.Folder`:
# ``<root>/partition_key=<int>/part-{epoch_ms}-{seed}.<ext>``.
# Same Hive-style partition shape the remote :class:`Tabular` cache
# uses, so the same lookup primitives — :meth:`CacheConfig.make_lookup_predicate`
# / :meth:`CacheConfig.make_batch_lookup_predicate` — prune both
# backends identically. The predicate's ``partition_key IN (...)``
# clause flows through :meth:`Folder.iter_children`'s candidate
# probe, so a batch lookup ``stat``s only the partition directories
# its requests touch instead of walking the whole tree.





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


# ---------------------------------------------------------------------------
# Retry tuning — backoff schedule for HTTPSession's connection pool.
# ---------------------------------------------------------------------------

# 429s get a longer schedule than 5xx because rate limits need wall-clock
# time to clear; both schedules are tight (we'd rather surface an error
# fast than mask a real outage with a minute-long retry storm).
# Server-supplied Retry-After always wins over these when present.
_RETRY_TOTAL = 4
_RETRY_CONNECT = 3
_RETRY_READ = 3

# 5xx schedule: 0.5, 1, 2 (capped at backoff_max). Worst-case ~3.5s.
_BACKOFF_5XX_FACTOR = 0.5
_BACKOFF_5XX_MAX = 5.0

# 429 schedule: 1, 2, 4 (capped at backoff_max). Worst-case ~7s.
_BACKOFF_429_FACTOR = 1.0
_BACKOFF_429_MAX = 5.0

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class _TieredRetry(Retry):
    """:class:`Retry` variant with status-aware backoff.

    Standard ``Retry`` exposes a single ``backoff_factor`` shared by every
    retry, so 429 (rate limit) and 503 (transient outage) get the same
    schedule. This subclass branches on the most recent response status:

    * **429** uses a longer, gentler exponential schedule, since rate-limit
      windows are typically wall-clock bound and respond poorly to tight
      retries.
    * **Everything else** (5xx, transport errors) uses a shorter schedule.
    * The server's ``Retry-After`` header — when present and respected via
      ``respect_retry_after_header=True`` — always overrides this, because
      the pool checks ``get_retry_after`` before ``get_backoff_time``.
    """

    BACKOFF_MAX = _BACKOFF_429_MAX

    def get_backoff_time(self) -> float:  # type: ignore[override]
        # Mirror urllib3's own short-circuit: no backoff before the second
        # consecutive error. ``history`` is a tuple of RequestHistory entries.
        consecutive_errors = list(
            takewhile(lambda x: x.redirect_location is None, reversed(self.history))
        )
        if len(consecutive_errors) <= 1:
            return 0.0

        last_status = consecutive_errors[0].status

        if last_status == 429:
            # Count *consecutive* 429s only — if the last attempt was a 503,
            # we want the 5xx schedule, not a 429 schedule inflated by older
            # rate-limit hits.
            n = 0
            for h in consecutive_errors:
                if h.status == 429:
                    n += 1
                else:
                    break
            backoff = _BACKOFF_429_FACTOR * (2 ** (n - 1))
            return float(min(_BACKOFF_429_MAX, backoff))

        # Default 5xx / transport-error schedule, mirroring urllib3's formula
        # but with our own factor and cap.
        backoff = _BACKOFF_5XX_FACTOR * (2 ** (len(consecutive_errors) - 1))
        return float(min(_BACKOFF_5XX_MAX, backoff))


class HTTPSession(Session):
    """HTTP/HTTPS session — singleton-keyed by ``(base_url, verify, pool_maxsize, headers, waiting, auth)``.

    Inherits the singleton + pickle + ``job_pool`` plumbing from
    :class:`~yggdrasil.io.session.Session` and adds every HTTP-specific
    concern: a stdlib-backed connection pool (HTTPSession owns the per-host
    socket cache directly — no separate ``PoolManager`` indirection)
    connection pool, the prepare → cache-lookup → wire-send → cache-writeback
    pipeline (both single-request :meth:`send` and bulk :meth:`send_many`),
    Spark fan-out via ``mapInArrow``, the verb sugar
    (:meth:`get` / :meth:`post` / :meth:`put` / :meth:`patch` /
    :meth:`delete` / :meth:`head` / :meth:`options` / :meth:`request`),
    cookie-header coercion, and the 403 auth-refresh retry loop.

    No User-Agent generator, cookie jar, or browser-emulation layering is
    built in — per-vendor integrations subclass this for their own auth /
    pagination / rate-limit policy (see :class:`yggdrasil.fxrate.FxRate` for
    a worked example).
    """

    _PREPARED_CLASS: ClassVar[type] = HTTPRequest
    _RESPONSE_CLASS: ClassVar[type] = HTTPResponse
    _BATCH_CLASS: ClassVar[type] = HTTPResponseBatch

    _TRANSIENT_STATE_ATTRS = Session._TRANSIENT_STATE_ATTRS | {"_connections", "_retry"}

    # Status codes that trigger an automatic redirect when ``redirect=True``.
    # 303 always falls back to GET (per RFC 7231); 307/308 preserve method.
    _REDIRECT_STATUSES: ClassVar[frozenset[int]] = frozenset({301, 302, 303, 307, 308})
    _MAX_REDIRECTS: ClassVar[int] = 10

    def __init__(
        self,
        base_url: Optional[URL | str] = None,
        verify: "bool | str | pathlib.Path" = True,
        pool_maxsize: int = 10,
        headers: "HTTPHeaders | Mapping[str, str] | None" = None,
        waiting: WaitingConfig = DEFAULT_WAITING_CONFIG,
        *,
        auth: Optional[Authorization] = None,
        proxy: Optional[URL | str] = None,
        no_proxy: Optional[str] = None,
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
        # Normalise pathlib.Path → str so the singleton key is hashable.
        if isinstance(verify, pathlib.Path):
            verify = str(verify)
        _warn_if_insecure(verify)
        # The pool caps idle sockets per host; 8 is plenty for our typical
        # workloads. Clamping here means the singleton key (built from
        # ``pool_maxsize``) collapses ``HTTPSession(pool_maxsize=20)`` and
        # ``HTTPSession()`` to one instance the way they always did.
        pool_maxsize = min(8, int(pool_maxsize)) if pool_maxsize else 8
        self.base_url = URL.from_(base_url) if base_url else None
        self.verify: bool | str = verify
        self.headers: HTTPHeaders = HTTPHeaders.from_(headers)
        self.waiting = waiting
        self.auth: Authorization | None = auth
        self.proxy: URL | None = URL.from_(proxy) if isinstance(proxy, str) else proxy
        self.no_proxy: str | None = no_proxy

        # Singleton-key probe path bails here — :class:`Session._singleton_key`
        # reads ``probe.__dict__`` and keeps only the keys whose names
        # appear in ``__init__``'s parameter list (``base_url`` / ``verify``
        # / ``pool_maxsize`` / ``headers`` / ``waiting`` / ``auth``), so
        # the lock + connection cache + retry policy + ``_job_pool`` build
        # below contribute nothing to identity. Skipping them on the probe
        # halves the singleton-hit cost (the bench was paying for an RLock
        # alloc + a ``_TieredRetry()`` per ``HTTPSession(base_url=…)`` call).
        if getattr(self, "_in_probe", False):
            self.pool_maxsize = pool_maxsize
            return

        super().__init__(pool_maxsize=pool_maxsize)
        # When a session-wide auth handler is bound at construction
        # time, pre-stamp ``self.headers["Authorization"]`` so anyone
        # inspecting the session sees the current credential without
        # going through a request first. :meth:`refresh_auth` keeps
        # the session header in sync on subsequent refreshes.
        if auth is not None:
            self.headers["Authorization"] = auth.authorization
        # Per-host idle-connection cache keyed by ``(scheme, host, port)``.
        # Sockets are recycled across requests so warm calls reuse the
        # existing TCP / TLS handshake instead of paying for a new one;
        # capped at ``pool_maxsize`` entries per host.
        self._connections: dict[tuple[str, str, int], "collections.deque[http.client.HTTPConnection]"] = {}
        # Retry policy is built once at init — same policy applies to
        # every wire send and the singleton key already pins
        # ``pool_maxsize`` / ``waiting`` / ``verify`` so two sessions
        # with the same identity share the same policy by construction.
        self._retry: Retry = self._build_retry()

    def __getnewargs_ex__(self):
        # Promote ``base_url`` to the positional arg slot so the receiver
        # can reproduce the canonical HTTPSession signature even when the
        # subclass introduces extra named knobs. Everything else rides as
        # kwargs through the inherited identity machinery.
        param_names = self._init_param_names()
        excluded = self._identity_excluded_attrs()
        state = {
            k: v for k, v in self.__dict__.items()
            if k in param_names and k not in excluded and k != "base_url"
        }
        return (self.base_url,), state

    def __setstate__(self, state):
        # Defer to :meth:`Session.__setstate__` for the live-singleton
        # short-circuit + lock rebuild; then promote ``_connections`` and
        # ``_retry`` from ``None`` (the transient default) to a usable
        # value. ``_connections`` starts empty — the receiver opens its
        # own sockets — and ``_retry`` is rebuilt fresh from the same
        # policy the sender used (it's stateless config-shaped).
        if getattr(self, "_initialized", False):
            return
        super().__setstate__(state)
        self._connections = {}
        self._retry = self._build_retry()

    # ------------------------------------------------------------------
    # Retry policy + connection cache
    # ------------------------------------------------------------------

    def _build_retry(self) -> Retry:
        """Build the :class:`Retry` policy applied to every wire send.

        Subclasses can override to swap the policy entirely, or call
        ``super()._build_retry().new(...)`` to tweak a single field.
        """
        return _TieredRetry(
            total=_RETRY_TOTAL,
            connect=_RETRY_CONNECT,
            read=_RETRY_READ,
            status=_RETRY_TOTAL,
            other=4,
            status_forcelist=_RETRY_STATUSES,
            allowed_methods=None,  # retry every method, incl. POST/PATCH
            respect_retry_after_header=True,
            raise_on_status=False,
            raise_on_redirect=False,
            # backoff_factor/backoff_max are unused — _TieredRetry overrides
            # get_backoff_time entirely — but we set sane defaults so any
            # fallback path (e.g. .new() that drops back to base behavior) is
            # still well-behaved.
            backoff_factor=_BACKOFF_5XX_FACTOR,
            backoff_max=_BACKOFF_429_MAX,
        )

    def _resolve_proxy_for(self, scheme: str, host: str) -> "URL | None":
        """Return the proxy URL for this request, or ``None`` to go direct."""
        if _should_bypass_proxy(host, self.no_proxy):
            return None
        return _resolve_proxy_url(self.proxy, target_scheme=scheme)

    def _build_connection(
        self,
        scheme: str,
        host: str,
        port: int,
        connect_timeout: Optional[float],
    ) -> http.client.HTTPConnection:
        """Open a fresh connection to *(scheme, host, port)*.

        When a proxy is configured and the target host is not bypassed,
        the connection routes through the proxy:

        - **HTTPS targets** use an HTTP CONNECT tunnel — the TCP socket
          connects to the proxy, sends ``CONNECT host:port``, then
          upgrades to TLS so the proxy sees only opaque bytes.
        - **HTTP targets** connect to the proxy directly; the caller
          must send the absolute URL as the request path (handled by
          :meth:`_send_once`).

        Honours ``self.verify``:

        - ``True`` — system default CA bundle.
        - ``False`` — no certificate verification.
        - ``str`` — path to a custom CA bundle file or directory.
        """
        connect_timeout = _floor_connect_timeout(connect_timeout)
        proxy = self._resolve_proxy_for(scheme, host)

        if scheme == "https":
            ssl_ctx: ssl.SSLContext = _make_ssl_context(self.verify)

            if proxy is not None:
                proxy_host = proxy.host
                proxy_port = proxy.port or (443 if proxy.scheme == "https" else 8080)
                conn = http.client.HTTPSConnection(
                    host, port=port, timeout=connect_timeout, context=ssl_ctx,
                )
                conn.set_tunnel(host, port, headers=self._proxy_auth_headers(proxy))
                conn._real_host = proxy_host  # type: ignore[attr-defined]
                conn._real_port = proxy_port  # type: ignore[attr-defined]
                try:
                    tunnel_conn = http.client.HTTPConnection(
                        proxy_host, port=proxy_port, timeout=connect_timeout,
                    )
                    tunnel_conn.connect()
                    tunnel_headers = {"Host": f"{host}:{port}"}
                    tunnel_headers.update(self._proxy_auth_headers(proxy))
                    tunnel_conn.request("CONNECT", f"{host}:{port}", headers=tunnel_headers)
                    tunnel_resp = tunnel_conn.getresponse()
                    if tunnel_resp.status != 200:
                        tunnel_conn.close()
                        raise ProxyError(
                            f"CONNECT tunnel to {host}:{port} via "
                            f"{proxy_host}:{proxy_port} failed: "
                            f"{tunnel_resp.status} {tunnel_resp.reason}"
                        )
                    # Upgrade the raw socket to TLS for the target host.
                    raw_sock = tunnel_conn.sock
                    conn = http.client.HTTPSConnection(
                        host, port=port, timeout=connect_timeout, context=ssl_ctx,
                    )
                    conn.sock = ssl_ctx.wrap_socket(raw_sock, server_hostname=host)
                    return conn
                except ProxyError:
                    raise
                except Exception as exc:
                    raise ProxyError(
                        f"Failed to establish CONNECT tunnel to {host}:{port} "
                        f"via {proxy_host}:{proxy_port}: {exc}",
                        error=exc,
                    ) from exc

            return http.client.HTTPSConnection(
                host, port=port, timeout=connect_timeout, context=ssl_ctx,
            )

        # Plain HTTP
        if proxy is not None:
            proxy_host = proxy.host
            proxy_port = proxy.port or (443 if proxy.scheme == "https" else 8080)
            conn = http.client.HTTPConnection(
                proxy_host, port=proxy_port, timeout=connect_timeout,
            )
            conn._ygg_proxy = True  # type: ignore[attr-defined]
            return conn

        return http.client.HTTPConnection(host, port=port, timeout=connect_timeout)

    @staticmethod
    def _proxy_auth_headers(proxy: URL) -> dict[str, str]:
        """Build ``Proxy-Authorization`` from userinfo in the proxy URL."""
        user = proxy.user
        if not user:
            return {}
        import base64
        password = proxy.password or ""
        cred = base64.b64encode(f"{user}:{password}".encode()).decode()
        return {"Proxy-Authorization": f"Basic {cred}"}

    def _get_connection(
        self,
        scheme: str,
        host: str,
        port: int,
        connect_timeout: Optional[float],
    ) -> http.client.HTTPConnection:
        """Pop an idle connection for *(scheme, host, port)* or build one."""
        key = (scheme, host, port)
        with self._lock:
            cached = self._connections.get(key)
            if cached:
                return cached.popleft()
        return self._build_connection(scheme, host, port, connect_timeout)

    def _release_connection(
        self,
        key: tuple[str, str, int],
        conn: http.client.HTTPConnection,
    ) -> None:
        """Return ``conn`` to the per-host idle cache or close it.

        Called by :meth:`HTTPResponse.release_conn` after a response is
        fully drained. Connections beyond ``pool_maxsize`` get closed
        instead of cached so a runaway caller can't leak sockets.
        """
        with self._lock:
            cached = self._connections.setdefault(key, collections.deque())
            if len(cached) < self.pool_maxsize:
                cached.append(conn)
                return
        try:
            conn.close()
        except Exception:
            pass

    def clear_connections(self) -> None:
        """Close every cached idle connection.

        Lifecycle convenience — closes the per-host sockets the session
        accumulated. Not called automatically; explicit cleanup is the
        caller's responsibility (or rely on process exit).
        """
        with self._lock:
            cached, self._connections = self._connections, {}
        for queue in cached.values():
            while queue:
                try:
                    queue.popleft().close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Direct HTTP send — no PoolManager indirection
    # ------------------------------------------------------------------

    def fetch(
        self,
        method: str,
        url: URL | str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        body: Any = None,
        timeout: Any = None,
        preload_content: bool = False,
        decode_content: bool = True,
        redirect: bool = True,
        tags: Optional[Mapping[str, str]] = None,
    ) -> HTTPResponse:
        """Low-level wire fetch — bypasses the cache / auth-refresh pipeline.

        Build a synthetic :class:`HTTPRequest` from ``method`` /
        ``url`` / ``headers`` / ``body`` and run it through the same
        retry + redirect machinery :meth:`_send_http` does for the
        regular :meth:`send` path. Useful when the caller just wants a
        raw byte stream off a URL — Databricks external-link readers
        feeding :func:`pa.input_stream` are the canonical case.
        """
        request = HTTPRequest.prepare(
            method=method,
            url=str(url) if isinstance(url, URL) else url,
            headers=dict(headers) if headers is not None else None,
            body=body,
        )
        return self._send_http(
            request,
            timeout=timeout,
            preload_content=preload_content,
            decode_content=decode_content,
            redirect=redirect,
            tags=tags,
        )

    def _send_http(
        self,
        request: HTTPRequest,
        *,
        timeout: Any = None,
        preload_content: bool = True,
        decode_content: bool = True,
        redirect: bool = True,
        tags: Optional[Mapping[str, str]] = None,
    ) -> HTTPResponse:
        """Drive one full HTTP request → response, retries + redirects included.

        :class:`HTTPSession` IS the connection pool — each retry attempt
        acquires a socket from :meth:`_get_connection`, fires
        ``conn.request`` + ``conn.getresponse`` once, and (on success)
        wraps the raw response in a high-level :class:`HTTPResponse`
        whose :meth:`release_conn` routes straight back into
        :meth:`_release_connection`. No intermediate transport class.
        """
        retries: Retry = self._retry.new()  # fresh history per call
        current_request = request
        visited_redirects = 0

        while True:
            try:
                response = self._send_once(
                    request=current_request,
                    timeout=timeout,
                    preload_content=preload_content,
                    decode_content=decode_content,
                    tags=tags,
                )
            except (socket.timeout, TimeoutError) as exc:
                url_str = current_request.url.to_string()
                wrapped: Exception = ReadTimeoutError(self, url_str, str(exc))
                retries = retries.increment(
                    method=current_request.method, url=url_str,
                    error=wrapped, _pool=self,
                )
                retries.sleep()
                continue
            except ssl.SSLError as exc:
                msg = str(exc)
                if "EOF" in msg or "UNEXPECTED_EOF" in msg or "Connection reset" in msg:
                    url_str = current_request.url.to_string()
                    wrapped = SSLError(msg)
                    wrapped.__cause__ = exc
                    retries = retries.increment(
                        method=current_request.method, url=url_str,
                        error=wrapped, _pool=self,
                    )
                    retries.sleep()
                    continue
                raise SSLError(msg) from exc
            except (OSError, http.client.HTTPException) as exc:
                url_str = current_request.url.to_string()
                wrapped = NewConnectionError(self, str(exc))
                retries = retries.increment(
                    method=current_request.method, url=url_str,
                    error=wrapped, _pool=self,
                )
                retries.sleep()
                continue

            # Redirect handling — drains the body, releases the socket,
            # rewrites method/body for 301/302/303 per RFC 7231.
            if redirect and response.status in self._REDIRECT_STATUSES:
                location = response.headers.get("Location")
                if location and visited_redirects < self._MAX_REDIRECTS:
                    response.drain_conn()
                    response.release_conn()
                    visited_redirects += 1
                    current_url = self._resolve_redirect(
                        current_request.url.to_string(), location,
                    )
                    if response.status in (301, 302, 303) and current_request.method.upper() != "HEAD":
                        redirect_headers = HTTPHeaders(current_request.headers)
                        redirect_headers.pop("Content-Length", None)
                        redirect_headers.pop("Content-Type", None)
                        current_request = current_request.copy(
                            url=current_url, method="GET", buffer=None,
                            headers=redirect_headers,
                        )
                    else:
                        current_request = current_request.copy(url=current_url)
                    continue

            # Retry on status_forcelist (5xx / 429 by default).
            if retries.is_retry(
                current_request.method,
                response.status,
                response.headers.get("Retry-After") is not None,
            ):
                try:
                    next_retries = retries.increment(
                        method=current_request.method,
                        url=current_request.url.to_string(),
                        response=response, _pool=self,
                    )
                except MaxRetryError:
                    if retries.raise_on_status:
                        raise
                    return response
                response.drain_conn()
                response.release_conn()
                next_retries.sleep(response=response)
                retries = next_retries
                continue

            return response

    def _send_once(
        self,
        *,
        request: HTTPRequest,
        timeout: Any,
        preload_content: bool,
        decode_content: bool,
        tags: Optional[Mapping[str, str]] = None,
    ) -> HTTPResponse:
        """Single wire send — one connection, one ``conn.getresponse``."""
        url = request.url
        scheme = url.scheme
        if scheme not in ("http", "https"):
            raise LocationValueError(f"Unsupported scheme: {scheme!r}")
        host = url.host
        if not host:
            raise LocationParseError(url.to_string())
        port = url.port or (443 if scheme == "https" else 80)
        path = url.path or "/"
        if url.query:
            path = f"{path}?{url.query}"

        connect_timeout, read_timeout = _resolve_timeout(timeout)
        connect_timeout = _floor_connect_timeout(connect_timeout)
        key = (scheme, host, port)
        conn = self._get_connection(scheme, host, port, connect_timeout)
        from_pool = conn.sock is not None

        # Plain HTTP through a proxy: send the absolute URL as the
        # request-target so the proxy knows where to forward.
        is_http_proxy = getattr(conn, "_ygg_proxy", False)
        if is_http_proxy:
            request_path = url.to_string()
        else:
            request_path = path

        try:
            # Establish the TCP+TLS connection with the connect timeout.
            # stdlib http.client only applies conn.timeout at connect() time,
            # so setting it after the socket exists is a no-op for reused
            # connections — and for fresh ones, request() would auto-connect
            # with conn.timeout which we must NOT overwrite with read_timeout.
            if not from_pool:
                conn.connect()
            # Switch the live socket to the read timeout for request IO +
            # response wait — the connect phase is done.
            if read_timeout is not None and conn.sock is not None:
                conn.sock.settimeout(read_timeout)
            send_headers = dict(request.headers) if request.headers else {}
            send_headers.setdefault(
                "Host", f"{host}:{port}" if port not in (80, 443) else host,
            )
            if is_http_proxy:
                proxy = self._resolve_proxy_for(scheme, host)
                if proxy:
                    send_headers.update(self._proxy_auth_headers(proxy))
            body = request.buffer.to_bytes() if request.buffer is not None else None
            if body is not None and "Content-Length" not in send_headers:
                send_headers["Content-Length"] = str(len(body))
            conn.request(request.method, request_path, body=body, headers=send_headers)
            raw = conn.getresponse()
        except socket.timeout as exc:
            try:
                conn.close()
            except Exception:
                pass
            raise ReadTimeoutError(self, url.to_string(), str(exc)) from exc
        except (OSError, http.client.HTTPException) as exc:
            try:
                conn.close()
            except Exception:
                pass
            # Stale pooled connection — the server closed the keep-alive
            # socket between requests. Retry once on a fresh connection
            # without charging the caller's retry budget.
            if from_pool:
                return self._send_once(
                    request=request,
                    timeout=timeout,
                    preload_content=preload_content,
                    decode_content=decode_content,
                    tags=tags,
                )
            raise
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

        return HTTPResponse.from_wire(
            request=request,
            raw=raw,
            session=self,
            connection=conn,
            pool_key=key,
            decode_content=decode_content,
            preload_content=preload_content,
            tags=tags,
        )

    @staticmethod
    def _resolve_redirect(current_url: str, location: str) -> str:
        """Resolve a redirect ``Location`` against the current URL."""
        if "://" in location:
            return location
        parts = urlsplit(current_url)
        if location.startswith("/"):
            return urlunsplit((parts.scheme, parts.netloc, location, "", ""))
        # Relative path — drop the trailing segment of the current path.
        base = parts.path.rsplit("/", 1)[0] + "/"
        return urlunsplit((parts.scheme, parts.netloc, base + location, "", ""))

    def _request_log_id(self, request: HTTPRequest) -> str:
        try:
            return request.xxh3_b64(url_safe=True)
        except Exception:
            return request.url.to_string()

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
            from yggdrasil.http_ import HTTPSession

            return HTTPSession(
                base_url=parsed,
                verify=verify,
                waiting=WaitingConfig.from_(waiting) if waiting is not None else None,
            )

        raise ValueError(f"Cannot build session from scheme: {parsed.scheme!r}")

    def insecure(self) -> "HTTPSession":
        """Return a session with SSL verification disabled.

        If ``self`` already has ``verify=False``, returns ``self``.
        Otherwise returns a new :class:`HTTPSession` with the same
        ``base_url`` / ``headers`` / ``waiting`` / ``auth`` / ``proxy``
        / ``no_proxy`` but ``verify=False``.

        Emits :class:`InsecureRequestWarning` on first construction.
        """
        if self.verify is False:
            return self
        return type(self)(
            base_url=self.base_url,
            verify=False,
            headers=self.headers,
            waiting=self.waiting,
            auth=self.auth,
            proxy=self.proxy,
            no_proxy=self.no_proxy,
        )

    def send(
        self,
        request: HTTPRequest,
        config: SendConfig | Mapping[str, Any] | None = None,
        *,
        wait: WaitingConfigArg = ...,
        raise_error: bool = ...,
        remote_cache: CacheConfig | Mapping[str, Any] | None = ...,
        local_cache: CacheConfig | Mapping[str, Any] | None = ...,
        cache_only: bool = ...,
        spark_session: Optional["SparkSession"] = ...,
        start: bool = True,
        **options,
    ) -> HTTPResponse:
        """Prepare, dispatch, and (optionally) await the response.

        Single-request entry point — always returns a :class:`Response`.
        adds no value.

        When ``spark_session`` is bound (or ``True`` / ``...`` resolved
        via :meth:`PyEnv.spark_session`), the wire send is fanned out
        to an executor through the same ``mapInArrow`` path
        :meth:`send_many` uses — the driver does not cross the wire.
        Otherwise the call runs synchronously on the local pool.

        ``start=True`` (default) fires the wire call. ``start=False``
        builds the prepared request + response shell without crossing
        the wire — the :class:`StatementExecutor` override uses the
        same knob to return an idled :class:`StatementResult` whose
        backend submission is deferred until
        :meth:`StatementResult.start` fires. Plain HTTP sessions don't
        need an idle :class:`Response` (the network call is
        synchronous), so the base raises a clean
        ``NotImplementedError`` via :meth:`_build_idle_response`.

        Per-request :attr:`HTTPRequest.send_config` is used as the
        base when no explicit *config* is passed — explicit kwargs
        still override individual fields.
        """
        overrides: dict[str, Any] = {**options}
        if wait is not ...:
            overrides["wait"] = wait
        if raise_error is not ...:
            overrides["raise_error"] = raise_error
        if remote_cache is not ...:
            overrides["remote_cache"] = remote_cache
        if local_cache is not ...:
            overrides["local_cache"] = local_cache
        if cache_only is not ...:
            overrides["cache_only"] = cache_only
        if spark_session is not ...:
            overrides["spark_session"] = spark_session
        base = config if config is not None else request.send_config
        cfg = SendConfig.from_(base, **overrides)
        # Per-request cache configs always win over session-level
        # fallbacks — stamp the request's own caches back on top.
        req_sc = request.send_config
        if req_sc is not None:
            merge_back: dict[str, Any] = {}
            if req_sc.local_cache is not None:
                merge_back["local_cache"] = req_sc.local_cache
            if req_sc.remote_cache is not None:
                merge_back["remote_cache"] = req_sc.remote_cache
            if merge_back:
                cfg = cfg.copy( **merge_back)
        request.send_config = cfg
        lc = cfg.local_cache
        if lc is not None and lc.tabular is None:
            lc.cache_tabular(session=self)
        if not start:
            return self._build_idle_response(request, cfg)
        for response in self._send_many(iter([request])):
            if cfg.raise_error:
                response.raise_for_status()
            return response

    def _build_idle_response(
        self,
        request: HTTPRequest,
        config: SendConfig,
    ) -> HTTPResponse:
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
        request: HTTPRequest,
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
            request.headers = HTTPHeaders()
        request.headers["Authorization"] = authorization
        # Mirror the refresh on the session-level header when the
        # session-wide handler is what we just ran — a per-request
        # override (request.auth) deliberately doesn't pollute the
        # session view.
        if handler is self.auth:
            self.headers["Authorization"] = authorization
        return self, True

    def prepare_request_before_send(self, request: HTTPRequest) -> HTTPRequest:
        """Session-wide request hook fired once per outbound request.

        Default returns *request* unchanged. Subclasses override to inject
        session-level concerns — auth, signing, correlation IDs, mandatory
        headers — that should apply to every request leaving this session.
        Runs in :meth:`_send` just before :meth:`_local_send`, so cache hits
        bypass it. Travels with the session into Spark workers via
        ``__getstate__`` / ``__setstate__``.
        """
        request.attach_session(self)
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

    def prepare_response_after_received(self, response: HTTPResponse) -> HTTPResponse:
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
        request: HTTPRequest,
    ) -> HTTPResponse:
        """Wire send — no cache logic, just prepare → send → post-process."""
        request = self.prepare_request_before_send(request)
        LOGGER.debug("Sending %s %s", request.method, request.url)
        response = self._local_send(request)
        response = self.prepare_response_after_received(response)
        LOGGER.info("Sent %s %s", request.method, request.url)
        return response

    # ------------------------------------------------------------------
    # Wire transport
    # ------------------------------------------------------------------

    def _local_send(
        self,
        request: HTTPRequest,
    ) -> HTTPResponse:
        config = request.send_config_or_default
        wait_cfg = config.wait if config.wait is not None else self.waiting

        result = self._wire_send(request, wait_cfg)

        # 403 → refresh auth and retry once. The pool's status_forcelist
        # covers 5xx / 429 transients; 403 is a deliberate auth signal
        # some vendors (Salesforce, M365 SharePoint, …) emit instead
        # of 401 when a previously-valid token has been silently
        # rotated upstream. Only worth retrying when an auth handler
        # is actually bound — otherwise the second attempt would
        # carry the same headers and 403 again.
        if result.status_code == 403 and (request.auth or self.auth) is not None:
            LOGGER.warning(
                "Refreshing auth after 403 for %s %s — retrying once",
                request.method, request.url,
            )
            _, refreshed = self.refresh_auth(request)  # force=True default
            if refreshed:
                result = self._wire_send(request, wait_cfg)

        x_current_page = result.headers.get("X-Current-Page")
        x_total_pages = result.headers.get("X-Last-Page")

        if x_current_page and x_total_pages:
            result = self._combine_paginated_pages(
                result=result,
                request=request,
                current_page=int(x_current_page),
                total_pages=int(x_total_pages),
                wait_cfg=wait_cfg,
                raise_error=config.raise_error,
            )

        if config.raise_error:
            result.raise_for_status()

        return result

    def _wire_send(
        self,
        request: HTTPRequest,
        wait_cfg: WaitingConfig,
    ) -> HTTPResponse:
        """Single wire-level send.

        :class:`HTTPSession` IS the pool now: :meth:`_send_http`
        returns the drained :class:`HTTPResponse` directly, so callers
        read ``X-Current-Page`` / ``X-Last-Page`` straight off
        ``response.headers`` without a parallel raw-response object.
        """
        result = self._send_http(
            request,
            timeout=wait_cfg.timeout_pool,
            preload_content=True,
            decode_content=False,
            redirect=True,
        )
        return result

    def _fetch_paginated_page(
        self,
        *,
        request: HTTPRequest,
        page_num: int,
        body_seed: bytes | None,
        wait_cfg: WaitingConfig,
        raise_error: bool,
    ) -> tuple[int, HTTPResponse]:
        page_url = request.url.add_param("page", str(page_num), replace=True)

        page_request = request.copy(
            url=page_url,
            buffer=Memory(binary=body_seed) if body_seed is not None else None,
        )

        page_result = self._send_http(
            page_request,
            timeout=wait_cfg.timeout_pool,
            preload_content=True,
            decode_content=False,
            redirect=True,
        )

        if raise_error:
            page_result.raise_for_status()

        return page_num, page_result

    def _combine_paginated_pages(
        self,
        *,
        result: HTTPResponse,
        request: HTTPRequest,
        current_page: int,
        total_pages: int,
        wait_cfg: WaitingConfig,
        raise_error: bool,
        pool: Optional[JobPoolExecutor | int] = None,
    ) -> HTTPResponse:
        if not isinstance(pool, JobPoolExecutor):
            with JobPoolExecutor.from_(pool) as parsed_pool:
                return self._combine_paginated_pages(
                    result=result,
                    request=request,
                    current_page=current_page,
                    total_pages=total_pages,
                    wait_cfg=wait_cfg,
                    raise_error=raise_error,
                    pool=parsed_pool,
                )

        from yggdrasil.lazy_imports import polars as pl

        init_df = result.to_polars(parse=True, lazy=False)
        if total_pages <= current_page:
            return result

        remaining_pages = list(range(current_page + 1, total_pages + 1))
        body_seed = request.buffer.to_bytes() if request.buffer else None

        def jobs():
            for pn in remaining_pages:
                yield Job.make(
                    self._fetch_paginated_page,
                    request=request,
                    page_num=pn,
                    body_seed=body_seed,
                    wait_cfg=wait_cfg,
                    raise_error=raise_error,
                )

        content_bytes = result.body_size
        frames = [init_df]
        for job_result in pool.as_completed(
            jobs(),
            ordered=False,
            max_in_flight=len(remaining_pages),
            cancel_on_exit=False,
            shutdown_on_exit=False,
            raise_error=True,
        ):
            _, page_resp = job_result.result
            content_bytes += page_resp.body_size
            frames.append(page_resp.to_polars(parse=True, lazy=False))

        final_df = pl.concat(frames, how="diagonal_relaxed", rechunk=False)
        combined_table = final_df.to_arrow(compat_level=pl.CompatLevel.newest())

        total_rows = combined_table.num_rows
        if total_rows > 0 and content_bytes > _PAGINATED_RECHUNK_BYTE_SIZE:
            max_chunksize = max(
                1, total_rows * _PAGINATED_RECHUNK_BYTE_SIZE // content_bytes,
            )
            batches = combined_table.to_batches(max_chunksize=max_chunksize)
        else:
            batches = combined_table.combine_chunks().to_batches()

        new_holder = Memory()
        new_holder.media_type = MediaTypes.ARROW_IPC
        with ArrowIPCFile(holder=new_holder, owns_holder=False, mode="wb") as new_buffer:
            new_buffer.write_arrow_batches(
                iter(batches),
                compression="zstd",
            )

        result.buffer.close()
        result.buffer = new_holder
        result.set_media_type(MediaTypes.ARROW_IPC)

        result.update_tags({
            "page_start": str(current_page),
            "page_total": str(total_pages),
        })

        return result

    def send_many(
        self,
        requests: Iterator[HTTPRequest],
        config: SendConfig | Mapping[str, Any] | None = None,
        *,
        wait: WaitingConfigArg = ...,
        raise_error: bool = ...,
        remote_cache: CacheConfig | Mapping[str, Any] | None = ...,
        local_cache: CacheConfig | Mapping[str, Any] | None = ...,
        cache_only: bool = ...,
        spark_session: Optional["SparkSession"] = ...,
        batch_size: int | None = None,
        ordered: bool = False,
        max_in_flight: int | None = None,
        max_batch_ttl: float | None = None,
        **options,
    ) -> Iterator[HTTPResponse]:
        """Stream responses for a batch of requests.

        Batch orchestration kwargs (``batch_size``, ``ordered``,
        ``max_in_flight``, ``max_batch_ttl``) control chunking and
        concurrency; everything else is folded into a :class:`SendConfig`
        that gets stamped on each request.

        Send-config kwargs default to ``...`` (Ellipsis).  When left
        unset the per-request :attr:`HTTPRequest.send_config` is
        preserved; an explicit value overrides that field on every
        request in the batch.
        """
        overrides: dict[str, Any] = {**options}
        if wait is not ...:
            overrides["wait"] = wait
        if raise_error is not ...:
            overrides["raise_error"] = raise_error
        if remote_cache is not ...:
            overrides["remote_cache"] = remote_cache
        if local_cache is not ...:
            overrides["local_cache"] = local_cache
        if cache_only is not ...:
            overrides["cache_only"] = cache_only
        if spark_session is not ...:
            overrides["spark_session"] = spark_session

        cfg = SendConfig.from_(config, **overrides)

        batch_kw: dict[str, Any] = {}
        if batch_size is not None:
            batch_kw["batch_size"] = batch_size
        if ordered:
            batch_kw["ordered"] = ordered
        if max_in_flight is not None:
            batch_kw["max_in_flight"] = max_in_flight
        if max_batch_ttl is not None:
            batch_kw["max_batch_ttl"] = max_batch_ttl

        lc = cfg.local_cache
        if lc is not None and lc.tabular is None:
            lc.cache_tabular(session=self)

        def _stamp(reqs: Iterator[HTTPRequest]) -> Iterator[HTTPRequest]:
            for r in reqs:
                if r.send_config is None:
                    r.send_config = cfg
                elif overrides:
                    req_sc = r.send_config
                    merged = SendConfig.from_(req_sc, **overrides)
                    # Per-request cache configs always win over
                    # call-level overrides.
                    merge_back: dict[str, Any] = {}
                    if req_sc.local_cache is not None:
                        merge_back["local_cache"] = req_sc.local_cache
                    if req_sc.remote_cache is not None:
                        merge_back["remote_cache"] = req_sc.remote_cache
                    if merge_back:
                        merged = merged.copy( **merge_back)
                    r.send_config = merged
                yield r

        return self._send_many(_stamp(requests), **batch_kw)

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

    def _send_batch(
        self,
        reqs: list[HTTPRequest],
        cfg: "SendConfig",
        *,
        ordered: bool = False,
        max_in_flight: int | None = None,
    ) -> HTTPResponseBatch:
        """Process one config group: split cache → fetch misses → writeback."""
        return HTTPResponseBatch(cfg, reqs, session=self)._fetch(
            ordered=ordered, max_in_flight=max_in_flight,
        )

    @staticmethod
    def _group_by_config(
        batch: list[HTTPRequest],
    ) -> dict["SendConfig", list[HTTPRequest]]:
        """Group requests by send config."""
        groups: dict[SendConfig, list[HTTPRequest]] = {}
        for r in batch:
            cfg = r.send_config_or_default
            groups.setdefault(cfg, []).append(r)
        return groups

    def _fetch_misses(
        self,
        misses: list[HTTPRequest],
        *,
        ordered: bool = False,
        max_in_flight: int | None = None,
    ) -> Iterator[HTTPResponse]:
        """Send misses through the job pool — no caching, just wire calls."""
        pool = self.job_pool
        for result in pool.as_completed(
            (Job.make(self._send, r) for r in misses),
            ordered=ordered,
            max_in_flight=max_in_flight or self.pool_maxsize,
            cancel_on_exit=False,
            shutdown_on_exit=False,
            raise_error=True,
        ):
            yield result.result

    @staticmethod
    def _run_concurrently(
        tasks: "Sequence[Callable[[], Any]]",
        *,
        max_workers: "int | None" = None,
        thread_name_prefix: str = "ygg-session",
    ) -> None:
        """Run *tasks* in parallel, re-raising the first failure.

        Used by the cache-write path (``_backfill_local_cache``,
        stage 4 remote persist) to fan out independent inserts
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

    def _send_many(
        self,
        requests: Iterator[HTTPRequest],
        **batch_kw: Any,
    ) -> Iterator[HTTPResponse]:
        """Stream responses, flattening the per-chunk :class:`HTTPResponseBatch`.

        In Spark mode ``raise_error`` is applied at the driver-iteration
        boundary so a single failure doesn't poison a whole partition.
        """
        for batch in self._send_many_batches(requests, **batch_kw):
            yield from batch.responses()

    def send_many_batches(
        self,
        requests: Iterator[HTTPRequest],
        *,
        wait: WaitingConfigArg = ...,
        raise_error: bool = ...,
        remote_cache: CacheConfig | Mapping[str, Any] | None = ...,
        local_cache: CacheConfig | Mapping[str, Any] | None = ...,
        cache_only: bool = ...,
        spark_session: Optional["SparkSession"] = ...,
        batch_size: int | None = None,
        ordered: bool = False,
        max_in_flight: int | None = None,
        max_batch_ttl: float | None = None,
    ) -> Iterator[HTTPResponseBatch]:
        """Yield one :class:`HTTPResponseBatch` per processed chunk.

        Send-config kwargs default to ``...`` (Ellipsis).  When left
        unset the per-request :attr:`HTTPRequest.send_config` is
        preserved; an explicit value overrides that field on every
        request in the batch.
        """
        overrides: dict[str, Any] = {}
        if wait is not ...:
            overrides["wait"] = wait
        if raise_error is not ...:
            overrides["raise_error"] = raise_error
        if remote_cache is not ...:
            overrides["remote_cache"] = remote_cache
        if local_cache is not ...:
            overrides["local_cache"] = local_cache
        if cache_only is not ...:
            overrides["cache_only"] = cache_only
        if spark_session is not ...:
            overrides["spark_session"] = spark_session

        if overrides:
            cfg = SendConfig.from_(None, **overrides)

            def _stamp(reqs: Iterator[HTTPRequest]) -> Iterator[HTTPRequest]:
                for r in reqs:
                    if r.send_config is None:
                        r.send_config = cfg
                    else:
                        req_sc = r.send_config
                        merged = SendConfig.from_(req_sc, **overrides)
                        merge_back: dict[str, Any] = {}
                        if req_sc.local_cache is not None:
                            merge_back["local_cache"] = req_sc.local_cache
                        if req_sc.remote_cache is not None:
                            merge_back["remote_cache"] = req_sc.remote_cache
                        if merge_back:
                            merged = merged.copy( **merge_back)
                        r.send_config = merged
                    yield r

            requests = _stamp(requests)

        yield from self._send_many_batches(
            requests,
            batch_size=batch_size,
            ordered=ordered,
            max_in_flight=max_in_flight,
            max_batch_ttl=max_batch_ttl,
        )

    def _send_many_batches(
        self,
        requests: Iterator[HTTPRequest],
        *,
        batch_size: int | None = None,
        max_batch_size: int | None = None,
        max_in_flight: int | None = None,
        ordered: bool = False,
        max_batch_ttl: float | None = DEFAULT_MAX_BATCH_TTL,
    ) -> Iterator[HTTPResponseBatch]:
        """Yield one :class:`HTTPResponseBatch` per processed chunk."""
        pool = self.job_pool
        if batch_size:
            eff_batch_size = batch_size
        else:
            eff_batch_size = min(max_batch_size or 1024, pool.max_workers * 10)

        ttl = max_batch_ttl
        chunk_index = 0
        total_cache_hits = 0
        total_network = 0
        total_failed = 0
        total_ignored = 0

        def _batched(
            it: Iterator[HTTPRequest],
            n: int,
            ttl_seconds: float | None,
        ) -> Iterator[list[HTTPRequest]]:
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
            buf: list[HTTPRequest] = []
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

        chunks = _batched(requests, eff_batch_size, ttl)

        for chunk in chunks:
            if not chunk:
                continue

            chunk_index += 1
            groups = self._group_by_config(chunk)
            LOGGER.debug(
                "Processing chunk #%d (requests=%d, groups=%d)",
                chunk_index, len(chunk), len(groups),
            )

            for cfg, reqs in groups.items():
                batch = self._send_batch(
                    reqs, cfg,
                    ordered=ordered, max_in_flight=max_in_flight,
                )
                total_cache_hits += len(reqs) - len(batch.misses)
                total_network += batch.new_tabular.count() if batch.new_tabular else 0
                total_failed += batch.failed_count
                total_ignored += batch.ignored_count
                yield batch

        LOGGER.info(
            "Finished send_many pipeline (chunks=%d, cache_hits=%d, "
            "network=%d, failed=%d, ignored=%d)",
            chunk_index, total_cache_hits,
            total_network, total_failed, total_ignored,
        )

    def get(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: IO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> HTTPResponse | HTTPRequest:
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
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def post(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: IO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> HTTPResponse | HTTPRequest:
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
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def put(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: IO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> HTTPResponse | HTTPRequest:
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
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def patch(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: IO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> HTTPResponse | HTTPRequest:
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
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def delete(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: IO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> HTTPResponse | HTTPRequest:
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
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def head(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: IO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> HTTPResponse | HTTPRequest:
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
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def options(
        self,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: IO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> HTTPResponse | HTTPRequest:
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
            normalize=normalize,
            remote_cache=remote_cache,
            local_cache=local_cache,
            send=send,
            **options,
        )

    def request(
        self,
        method: str,
        url: URL | str | None = None,
        *,
        config: SendConfig | Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: IO | bytes | None = None,
        data: Any = None,
        tags: Mapping[str, str] | None = None,
        json: Any | None = None,
        cookies: "Mapping[str, str] | None" = None,
        wait: WaitingConfigArg = None,
        timeout: WaitingConfigArg = None,
        raise_error: bool = True,
        normalize: bool = True,
        remote_cache: CacheConfig | Mapping[str, Any] | None = None,
        local_cache: CacheConfig | Mapping[str, Any] | None = None,
        send: bool = True,
        **options,
    ) -> HTTPResponse | HTTPRequest:
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
                    "native IO/bytes path."
                )
            body, form_content_type = _encode_request_data(data)
            if form_content_type is not None:
                merged = HTTPHeaders(headers) if headers else HTTPHeaders()
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
                merged = HTTPHeaders(headers) if headers else HTTPHeaders()
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

        if not send:
            return prepared

        return self.send(
            prepared,
            config=config,
            wait=wait,
            raise_error=raise_error,
            remote_cache=remote_cache,
            local_cache=local_cache,
            **options,
        )

    def prepare_request(
        self,
        method: str,
        url: URL | str | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: IO | bytes | None = None,
        tags: Mapping[str, str] | None = None,
        *,
        json: Any | None = None,
        normalize: bool = True,
        send_config: SendConfig | None = None,
        **send_kwargs: Any,
    ) -> HTTPRequest:
        full_url: URL | str | None = url

        if self.base_url:
            full_url = self.base_url.join(url) if url else self.base_url
        elif url is None:
            raise ValueError("url is required when base_url is not set on the session.")

        if params:
            parsed = URL.from_(full_url, normalize=normalize)
            full_url = parsed.with_query_items(params)

        if send_kwargs:
            send_config = SendConfig.from_(send_config, **send_kwargs)

        if send_config is not None and send_config.local_cache is not None:
            lc = send_config.local_cache
            if lc.tabular is None and (
                lc.received_from is not None or lc.received_to is not None
            ):
                object.__setattr__(lc, "tabular", self.local_cache())

        request = HTTPRequest.prepare(
            method=method,
            url=full_url,
            headers=headers,
            body=body,
            tags=tags,
            json=json,
            normalize=normalize,
            send_config=send_config,
            session=self
        )
        return request
