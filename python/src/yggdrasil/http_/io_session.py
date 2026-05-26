"""Abstract :class:`Session` shell — singleton + pickle + concurrency-pool plumbing.

This module hosts only the generic per-transport machinery every concrete
session needs: process-lifetime singleton caching keyed off the post-init
``__dict__``, pickle/getnewargs/setstate that round-trips that identity
across process boundaries, and a lazy :class:`JobPoolExecutor` for any
parallel work the subclass dispatches.

Transport-specific surface — URL handling, headers, auth, verb methods,
``send`` / ``send_many``, local/remote cache pipeline, Spark integration —
lives on the concrete subclass:

* :class:`yggdrasil.http_.HTTPSession` for HTTP/HTTPS;
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

import logging
import pathlib
import threading
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional

from yggdrasil.concurrent.threading import JobPoolExecutor
from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.url import URL

if TYPE_CHECKING:
    from yggdrasil.io.nested.folder_path import FolderPath

__all__ = ["Session"]


LOGGER = logging.getLogger(__name__)


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
    # :class:`HTTPSession` to :class:`PreparedRequest` / :class:`Response`
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
        self._local_cache: Optional["FolderPath"] = None
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

    def local_cache(self) -> "FolderPath":
        """Return the session-scoped local cache folder, creating the directory on first access.

        The folder lives under ``~/.cache/http/<host>/<path>``
        when :attr:`base_url` is set (so different APIs on the same machine
        don't collide), or ``~/.cache/http/default`` otherwise.

        Thread-safe: the directory is created under the session lock and
        the resulting :class:`FolderPath` is cached for the lifetime of
        the singleton.
        """
        cached = getattr(self, "_local_cache", None)
        if cached is not None:
            return cached
        with self._lock:
            cached = getattr(self, "_local_cache", None)
            if cached is not None:
                return cached
            from yggdrasil.io.nested.folder_path import FolderPath
            from yggdrasil.io.path import Path

            root = pathlib.Path.home() / ".cache" / "http"
            base_url = getattr(self, "base_url", None)
            host = getattr(base_url, "host", None) if base_url is not None else None
            if not host:
                folder = root / "default"
            else:
                url_path = (getattr(base_url, "path", "") or "").strip("/")
                folder = root / host / url_path if url_path else root / host
            path = Path.from_(folder)
            self._local_cache: "FolderPath" = FolderPath(path=path)
            return self._local_cache
