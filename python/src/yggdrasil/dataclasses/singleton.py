"""Hash-keyed singleton base class — process-wide instance cache.

Pairs ``__new__`` with an :class:`ExpiringDict` so two constructor
calls that produce the same hashable key collapse to one instance.
The singleton key is whatever :meth:`Singleton._singleton_key`
projects out of the constructor arguments — by default ``(cls,
args, sorted kwargs items)``, which works for any constructor
whose arguments are hashable.

Subclasses with non-trivial constructors override
:meth:`_singleton_key` to:

- drop fields that don't participate in instance identity;
- normalize unhashable inputs (``dict`` → ``frozenset`` of items,
  ``list`` → ``tuple``);
- canonicalize semantically equivalent forms (e.g. trailing slash
  on a URL) so different spellings of the same identity collapse.

``__init__`` must be idempotent: Python re-invokes it on every
constructor call, even when ``__new__`` returns the cached
instance, so subclasses guard with ``if getattr(self,
"_initialized", False): return`` to avoid clobbering live state
on the second pass.

Pickling is wired through :meth:`__getstate__` / :meth:`__setstate__`:
attribute names listed in ``_TRANSIENT_STATE_ATTRS`` are excluded
from the payload (live SDK clients, sockets, lazy service
caches), and ``__setstate__`` short-circuits when the receiver is
already initialized so unpickling collapses to the live
in-process singleton.
"""

from __future__ import annotations

import logging
from threading import RLock
from typing import Any, ClassVar

from .expiring import ExpiringDict

__all__ = ["Singleton"]


LOGGER = logging.getLogger(__name__)


class Singleton:
    """Base class that caches one instance per hashable constructor key.

    The cache is shared across every subclass (the default
    ``_singleton_key`` includes ``cls`` so different subclasses can
    coexist in one dict). A subclass that wants a private cache
    re-declares its own ``_INSTANCES`` / ``_INSTANCES_LOCK``
    ClassVars.
    """

    # ``default_ttl=None`` keeps singletons live for the process
    # lifetime — same shape as the MSAL / Databricks SDK
    # ``_INSTANCES`` caches throughout the codebase.
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(default_ttl=None)
    _INSTANCES_LOCK: ClassVar[RLock] = RLock()

    # Class-level default for the per-call ``singleton_ttl`` kwarg.
    # ``...`` (the default on this base) keeps caching strictly
    # opt-in — subclasses that always want process-lifetime
    # caching set this to ``None``; subclasses that want a
    # bounded cache lifetime set it to a number of seconds.
    _SINGLETON_TTL: ClassVar[Any] = ...

    # Attribute names that don't survive pickling. Subclasses
    # extend with their own non-picklable handles (live SDK clients,
    # connection pools, lazy service caches).
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        """Project constructor inputs into a hashable singleton key.

        Default: ``(cls, args, tuple(sorted(kwargs.items())))``.
        Override when constructor args contain unhashable values
        (``dict``, ``list``, ``set``) or when only a subset of args
        contributes to instance identity.
        """
        return (cls, args, tuple(sorted(kwargs.items())))

    def __new__(
        cls,
        *args: Any,
        singleton_ttl: "int | None" = ...,
        **kwargs: Any,
    ) -> "Singleton":
        # ``singleton_ttl`` is the opt-in cache switch:
        #   - omitted (``...``) → fall back to the subclass's
        #     ``_SINGLETON_TTL`` ClassVar (default ``...`` on the
        #     base = no caching at all).
        #   - ``False``          → skip the cache for this call, even
        #     when the subclass default would cache. Used by hot
        #     listing paths (``iterdir`` / ``_ls`` / ``glob``) that
        #     produce many short-lived child instances we don't want
        #     to keep around. The instance is still stamped with its
        #     singleton key so :meth:`to_singleton` can promote it
        #     into the cache later if the caller decides to keep it.
        #   - ``None``           → register without expiry (live for
        #     the process lifetime); same shape as the long-running
        #     MSAL / Databricks SDK ``_INSTANCES`` caches.
        #   - ``int`` (seconds)  → register with that TTL; the entry
        #     auto-evicts after the window so callers building one
        #     instance per short-lived workload don't leak.
        if singleton_ttl is ...:
            singleton_ttl = cls._SINGLETON_TTL

        key = cls._singleton_key(*args, **kwargs)

        if singleton_ttl is ... or singleton_ttl is False:
            instance = super().__new__(cls)
            object.__setattr__(instance, "_singleton_key_", key)
            return instance

        with cls._INSTANCES_LOCK:
            existing = cls._INSTANCES.get(key)
            if existing is not None:
                return existing
            instance = super().__new__(cls)
            object.__setattr__(instance, "_singleton_key_", key)
            # ``ExpiringDict.set`` reads bare ``int`` as nanoseconds —
            # promote to float so the user-facing seconds contract
            # holds. ``None`` passes through unchanged (no expiry).
            ttl_arg = (
                float(singleton_ttl)
                if isinstance(singleton_ttl, int) and not isinstance(singleton_ttl, bool)
                else singleton_ttl
            )
            cls._INSTANCES.set(key, instance, ttl=ttl_arg)
            return instance

    def to_singleton(self, ttl: Any = ...) -> "Singleton":
        """Promote this instance into the per-class ``_INSTANCES`` cache.

        Hot listing paths (``iterdir`` / ``_ls`` / ``glob``) build
        children with ``singleton_ttl=False`` so the bounded cache
        doesn't fill up with thousands of short-lived entries. When a
        caller decides one of those children is worth keeping around
        (handing it to a long-running worker, returning it from an
        API), :meth:`to_singleton` registers ``self`` into the cache
        so the next constructor call with the same key collapses to
        the same instance.

        ``ttl`` defaults to the subclass's ``_SINGLETON_TTL``
        (``...`` = no caching, ``None`` = process lifetime, or a
        seconds count). When a different instance is already cached
        under this key, that pre-existing one wins and is returned
        unchanged — the cache is the source of truth.
        """
        if ttl is ...:
            ttl = type(self)._SINGLETON_TTL
        if ttl is ... or ttl is False:
            return self

        key = getattr(self, "_singleton_key_", None)
        if key is None:
            # Instance bypassed ``__new__`` (e.g. raw ``object.__new__``
            # or a hand-rolled deserialiser); without the key we can't
            # register it. Leave it uncached.
            return self

        cls = type(self)
        with cls._INSTANCES_LOCK:
            existing = cls._INSTANCES.get(key)
            if existing is not None:
                return existing
            ttl_arg = (
                float(ttl)
                if isinstance(ttl, int) and not isinstance(ttl, bool)
                else ttl
            )
            cls._INSTANCES.set(key, self, ttl=ttl_arg)
            return self

    def _invalidate_singleton(self, remove_global: bool = True) -> None:
        """Pop ``self`` from the per-class ``_INSTANCES`` cache.

        Mutating ops on a Singleton-cached object (writes, deletes,
        schema invalidations on a Databricks table, ``put_object`` on
        an :class:`S3Path`) want to make sure the next caller asking
        for the same key gets a fresh build rather than collapsing
        onto this stale handle — that's what ``remove_global=True``
        (the default) does. The pop is :meth:`identity-guarded`:
        only an entry that still points at ``self`` is removed, so
        a concurrent re-construction that already raced past this
        thread is left alone.

        ``remove_global=False`` is a no-op. The keyword exists so
        subclass invalidators (``_invalidate_singleton``,
        ``_invalidate_entity_tag_cache``, …) can offer the same
        switch without branching at the call site.
        """
        if not remove_global:
            return
        key = getattr(self, "_singleton_key_", None)
        if key is None:
            return
        cls = type(self)
        with cls._INSTANCES_LOCK:
            cached = cls._INSTANCES.get(key)
            if cached is self:
                cls._INSTANCES.pop(key, None)
        LOGGER.debug("Invalidated singleton for %r", self)

    def __hash__(self) -> int:
        return hash(getattr(self, "_singleton_key_", id(self)))

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return (
            getattr(self, "_singleton_key_", id(self))
            == getattr(other, "_singleton_key_", id(other))
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items()
            if k not in self._TRANSIENT_STATE_ATTRS
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        # ``__new__`` may have returned a live in-process singleton
        # already populated by an earlier construction or unpickle —
        # leave it untouched so live SDK handles, lazy caches, and
        # in-flight init state survive.
        if getattr(self, "_initialized", False):
            return
        self.__dict__.update(state)
        for attr in self._TRANSIENT_STATE_ATTRS:
            self.__dict__.setdefault(attr, None)
