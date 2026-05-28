"""
yggdrasil/dataclasses/expiring.py

Thread-safe expiring cache primitives.

Classes
-------
Expiring[T]
    Abstract base: subclasses implement ``_refresh()`` to reload a single value
    on demand.

ExpiringDict[K, V]
    Thread-safe dict where every key has an individual TTL.  No subclassing
    required — pass a ``refresher`` callable if per-key auto-refresh is wanted.
    Pass ``on_evict`` to receive notifications when an entry leaves the cache,
    so values that own external resources (file handles, spilled IO temp
    files, GPU buffers, …) can release them deterministically.

Change log vs previous version
- ``Expiring``: no ``new_instance`` callable field; subclasses implement
  ``_refresh()``.
- ``ExpiringDict``: new; built on the same time-utils / lock discipline as
  ``Expiring``.
- ``ExpiringDict``: new ``on_evict`` callback fires for every removal path
  (explicit delete, TTL expiry sweep, capacity eviction, refresher replace,
  pop, clear, etc). Callback runs OUTSIDE the lock so it can do real work
  without serializing other dict ops; exceptions are caught and discarded
  so a bad callback can't poison the cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from time import time_ns
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

__all__ = [
    # single-value cache
    "Expiring",
    "ExpiredError",
    "RefreshResult",
    # dict cache
    "ExpiringDict",
    # time utils
    "now_utc_ns",
    "datetime_to_epoch_ns",
    "timedelta_to_ns",
]

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
_IntLike = Union[int, str]

# ``...`` (Ellipsis) is the project-wide "missing / unset" sentinel
# — see the convention note in ``AGENTS.md`` / ``CLAUDE.md``. Used as
# a kwarg default and as a ``dict.get(...)`` miss marker so callers
# can distinguish "key absent" from "key present with value None".

# 15 minutes expressed in nanoseconds — the background-purge check interval.
_PURGE_INTERVAL_NS: int = 15 * 60 * 1_000_000_000


# ═══════════════════════════════════════════════════════════
# Time utilities  (shared by both Expiring and ExpiringDict)
# ═══════════════════════════════════════════════════════════

def now_utc_ns() -> int:
    """UTC epoch nanoseconds (monotonic-wall hybrid via time_ns)."""
    return time_ns()


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _coerce_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def datetime_to_epoch_ns(dt: datetime) -> int:
    """datetime → epoch ns (microsecond precision ⇒ trailing *000)."""
    dt_utc = _coerce_utc(dt)
    delta = dt_utc - _EPOCH
    return (
        delta.days * 86_400 * 1_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
    )


def timedelta_to_ns(td: timedelta) -> int:
    """timedelta → ns (microsecond precision ⇒ trailing *000)."""
    return (
        td.days * 86_400 * 1_000_000_000
        + td.seconds * 1_000_000_000
        + td.microseconds * 1_000
    )


# ── casting helpers ──────────────────────────────────────────

def _to_intlike(x: Any, name: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise TypeError(f"{name} must be int-like, got {type(x).__name__}") from e


def _to_epoch_ns(x: Any, name: str) -> int:
    if isinstance(x, datetime):
        return datetime_to_epoch_ns(x)
    return _to_intlike(x, name)


def _to_ttl_ns(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, timedelta):
        return timedelta_to_ns(x)
    return _to_intlike(x, "ttl_ns")


def _to_expires_at_ns(x: Any, *, now_ns: int) -> Optional[int]:
    """
    Convert an expiry input to an absolute epoch-ns timestamp.

    Accepts:
    - ``None`` → no expiry
    - ``int``-like absolute epoch ns
    - ``datetime`` absolute time
    - ``timedelta`` → now + delta
    """
    if x is None:
        return None
    if isinstance(x, timedelta):
        return now_ns + timedelta_to_ns(x)
    if isinstance(x, datetime):
        return datetime_to_epoch_ns(x)
    return _to_intlike(x, "expires_at_ns")


def _ttl_ns_to_seconds(ttl_ns: Optional[int]) -> Optional[float]:
    return None if ttl_ns is None else ttl_ns / 1_000_000_000


def _seconds_to_ns(seconds: Optional[float]) -> Optional[int]:
    return None if seconds is None else int(seconds * 1_000_000_000)


# ═══════════════════════════════════════════════════════════
# Expiring[T]  — single-value abstract cache
# ═══════════════════════════════════════════════════════════

class ExpiredError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class RefreshResult(Generic[T]):
    """
    Output of ``_refresh()``.

    Provide either:
    - ``ttl_ns`` (duration) OR
    - ``expires_at_ns`` (absolute time)

    If both are ``None`` → non-expiring.
    """
    value: Optional[T]
    created_at_ns: int | None = None
    ttl_ns: int | None = None
    expires_at_ns: int | None = None

    @classmethod
    def make(
        cls,
        value: Optional[T] = None,
        *,
        created_at_ns: Optional[_IntLike] = None,
        ttl_ns: Optional[Union[_IntLike, timedelta]] = None,
        expires_at_ns: Optional[_IntLike] = None,
    ) -> "RefreshResult[T]":
        """Convenience constructor with defaults + light casting."""
        created = None if created_at_ns is None else _to_intlike(created_at_ns, "created_at_ns")
        ttl = _to_ttl_ns(ttl_ns)
        exp = None if expires_at_ns is None else _to_intlike(expires_at_ns, "expires_at_ns")
        return cls(value=value, created_at_ns=created, ttl_ns=ttl, expires_at_ns=exp)


@dataclass(slots=True)
class Expiring(Generic[T], ABC):
    """
    Abstract thread-safe expiring cache holder.

    Subclasses implement ``_refresh()`` which is called when ``.value`` is
    accessed and the cache is expired.
    """

    _value: Optional[T] = None
    _created_at_ns: int = field(default_factory=now_utc_ns, repr=False)
    _expires_at_ns: Optional[int] = field(default=None, repr=False)
    _ttl_ns: Optional[int] = field(default=None, repr=False)

    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    # ── abstract ────────────────────────────────────────────

    @abstractmethod
    def _refresh(self) -> RefreshResult[T]:
        """
        Fetch a fresh value and return refresh metadata.

        Called **outside** the internal lock.
        Must not mutate ``self`` directly — return a ``RefreshResult`` instead.
        """

    # ── constructors ────────────────────────────────────────

    @classmethod
    def create(
        cls,
        value: Optional[Union[T, RefreshResult[T]]] = None,
        *args,
        ttl: Optional[Union[_IntLike, timedelta]] = None,
        expires_at: Optional[Union[_IntLike, datetime, timedelta]] = None,
        created_at: Optional[Union[_IntLike, datetime]] = None,
        **kwargs,
    ) -> "Expiring[T]":
        if isinstance(value, RefreshResult):
            created_at = value.created_at_ns
            exp_val = value.expires_at_ns
            ttl_val = value.ttl_ns
        else:
            created_at = _to_epoch_ns(
                created_at if created_at is not None else now_utc_ns(), "now_ns"
            )
            ttl_val = _to_ttl_ns(ttl)
            exp_val = _to_expires_at_ns(expires_at, now_ns=created_at)

        if exp_val is None and created_at is not None and ttl_val is not None:
            exp_val = created_at + ttl_val

        return cls(  # type: ignore[misc]
            _value=value,
            _created_at_ns=created_at,
            _expires_at_ns=exp_val,
            _ttl_ns=ttl_val,
            *args,
            **kwargs,
        )

    # ── value property ───────────────────────────────────────

    @property
    def value(self) -> Optional[T]:
        """Cached value with auto-refresh on expiry."""
        if self._value is None:
            result = self._refresh()
            self._apply_refresh_result(result)
            with self._lock:
                return self._value

        now_val = now_utc_ns()
        with self._lock:
            if not self._is_expired_locked(now_val):
                return self._value

        result = self._refresh()
        self._apply_refresh_result(result)
        with self._lock:
            return self._value

    @value.setter
    def value(self, v: Optional[T]) -> None:
        with self._lock:
            self._value = v

    # ── tuning properties ────────────────────────────────────

    @property
    def ttl_ns(self) -> Optional[int]:
        with self._lock:
            return self._ttl_ns

    @ttl_ns.setter
    def ttl_ns(self, v: Any) -> None:
        ttl_val = _to_ttl_ns(v)
        with self._lock:
            self._ttl_ns = ttl_val
            if ttl_val is None:
                return
            self._expires_at_ns = self._created_at_ns + ttl_val

    @property
    def expires_at_ns(self) -> Optional[int]:
        with self._lock:
            return self._expires_at_ns

    @expires_at_ns.setter
    def expires_at_ns(self, v: Any) -> None:
        with self._lock:
            self._expires_at_ns = _to_expires_at_ns(v, now_ns=now_utc_ns())

    # ── convenience ──────────────────────────────────────────

    def is_expired(self, *, now_ns: Optional[Union[_IntLike, datetime]] = None) -> bool:
        now_val = _to_epoch_ns(now_ns if now_ns is not None else now_utc_ns(), "now_ns")
        with self._lock:
            return self._is_expired_locked(now_val)

    def refresh(self) -> None:
        """Force refresh now (same as accessing ``.value`` and discarding it)."""
        _ = self.value

    # ── internals ────────────────────────────────────────────

    def _apply_refresh_result(self, rr: RefreshResult[T]) -> None:
        created = int(rr.created_at_ns) if rr.created_at_ns is not None else now_utc_ns()
        ttl = int(rr.ttl_ns) if rr.ttl_ns is not None else None
        exp = int(rr.expires_at_ns) if rr.expires_at_ns is not None else None
        if exp is None and ttl is not None:
            exp = created + ttl
        with self._lock:
            self._value = rr.value
            self._created_at_ns = created
            self._ttl_ns = ttl
            self._expires_at_ns = exp

    def _effective_expires_at_ns_locked(self) -> Optional[int]:
        if self._expires_at_ns is not None:
            return int(self._expires_at_ns)
        if self._ttl_ns is None:
            return None
        return int(self._created_at_ns) + int(self._ttl_ns)

    def _is_expired_locked(self, now_val: int) -> bool:
        exp = self._effective_expires_at_ns_locked()
        return exp is not None and now_val >= exp


# ═══════════════════════════════════════════════════════════
# ExpiringDict[K, V]  — per-key TTL dict
# ═══════════════════════════════════════════════════════════

class ExpiringDict(Generic[K, V]):
    """
    Thread-safe dictionary where every key carries an individual TTL.

    Built on the same nanosecond time-utils and ``RLock`` discipline as
    ``Expiring``.  No subclassing required.

    Parameters
    ----------
    default_ttl :
        Default TTL as seconds (``float``), nanoseconds (``int``),
        ``timedelta``, or ``None`` (keys never expire unless given a per-key
        TTL).
    max_size :
        Evict the soonest-to-expire key when capacity is reached.
    refresher :
        Optional ``Callable[[key], RefreshResult[V]]``.  When supplied,
        an expired get will call ``refresher(key)`` and atomically replace the
        entry rather than returning the default/raising.
    on_evict :
        Optional ``Callable[[key, value], None]`` invoked after every
        removal — TTL-driven sweeps, capacity evictions, explicit deletes,
        ``pop``, ``clear``, refresher replacements, ``__setitem__``
        overwrites of an existing key. The callback runs **outside** the
        cache's lock so it can do real work (close a file handle, unlink
        a temp file, decrement a refcount) without serializing other
        cache ops. Exceptions raised by the callback are swallowed so a
        bad callback can't poison the cache; if the callback's failure
        matters to the caller, they should log it themselves.

    Serialization
    -------------
    ``__getstate__`` / ``__setstate__`` are implemented: only live (non-expired)
    entries are persisted; the lock is recreated on load.  The
    ``on_evict`` and ``refresher`` callbacks are NOT persisted — they're
    typically closures over runtime state. Compatible with ``pickle``,
    ``copy.deepcopy``, and ``joblib``.
    """

    # ``ExpiringDict`` is the cache primitive behind every Databricks
    # SDK / MSAL singleton in the codebase, so per-call overhead on the
    # hot path matters. Slots cut the per-attribute __dict__ lookup
    # (``self._store`` etc.) down to a slot descriptor read, which
    # shows up as a measurable shave on ``get`` / ``set``.
    #
    # ``_lock`` is retained as a slot for backward-compat with any
    # caller that reaches into ``cache._lock`` (no such caller exists
    # in the repo); internal code no longer acquires it. Every
    # mutation path relies on the CPython GIL atomicity of ``dict.get``
    # / ``dict.pop`` / ``dict.__setitem__`` / ``list(dict.items())``
    # and accepts permissive race semantics in exchange (a cache that
    # briefly over-counts by one entry or loses a single write under
    # contention is still a correct cache; a cache that deadlocks
    # isn't).
    __slots__ = (
        "_default_ttl_ns",
        "_max_size",
        "_refresher",
        "_on_evict",
        "_store",
        "_lock",
        "_last_purge_ns",
        "_purge_pending",
    )

    # ── construction ─────────────────────────────────────────

    def __init__(
        self,
        default_ttl: Optional[Union[float, int, timedelta]] = 300.0,
        *,
        max_size: int | None = None,
        refresher: Optional[Callable[[K], RefreshResult[V]]] = None,
        on_evict: Optional[Callable[[K, V], None]] = None,
    ) -> None:
        self._default_ttl_ns: Optional[int] = self._parse_ttl(default_ttl)
        self._max_size = max_size
        self._refresher = refresher
        self._on_evict = on_evict
        self._store: Dict[K, Tuple[V, Optional[int]]] = {}
        # Retained for backward compatibility; never acquired by the
        # class itself. External code that reached for ``cache._lock``
        # still finds a real ``RLock``.
        self._lock: RLock = RLock()
        # Background-purge tracking (ns epoch integers — GIL-atomic reads on CPython)
        self._last_purge_ns: int = now_utc_ns()
        self._purge_pending: bool = False

    # ── eviction notification ────────────────────────────────

    def _notify_evicted(self, evicted: List[Tuple[K, V]]) -> None:
        """Fire the ``on_evict`` callback for every entry in *evicted*.

        Called from the various removal paths AFTER the lock has been
        released, so a slow callback (closing a file, unlinking a temp,
        flushing a remote handle) doesn't hold up other cache ops.

        Exceptions are caught and discarded — a faulty callback shouldn't
        be able to wedge the cache or mask the original removal. Callers
        who want strict-mode delivery can wrap the callback in their
        own try/except and log/raise as desired.
        """
        if self._on_evict is None or not evicted:
            return
        for key, value in evicted:
            try:
                self._on_evict(key, value)
            except Exception:
                pass

    # ── serialization ────────────────────────────────────────

    def __getstate__(self) -> Dict[str, Any]:
        # ``list(dict.items())`` is a single GIL-atomic C call, so we
        # don't need to serialize pickle/deepcopy against concurrent
        # writers — they either land before or after the snapshot.
        now = now_utc_ns()
        snapshot = list(self._store.items())
        return {
            "default_ttl_ns": self._default_ttl_ns,
            "max_size": self._max_size,
            # refresher and on_evict are intentionally excluded —
            # not picklable in general (closures, bound methods, ...).
            "store": {
                k: (v, exp)
                for k, (v, exp) in snapshot
                if exp is None or now < exp
            },
            "last_purge_ns": now,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._default_ttl_ns = state.get("default_ttl_ns")
        self._max_size = state.get("max_size")
        self._refresher = None
        self._on_evict = None
        self._store = state.get("store", {})
        self._lock = RLock()
        self._last_purge_ns = int(state.get("last_purge_ns", now_utc_ns()))
        self._purge_pending = False

    # ── background purge ──────────────────────────────────────

    def _background_purge(self) -> None:
        """Evict all expired keys; runs in a background daemon thread.

        Lock-free: iterate a GIL-atomic snapshot, atomically pop each
        candidate, and re-check expiry on the popped tuple so a
        concurrent refresh that landed between snapshot and pop is
        restored (rare; permissive on the rare race).
        """
        now = now_utc_ns()
        evicted: List[Tuple[K, V]] = []
        for k, (_, exp) in list(self._store.items()):
            if exp is None or now < exp:
                continue
            removed = self._store.pop(k, ...)
            if removed is ...:
                continue
            popped_value, popped_exp = removed  # type: ignore[misc]
            if popped_exp is None or now < popped_exp:
                # Refreshed in flight — best-effort restore.
                self._store[k] = removed  # type: ignore[assignment]
                continue
            evicted.append((k, popped_value))
        self._purge_pending = False
        self._notify_evicted(evicted)

    def _maybe_schedule_purge(self) -> None:
        """Schedule an async background sweep if ``_PURGE_INTERVAL_NS`` has elapsed.

        Fully lock-free. The race window is permissive: under heavy
        contention two threads may both pass the gates and both
        spawn a background purge. Each purge is independently
        idempotent (atomic per-key pops), so the worst case is a
        sliver of duplicated CPU work — never a correctness issue.

        The background thread is a daemon :class:`~yggdrasil.concurrent.threading.ThreadJob`
        so it never blocks interpreter shutdown.
        """
        now = now_utc_ns()
        # Fast path — single integer compare, GIL-atomic on CPython.
        if now - self._last_purge_ns < _PURGE_INTERVAL_NS:
            return
        # Second test: bail if another thread has already claimed the
        # slot. Races here are harmless (see docstring).
        if self._purge_pending:
            return
        # Claim — permissive; under a tight race two threads can both
        # set this True and both spawn a purge. Atomic ``dict.pop``
        # inside ``_background_purge`` keeps eviction one-shot regardless.
        self._purge_pending = True
        self._last_purge_ns = now
        from yggdrasil.concurrent.job import Job  # lazy import — avoids circular deps
        Job.make(self._background_purge).fire_and_forget()

    # ── core helpers ─────────────────────────────────────────

    @staticmethod
    def _parse_ttl(ttl: Any) -> Optional[int]:
        """Convert seconds/timedelta/int-ns/None → nanoseconds int or None."""
        if ttl is None:
            return None
        if isinstance(ttl, timedelta):
            return timedelta_to_ns(ttl)
        if isinstance(ttl, float):
            return _seconds_to_ns(ttl)
        return int(ttl)  # treat bare int as nanoseconds for internal use

    def _expires_at_ns(self, ttl: Any) -> Optional[int]:
        """
        Compute absolute expiry from a per-call TTL override or fall back to
        the instance default.  Returns ``None`` if no TTL is set.
        """
        if ttl is ...:
            ns = self._default_ttl_ns
        else:
            ns = self._parse_ttl(ttl)
        return None if ns is None else now_utc_ns() + ns

    def _is_expired(self, expires_at: Optional[int]) -> bool:
        return expires_at is not None and now_utc_ns() >= expires_at

    def _evict_expired_lockless(self) -> List[Tuple[K, V]]:
        """Drop every expired entry from the store via atomic pops.

        Iterates a ``list(dict.items())`` snapshot (GIL-atomic), then
        ``dict.pop`` each candidate. The popped tuple is re-checked
        against ``now`` so a concurrent refresh that landed between
        the snapshot and the pop is restored (rare permissive race).
        """
        now = now_utc_ns()
        evicted: List[Tuple[K, V]] = []
        for k, (_, exp) in list(self._store.items()):
            if exp is None or now < exp:
                continue
            removed = self._store.pop(k, ...)
            if removed is ...:
                continue
            popped_value, popped_exp = removed  # type: ignore[misc]
            if popped_exp is None or now < popped_exp:
                self._store[k] = removed  # type: ignore[assignment]
                continue
            evicted.append((k, popped_value))
        return evicted

    def _evict_one_for_capacity_lockless(self) -> Optional[Tuple[K, V]]:
        """Pop the soonest-expiring entry for capacity reasons.

        Snapshots via ``list(dict.items())`` to avoid ``dict changed
        size during iteration`` under concurrent writers; the
        chosen victim is then popped atomically. Returns ``None``
        if the store is empty or the chosen victim was already
        removed by a racing caller.
        """
        items = list(self._store.items())
        if not items:
            return None
        victim_key = min(
            items,
            key=lambda kv: kv[1][1] if kv[1][1] is not None else float("inf"),
        )[0]
        popped = self._store.pop(victim_key, ...)
        if popped is ...:
            return None
        return (victim_key, popped[0])  # type: ignore[index]

    def _put_lockless(
        self, key: K, value: V, expires_at: Optional[int]
    ) -> List[Tuple[K, V]]:
        """Write a key into the store, respecting max_size.

        Returns a list of (key, value) pairs that were evicted to
        make room or because they got overwritten. The caller fires
        ``on_evict`` for them after the put settles.

        Lock-free with permissive race semantics — concurrent puts
        at capacity may briefly leave the store at ``max_size + 1``;
        the next put rebalances. Overwrite-notify can fire for a
        value that was already replaced by a concurrent writer.
        """
        evicted: List[Tuple[K, V]] = []

        # Capacity check — best-effort; permissive on races.
        if (
            self._max_size is not None
            and key not in self._store
            and len(self._store) >= self._max_size
        ):
            cap_victim = self._evict_one_for_capacity_lockless()
            if cap_victim is not None:
                evicted.append(cap_victim)

        # Overwrite-notify only when ``on_evict`` is wired up.
        if self._on_evict is not None:
            existing = self._store.get(key)
            if existing is not None:
                evicted.append((key, existing[0]))

        self._store[key] = (value, expires_at)
        return evicted

    # ── dict-style interface ──────────────────────────────────

    def set(
        self,
        key: K,
        value: V,
        ttl: Any = ...,
    ) -> None:
        """Insert or overwrite *key*.

        ``ttl`` accepts seconds (``float``), nanoseconds (``int``),
        ``timedelta``, or ``None`` (no expiry).  Omitting ``ttl`` uses the
        instance default.  Schedules a background purge every 15 minutes.
        Fires ``on_evict`` for any entry that was capacity-evicted to
        make room or whose key got overwritten.

        Lock-free. ``dict.__setitem__`` / ``dict.pop`` /
        ``dict.get`` are GIL-atomic; the capacity check and
        overwrite-notify bookkeeping run under a permissive race
        regime (under heavy contention the cache may briefly hold
        ``max_size + 1`` entries or notify a value that another
        thread already replaced — both harmless).
        """
        default_ttl_ns = self._default_ttl_ns
        # Fast-path the dominant call shape: default-TTL set on a
        # cache without ``on_evict`` or ``max_size``. Covers every
        # singleton-style cache (MSAL, Databricks SDK ``_INSTANCES``)
        # and the bulk of warehouse / catalog caches.
        if (
            ttl is ...
            and self._on_evict is None
            and self._max_size is None
        ):
            exp = None if default_ttl_ns is None else now_utc_ns() + default_ttl_ns
            self._store[key] = (value, exp)
            if default_ttl_ns is not None:
                self._maybe_schedule_purge()
            return

        exp = self._expires_at_ns(ttl)
        evicted = self._put_lockless(key, value, exp)
        if evicted:
            self._notify_evicted(evicted)
        self._maybe_schedule_purge()

    def __setitem__(self, key: K, value: V) -> None:
        self.set(key, value)

    def get(self, key: K, default: Any = None) -> Optional[V]:
        """Return value for *key*, or *default* if missing / expired."""
        # Lock-free hot path. ``dict.get`` is a single C call, atomic
        # under the CPython GIL, and the tuple it returns is immutable
        # — so a concurrent eviction either lands before our read (we
        # see ``None``) or after (we see the soon-to-be-evicted value,
        # same outcome a barely-earlier reader would see). For caches
        # where a stale-by-a-microsecond read is fine (Databricks SDK
        # warehouse / table catalogs, MSAL singletons, the rest), this
        # collapses get-hit cost to one dict lookup + one wall-clock
        # read, no ``RLock`` traffic.
        entry = self._store.get(key)
        default_ttl_ns = self._default_ttl_ns
        if entry is not None:
            value, expires_at = entry
            if expires_at is None or now_utc_ns() < expires_at:
                if default_ttl_ns is not None:
                    self._maybe_schedule_purge()
                return value
        elif self._refresher is None:
            # Miss without a refresher to repopulate — no lock needed.
            # Subsequent ``set`` from another thread is fine: a future
            # ``get`` will see it via the same lock-free read.
            if default_ttl_ns is not None:
                self._maybe_schedule_purge()
            return default

        # Slow path — expired entry, or miss with a refresher to invoke.
        # Lock-free: atomic ``dict.pop`` removes the expired entry
        # exactly once, and we re-check the popped tuple's expiry so
        # a refresh that landed between the read above and this pop
        # is restored permissively.
        evicted: List[Tuple[K, V]] = []
        if entry is not None:
            removed = self._store.pop(key, ...)
            if removed is not ...:
                popped_value, popped_exp = removed  # type: ignore[misc]
                if popped_exp is None or now_utc_ns() < popped_exp:
                    # Concurrent writer refreshed us; restore and use it.
                    # Race: a third writer may overwrite in this window.
                    # Permissive — caller still gets a live value.
                    self._store[key] = removed  # type: ignore[assignment]
                    if default_ttl_ns is not None:
                        self._maybe_schedule_purge()
                    return popped_value
                evicted.append((key, popped_value))

        if evicted:
            self._notify_evicted(evicted)
        if default_ttl_ns is not None:
            self._maybe_schedule_purge()

        # Expired or missing — try refresher if configured.
        if self._refresher is not None:
            try:
                rr = self._refresher(key)
            except Exception:
                return default
            exp = rr.expires_at_ns
            if exp is None and rr.ttl_ns is not None:
                created = rr.created_at_ns or now_utc_ns()
                exp = created + rr.ttl_ns
            # Lockless insert. Overwrite-notify is best-effort.
            overwritten = self._store.get(key)
            self._store[key] = (rr.value, exp)
            if overwritten is not None and self._on_evict is not None:
                self._notify_evicted([(key, overwritten[0])])
            return rr.value

        return default

    def __getitem__(self, key: K) -> V:
        # Inlined version of ``get(key, ...)`` so the dunder doesn't
        # pay an extra Python frame on every ``cache[key]`` access.
        # Same lock-free fast path as ``get``.
        entry = self._store.get(key)
        if entry is not None:
            value, expires_at = entry
            if expires_at is None or now_utc_ns() < expires_at:
                if self._default_ttl_ns is not None:
                    self._maybe_schedule_purge()
                return value
        val = self.get(key, ...)
        if val is ...:
            raise KeyError(key)
        return val  # type: ignore[return-value]

    def __delitem__(self, key: K) -> None:
        # ``dict.pop`` is GIL-atomic on CPython — no lock needed for the
        # single-key removal, which shrinks the lock footprint and
        # rules out a class of deadlocks where the on_evict callback
        # or a concurrent writer happens to contend on this RLock.
        entry = self._store.pop(key, ...)
        if entry is ...:
            raise KeyError(key)
        value, _ = entry  # type: ignore[misc]
        self._notify_evicted([(key, value)])

    def pop(self, key: K, *args) -> V:
        entry = self._store.pop(key, ...)
        if entry is ...:
            if args:
                return args[0]
            raise KeyError(key)
        value, expires_at = entry  # type: ignore[misc]
        self._notify_evicted([(key, value)])
        if self._is_expired(expires_at):
            return args[0] if args else None  # type: ignore[return-value]
        return value

    def __contains__(self, key: object) -> bool:
        # Lock-free fast path mirroring ``get`` — avoids the extra
        # Python call into ``get`` and the sentinel comparison on every
        # ``key in cache`` test (used heavily by SDK schema/table
        # negative-lookup paths).
        entry = self._store.get(key)  # type: ignore[arg-type]
        if entry is None:
            return False
        _, expires_at = entry
        return expires_at is None or now_utc_ns() < expires_at

    def __len__(self) -> int:
        # ``len(dict)`` is a single C call, GIL-atomic on CPython.
        return len(self._store)

    def __iter__(self) -> Iterator[K]:
        return iter(self.keys())

    def __repr__(self) -> str:
        default_s = (
            f"{self._default_ttl_ns / 1e9:.3g}s"
            if self._default_ttl_ns is not None
            else "∞"
        )
        return f"{self.__class__.__name__}(len={len(self)}, default_ttl={default_s})"

    # ── bulk operations ───────────────────────────────────────

    def set_many(
        self,
        mapping: Dict[K, V],
        ttl: Any = ...,
    ) -> None:
        """Insert multiple key-value pairs sharing a TTL.

        Lock-free; each insert lands via GIL-atomic
        ``dict.__setitem__``. The batch is no longer atomic
        end-to-end — a concurrent reader may observe a partially
        applied batch — but each individual entry's visibility is
        atomic, which is all a cache needs.

        Fires ``on_evict`` (after the writes) for any entries that
        got capacity-evicted or overwritten.
        """
        exp = self._expires_at_ns(ttl)
        evicted_all: List[Tuple[K, V]] = []
        for key, value in mapping.items():
            evicted_all.extend(self._put_lockless(key, value, exp))
        if evicted_all:
            self._notify_evicted(evicted_all)
        self._maybe_schedule_purge()

    def update(
        self,
        other: Union[Dict[K, V], "ExpiringDict[K, V]", None] = None,
        ttl: Any = ...,
        **kwargs: V,
    ) -> None:
        """
        Update the dict from a mapping and/or keyword arguments, mirroring
        the stdlib ``dict.update`` signature.

        Parameters
        ----------
        other :
            A plain ``dict``, another ``ExpiringDict``, or any object with an
            ``.items()`` method.  When the source is an ``ExpiringDict`` and no
            explicit *ttl* is given, each key's **remaining TTL is preserved**
            so expiry semantics survive a copy.  Already-expired source keys
            are silently skipped.
        ttl :
            Override TTL for every key written.  Accepts seconds (``float``),
            nanoseconds (``int``), ``timedelta``, or ``None`` (no expiry).
            When omitted, the instance default is used for plain-dict sources,
            or the source key's remaining TTL is used for ``ExpiringDict``
            sources.
        **kwargs :
            Additional key-value pairs merged after *other*, using *ttl* /
            the instance default.
        """
        explicit_ttl = ttl is not ...

        # Snapshot source via the GIL-atomic ``list(dict.items())`` —
        # no need to take ``other._lock``. Cross-instance lock
        # acquisition was the only place this class could land in a
        # lock-ordering hazard with another ExpiringDict, so dropping
        # it removes the last residual deadlock surface.
        if isinstance(other, ExpiringDict):
            snapshot_ts = now_utc_ns()
            source_raw: list = list(other._store.items())
            is_expiring_dict = True
        elif other is not None:
            if hasattr(other, "items"):
                source_raw = list(other.items())
            else:
                source_raw = list(other)
            is_expiring_dict = False
        else:
            source_raw = []
            is_expiring_dict = False

        evicted_all: List[Tuple[K, V]] = []
        # Lock-free batch — each ``_put_lockless`` call uses
        # GIL-atomic dict ops. The batch isn't atomic end-to-end
        # (a concurrent reader may observe a partial merge) but
        # each key's visibility is atomic.
        for key, raw in source_raw:
            if is_expiring_dict:
                value, src_exp = raw
                if src_exp is not None and snapshot_ts >= src_exp:
                    continue
                if explicit_ttl:
                    exp = self._expires_at_ns(ttl)
                elif src_exp is None:
                    exp = self._expires_at_ns(...)
                else:
                    remaining_ns = src_exp - snapshot_ts
                    exp = now_utc_ns() + remaining_ns
            else:
                value, exp = raw, self._expires_at_ns(ttl)

            evicted_all.extend(self._put_lockless(key, value, exp))

        for key, value in kwargs.items():
            evicted_all.extend(
                self._put_lockless(key, value, self._expires_at_ns(ttl))
            )

        if evicted_all:
            self._notify_evicted(evicted_all)
        self._maybe_schedule_purge()

    def get_many(self, keys: Iterable[K]) -> Dict[K, V]:
        """Return ``{key: value}`` for all live keys in *keys*.

        Pure read — no eviction side-effect. Lock-free; relies on
        the GIL atomicity of ``dict.get``. Expired entries are skipped
        but left in place for the background purge / next ``get`` to
        evict (the previous pop-on-expire path raced with concurrent
        refreshers).
        """
        result: Dict[K, V] = {}
        now = now_utc_ns()
        for key in keys:
            entry = self._store.get(key)
            if entry is None:
                continue
            value, exp = entry
            if exp is None or now < exp:
                result[key] = value
        return result

    def delete_many(self, keys: Iterable[K]) -> int:
        """Delete *keys*; returns count of keys actually removed."""
        evicted: List[Tuple[K, V]] = []
        for key in keys:
            entry = self._store.pop(key, ...)
            if entry is not ...:
                value, _ = entry  # type: ignore[misc]
                evicted.append((key, value))
        self._notify_evicted(evicted)
        return len(evicted)

    # ── inspection ───────────────────────────────────────────

    def ttl(self, key: K) -> Optional[float]:
        """Remaining TTL in **seconds** for *key*, or ``None`` if gone/expired.

        Pure inspection — no eviction side-effect. (The previous
        evict-on-expired variant raced with concurrent refresh and
        could drop a freshly-set value; lazy eviction happens in
        ``get`` and the background sweep.)
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        _, expires_at = entry
        if expires_at is None:
            return None  # non-expiring
        remaining_ns = expires_at - now_utc_ns()
        if remaining_ns <= 0:
            return None
        return remaining_ns / 1_000_000_000

    def ttl_ns(self, key: K) -> Optional[int]:
        """Remaining TTL in **nanoseconds** for *key*, or ``None``."""
        entry = self._store.get(key)
        if entry is None:
            return None
        _, expires_at = entry
        if expires_at is None:
            return None
        remaining = expires_at - now_utc_ns()
        if remaining <= 0:
            return None
        return remaining

    def keys(self) -> list[K]:
        # ``list(dict.items())`` is a single C call — atomic snapshot
        # under the CPython GIL, no risk of "dict changed size during
        # iteration" and no lock contention with writers.
        now = now_utc_ns()
        return [k for k, (_, exp) in list(self._store.items()) if exp is None or now < exp]

    def values(self) -> list[V]:
        now = now_utc_ns()
        return [v for v, exp in list(self._store.values()) if exp is None or now < exp]

    def items(self) -> list[Tuple[K, V]]:
        now = now_utc_ns()
        return [(k, v) for k, (v, exp) in list(self._store.items()) if exp is None or now < exp]

    def snapshot(self) -> Dict[K, Tuple[V, Optional[int]]]:
        """
        Shallow copy of live entries as ``{key: (value, expires_at_ns)}``.
        Useful for debugging or external persistence.
        """
        now = now_utc_ns()
        return {
            k: (v, exp)
            for k, (v, exp) in list(self._store.items())
            if exp is None or now < exp
        }

    def clear(self) -> None:
        """Drop every entry. Fires ``on_evict`` for each removed entry.

        Lock-free. When ``on_evict`` is wired up we snapshot first,
        then call ``dict.clear()`` — entries inserted between the
        snapshot and the clear get cleared too but won't trigger the
        callback (permissive). When no callback is set, the fast
        path is a single GIL-atomic ``dict.clear()`` call.
        """
        if self._on_evict is None:
            self._store.clear()
            return
        evicted = [(k, v) for k, (v, _) in list(self._store.items())]
        self._store.clear()
        self._notify_evicted(evicted)

    def purge_expired(self) -> int:
        """Explicitly evict all expired keys; returns count removed.

        Fires ``on_evict`` for every entry that expired.
        """
        evicted = self._evict_expired_lockless()
        self._notify_evicted(evicted)
        return len(evicted)

    # ── refresh / touch ───────────────────────────────────────

    def refresh_key(self, key: K, ttl: Any = ...) -> bool:
        """
        Reset the TTL of an existing, live key.
        Returns ``True`` if key existed and was refreshed, ``False`` otherwise.
        Fires ``on_evict`` ONLY when the call discovers a silently
        expired entry and drops it; never when the key is genuinely
        refreshed.

        Lock-free pop+reinsert. Permissive race: a concurrent writer
        that lands between our pop and our reinsert will be
        overwritten by ours (we restore the previous value with a
        new TTL). Acceptable for a TTL-bump operation.
        """
        new_exp = self._expires_at_ns(ttl)
        popped = self._store.pop(key, ...)
        if popped is ...:
            return False
        value, old_exp = popped  # type: ignore[misc]
        if self._is_expired(old_exp):
            self._notify_evicted([(key, value)])
            return False
        self._store[key] = (value, new_exp)
        return True

    def get_or_set(
        self,
        key: K,
        default: Union[V, Callable[[], V]],
        ttl: Any = ...,
    ) -> V:
        """
        Return the live value for *key*; if missing/expired, store *default*
        (or the result of calling it) and return that.

        Fires ``on_evict`` if a previously-stored expired entry got
        replaced — the old value is leaving the cache.

        Note: the ``default`` callable is invoked OUTSIDE the internal
        lock. This rules out a class of deadlocks where ``default()``
        takes another lock (or re-enters this cache from a different
        thread). The tradeoff is that under contention two threads
        may both invoke ``default()`` — the first writer wins, the
        loser silently discards its own computation.
        """
        # Lock-free fast path for the hit case — ``get_or_set`` is
        # the workhorse of every "build it once, cache it forever"
        # pattern (Databricks SDK schema / table / catalog caches),
        # so the hit case dominates the miss case by orders of magnitude.
        entry = self._store.get(key)
        if entry is not None:
            value, expires_at = entry
            if expires_at is None or now_utc_ns() < expires_at:
                return value

        # Build the new value first; no lock is held anywhere, so
        # arbitrary user code in ``default()`` is automatically safe
        # against deadlock.
        new_value: V = default() if callable(default) else default  # type: ignore[assignment]
        exp = self._expires_at_ns(ttl)

        # Re-check after the (potentially slow) ``default()`` call —
        # another thread may have populated in the meantime. If we
        # see a live entry, honor it and discard ours.
        entry = self._store.get(key)
        if entry is not None:
            existing_value, existing_exp = entry
            if existing_exp is None or now_utc_ns() < existing_exp:
                return existing_value

        evicted = self._put_lockless(key, new_value, exp)
        if evicted:
            self._notify_evicted(evicted)
        self._maybe_schedule_purge()
        return new_value

    # ── RefreshResult convenience ─────────────────────────────

    def apply_refresh_result(self, key: K, rr: RefreshResult[V]) -> None:
        """
        Store the outcome of an external ``RefreshResult`` for *key*.
        Mirrors the contract from ``Expiring._apply_refresh_result``.
        """
        created = int(rr.created_at_ns) if rr.created_at_ns is not None else now_utc_ns()
        ttl_ns_val = int(rr.ttl_ns) if rr.ttl_ns is not None else None
        exp = int(rr.expires_at_ns) if rr.expires_at_ns is not None else None
        if exp is None and ttl_ns_val is not None:
            exp = created + ttl_ns_val
        evicted = self._put_lockless(key, rr.value, exp)
        if evicted:
            self._notify_evicted(evicted)
        self._maybe_schedule_purge()