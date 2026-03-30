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

Change log vs previous version
- ``Expiring``: no ``new_instance`` callable field; subclasses implement
  ``_refresh()``.
- ``ExpiringDict``: new; built on the same time-utils / lock discipline as
  ``Expiring``.
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

_MISSING = object()

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
    created_at_ns: Optional[int] = None
    ttl_ns: Optional[int] = None
    expires_at_ns: Optional[int] = None

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

    # ── pickle ───────────────────────────────────────────────

    def __getstate__(self) -> dict[str, Any]:
        with self._lock:
            return {
                "_value": self._value,
                "_created_at_ns": self._created_at_ns,
                "_expires_at_ns": self._expires_at_ns,
                "_ttl_ns": self._ttl_ns,
            }

    def __setstate__(self, state: dict[str, Any]) -> None:
        object.__setattr__(self, "_lock", RLock())
        with self._lock:
            self._value = state.get("_value")
            self._created_at_ns = _to_epoch_ns(
                state.get("_created_at_ns", now_utc_ns()), "_created_at_ns"
            )
            self._ttl_ns = _to_ttl_ns(state.get("_ttl_ns"))
            self._expires_at_ns = _to_expires_at_ns(
                state.get("_expires_at_ns"), now_ns=self._created_at_ns
            )

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

    Serialization
    -------------
    ``__getstate__`` / ``__setstate__`` are implemented: only live (non-expired)
    entries are persisted; the lock is recreated on load.  Compatible with
    ``pickle``, ``copy.deepcopy``, and ``joblib``.
    """

    # ── construction ─────────────────────────────────────────

    def __init__(
        self,
        default_ttl: Optional[Union[float, int, timedelta]] = 300.0,
        *,
        max_size: Optional[int] = None,
        refresher: Optional[Callable[[K], RefreshResult[V]]] = None,
    ) -> None:
        self._default_ttl_ns: Optional[int] = self._parse_ttl(default_ttl)
        self._max_size = max_size
        self._refresher = refresher
        self._store: Dict[K, Tuple[V, Optional[int]]] = {}
        self._lock: RLock = RLock()
        # Background-purge tracking (ns epoch integers — GIL-atomic reads on CPython)
        self._last_purge_ns: int = now_utc_ns()
        self._purge_pending: bool = False

    # ── serialization ────────────────────────────────────────

    def __getstate__(self) -> Dict[str, Any]:
        with self._lock:
            now = now_utc_ns()
            return {
                "default_ttl_ns": self._default_ttl_ns,
                "max_size": self._max_size,
                # refresher is intentionally excluded (not picklable in general)
                "store": {
                    k: (v, exp)
                    for k, (v, exp) in self._store.items()
                    if exp is None or now < exp
                },
                "last_purge_ns": now,
            }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._default_ttl_ns = state.get("default_ttl_ns")
        self._max_size = state.get("max_size")
        self._refresher = None
        self._store = state.get("store", {})
        self._lock = RLock()
        self._last_purge_ns = int(state.get("last_purge_ns", now_utc_ns()))
        self._purge_pending = False

    # ── background purge ──────────────────────────────────────

    def _background_purge(self) -> None:
        """Evict all expired keys; runs in a background daemon thread."""
        with self._lock:
            self._evict_expired_locked()
            self._purge_pending = False

    def _maybe_schedule_purge(self) -> None:
        """Schedule an async background sweep if ``_PURGE_INTERVAL_NS`` has elapsed.

        Uses **double-checked locking** for a near-zero hot-path cost:

        1. **Outer check** — lock-free integer comparison (CPython GIL makes
           single-int reads atomic).  Exits immediately on the common path
           (interval not elapsed).
        2. **Inner check** — under ``RLock`` to atomically verify the condition
           and claim the purge slot, preventing duplicate threads.

        The background thread is a daemon :class:`~yggdrasil.concurrent.threading.ThreadJob`
        so it never blocks interpreter shutdown.
        """
        # Fast path — no lock (effectively atomic in CPython via the GIL)
        if now_utc_ns() - self._last_purge_ns < _PURGE_INTERVAL_NS:
            return

        # Slow path — claim the purge slot under the lock
        now = now_utc_ns()
        with self._lock:
            if self._purge_pending or (now - self._last_purge_ns) < _PURGE_INTERVAL_NS:
                return  # another thread beat us or interval not yet elapsed
            self._purge_pending = True
            self._last_purge_ns = now

        # Spawn outside the lock so we don't hold it while creating the thread
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
        if ttl is _MISSING:
            ns = self._default_ttl_ns
        else:
            ns = self._parse_ttl(ttl)
        return None if ns is None else now_utc_ns() + ns

    def _is_expired(self, expires_at: Optional[int]) -> bool:
        return expires_at is not None and now_utc_ns() >= expires_at

    def _evict_expired_locked(self) -> None:
        now = now_utc_ns()
        dead = [k for k, (_, exp) in self._store.items() if exp is not None and now >= exp]
        for k in dead:
            del self._store[k]

    def _evict_one_for_capacity_locked(self) -> None:
        if not self._store:
            return
        # Prefer keys with the soonest expiry; treat None (non-expiring) as ∞
        victim = min(
            self._store,
            key=lambda k: self._store[k][1] if self._store[k][1] is not None else float("inf"),
        )
        del self._store[victim]

    def _put_locked(self, key: K, value: V, expires_at: Optional[int]) -> None:
        """Write a key into the store, respecting max_size."""
        if (
            self._max_size is not None
            and key not in self._store
            and len(self._store) >= self._max_size
        ):
            self._evict_one_for_capacity_locked()
        self._store[key] = (value, expires_at)

    # ── dict-style interface ──────────────────────────────────

    def set(
        self,
        key: K,
        value: V,
        ttl: Any = _MISSING,
    ) -> None:
        """Insert or overwrite *key*.

        ``ttl`` accepts seconds (``float``), nanoseconds (``int``),
        ``timedelta``, or ``None`` (no expiry).  Omitting ``ttl`` uses the
        instance default.  Schedules a background purge every 15 minutes.
        """
        exp = self._expires_at_ns(ttl)
        with self._lock:
            self._put_locked(key, value, exp)
        self._maybe_schedule_purge()

    def __setitem__(self, key: K, value: V) -> None:
        self.set(key, value)

    def get(self, key: K, default: Any = None) -> Optional[V]:
        """Return value for *key*, or *default* if missing / expired."""
        refresher_key: Any = _MISSING  # sentinel: _MISSING → don't run refresher

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                result = default
            else:
                value, expires_at = entry
                if not self._is_expired(expires_at):
                    result = value
                else:
                    del self._store[key]
                    result = default
                    refresher_key = key  # expired — may need refresher

        # Check purge interval on every get (lock-free fast path)
        self._maybe_schedule_purge()

        # Expired key — try refresher if configured
        if refresher_key is not _MISSING and self._refresher is not None:
            try:
                rr = self._refresher(refresher_key)
            except Exception:
                return default
            exp = rr.expires_at_ns
            if exp is None and rr.ttl_ns is not None:
                created = rr.created_at_ns or now_utc_ns()
                exp = created + rr.ttl_ns
            with self._lock:
                self._put_locked(refresher_key, rr.value, exp)
            return rr.value

        return result

    def __getitem__(self, key: K) -> V:
        val = self.get(key, _MISSING)
        if val is _MISSING:
            raise KeyError(key)
        return val  # type: ignore[return-value]

    def __delitem__(self, key: K) -> None:
        with self._lock:
            if key not in self._store:
                raise KeyError(key)
            del self._store[key]

    def pop(self, key: K, *args) -> V:
        with self._lock:
            entry = self._store.pop(key, _MISSING)
        if entry is _MISSING:
            if args:
                return args[0]
            raise KeyError(key)
        value, expires_at = entry
        if self._is_expired(expires_at):
            return args[0] if args else None  # type: ignore[return-value]
        return value

    def __contains__(self, key: object) -> bool:
        return self.get(key, _MISSING) is not _MISSING  # type: ignore[arg-type]

    def __len__(self) -> int:
        with self._lock:
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
        ttl: Any = _MISSING,
    ) -> None:
        """Atomically insert multiple key-value pairs sharing a TTL."""
        exp = self._expires_at_ns(ttl)
        with self._lock:
            for key, value in mapping.items():
                self._put_locked(key, value, exp)
        self._maybe_schedule_purge()

    def update(
        self,
        other: Union[Dict[K, V], "ExpiringDict[K, V]", None] = None,
        ttl: Any = _MISSING,
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
        explicit_ttl = ttl is not _MISSING

        # Snapshot source outside our lock to avoid lock-ordering deadlocks
        if isinstance(other, ExpiringDict):
            snapshot_ts = now_utc_ns()
            with other._lock:
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

        with self._lock:
            for key, raw in source_raw:
                if is_expiring_dict:
                    value, src_exp = raw
                    if src_exp is not None and snapshot_ts >= src_exp:
                        continue
                    if explicit_ttl:
                        exp = self._expires_at_ns(ttl)
                    elif src_exp is None:
                        exp = self._expires_at_ns(_MISSING)
                    else:
                        remaining_ns = src_exp - snapshot_ts
                        exp = now_utc_ns() + remaining_ns
                else:
                    value, exp = raw, self._expires_at_ns(ttl)

                self._put_locked(key, value, exp)

            for key, value in kwargs.items():
                self._put_locked(key, value, self._expires_at_ns(ttl))

        self._maybe_schedule_purge()

    def get_many(self, keys: Iterable[K]) -> Dict[K, V]:
        """Return ``{key: value}`` for all live keys in *keys*."""
        result: Dict[K, V] = {}
        now = now_utc_ns()
        with self._lock:
            for key in keys:
                entry = self._store.get(key)
                if entry is not None:
                    value, exp = entry
                    if exp is None or now < exp:
                        result[key] = value
                    else:
                        del self._store[key]
        return result

    def delete_many(self, keys: Iterable[K]) -> int:
        """Delete *keys*; returns count of keys actually removed."""
        removed = 0
        with self._lock:
            for key in keys:
                if self._store.pop(key, _MISSING) is not _MISSING:
                    removed += 1
        return removed

    # ── inspection ───────────────────────────────────────────

    def ttl(self, key: K) -> Optional[float]:
        """Remaining TTL in **seconds** for *key*, or ``None`` if gone/expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            _, expires_at = entry
            if expires_at is None:
                return None  # non-expiring
            remaining_ns = expires_at - now_utc_ns()
            if remaining_ns <= 0:
                del self._store[key]
                return None
            return remaining_ns / 1_000_000_000

    def ttl_ns(self, key: K) -> Optional[int]:
        """Remaining TTL in **nanoseconds** for *key*, or ``None``."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            _, expires_at = entry
            if expires_at is None:
                return None
            remaining = expires_at - now_utc_ns()
            if remaining <= 0:
                del self._store[key]
                return None
            return remaining

    def keys(self) -> list[K]:
        now = now_utc_ns()
        with self._lock:
            return [k for k, (_, exp) in self._store.items() if exp is None or now < exp]

    def values(self) -> list[V]:
        now = now_utc_ns()
        with self._lock:
            return [v for v, exp in self._store.values() if exp is None or now < exp]

    def items(self) -> list[Tuple[K, V]]:
        now = now_utc_ns()
        with self._lock:
            return [(k, v) for k, (v, exp) in self._store.items() if exp is None or now < exp]

    def snapshot(self) -> Dict[K, Tuple[V, Optional[int]]]:
        """
        Shallow copy of live entries as ``{key: (value, expires_at_ns)}``.
        Useful for debugging or external persistence.
        """
        now = now_utc_ns()
        with self._lock:
            return {
                k: (v, exp)
                for k, (v, exp) in self._store.items()
                if exp is None or now < exp
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def purge_expired(self) -> int:
        """Explicitly evict all expired keys; returns count removed."""
        with self._lock:
            before = len(self._store)
            self._evict_expired_locked()
            return before - len(self._store)

    # ── refresh / touch ───────────────────────────────────────

    def refresh_key(self, key: K, ttl: Any = _MISSING) -> bool:
        """
        Reset the TTL of an existing, live key.
        Returns ``True`` if key existed and was refreshed, ``False`` otherwise.
        """
        exp = self._expires_at_ns(ttl)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            value, old_exp = entry
            if self._is_expired(old_exp):
                del self._store[key]
                return False
            self._store[key] = (value, exp)
            return True

    def get_or_set(
        self,
        key: K,
        default: Union[V, Callable[[], V]],
        ttl: Any = _MISSING,
    ) -> V:
        """
        Return the live value for *key*; if missing/expired, store *default*
        (or the result of calling it) and return that.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                value, expires_at = entry
                if not self._is_expired(expires_at):
                    return value
            value = default() if callable(default) else default
            exp = self._expires_at_ns(ttl)
            self._put_locked(key, value, exp)
        self._maybe_schedule_purge()
        return value

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
        with self._lock:
            self._put_locked(key, rr.value, exp)
        self._maybe_schedule_purge()

