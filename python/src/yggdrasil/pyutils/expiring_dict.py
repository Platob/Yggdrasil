from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections.abc import MutableMapping, Iterator
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, Dict, Tuple

K = TypeVar("K")
V = TypeVar("V")


__all__ = [
    "ExpiringDict"
]


@dataclass(frozen=True)
class _Entry(Generic[V]):
    value: V
    expires_at: float  # monotonic timestamp


class ExpiringDict(MutableMapping[K, V]):
    """
    Dict with per-key TTL expiration.

    - Uses time.monotonic() (safe against system clock changes)
    - O(log n) cleanup amortized via a min-heap of expirations
    - Overwrites are handled (stale heap entries are ignored)
    - Optional refresh_on_get: touching a key extends its TTL
    """

    def __init__(
        self,
        default_ttl: Optional[float] = None,
        *,
        refresh_on_get: bool = False,
        on_expire: Optional[Callable[[K, V], None]] = None,
        thread_safe: bool = False,
    ) -> None:
        """
        default_ttl: seconds, if provided used when ttl isn't passed to set()
        refresh_on_get: if True, get()/__getitem__ extends TTL using default_ttl
        on_expire: callback(key, value) called when an item is expired during cleanup
        thread_safe: wrap operations in an RLock (extra overhead, but safe)
        """
        self.default_ttl = default_ttl
        self.refresh_on_get = refresh_on_get
        self.on_expire = on_expire

        self._store: Dict[K, _Entry[V]] = {}
        self._heap: list[Tuple[float, int, K]] = []  # (expires_at, seq, key)
        self._seq = itertools.count()

        self._lock = threading.RLock() if thread_safe else None

    def _now(self) -> float:
        return time.monotonic()

    def _with_lock(self):
        # tiny helper to avoid repeating if/else everywhere
        return self._lock or _NoopLock()

    def _prune(self) -> None:
        """Remove expired entries. Ignores stale heap rows from overwrites."""
        now = self._now()
        while self._heap and self._heap[0][0] <= now:
            exp, _, key = heapq.heappop(self._heap)
            entry = self._store.get(key)
            if entry is None:
                continue
            # Only expire if this heap expiry matches current entry expiry
            if entry.expires_at == exp:
                del self._store[key]
                if self.on_expire:
                    self.on_expire(key, entry.value)

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        with self._with_lock():
            self._prune()
            if ttl is None:
                ttl = self.default_ttl
            if ttl is None:
                # no expiration
                expires_at = float("inf")
            else:
                if ttl <= 0:
                    # immediate expiration: just delete if exists
                    self._store.pop(key, None)
                    return
                expires_at = self._now() + ttl

            self._store[key] = _Entry(value=value, expires_at=expires_at)
            heapq.heappush(self._heap, (expires_at, next(self._seq), key))

    # --- MutableMapping interface ---
    def __setitem__(self, key: K, value: V) -> None:
        # Uses default_ttl (if any)
        self.set(key, value, ttl=self.default_ttl)

    def __getitem__(self, key: K) -> V:
        with self._with_lock():
            self._prune()
            entry = self._store[key]  # may raise KeyError
            if entry.expires_at <= self._now():
                # edge case: expired but not yet pruned (rare)
                del self._store[key]
                raise KeyError(key)

            if self.refresh_on_get:
                if self.default_ttl is None:
                    raise ValueError("refresh_on_get=True requires default_ttl")
                # refresh TTL
                self.set(key, entry.value, ttl=self.default_ttl)
                return entry.value

            return entry.value

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        try:
            return self[key]
        except KeyError:
            return default

    def __delitem__(self, key: K) -> None:
        with self._with_lock():
            self._prune()
            del self._store[key]
            # heap keeps stale rows; they'll be ignored during prune

    def __iter__(self) -> Iterator[K]:
        with self._with_lock():
            self._prune()
            return iter(list(self._store.keys()))

    def __len__(self) -> int:
        with self._with_lock():
            self._prune()
            return len(self._store)

    def __contains__(self, key: object) -> bool:
        with self._with_lock():
            self._prune()
            if key in self._store:
                entry = self._store[key]  # type: ignore[index]
                return entry.expires_at > self._now()
            return False

    def cleanup(self) -> int:
        """Force prune and return number of remaining items."""
        with self._with_lock():
            self._prune()
            return len(self._store)

    def items(self):
        with self._with_lock():
            self._prune()
            return [(k, e.value) for k, e in self._store.items()]

    def keys(self):
        with self._with_lock():
            self._prune()
            return list(self._store.keys())

    def values(self):
        with self._with_lock():
            self._prune()
            return [e.value for e in self._store.values()]


class _NoopLock:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
