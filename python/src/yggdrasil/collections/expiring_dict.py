from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections.abc import MutableMapping, Iterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Generic, Optional, TypeVar, Dict, Tuple, Any, Mapping, Union

K = TypeVar("K")
V = TypeVar("V")

__all__ = ["ExpiringDict"]

# 1 second = 10^9 nanoseconds
SEC_TO_NS = 1_000_000_000


@dataclass(frozen=True)
class _Entry(Generic[V]):
    value: V
    expires_at: int  # Nanosecond epoch timestamp (UTC)


class ExpiringDict(MutableMapping[K, V]):
    """
    Dict with per-key TTL expiration using Nanosecond UTC Epoch timestamps.
    Supports datetime.timedelta for TTL durations.
    """

    _SER_VERSION = 3

    def __init__(
        self,
        default_ttl: Optional[Union[int, float, timedelta]] = None,
        *,
        refresh_on_get: bool = False,
        on_expire: Optional[Callable[[K, V], None]] = None,
        thread_safe: bool = True,
    ) -> None:
        self.default_ttl = default_ttl
        self.refresh_on_get = refresh_on_get
        self.on_expire = on_expire

        self._store: Dict[K, _Entry[V]] = {}
        self._heap: list[Tuple[int, int, K]] = []  # (expires_at_ns, seq, key)
        self._seq = itertools.count()

        self._thread_safe = thread_safe
        self._lock = threading.RLock() if thread_safe else _NoopLock()

    # --- Serialization ---
    def to_state(self) -> dict[str, Any]:
        with self._lock:
            self._prune()
            # NOTE: do NOT serialize on_expire or the lock
            return {
                "v": self._SER_VERSION,
                "default_ttl": self.default_ttl,
                "refresh_on_get": self.refresh_on_get,
                "thread_safe": self._thread_safe,
                "items": [(k, e.value, e.expires_at) for k, e in self._store.items()],
            }

    @classmethod
    def from_state(cls, state: Mapping[str, Any], on_expire=None) -> "ExpiringDict[K, V]":
        d = cls(
            default_ttl=state.get("default_ttl"),
            refresh_on_get=bool(state.get("refresh_on_get", False)),
            on_expire=on_expire,
            thread_safe=bool(state.get("thread_safe", True)),
        )

        now = d._now_ns()
        max_seq = -1

        # state["items"] is [(k, v, exp_ns), ...]
        for k, v, exp_ns in state.get("items", []):
            if exp_ns > now:
                d._store[k] = _Entry(value=v, expires_at=exp_ns)
                # Rebuild heap with explicit seq
                max_seq += 1
                heapq.heappush(d._heap, (exp_ns, max_seq, k))

        # Restart seq after whatever we used
        d._seq = itertools.count(max_seq + 1)
        return d

    def __getstate__(self) -> dict[str, Any]:
        # what pickle stores
        return self.to_state()

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Rebuild a fresh instance, then transplant internals into *this* instance.
        # Important: caller can re-attach on_expire after unpickling if desired.
        rebuilt = type(self).from_state(state, on_expire=None)
        self.__dict__.clear()
        self.__dict__.update(rebuilt.__dict__)

    @property
    def lock(self):
        """Expose the lock for external atomic operations."""
        return self._lock

    def _now_ns(self) -> int:
        return time.time_ns()

    def _ttl_to_ns(self, ttl: Union[int, float, timedelta]) -> int:
        if isinstance(ttl, timedelta):
            return int(ttl.total_seconds() * SEC_TO_NS)
        return int(ttl * SEC_TO_NS)

    def _prune(self) -> None:
        now = self._now_ns()
        while self._heap and self._heap[0][0] <= now:
            exp_ns, _, key = heapq.heappop(self._heap)
            entry = self._store.get(key)
            if entry is not None and entry.expires_at == exp_ns:
                del self._store[key]
                if self.on_expire:
                    self.on_expire(key, entry.value)

    def set(self, key: K, value: V, ttl: Optional[Union[int, float, timedelta]] = None) -> None:
        with self._lock:
            self._prune()
            target_ttl = ttl if ttl is not None else self.default_ttl

            if target_ttl is None:
                expires_at = 9_223_372_036_854_775_807  # Max int64
            else:
                ttl_ns = self._ttl_to_ns(target_ttl)
                if ttl_ns <= 0:
                    self._store.pop(key, None)
                    return
                expires_at = self._now_ns() + ttl_ns

            self._store[key] = _Entry(value=value, expires_at=expires_at)
            heapq.heappush(self._heap, (expires_at, next(self._seq), key))

    # --- MutableMapping interface ---
    def __setitem__(self, key: K, value: V) -> None:
        self.set(key, value, ttl=self.default_ttl)

    def __getitem__(self, key: K) -> V:
        with self._lock:
            self._prune()
            if key not in self._store:
                raise KeyError(key)

            entry = self._store[key]
            # Double check expiration in case prune missed it
            if entry.expires_at <= self._now_ns():
                del self._store[key]
                raise KeyError(key)

            if self.refresh_on_get:
                if self.default_ttl is None:
                    raise ValueError("refresh_on_get=True requires default_ttl")
                self.set(key, entry.value, ttl=self.default_ttl)

            return entry.value

    def __delitem__(self, key: K) -> None:
        with self._lock:
            # We don't strictly need to prune here, but it keeps the heap clean
            self._prune()
            if key not in self._store:
                raise KeyError(key)
            del self._store[key]

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            self._prune()
            return iter(list(self._store.keys()))

    def __len__(self) -> int:
        with self._lock:
            self._prune()
            return len(self._store)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            self._prune()
            entry = self._store.get(key)  # type: ignore
            return entry is not None and entry.expires_at > self._now_ns()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._heap.clear()


class _NoopLock:
    def __enter__(self): return self

    def __exit__(self, *args): return False