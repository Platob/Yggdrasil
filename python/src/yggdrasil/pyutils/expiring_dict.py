from __future__ import annotations

import heapq
import itertools
import threading
import time
import pickle
from collections.abc import MutableMapping, Iterator
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, Dict, Tuple, Any, Mapping

K = TypeVar("K")
V = TypeVar("V")


__all__ = ["ExpiringDict"]


@dataclass(frozen=True)
class _Entry(Generic[V]):
    value: V
    expires_at: float  # monotonic timestamp


class ExpiringDict(MutableMapping[K, V]):
    """
    Dict with per-key TTL expiration.

    Serialization note:
    - Internally uses time.monotonic() for expires_at (great locally).
    - For serialization, we store remaining TTLs so it can be reconstructed
      in a different process with a different monotonic clock origin.
    """

    _SER_VERSION = 1

    def __init__(
        self,
        default_ttl: Optional[float] = None,
        *,
        refresh_on_get: bool = False,
        on_expire: Optional[Callable[[K, V], None]] = None,
        thread_safe: bool = False,
    ) -> None:
        self.default_ttl = default_ttl
        self.refresh_on_get = refresh_on_get
        self.on_expire = on_expire

        self._store: Dict[K, _Entry[V]] = {}
        self._heap: list[Tuple[float, int, K]] = []  # (expires_at, seq, key)
        self._seq = itertools.count()

        self._thread_safe = thread_safe
        self._lock = threading.RLock() if thread_safe else None

    def _now(self) -> float:
        return time.monotonic()

    def _with_lock(self):
        return self._lock or _NoopLock()

    def _prune(self) -> None:
        now = self._now()
        while self._heap and self._heap[0][0] <= now:
            exp, _, key = heapq.heappop(self._heap)
            entry = self._store.get(key)
            if entry is None:
                continue
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
                expires_at = float("inf")
            else:
                if ttl <= 0:
                    self._store.pop(key, None)
                    return
                expires_at = self._now() + ttl

            self._store[key] = _Entry(value=value, expires_at=expires_at)
            heapq.heappush(self._heap, (expires_at, next(self._seq), key))

    # --- MutableMapping interface ---
    def __setitem__(self, key: K, value: V) -> None:
        self.set(key, value, ttl=self.default_ttl)

    def __getitem__(self, key: K) -> V:
        with self._with_lock():
            self._prune()
            entry = self._store[key]
            if entry.expires_at <= self._now():
                del self._store[key]
                raise KeyError(key)

            if self.refresh_on_get:
                if self.default_ttl is None:
                    raise ValueError("refresh_on_get=True requires default_ttl")
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

    # ----------------------------
    # Serialization / Pickling
    # ----------------------------
    def to_state(self) -> dict[str, Any]:
        """
        Returns a JSON-ish friendly state (assuming keys/values are JSON-friendly).
        Stores remaining TTL (seconds) per key; None => no expiration.
        Expired entries are dropped.
        """
        with self._with_lock():
            self._prune()
            now = self._now()

            items: list[tuple[Any, Any, Optional[float]]] = []
            for k, e in self._store.items():
                if e.expires_at == float("inf"):
                    rem = None
                else:
                    rem = e.expires_at - now
                    if rem <= 0:
                        continue
                items.append((k, e.value, rem))

            return {
                "v": self._SER_VERSION,
                "default_ttl": self.default_ttl,
                "refresh_on_get": self.refresh_on_get,
                "thread_safe": self._thread_safe,
                # NOTE: on_expire is intentionally not serialized
                "items": items,
            }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        on_expire: Optional[Callable[[K, V], None]] = None,
    ) -> "ExpiringDict[K, V]":
        """
        Rebuild from state produced by to_state().
        """
        if state.get("v") != cls._SER_VERSION:
            raise ValueError(f"Unsupported serialized version: {state.get('v')}")

        d = cls(
            default_ttl=state.get("default_ttl"),
            refresh_on_get=bool(state.get("refresh_on_get", False)),
            on_expire=on_expire,
            thread_safe=bool(state.get("thread_safe", False)),
        )

        now = d._now()
        items = state.get("items", [])
        for k, v, rem in items:
            if rem is None:
                expires_at = float("inf")
            else:
                if rem <= 0:
                    continue
                expires_at = now + float(rem)

            d._store[k] = _Entry(value=v, expires_at=expires_at)
            heapq.heappush(d._heap, (expires_at, next(d._seq), k))

        return d

    def to_bytes(self, protocol: int = pickle.HIGHEST_PROTOCOL) -> bytes:
        """
        Pickle to bytes using the portable state representation.
        """
        return pickle.dumps(self.to_state(), protocol=protocol)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        on_expire: Optional[Callable[[K, V], None]] = None,
    ) -> "ExpiringDict[K, V]":
        state = pickle.loads(data)
        return cls.from_state(state, on_expire=on_expire)

    def __getstate__(self) -> dict[str, Any]:
        # Called by pickle
        return self.to_state()

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Called by pickle; rebuild "self" in-place
        rebuilt = self.from_state(state, on_expire=None)

        # Copy rebuilt internals into self (keep pickling contract)
        self.default_ttl = rebuilt.default_ttl
        self.refresh_on_get = rebuilt.refresh_on_get
        self.on_expire = None  # not serialized

        self._store = rebuilt._store
        self._heap = rebuilt._heap
        self._seq = rebuilt._seq

        self._thread_safe = rebuilt._thread_safe
        self._lock = threading.RLock() if self._thread_safe else None


class _NoopLock:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
