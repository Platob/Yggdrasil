"""Case-insensitive, multi-value HTTP header dict — :class:`HTTPHeaderDict`.

The transport surface (raw socket writes, retry-after parsing, the
urllib3-shim error path in :mod:`yggdrasil.exceptions.http`) speaks the
urllib3-shaped :class:`HTTPHeaderDict`: same name, same multi-value
semantics, lowercase-keyed storage with first-seen original casing
preserved on iteration. The high-level
:class:`yggdrasil.io.headers.Headers` is a different abstraction
(normalised, anonymisation-aware, hash-stable) and isn't a drop-in
replacement at the transport layer.
"""
from __future__ import annotations

import collections.abc
from typing import Any, Iterator, Tuple


__all__ = ["HTTPHeaderDict"]


class HTTPHeaderDict(collections.abc.MutableMapping):
    """Case-insensitive, multi-value header dict mirroring ``urllib3``'s.

    Stores values per lowercase key but preserves the first-seen original
    casing when iterating. Multi-value headers (Set-Cookie, …) are joined
    with ``, `` on read, matching urllib3's collapsing behavior.
    """

    def __init__(self, headers: Any = None, **kwargs: str) -> None:
        # _store maps lowercase key -> (original_case, [values])
        self._store: dict[str, Tuple[str, list[str]]] = {}
        if headers is not None:
            self.extend(headers)
        if kwargs:
            self.extend(kwargs)

    # MutableMapping protocol -------------------------------------------------
    def __setitem__(self, key: str, value: str) -> None:
        self._store[key.lower()] = (key, [value])

    def __getitem__(self, key: str) -> str:
        _, values = self._store[key.lower()]
        return ", ".join(values)

    def __delitem__(self, key: str) -> None:
        del self._store[key.lower()]

    def __iter__(self) -> Iterator[str]:
        return (original for original, _ in self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.lower() in self._store

    # Multi-value helpers -----------------------------------------------------
    def add(self, key: str, value: str) -> None:
        slot = self._store.get(key.lower())
        if slot is None:
            self._store[key.lower()] = (key, [value])
        else:
            slot[1].append(value)

    def extend(self, other: Any) -> None:
        if isinstance(other, HTTPHeaderDict):
            for original, values in other._store.values():
                for v in values:
                    self.add(original, v)
            return
        if hasattr(other, "items"):
            other = other.items()
        for k, v in other:
            self.add(k, v)

    def getlist(self, key: str) -> list[str]:
        slot = self._store.get(key.lower())
        return list(slot[1]) if slot is not None else []

    def __repr__(self) -> str:
        return f"HTTPHeaderDict({dict(self.items())!r})"
