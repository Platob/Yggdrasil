from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List


class HttpMongoCursor:
    """Minimal eager cursor wrapper good enough for MongoEngine aggregation use."""

    def __init__(self, documents: Iterable[Dict[str, Any]]) -> None:
        self._documents: List[Dict[str, Any]] = list(documents)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._documents)

    def __len__(self) -> int:
        return len(self._documents)

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self._documents)
