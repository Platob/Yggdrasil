"""Shared base for Loki's typed result objects.

The loki layer returns a handful of structured values — an act transcript, a
fleet agent summary, a scaffold receipt — that used to be loose dicts read by
static string keys. :class:`DictResult` makes them ``slots`` dataclasses that
still *read* like the old dicts (``obj["x"]`` / ``obj.get("x")``) so every
existing caller keeps working, while new code uses real attributes; ``to_dict``
serializes them at a JSON edge. Mirrors the mapping shim on
:class:`~yggdrasil.loki.planning.AgentPlan`.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

__all__ = ["DictResult"]


class DictResult:
    """Mapping-compatible mixin for a ``slots`` dataclass.

    Declares empty ``__slots__`` so a ``@dataclass(slots=True)`` subclass keeps
    its slots — a non-slotted base would re-introduce a per-instance ``__dict__``
    and defeat the point.
    """

    __slots__ = ()

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
