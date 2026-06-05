"""Object namespace utilities.

Provides:
    ObjectSerde — module/qualname resolution helpers used across the
                  yggdrasil type-dispatch machinery.
"""
from __future__ import annotations

import inspect
import logging
from types import ModuleType
from typing import Any

__all__ = [
    "ObjectSerde",
]

logger = logging.getLogger(__name__)


class ObjectSerde:
    """Module/qualname resolution helpers."""

    @staticmethod
    def module_and_name(obj: Any, *, fallback: str = "") -> tuple[str, str]:
        """Return ``(module, qualname)`` for *obj*.

        Robust across Python objects including C-extension / PyArrow objects
        where ``__module__`` may be missing or misleading.
        """
        if obj is None:
            return "builtins", fallback or "None"

        if isinstance(obj, ModuleType):
            return (
                getattr(obj, "__name__", None) or "builtins",
                fallback or getattr(obj, "__name__", "module"),
            )

        try:
            unwrapped = inspect.unwrap(obj)  # type: ignore[arg-type]
        except Exception:
            unwrapped = obj

        if inspect.isclass(unwrapped):
            cls = unwrapped
        else:
            cls = getattr(unwrapped, "__objclass__", None) or type(unwrapped)

        def _qualname(x: Any) -> str:
            return (
                getattr(x, "__qualname__", None)
                or getattr(x, "__name__", None)
                or getattr(type(x), "__qualname__", None)
                or getattr(type(x), "__name__", None)
                or fallback
            )

        mod = getattr(unwrapped, "__module__", None)
        qual = getattr(unwrapped, "__qualname__", None) or getattr(unwrapped, "__name__", None)

        if not mod:
            mod = getattr(cls, "__module__", None)
        if not qual:
            qual = _qualname(unwrapped)

        if not mod or mod == "builtins":
            cls_mod = getattr(getattr(unwrapped, "__class__", None), "__module__", None)
            if cls_mod and cls_mod != "builtins":
                mod = cls_mod

        dotted_candidate = None
        for cand in (
            getattr(cls, "__qualname__", None),
            getattr(cls, "__name__", None),
            qual,
        ):
            if isinstance(cand, str) and "." in cand:
                dotted_candidate = cand
                break

        if dotted_candidate:
            left, right = dotted_candidate.rsplit(".", 1)
            if (not mod) or mod == "builtins":
                mod = left
                qual = right

        if not mod:
            mod = "builtins"
        if not qual:
            qual = fallback

        return mod, qual

    @staticmethod
    def full_namespace(obj: Any, *, fallback: str = "") -> str:
        """Return the fully-qualified dotted name for *obj*.

        Examples
        --------
        >>> ObjectSerde.full_namespace(int)
        'builtins.int'
        """
        mod, qual = ObjectSerde.module_and_name(obj, fallback=fallback)
        return f"{mod}.{qual}"
