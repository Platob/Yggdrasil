# yggdrasil.pyutils.serde.py

from typing import Any

__all__ = [
    "ObjectSerde",
]

class ObjectSerde:

    @staticmethod
    def full_namespace(obj: Any, *, fallback: str = "") -> str:
        cls = obj if isinstance(obj, type) else getattr(obj, "__class__", None)
        if cls is None:
            return fallback

        mod = getattr(cls, "__module__", None)
        qual = getattr(cls, "__qualname__", None) or getattr(cls, "__name__", None)
        if not mod or not qual:
            return fallback

        return f"{mod}.{qual}"
