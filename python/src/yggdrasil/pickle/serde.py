# yggdrasil/pyutils/serde.py
"""
Object serialisation / deserialisation utilities.

Provides:
    ObjectSerde   — namespace / type-identity helpers + full smartpickle
                    integration with PickleOptions, Codec auto-compression,
                    and yggdrasil BytesIO support.
"""

from __future__ import annotations

import inspect
import logging
from types import ModuleType

from typing import Any, Tuple

__all__ = [
    "ObjectSerde",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ObjectSerde
# ---------------------------------------------------------------------------

class ObjectSerde:
    # ==================================================================
    # Namespace utilities (unchanged from original)
    # ==================================================================

    @staticmethod
    def module_and_name(obj: Any, *, fallback: str = "") -> tuple[str, str]:
        """
        Return (module, qualname) for obj.

        Robust across Python objects, including many C-extension / PyArrow objects
        where __module__ may be missing or misleading.

        Notes
        -----
        - For instances, we primarily identify the type (cls = type(obj)).
        - For method descriptors, we use __objclass__ when available.
        - For extension types, we may parse a dotted qualname (e.g. "pyarrow.lib.Table")
          to recover a better module path.
        """
        if obj is None:
            return "builtins", fallback or "None"

        # Modules: easy mode
        if isinstance(obj, ModuleType):
            return (getattr(obj, "__name__", None) or "builtins"), fallback or getattr(obj, "__name__", "module")

        # Unwrap bound methods / functions / descriptors when possible
        # (inspect.unwrap helps for decorated callables)
        try:
            unwrapped = inspect.unwrap(obj)  # type: ignore[arg-type]
        except Exception:
            unwrapped = obj

        # Determine a "class-like" anchor for naming
        cls = None
        if inspect.isclass(unwrapped):
            cls = unwrapped
        else:
            # Method descriptors often expose __objclass__
            cls = getattr(unwrapped, "__objclass__", None) or type(unwrapped)

        # Helper: pick a reasonable qualname for any object
        def _qualname(x: Any) -> str:
            return (
                getattr(x, "__qualname__", None)
                or getattr(x, "__name__", None)
                or getattr(type(x), "__qualname__", None)
                or getattr(type(x), "__name__", None)
                or fallback
            )

        # 1) Try the obvious: object/module metadata
        mod = getattr(unwrapped, "__module__", None)
        qual = getattr(unwrapped, "__qualname__", None) or getattr(unwrapped, "__name__", None)

        # 2) If missing, use class metadata
        if not mod:
            mod = getattr(cls, "__module__", None)
        if not qual:
            qual = _qualname(unwrapped)

        # 3) If module is still suspicious (None, "builtins"), try harder.
        # Some extension types report __module__="builtins" but show a dotted qualname.
        if not mod or mod == "builtins":
            # Attempt: class module first (often correct for extension instances)
            cls_mod = getattr(getattr(unwrapped, "__class__", None), "__module__", None)
            if cls_mod and cls_mod != "builtins":
                mod = cls_mod

        # 4) Recover module by parsing dotted names like "pyarrow.lib.Table"
        # We try: type(obj).__module__/__qualname__ are sometimes wrong, but the repr string isn't.
        # For classes, __qualname__ might already be dotted; for extension types, __name__ can be dotted.
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
            # If we have something like "pyarrow.lib.Table" and module is missing or "builtins",
            # split at last dot.
            left, right = dotted_candidate.rsplit(".", 1)
            if (not mod) or mod == "builtins":
                mod = left
                qual = right

        # 5) Final fallback: builtins if still empty
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
        >>> ObjectSerde.full_namespace(pd.DataFrame)
        'pandas.core.frame.DataFrame'
        """
        mod, qual = ObjectSerde.module_and_name(obj, fallback=fallback)
        return f"{mod}.{qual}"
