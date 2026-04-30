import hashlib
import re

try:
    from yggdrasil.data.cast.options import CastOptions
except ImportError:
    CastOptions = None


__all__ = [
    "get_cast_options_class",
    "safe_constraint_name",
]


_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_]+")
_MULTI_UNDERSCORE = re.compile(r"_+")


def get_cast_options_class():
    global CastOptions
    if CastOptions is None:
        from yggdrasil.data.cast.options import CastOptions as _CastOptions
        CastOptions = _CastOptions
    return CastOptions


def _sanitize(s: str) -> str:
    """Normalize a string to SQL-identifier-safe chars: [A-Za-z0-9_]."""
    s = _INVALID_CHARS.sub("_", s)
    s = _MULTI_UNDERSCORE.sub("_", s)
    return s.strip("_")


def safe_constraint_name(
    obj: str | list[str],
    limit: int = 256,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Generate a deterministic, length-bounded, identifier-safe constraint name.

    Sanitizes all parts to ``[A-Za-z0-9_]``, joins with ``__``, wraps with
    ``prefix`` and ``suffix``, and returns it verbatim if it fits within
    ``limit``. Otherwise, replaces the middle with a BLAKE2b hex digest
    sized to fit, keeping the prefix and suffix intact.
    """
    parts = [obj] if isinstance(obj, str) else list(obj)
    name = "__".join(filter(None, (_sanitize(p) for p in parts)))

    prefix = _sanitize(prefix)
    suffix = _sanitize(suffix)

    sep_p = "_" if prefix and name else ""
    sep_s = "_" if suffix and name else ""
    full = f"{prefix}{sep_p}{name}{sep_s}{suffix}"

    if len(full) <= limit:
        return full

    budget = limit - len(prefix) - len(suffix) - len(sep_p) - len(sep_s)
    if budget <= 0:
        raise ValueError(
            f"prefix ({len(prefix)}) + suffix ({len(suffix)}) "
            f"leave no room for a hash within limit={limit}"
        )

    digest_size = min(64, budget // 2)
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=digest_size).hexdigest()
    return f"{prefix}{sep_p}{digest}{sep_s}{suffix}"