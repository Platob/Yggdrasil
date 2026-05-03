"""Save mode enum — write-time disposition for any sink.

Covers the standard Spark / Delta / write-API set
(``OVERWRITE`` / ``APPEND`` / ``IGNORE`` / ``UPSERT`` /
``TRUNCATE`` / ``ERROR_IF_EXISTS``) plus an ``AUTO`` sentinel that
lets the implementation pick.

Parsing accepts three input shapes:

1. **A :class:`Mode`** — returned as-is (idempotent).
2. **A human-readable alias** — ``"overwrite"``, ``"replace"``,
   ``"append"``, ``"add"``, … (full list below).
3. **A POSIX/stdlib ``open()`` mode string** — ``"rb"``, ``"wb"``,
   ``"ab"``, ``"xb"``, plus ``"+"`` / ``"t"`` / ``"b"`` variants.
   Parsed structurally rather than enumerated, so ``"rb+"`` and
   ``"r+b"`` and ``"+rb"`` all resolve correctly.

Mapping for OS modes:

- ``r`` / ``rb`` / ``rt`` (no ``+``)        → :data:`AUTO`
  (read-only — there's no write to dispose of).
- ``r+`` / ``rb+`` / ``rt+``                → :data:`AUTO`
  (in-place edit — neither truncate nor append nor exclusive).
- ``w`` / ``wb`` / ``wt`` / ``w+`` / …      → :data:`OVERWRITE`
  (POSIX ``O_TRUNC``).
- ``a`` / ``ab`` / ``at`` / ``a+`` / …      → :data:`APPEND`
  (POSIX ``O_APPEND``).
- ``x`` / ``xb`` / ``xt`` / ``x+`` / …      → :data:`ERROR_IF_EXISTS`
  (POSIX ``O_EXCL``).

Note: there's no OS-mode counterpart to :data:`UPSERT` or
:data:`TRUNCATE` (Spark-style "wipe and keep structure") — those
are SQL/dataset-level concepts and have no `open(2)` analog.
"""

from enum import Enum
from typing import Optional, Union

__all__ = ["Mode", "ModeLike", "STR_MAPPING"]


ModeLike = Union["Mode", str]


class Mode(str, Enum):
    AUTO = "auto"
    OVERWRITE = "overwrite"
    APPEND = "append"
    IGNORE = "ignore"
    UPSERT = "upsert"
    MERGE = "merge"
    TRUNCATE = "truncate"
    ERROR_IF_EXISTS = "error_if_exists"

    @classmethod
    def from_(
        cls,
        value: Optional[ModeLike],
        default: Optional["Mode"] = None,
    ) -> "Mode":
        """Normalize *value* into a :class:`Mode`.

        Accepts:

        - :class:`Mode` (returned as-is, idempotent).
        - Aliases like ``"overwrite"``, ``"OVERWRITE"``,
          ``"error-if-exists"``, ``"replace"``, ``"add"`` — see
          :data:`STR_MAPPING`.
        - POSIX-style mode strings — ``"rb"``, ``"wb"``, ``"ab+"``,
          ``"x"``, ``"r+b"`` — parsed structurally; any combination
          of one primary character (``r``/``w``/``a``/``x``) plus
          optional ``b``/``t``/``+`` flags is accepted.
        - ``None`` → returns *default* if supplied, else
          :data:`Mode.AUTO`.

        Falls back to :class:`ValueError` for unrecognized strings.
        Numeric / non-string non-Mode inputs raise
        :class:`TypeError` — the input grammar is "string or enum,"
        not "anything stringifiable."
        """
        if isinstance(value, cls):
            return value

        if value is None:
            return default if default is not None else cls.AUTO

        if not isinstance(value, str):
            raise TypeError(
                f"Mode.parse expected a string or Mode, got "
                f"{type(value).__name__}: {value!r}"
            )

        # Normalize once. Lower-case, strip whitespace, replace
        # "error-if-exists" → "error_if_exists" style.
        normalized = value.strip().lower().replace("-", "_")
        if not normalized:
            return default if default is not None else cls.AUTO

        # Alias table first (covers "overwrite", "replace", "fail",
        # plus the simple OS modes that are also direct keys —
        # "w", "wb", "a", "ab", "i", "x", "xb").
        hit = STR_MAPPING.get(normalized)
        if hit is not None:
            return hit

        # OS-mode parser. Handles the full open() grammar including
        # `+` variants ("rb+", "wt+", "ab+") and any character order
        # ("r+b" / "+rb"). Raises ValueError if the string isn't a
        # valid mode.
        os_match = _parse_os_mode(normalized)
        if os_match is not None:
            return os_match

        # Last resort: try the enum's own value lookup. Mode
        # values are lower-snake-case, so "overwrite" / "append" /
        # "error_if_exists" land here cleanly. Anything else raises
        # ValueError with a helpful message.
        try:
            return cls(normalized)
        except ValueError:
            raise ValueError(
                f"Cannot parse {value!r} as a Mode. Accepted "
                f"values: {sorted(m.value for m in cls)} or aliases "
                f"like {sorted(STR_MAPPING)}."
            )


# ---------------------------------------------------------------------------
# OS-mode parser — POSIX open(2) / stdlib open() grammar
# ---------------------------------------------------------------------------


def _parse_os_mode(s: str) -> Optional[Mode]:
    """Parse an OS-style mode string into a :class:`Mode`.

    Returns ``None`` if *s* doesn't look like an OS mode at all
    (mix of unknown characters), letting the caller fall through
    to other parsing strategies. Raises :class:`ValueError` only
    if *s* looks OS-mode-shaped but is actually invalid (multiple
    primary mode characters, etc.).

    The grammar:

    - exactly one primary mode character: ``r``, ``w``, ``a``, ``x``
    - any subset of these flags, any order:
      ``b`` (binary), ``t`` (text), ``+`` (read+write)
    - no other characters

    Returns:

    - ``r``, ``r+``    → :data:`AUTO` (read-only or in-place edit)
    - ``w``, ``w+``    → :data:`OVERWRITE`
    - ``a``, ``a+``    → :data:`APPEND`
    - ``x``, ``x+``    → :data:`ERROR_IF_EXISTS`
    """
    if not s:
        return None

    # Quick rejection: if any character is not a valid mode letter,
    # this isn't an OS mode at all — let the caller try other
    # strategies (alias map, enum value, etc.).
    if not all(c in "rwaxbt+" for c in s):
        return None

    # Count primary chars. Exactly one must be present.
    primaries = [c for c in s if c in "rwax"]
    if len(primaries) != 1:
        # 0 primaries: just "+" or "b" — not a valid mode.
        # 2+ primaries: "rw" / "wa" — invalid.
        # In both cases this looks like a mode but isn't a valid
        # one; raise so the caller doesn't silently fall through to
        # the enum lookup which would also fail with a less
        # helpful message.
        raise ValueError(
            f"Invalid OS mode string {s!r}: must contain exactly one "
            f"of 'r', 'w', 'a', 'x' (got {len(primaries)})."
        )

    # Reject mixing 'b' and 't' (binary AND text) — POSIX rejects
    # this in stdlib open() too. Not strictly relevant for
    # Mode (we don't care whether bytes are wrapped in text)
    # but it's a sign of caller confusion worth surfacing.
    if "b" in s and "t" in s:
        raise ValueError(
            f"Invalid OS mode string {s!r}: 'b' and 't' are mutually "
            "exclusive."
        )

    primary = primaries[0]
    if primary == "r":
        # 'r', 'rb', 'rt', 'r+', 'rb+', 'rt+' — no write disposition
        # (or in-place edit for '+'). Both collapse to AUTO at this
        # layer; the caller's write semantics happen elsewhere.
        return Mode.AUTO
    if primary == "w":
        return Mode.OVERWRITE
    if primary == "a":
        return Mode.APPEND
    if primary == "x":
        return Mode.ERROR_IF_EXISTS

    # Unreachable — len(primaries)==1 and primary in "rwax" is enforced.
    return None


# ---------------------------------------------------------------------------
# Alias table — explicit shorthands and human-readable forms
# ---------------------------------------------------------------------------
#
# Most OS-mode strings are also handled by _parse_os_mode (the
# structural parser), but we keep the simplest forms in here too
# so the table remains a one-glance summary of every accepted
# shorthand. The two paths produce the same result for any mode
# they both handle.

STR_MAPPING = {
    # OS modes — simple primary forms (parser handles the rest).
    "r": Mode.AUTO,
    "rb": Mode.AUTO,
    "rt": Mode.AUTO,
    "w": Mode.OVERWRITE,
    "wb": Mode.OVERWRITE,
    "wt": Mode.OVERWRITE,
    "a": Mode.APPEND,
    "ab": Mode.APPEND,
    "at": Mode.APPEND,
    "x": Mode.ERROR_IF_EXISTS,
    "xb": Mode.ERROR_IF_EXISTS,
    "xt": Mode.ERROR_IF_EXISTS,

    # Human-readable / Spark / SQL aliases.
    "write": Mode.OVERWRITE,
    "overwrite": Mode.OVERWRITE,
    "replace": Mode.OVERWRITE,
    "clobber": Mode.OVERWRITE,

    "append": Mode.APPEND,
    "add": Mode.APPEND,

    "i": Mode.IGNORE,
    "ignore": Mode.IGNORE,
    "skip": Mode.IGNORE,

    "up": Mode.UPSERT,
    "update": Mode.UPSERT,
    "upsert": Mode.UPSERT,

    # MERGE — try the engine-native MERGE statement first; the backend may
    # fall back to a delete-then-insert if MERGE is unavailable.
    "merge": Mode.MERGE,

    "trunc": Mode.TRUNCATE,
    "truncate": Mode.TRUNCATE,

    "error": Mode.ERROR_IF_EXISTS,
    "fail": Mode.ERROR_IF_EXISTS,
    "raise": Mode.ERROR_IF_EXISTS,
    "errorifexists": Mode.ERROR_IF_EXISTS,
    "error_if_exists": Mode.ERROR_IF_EXISTS,

    # AUTO — empty / "default" / "auto".
    "": Mode.AUTO,
    "auto": Mode.AUTO,
    "default": Mode.AUTO,
}