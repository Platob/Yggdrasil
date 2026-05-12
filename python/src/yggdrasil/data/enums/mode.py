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

from enum import IntEnum
from typing import Optional, Union

__all__ = ["Mode", "ModeLike", "STR_MAPPING"]


ModeLike = Union["Mode", str, int]


class Mode(IntEnum):
    AUTO = 0
    READ_ONLY = 1
    OVERWRITE = 2
    APPEND = 3
    IGNORE = 4
    UPSERT = 5
    MERGE = 6
    TRUNCATE = 7
    ERROR_IF_EXISTS = 8

    @property
    def is_read_only(self) -> bool:
        """True for read-only modes — i.e. no write disposition."""
        return self is Mode.READ_ONLY

    @property
    def allows_write(self) -> bool:
        """True when the mode admits any write disposition."""
        return self is not Mode.READ_ONLY

    @property
    def readable(self) -> bool:
        """True when the mode admits reads.

        Every :class:`Mode` canonically resolves to a ``+`` POSIX form
        (``rb``, ``rb+``, ``wb+``, ``ab+``, ``xb+``) — all of those
        admit reads. Only the strict :data:`READ_ONLY` ``rb`` and
        :data:`IGNORE` (which is no-op) deny writes; nothing here
        denies reads.
        """
        return True

    @property
    def writable(self) -> bool:
        """True when the mode admits writes — alias of :attr:`allows_write`."""
        return self is not Mode.READ_ONLY

    @property
    def appendable(self) -> bool:
        """True when writes append at EOF rather than at the cursor.

        Only :data:`APPEND` carries POSIX ``O_APPEND`` semantics; every
        other write mode positions writes at the explicit cursor.
        """
        return self is Mode.APPEND

    def __contains__(self, item: object) -> bool:
        """Delegate substring checks to :attr:`os_mode`.

        pandas / pyarrow / zipfile inspect file-like ``.mode`` with
        ``"b" in handle.mode`` to dispatch binary vs text reads. The
        IO surface returns the typed :class:`Mode` enum, so we make
        the enum behave like its POSIX form for these checks instead
        of forcing every consumer to reach for ``mode.os_mode``.
        """
        return item in self.os_mode

    @property
    def os_mode(self) -> str:
        """Stdlib :func:`open` mode string for this :class:`Mode`.

        - :attr:`READ_ONLY` → ``"rb"``
        - :attr:`OVERWRITE` / :attr:`TRUNCATE` → ``"wb+"``
        - :attr:`APPEND`    → ``"ab+"``
        - :attr:`ERROR_IF_EXISTS` → ``"xb+"``
        - everything else (AUTO, IGNORE, UPSERT, MERGE) → ``"rb+"``
          (in-place edit; the disposition is enforced higher up).
        """
        return _MODE_TO_OS.get(self, "rb+")

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

        # IntEnum members compare equal to ints; allow lookup by the
        # integer value too so persisted Mode codes round-trip.
        if isinstance(value, int) and not isinstance(value, bool):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(
                    f"Cannot parse {value!r} as a Mode. Accepted "
                    f"integer codes: {sorted(int(m) for m in cls)}."
                )

        if not isinstance(value, str):
            raise TypeError(
                f"Mode.parse expected a string or Mode, got "
                f"{type(value).__name__}: {value!r}"
            )

        # Fast path: most callers pass an already-canonical token
        # (``"overwrite"`` / ``"OVERWRITE"`` / ``"rb"`` / ``"wb+"``).
        # A single dict probe resolves them without paying any string
        # normalisation cost.
        hit = _MODE_LOOKUP.get(value)
        if hit is not None:
            return hit

        # Normalize once. Lower-case, strip whitespace, replace
        # "error-if-exists" → "error_if_exists" style.
        normalized = value.strip().lower().replace("-", "_")
        if not normalized:
            return default if default is not None else cls.AUTO

        hit = _MODE_LOOKUP.get(normalized)
        if hit is not None:
            return hit

        # OS-mode parser. Handles the full open() grammar including
        # `+` variants ("rb+", "wt+", "ab+") and any character order
        # ("r+b" / "+rb"). Raises ValueError if the string isn't a
        # valid mode.
        os_match = _parse_os_mode(normalized)
        if os_match is not None:
            return os_match

        raise ValueError(
            f"Cannot parse {value!r} as a Mode. Accepted "
            f"values: {sorted(m.name for m in cls)} or aliases "
            f"like {sorted(STR_MAPPING)}."
        )


# ---------------------------------------------------------------------------
# Mode → POSIX open() string lookup
# ---------------------------------------------------------------------------
#
# Built once at import; consulted by :attr:`Mode.os_mode` and the
# :class:`Holder.open(mode=...)` plumbing in ``yggdrasil.io.holder``.

_MODE_TO_OS: dict["Mode", str] = {}


def _seed_mode_to_os() -> None:
    global _MODE_TO_OS
    _MODE_TO_OS = {
        Mode.READ_ONLY: "rb",
        Mode.OVERWRITE: "wb+",
        Mode.TRUNCATE: "wb+",
        Mode.APPEND: "ab+",
        Mode.ERROR_IF_EXISTS: "xb+",
        Mode.AUTO: "rb+",
        Mode.IGNORE: "rb+",
        Mode.UPSERT: "rb+",
        Mode.MERGE: "rb+",
    }


_seed_mode_to_os()


# ---------------------------------------------------------------------------
# OS-mode parser — POSIX open(2) / stdlib open() grammar
# ---------------------------------------------------------------------------


_OS_MODE_CHARS = frozenset("rwaxbt+")


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
    # strategies (alias map, enum value, etc.). The set check beats
    # an ``all(...)`` generator over the string for the typical 1-3
    # character mode strings ``open(2)`` accepts.
    if not _OS_MODE_CHARS.issuperset(s):
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
        # 'r', 'rb', 'rt' — read-only. '+' variants ('r+', 'rb+',
        # 'rt+') admit in-place writes and resolve to AUTO so a
        # write-side caller can pick a disposition. Pure-read forms
        # surface as Mode.READ_ONLY so callers can refuse writes
        # explicitly without parsing the mode string themselves.
        if "+" in s:
            return Mode.AUTO
        return Mode.READ_ONLY
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
    "r": Mode.READ_ONLY,
    "rb": Mode.READ_ONLY,
    "rt": Mode.READ_ONLY,
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

    # Read-only — no write disposition admitted.
    "ro": Mode.READ_ONLY,
    "read": Mode.READ_ONLY,
    "readonly": Mode.READ_ONLY,
    "read_only": Mode.READ_ONLY,

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


def _build_mode_lookup() -> dict[str, Mode]:
    """Pre-compute every accepted spelling → :class:`Mode` member.

    Folds :data:`STR_MAPPING` with the canonical member names
    (``"OVERWRITE"`` / ``"APPEND"`` / …), every OS-mode permutation
    (``"rb"``, ``"rb+"``, ``"r+b"``, ``"+rb"``, ``"wb+"``, …), and the
    upper-case variant of each alias so :meth:`Mode.from_` resolves
    common shapes with a single ``dict.get`` and no string allocation.
    """
    out: dict[str, Mode] = {}
    for alias, mode in STR_MAPPING.items():
        out[alias] = mode
        if alias and alias != alias.upper():
            out[alias.upper()] = mode

    # Member names — "OVERWRITE" / "APPEND" / "ERROR_IF_EXISTS" / ...
    for member in Mode:
        out[member.name] = member
        out[member.name.lower()] = member

    # Full POSIX permutations: every primary (r/w/a/x) × every subset
    # of {b, t, +} in every order. There are 1 × (1 + 3 + 3*2 + 3*2*1)
    # × 4 = 64 strings total — small enough to enumerate.
    from itertools import permutations
    for primary, target in (
        ("r", Mode.READ_ONLY),
        ("w", Mode.OVERWRITE),
        ("a", Mode.APPEND),
        ("x", Mode.ERROR_IF_EXISTS),
    ):
        for flags_subset in ((), ("b",), ("t",), ("+",),
                             ("b", "+"), ("t", "+"),
                             ("b", "t"), ("b", "t", "+")):
            # ``b`` and ``t`` are mutually exclusive — skip mixed.
            if "b" in flags_subset and "t" in flags_subset:
                continue
            for perm in permutations((primary,) + flags_subset):
                s = "".join(perm)
                # ``r`` variants: only the bare-read forms resolve to
                # READ_ONLY; the ``+`` family resolves to AUTO.
                if primary == "r" and "+" in s:
                    out.setdefault(s, Mode.AUTO)
                else:
                    out.setdefault(s, target)
    return out


_MODE_LOOKUP: dict[str, Mode] = _build_mode_lookup()