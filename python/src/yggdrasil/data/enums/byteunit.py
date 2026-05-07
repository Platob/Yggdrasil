"""Centralized byte-unit enum for memory / capacity / spill thresholds.

Across Yggdrasil, a "size in bytes" appears in many places: ``Memory``'s
``spill_bytes`` threshold, the Arrow tabular spill cutoff, request /
response body limits, codec hints, hash batch sizes. Before this
module each call site rolled its own ``128 * 1024 * 1024`` /
``"128MB"`` parsing and the convention drifted between SI (base
1000) and IEC (base 1024).

:class:`ByteUnit` pins the convention to **base 1024 — IEC binary
units** (``KiB`` / ``MiB`` / ``GiB`` / …) and exposes:

* canonical members for ``B`` through ``PiB``;
* :meth:`from_` for forgiving string / int / ``ByteUnit`` input;
* :meth:`parse_size` for "give me an integer byte count" — the most
  common need at config / API boundaries (``"128 MB"`` → ``134217728``);
* :attr:`bytes` (== member value) for "multiply this by N to get N
  units of bytes": ``128 * ByteUnit.MIB`` is just an int and reads at
  the call site.

The enum subclasses :class:`int` so a member is interchangeable with
its byte count everywhere a plain integer would land — annotations
like ``spill_bytes: int = 128 * ByteUnit.MIB`` work without a cast.

SI (base-1000) units are intentionally **not** included. Memory
sizing in this codebase is uniformly binary; mixing 10⁶ vs 2²⁰ at
threshold boundaries is the kind of subtle bug this enum exists to
prevent.
"""
from __future__ import annotations

import re
from enum import IntEnum
from typing import Any, ClassVar, Union

__all__ = ["ByteUnit"]


# ---------------------------------------------------------------------------
# Alias table — accepted spellings that normalize to a canonical member.
# Both IEC (``KiB``) and the colloquial short forms (``KB`` meaning
# kibibyte in the binary-memory sense) are accepted; case-insensitive.
# ---------------------------------------------------------------------------

_BYTEUNIT_ALIASES: dict[str, str] = {
    "":     "B",
    "b":    "B",
    "byte": "B",
    "bytes": "B",
    "k":    "KIB",
    "kb":   "KIB",
    "kib":  "KIB",
    "kibibyte":  "KIB",
    "kibibytes": "KIB",
    "kilobyte":  "KIB",
    "kilobytes": "KIB",
    "m":    "MIB",
    "mb":   "MIB",
    "mib":  "MIB",
    "mebibyte":  "MIB",
    "mebibytes": "MIB",
    "megabyte":  "MIB",
    "megabytes": "MIB",
    "g":    "GIB",
    "gb":   "GIB",
    "gib":  "GIB",
    "gibibyte":  "GIB",
    "gibibytes": "GIB",
    "gigabyte":  "GIB",
    "gigabytes": "GIB",
    "t":    "TIB",
    "tb":   "TIB",
    "tib":  "TIB",
    "tebibyte":  "TIB",
    "tebibytes": "TIB",
    "terabyte":  "TIB",
    "terabytes": "TIB",
    "p":    "PIB",
    "pb":   "PIB",
    "pib":  "PIB",
    "pebibyte":  "PIB",
    "pebibytes": "PIB",
    "petabyte":  "PIB",
    "petabytes": "PIB",
}


# Quantity regex: ``"128 MB"`` / ``"128MB"`` / ``"1.5 GiB"`` / ``"1024"``.
# Anchored so junk on either side fails fast.
_QUANTITY_RE = re.compile(
    r"""\s*
        (?P<value>[+-]?\d+(?:\.\d+)?)
        \s*
        (?P<unit>[A-Za-z]*)
        \s*$
    """,
    re.VERBOSE,
)


class ByteUnit(IntEnum):
    """Canonical IEC binary byte-unit token + scalar value.

    Each member's value IS the byte count for one unit, so ``128 *
    ByteUnit.MIB`` reads naturally at the call site and slots into
    any ``int`` field. Use :meth:`parse_size` when accepting external
    config / API input — it canonicalizes ``"128 MB"`` /
    ``"1.5 GiB"`` / raw integers / ``ByteUnit`` members all to a
    plain integer byte count.
    """

    B   = 1
    KIB = 1024
    MIB = 1024 ** 2
    GIB = 1024 ** 3
    TIB = 1024 ** 4
    PIB = 1024 ** 5

    # Short colloquial aliases — these refer to the binary forms
    # (``KB == KiB == 1024``). Yggdrasil sizing is uniformly base-1024;
    # SI 10³ semantics are not supported anywhere in this enum.
    KB: ClassVar["ByteUnit"]
    MB: ClassVar["ByteUnit"]
    GB: ClassVar["ByteUnit"]
    TB: ClassVar["ByteUnit"]
    PB: ClassVar["ByteUnit"]

    # ── Coercion ────────────────────────────────────────────────────────────

    @classmethod
    def from_(cls, value: Any, *, default: Any = ...) -> "ByteUnit":
        """Coerce any Python value into a :class:`ByteUnit` member.

        Accepts:

        * :class:`ByteUnit` (returned as-is);
        * any string the alias table knows — ``B`` / ``KB`` / ``MiB`` /
          ``gigabyte`` / mixed case / trailing ``s``;
        * an integer matching a member's byte value (``1024`` →
          :attr:`KIB`);
        * ``None`` — returns *default* if supplied, else raises.

        ``default`` swallows unknown / unparseable input. Without it,
        unknown tokens raise :class:`ValueError` and unsupported types
        raise :class:`TypeError`.
        """
        if isinstance(value, cls):
            return value

        if value is None:
            if default is not ...:
                return default
            raise ValueError("ByteUnit cannot be derived from None")

        if isinstance(value, str):
            return cls._from_str(value, default=default)

        if isinstance(value, bool):
            # bool is-a int — reject explicitly to prevent
            # ByteUnit.from_(True) silently mapping to ``B``.
            if default is not ...:
                return default
            raise TypeError(
                f"Cannot derive ByteUnit from bool: {value!r}"
            )

        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError:
                if default is not ...:
                    return default
                raise ValueError(
                    f"Unknown ByteUnit value: {value!r}. "
                    f"Valid values: {sorted({m.value for m in cls})!r}"
                )

        if default is not ...:
            return default
        raise TypeError(
            f"Cannot derive ByteUnit from {type(value).__name__}: {value!r}"
        )

    @classmethod
    def _from_str(cls, value: str, *, default: Any = ...) -> "ByteUnit":
        token = value.strip().lower()
        canonical = _BYTEUNIT_ALIASES.get(token)
        if canonical is not None:
            return cls[canonical]

        # Bare uppercase member name (``"MIB"``).
        try:
            return cls[value.strip().upper()]
        except KeyError:
            pass

        if default is not ...:
            return default
        raise ValueError(
            f"Unknown byte unit: {value!r}. "
            f"Valid units: {sorted(set(_BYTEUNIT_ALIASES.values()))!r}"
        )

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """``True`` when :meth:`from_` would succeed for *value*."""
        try:
            cls.from_(value)
            return True
        except (TypeError, ValueError):
            return False

    @classmethod
    def parse_size(
        cls,
        value: Union[int, str, "ByteUnit", None],
        *,
        default: Any = ...,
    ) -> int:
        """Coerce a size-like value to an integer byte count.

        The single entry point for "config gave me a size, give me
        bytes." Accepts:

        * :class:`int` — passed through (must be non-negative);
        * :class:`ByteUnit` — its scalar value;
        * a quantity string ``"128 MB"`` / ``"1.5 GiB"`` / ``"512"``
          — parsed with this enum's IEC conventions;
        * a bare unit string ``"MiB"`` — yields one unit (``1024**2``);
        * ``None`` — returns *default* if supplied, else raises.

        Floating-point quantities round to the nearest byte (``"1.5 KiB"``
        → ``1536``). Negative values raise :class:`ValueError`.
        """
        if value is None:
            if default is not ...:
                return default
            raise ValueError("parse_size cannot accept None")

        if isinstance(value, bool):
            raise TypeError(
                f"parse_size: bool is not a byte size ({value!r})"
            )

        if isinstance(value, cls):
            return int(value.value)

        if isinstance(value, int):
            if value < 0:
                raise ValueError(
                    f"parse_size: negative byte count {value!r}"
                )
            return int(value)

        if isinstance(value, float):
            if value < 0:
                raise ValueError(
                    f"parse_size: negative byte count {value!r}"
                )
            return int(round(value))

        if isinstance(value, str):
            return cls._parse_size_str(value, default=default)

        if default is not ...:
            return default
        raise TypeError(
            f"parse_size: cannot derive byte count from "
            f"{type(value).__name__}: {value!r}"
        )

    @classmethod
    def _parse_size_str(
        cls, value: str, *, default: Any = ...,
    ) -> int:
        match = _QUANTITY_RE.match(value)
        if match is not None:
            scalar = float(match.group("value"))
            unit_token = match.group("unit") or "B"
            if scalar < 0:
                raise ValueError(
                    f"parse_size: negative byte count {value!r}"
                )
            unit = cls._from_str(unit_token, default=default)
            if unit is default and default is not ...:
                return default
            return int(round(scalar * int(unit.value)))

        # No leading number — accept a bare unit token as "one of that
        # unit" so ``"MiB"`` reads naturally as ``1024**2``.
        stripped = value.strip()
        if stripped:
            unit = cls._from_str(stripped, default=default)
            if unit is default and default is not ...:
                return default
            return int(unit.value)

        if default is not ...:
            return default
        raise ValueError(
            f"parse_size: cannot parse {value!r} as a byte quantity"
        )

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def bytes(self) -> int:
        """Bytes per one of this unit (== member value)."""
        return int(self.value)

    @property
    def short(self) -> str:
        """Short colloquial token: ``"B"`` / ``"KB"`` / ``"MB"`` / …."""
        return _SHORT_TOKENS[self.name]

    @property
    def iec(self) -> str:
        """Strict IEC token: ``"B"`` / ``"KiB"`` / ``"MiB"`` / …."""
        return _IEC_TOKENS[self.name]

    # ── Formatting ──────────────────────────────────────────────────────────

    @classmethod
    def format(cls, n: int, *, iec: bool = True, precision: int = 1) -> str:
        """Format an integer byte count as a human-readable string.

        Picks the largest unit at which *n* divides cleanly into a
        scalar ≥ 1, defaulting to IEC tokens (``"128 MiB"``); pass
        ``iec=False`` for the colloquial short form (``"128 MB"``).
        ``precision`` controls fractional digits.
        """
        if n < 0:
            return "-" + cls.format(-n, iec=iec, precision=precision)
        # Walk from largest to smallest until n / unit >= 1.
        for unit in (cls.PIB, cls.TIB, cls.GIB, cls.MIB, cls.KIB):
            if n >= unit.value:
                token = unit.iec if iec else unit.short
                return f"{n / unit.value:.{precision}f} {token}"
        return f"{n} B"

    # ── Dunder ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"ByteUnit.{self.name}"


# Convenience short-name aliases — assigned after the class body so the
# Enum machinery doesn't try to register them as separate members.
ByteUnit.KB = ByteUnit.KIB
ByteUnit.MB = ByteUnit.MIB
ByteUnit.GB = ByteUnit.GIB
ByteUnit.TB = ByteUnit.TIB
ByteUnit.PB = ByteUnit.PIB


_SHORT_TOKENS: dict[str, str] = {
    "B":   "B",
    "KIB": "KB",
    "MIB": "MB",
    "GIB": "GB",
    "TIB": "TB",
    "PIB": "PB",
}


_IEC_TOKENS: dict[str, str] = {
    "B":   "B",
    "KIB": "KiB",
    "MIB": "MiB",
    "GIB": "GiB",
    "TIB": "TiB",
    "PIB": "PiB",
}
