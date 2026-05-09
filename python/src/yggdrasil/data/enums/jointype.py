"""Centralized join-type enum shared by tabular join surfaces.

Different engines spell the same join concept differently — pyarrow
uses space-separated lower-case (``"left anti"``, ``"left outer"``),
polars collapses to a single token (``"left"`` / ``"anti"``), SQL
uppercases and appends ``JOIN`` (``"LEFT ANTI JOIN"``). Centralizing
the canonical token set here lets a cross-engine surface accept any
of those spellings and route through a single normalized member.

:class:`JoinType` provides:

* canonical members for the nine join kinds the supported engines
  understand (``inner``, ``left/right/full outer``,
  ``left/right semi``, ``left/right anti``, ``cross``);
* :meth:`from_` for forgiving string / integer / :class:`JoinType` /
  ``None`` input with alias support (``"left"`` → ``LEFT_OUTER``,
  ``"anti"`` → ``LEFT_ANTI``, ``"LEFT JOIN"`` → ``LEFT_OUTER``);
* :meth:`is_valid` for boolean checks without raising;
* :attr:`arrow` / :attr:`polars` / :attr:`sql` for engine-specific
  spellings, plus :attr:`is_outer` / :attr:`is_semi` / :attr:`is_anti`
  / :attr:`is_inner` / :attr:`is_cross` predicates.

Subclasses :class:`IntEnum` so members are stable, ordered, and
serializable as integer codes — pass ``join_type.arrow`` to
``pa.Table.join`` and ``join_type.polars`` to ``pl.DataFrame.join``.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any, ClassVar

__all__ = [
    "JoinType",
]


# ---------------------------------------------------------------------------
# Alias table — accepted spellings that normalize to a canonical Arrow token.
# Covers Arrow's space-separated form, polars' single-word form,
# SQL's uppercase + ``JOIN`` form, plus the unqualified ``semi`` /
# ``anti`` / ``outer`` shorthands (which conventionally mean LEFT
# semi / LEFT anti / FULL outer).
# ---------------------------------------------------------------------------

_JOINTYPE_ALIASES: dict[str, str] = {
    # Inner.
    "inner":             "inner",
    "inner_join":        "inner",

    # Left outer.
    "left":              "left outer",
    "left_outer":        "left outer",
    "left_outer_join":   "left outer",
    "left_join":         "left outer",
    "left outer":        "left outer",
    "left outer join":   "left outer",
    "left join":         "left outer",

    # Right outer.
    "right":             "right outer",
    "right_outer":       "right outer",
    "right_outer_join":  "right outer",
    "right_join":        "right outer",
    "right outer":       "right outer",
    "right outer join":  "right outer",
    "right join":        "right outer",

    # Full outer (a.k.a. outer / full).
    "full":              "full outer",
    "outer":             "full outer",
    "full_outer":        "full outer",
    "full_outer_join":   "full outer",
    "full_join":         "full outer",
    "full outer":        "full outer",
    "full outer join":   "full outer",
    "full join":         "full outer",
    "outer join":        "full outer",

    # Cross.
    "cross":             "cross",
    "cross_join":        "cross",
    "cross join":        "cross",

    # Left semi (also the unqualified ``semi``).
    "semi":              "left semi",
    "left_semi":         "left semi",
    "left_semi_join":    "left semi",
    "left semi":         "left semi",
    "left semi join":    "left semi",
    "semi_join":         "left semi",

    # Right semi.
    "right_semi":        "right semi",
    "right_semi_join":   "right semi",
    "right semi":        "right semi",
    "right semi join":   "right semi",

    # Left anti (also the unqualified ``anti``).
    "anti":              "left anti",
    "left_anti":         "left anti",
    "left_anti_join":    "left anti",
    "left anti":         "left anti",
    "left anti join":    "left anti",
    "anti_join":         "left anti",

    # Right anti.
    "right_anti":        "right anti",
    "right_anti_join":   "right anti",
    "right anti":        "right anti",
    "right anti join":   "right anti",
}


class JoinType(IntEnum):
    """Canonical join kind for tabular join surfaces.

    Use :meth:`from_` when accepting external input — it canonicalizes
    aliases (``"left"``, ``"LEFT JOIN"``, ``"anti"``, ``"outer"``,
    integer codes) to a member and raises :class:`ValueError` for
    unknown tokens.

    Pass :attr:`arrow` / :attr:`polars` / :attr:`sql` to engine join
    APIs — pyarrow's :meth:`pa.Table.join` and polars'
    :meth:`DataFrame.join` accept different spellings of the same
    concept, and storing the integer code keeps the enum a clean
    discriminator without binding to either spelling.
    """

    INNER = 0
    LEFT_OUTER = 1
    RIGHT_OUTER = 2
    FULL_OUTER = 3
    LEFT_SEMI = 4
    RIGHT_SEMI = 5
    LEFT_ANTI = 6
    RIGHT_ANTI = 7
    CROSS = 8

    # ── Convenience aliases for callers that prefer the short name ──
    LEFT: ClassVar["JoinType"]
    RIGHT: ClassVar["JoinType"]
    FULL: ClassVar["JoinType"]
    OUTER: ClassVar["JoinType"]
    SEMI: ClassVar["JoinType"]
    ANTI: ClassVar["JoinType"]

    # ── Coercion ────────────────────────────────────────────────────────────

    @classmethod
    def from_(cls, value: Any, *, default: Any = ...) -> "JoinType":
        """Coerce any Python value into a :class:`JoinType`.

        Accepts:

        * :class:`JoinType` (returned as-is);
        * any string the alias table or canonical Arrow tokens know —
          ``"inner"``, ``"left"``, ``"left outer"``, ``"LEFT JOIN"``,
          ``"anti"``, ``"left_anti"``, ``"outer"``, ``"cross"``, …;
          mixed case, hyphens / underscores / spaces all normalize;
        * an integer code matching a member's value (round-trips with
          ``int(JoinType.X)``);
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
            raise ValueError("JoinType cannot be derived from None")

        # IntEnum members compare equal to ints; allow integer-code
        # lookups too so persisted JoinType codes round-trip.
        if isinstance(value, int) and not isinstance(value, bool):
            try:
                return cls(value)
            except ValueError:
                if default is not ...:
                    return default
                raise ValueError(
                    f"Cannot parse {value!r} as a JoinType. Accepted "
                    f"integer codes: {sorted(int(m) for m in cls)}."
                )

        if isinstance(value, str):
            return cls._from_str(value, default=default)

        if default is not ...:
            return default
        raise TypeError(
            f"Cannot derive JoinType from {type(value).__name__}: {value!r}"
        )

    @classmethod
    def _from_str(cls, value: str, *, default: Any = ...) -> "JoinType":
        token = value.strip().lower().replace("-", "_")
        if not token:
            if default is not ...:
                return default
            raise ValueError("JoinType string cannot be empty")

        # Try the alias table both with underscores intact and with
        # them flipped to spaces — covers Arrow's space-separated form
        # when the caller hands in ``"left_anti"`` style.
        canonical = (
            _JOINTYPE_ALIASES.get(token)
            or _JOINTYPE_ALIASES.get(token.replace("_", " "))
        )
        if canonical is not None:
            return _ARROW_TO_MEMBER[canonical]

        # Last resort: try the enum's name lookup ("LEFT_OUTER", …).
        try:
            return cls[token.upper().replace(" ", "_")]
        except KeyError:
            pass

        if default is not ...:
            return default
        raise ValueError(
            f"Unknown join type: {value!r}. "
            f"Valid tokens are: {sorted(_JOINTYPE_ALIASES)!r} "
            f"or member names {sorted(m.name for m in cls)!r}."
        )

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """Return ``True`` when :meth:`from_` would succeed for *value*."""
        try:
            cls.from_(value)
            return True
        except (TypeError, ValueError):
            return False

    # ── Engine-specific spellings ───────────────────────────────────────────

    @property
    def arrow(self) -> str:
        """pyarrow's :meth:`pa.Table.join` ``join_type`` token."""
        return _ARROW_MAP[self]

    @property
    def polars(self) -> str:
        """polars' :meth:`DataFrame.join` ``how`` token.

        Polars has no built-in ``right semi`` / ``right anti`` form —
        those raise :class:`NotImplementedError`. Swap operands and use
        the left-side equivalent.
        """
        try:
            return _POLARS_MAP[self]
        except KeyError:
            raise NotImplementedError(
                f"polars has no equivalent of JoinType.{self.name!r}; "
                f"swap operands and use {_polars_swap_hint(self)!r} "
                "instead."
            )

    @property
    def sql(self) -> str:
        """SQL ``JOIN`` clause keyword (uppercase, with trailing ``JOIN``)."""
        return _SQL_MAP[self]

    # ── Predicates ──────────────────────────────────────────────────────────

    @property
    def is_inner(self) -> bool:
        """``True`` for :attr:`INNER`."""
        return self is JoinType.INNER

    @property
    def is_outer(self) -> bool:
        """``True`` for any of the three outer joins."""
        return self in (
            JoinType.LEFT_OUTER,
            JoinType.RIGHT_OUTER,
            JoinType.FULL_OUTER,
        )

    @property
    def is_semi(self) -> bool:
        """``True`` for :attr:`LEFT_SEMI` / :attr:`RIGHT_SEMI`."""
        return self in (JoinType.LEFT_SEMI, JoinType.RIGHT_SEMI)

    @property
    def is_anti(self) -> bool:
        """``True`` for :attr:`LEFT_ANTI` / :attr:`RIGHT_ANTI`."""
        return self in (JoinType.LEFT_ANTI, JoinType.RIGHT_ANTI)

    @property
    def is_cross(self) -> bool:
        """``True`` for :attr:`CROSS`."""
        return self is JoinType.CROSS

    # ── Dunder ──────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return self.arrow


# Convenience short-name aliases — assigned after the class body so the
# Enum machinery doesn't try to register them as separate members.
JoinType.LEFT = JoinType.LEFT_OUTER
JoinType.RIGHT = JoinType.RIGHT_OUTER
JoinType.FULL = JoinType.FULL_OUTER
JoinType.OUTER = JoinType.FULL_OUTER
JoinType.SEMI = JoinType.LEFT_SEMI
JoinType.ANTI = JoinType.LEFT_ANTI


_ARROW_MAP: dict["JoinType", str] = {
    JoinType.INNER:       "inner",
    JoinType.LEFT_OUTER:  "left outer",
    JoinType.RIGHT_OUTER: "right outer",
    JoinType.FULL_OUTER:  "full outer",
    JoinType.LEFT_SEMI:   "left semi",
    JoinType.RIGHT_SEMI:  "right semi",
    JoinType.LEFT_ANTI:   "left anti",
    JoinType.RIGHT_ANTI:  "right anti",
    JoinType.CROSS:       "cross",
}


_ARROW_TO_MEMBER: dict[str, "JoinType"] = {v: k for k, v in _ARROW_MAP.items()}


_POLARS_MAP: dict["JoinType", str] = {
    JoinType.INNER:       "inner",
    JoinType.LEFT_OUTER:  "left",
    JoinType.RIGHT_OUTER: "right",
    JoinType.FULL_OUTER:  "full",
    JoinType.CROSS:       "cross",
    JoinType.LEFT_SEMI:   "semi",
    JoinType.LEFT_ANTI:   "anti",
}


_SQL_MAP: dict["JoinType", str] = {
    JoinType.INNER:       "INNER JOIN",
    JoinType.LEFT_OUTER:  "LEFT OUTER JOIN",
    JoinType.RIGHT_OUTER: "RIGHT OUTER JOIN",
    JoinType.FULL_OUTER:  "FULL OUTER JOIN",
    JoinType.CROSS:       "CROSS JOIN",
    JoinType.LEFT_SEMI:   "LEFT SEMI JOIN",
    JoinType.RIGHT_SEMI:  "RIGHT SEMI JOIN",
    JoinType.LEFT_ANTI:   "LEFT ANTI JOIN",
    JoinType.RIGHT_ANTI:  "RIGHT ANTI JOIN",
}


def _polars_swap_hint(jt: "JoinType") -> str:
    """Suggest the polars-supported flip of an unsupported right-side join."""
    if jt is JoinType.RIGHT_SEMI:
        return _POLARS_MAP[JoinType.LEFT_SEMI]
    if jt is JoinType.RIGHT_ANTI:
        return _POLARS_MAP[JoinType.LEFT_ANTI]
    return jt.arrow
