"""Per-statistic descriptors: what to compute, on which field(s), labeled.

A :class:`DataStatistic` bundles three things:

- **fields**: one or more :class:`~yggdrasil.data.data_field.Field` targets
  (a single column, or a compound key for partition-pruning).
- **kpis**: a set of :class:`KPI` values to compute over those fields —
  ``MIN``, ``MAX``, ``COUNT``, ``DISTINCT``, and the others.
- **label**: a human-readable identifier for the result row. Auto-derived
  from ``fields``/``kpis`` when the caller doesn't supply one.

It is frozen, so callers can share instances across threads without
defensive copies, and fully parseable from a string DSL::

    "amount.min"                  # scalar min
    "amount.min,max,count"        # three KPIs, one stat
    "(year,month).distinct"       # distinct tuple values, for partition pruning
    "name.null_count"

Downstream SQL engines read the resulting KPIs to decide insertion
strategy — bulk INSERT, MERGE, partition-pruned APPEND, COPY staging
sizing.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from enum import Enum
from typing import Any, Iterable, Union

import pyarrow as pa

from yggdrasil.data.data_field import Field

__all__ = ["DataStatistic", "KPI"]


class KPI(str, Enum):
    """Column-level KPI name. The string value doubles as the DSL token."""

    COUNT = "count"
    NULL_COUNT = "null_count"
    DISTINCT_COUNT = "distinct_count"
    DISTINCT = "distinct"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    MEAN = "mean"
    BYTE_SIZE = "byte_size"

    @classmethod
    def parse(cls, value: "str | KPI") -> "KPI":
        """Case-insensitive parse with a couple of tolerated aliases."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(f"KPI must be str or KPI, got {type(value).__name__}")
        key = value.strip().lower()
        aliases = {
            "distinct_values": cls.DISTINCT,
            "avg": cls.MEAN,
            "average": cls.MEAN,
            "nulls": cls.NULL_COUNT,
            "nunique": cls.DISTINCT_COUNT,
            "size": cls.BYTE_SIZE,
        }
        if key in aliases:
            return aliases[key]
        try:
            return cls(key)
        except ValueError:
            valid = sorted(m.value for m in cls)
            raise ValueError(
                f"Unknown KPI {value!r}. Valid KPIs: {valid}. "
                "Aliases accepted: distinct_values→distinct, avg→mean, nulls→null_count."
            )


# KPIs that need a scalar column. Trying to compute MIN over a compound
# ``(year, month)`` key makes no sense — the SQL planner wants per-column
# ranges; caller should declare a second stat for that.
_COMPOUND_UNSAFE_KPIS: frozenset[KPI] = frozenset({
    KPI.MIN,
    KPI.MAX,
    KPI.SUM,
    KPI.MEAN,
    KPI.BYTE_SIZE,
})


FieldLike = Union[Field, str, pa.Field]


@dataclass(frozen=True, slots=True)
class DataStatistic:
    """One labeled statistic: what to compute over which field(s).

    Parameters
    ----------
    fields:
        One or more :class:`Field` targets. At the door we accept a
        :class:`Field`, a bare column name (``str``), a :class:`pa.Field`,
        or a tuple/list mixing those — strings are promoted to
        :class:`Field` with a placeholder ``null`` dtype so the config
        stays lightweight; the compute layer can resolve real dtypes
        against the batch schema.
    kpis:
        One or more :class:`KPI` values. Accepts a single KPI / string,
        or an iterable. Stored as a sorted, frozen set.
    label:
        Human-readable identifier for the result. Defaults to
        ``"{field_spec}.{kpi_spec}"`` — e.g. ``"amount.min"`` or
        ``"(year,month).distinct"``.
    """

    fields: tuple[Field, ...]
    kpis: frozenset[KPI]
    label: str = _dc_field(default="")

    def __post_init__(self) -> None:
        normalized_fields = self._normalize_fields(self.fields)
        object.__setattr__(self, "fields", normalized_fields)

        normalized_kpis = self._normalize_kpis(self.kpis)
        object.__setattr__(self, "kpis", normalized_kpis)

        if self.is_compound:
            bad = normalized_kpis & _COMPOUND_UNSAFE_KPIS
            if bad:
                names = ", ".join(sorted(k.value for k in bad))
                raise ValueError(
                    f"KPIs {{{names}}} need a scalar column but this stat targets "
                    f"compound key ({', '.join(f.name for f in normalized_fields)}). "
                    "Declare a separate DataStatistic per column for per-component "
                    "ranges."
                )

        if not self.label:
            object.__setattr__(self, "label", self._default_label())
        elif not isinstance(self.label, str):
            raise TypeError(
                f"DataStatistic.label must be str, got {type(self.label).__name__}"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_compound(self) -> bool:
        """``True`` when this stat targets more than one column."""
        return len(self.fields) > 1

    @property
    def field_names(self) -> tuple[str, ...]:
        """Plain column names in declaration order."""
        return tuple(f.name for f in self.fields)

    # ------------------------------------------------------------------
    # Default label
    # ------------------------------------------------------------------

    def _field_spec(self) -> str:
        names = self.field_names
        if len(names) == 1:
            return names[0]
        return "(" + ",".join(names) + ")"

    def _kpi_spec(self) -> str:
        return ",".join(sorted(k.value for k in self.kpis))

    def _default_label(self) -> str:
        return f"{self._field_spec()}.{self._kpi_spec()}"

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _field_from(value: FieldLike) -> Field:
        if isinstance(value, Field):
            return value
        if isinstance(value, str):
            name = value.strip()
            if not name:
                raise ValueError("DataStatistic: empty string is not a valid column name")
            # Placeholder dtype — the compute layer re-resolves against the
            # batch schema. Keeps the config cheap to write.
            return Field(name=name, dtype=pa.null())
        if isinstance(value, pa.Field):
            return Field.from_arrow_field(value)
        raise TypeError(
            "DataStatistic.fields entries must be a Field, column name (str), or "
            f"pa.Field — got {type(value).__name__}"
        )

    @classmethod
    def _normalize_fields(cls, value: Any) -> tuple[Field, ...]:
        if isinstance(value, (Field, str, pa.Field)):
            return (cls._field_from(value),)
        if isinstance(value, (tuple, list)):
            if not value:
                raise ValueError("DataStatistic.fields must contain at least one field")
            out = tuple(cls._field_from(v) for v in value)
            seen: set[str] = set()
            for f in out:
                if f.name in seen:
                    raise ValueError(
                        f"DataStatistic.fields contains duplicate column {f.name!r}"
                    )
                seen.add(f.name)
            return out
        raise TypeError(
            "DataStatistic.fields must be a Field / str / pa.Field or a "
            f"tuple/list of those, got {type(value).__name__}"
        )

    @staticmethod
    def _normalize_kpis(value: Any) -> frozenset[KPI]:
        if isinstance(value, KPI):
            return frozenset({value})
        if isinstance(value, str):
            return frozenset({KPI.parse(value)})
        if isinstance(value, (frozenset, set, tuple, list)):
            items = [KPI.parse(v) for v in value]
            if not items:
                raise ValueError("DataStatistic.kpis must contain at least one KPI")
            return frozenset(items)
        raise TypeError(
            "DataStatistic.kpis must be a KPI, str, or an iterable of those; "
            f"got {type(value).__name__}"
        )

    # ------------------------------------------------------------------
    # String DSL
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, spec: "str | DataStatistic") -> "DataStatistic":
        """Parse a single DSL spec into one :class:`DataStatistic`.

        Grammar::

            spec        = field_spec '.' kpi_list
            field_spec  = column_name | '(' column_name (',' column_name)* ')'
            kpi_list    = kpi (',' kpi)*

        The KPI list is the tail after the **last** top-level ``.``,
        so dotted column names are still possible by wrapping them in
        parentheses: ``"(ns.col).min"``. Whitespace around separators
        is ignored.

        Examples
        --------
        >>> DataStatistic.parse("amount.min")
        >>> DataStatistic.parse("amount.min,max,count")
        >>> DataStatistic.parse("(year,month).distinct")
        """
        if isinstance(spec, cls):
            return spec
        if not isinstance(spec, str):
            raise TypeError(
                f"DataStatistic.parse expects a str, got {type(spec).__name__}"
            )

        text = spec.strip()
        if not text:
            raise ValueError("DataStatistic.parse: empty spec")

        field_text, kpi_text = cls._split_field_and_kpis(text)
        field_names = cls._parse_field_spec(field_text)
        kpi_tokens = [tok.strip() for tok in kpi_text.split(",") if tok.strip()]
        if not kpi_tokens:
            raise ValueError(
                f"DataStatistic.parse({spec!r}): no KPI after '.'. "
                "Example: 'amount.min' or 'amount.min,max,count'."
            )

        return cls(
            fields=tuple(field_names),
            kpis=frozenset(KPI.parse(t) for t in kpi_tokens),
        )

    @classmethod
    def parse_many(
        cls,
        values: "Iterable[DataStatistic | str] | None",
    ) -> list["DataStatistic"] | None:
        """Normalize a list of stats (mixed DSL strings / instances)."""
        if values is None:
            return None
        if isinstance(values, (str, cls)):
            raise TypeError(
                "statistics must be a list of DataStatistic / DSL strings, not a "
                f"single {type(values).__name__}. Wrap it in a list."
            )

        try:
            items = list(values)
        except TypeError as e:
            raise TypeError("statistics must be an iterable") from e

        out: list[DataStatistic] = []
        seen: set[str] = set()
        for entry in items:
            stat = cls.parse(entry) if isinstance(entry, str) else entry
            if not isinstance(stat, cls):
                raise TypeError(
                    "statistics entries must be DataStatistic or DSL strings, got "
                    f"{type(entry).__name__}"
                )
            if stat.label in seen:
                raise ValueError(
                    f"Duplicate statistics entry with label {stat.label!r}. "
                    "Give the colliding stat a distinct label or merge them."
                )
            seen.add(stat.label)
            out.append(stat)
        return out

    # ------------------------------------------------------------------
    # DSL helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_field_and_kpis(text: str) -> tuple[str, str]:
        """Split on the last top-level ``.`` (ignoring dots inside parens)."""
        depth = 0
        split_at = -1
        for i, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError(
                        f"DataStatistic.parse({text!r}): unbalanced ')'"
                    )
            elif ch == "." and depth == 0:
                split_at = i
        if depth != 0:
            raise ValueError(
                f"DataStatistic.parse({text!r}): unbalanced '(' — close it with ')'"
            )
        if split_at < 0:
            raise ValueError(
                f"DataStatistic.parse({text!r}): missing '.kpi' suffix. "
                "Example: 'amount.min' or '(year,month).distinct'."
            )
        return text[:split_at].strip(), text[split_at + 1:].strip()

    @staticmethod
    def _parse_field_spec(text: str) -> list[str]:
        """Parse ``"col"`` or ``"(a, b, c)"`` into a list of names."""
        if not text:
            raise ValueError("DataStatistic.parse: empty field spec before '.'")
        if text.startswith("(") and text.endswith(")"):
            inner = text[1:-1]
            names = [n.strip() for n in inner.split(",") if n.strip()]
            if not names:
                raise ValueError(
                    "DataStatistic.parse: empty '()' — name at least one column"
                )
            return names
        if "(" in text or ")" in text or "," in text:
            raise ValueError(
                f"DataStatistic.parse({text!r}): mixed parens / commas outside a "
                "compound-field group. Use '(a,b)' for multi-column specs."
            )
        return [text]
