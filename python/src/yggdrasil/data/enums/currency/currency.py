"""Currency normalisation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import polars

__all__ = ["Currency"]

_CURRENCY_ALIASES: dict[str, str] = {
    "$": "USD",
    "US$": "USD",
    "DOLLAR": "USD",
    "EURO": "EUR",
    "€": "EUR",
    "£": "GBP",
    "POUND": "GBP",
    "YEN": "JPY",
    "¥": "JPY",
    "FRANC": "CHF",
    "YUAN": "CNY",
}


@dataclass(slots=True, frozen=True)
class Currency:
    code: str

    USD: ClassVar["Currency"]
    EUR: ClassVar["Currency"]
    GBP: ClassVar["Currency"]
    CHF: ClassVar["Currency"]
    JPY: ClassVar["Currency"]

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        return f"Currency({self.code!r})"

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", self.code.strip().upper())
        if len(self.code) != 3:
            raise ValueError(f"Currency code must be ISO-4217 alpha-3, got {self.code!r}")

    @classmethod
    def parse(cls, obj: Any) -> "Currency":
        if isinstance(obj, cls):
            return obj
        if obj is None:
            return cls.USD
        if isinstance(obj, str):
            return cls.parse_str(obj)
        raise TypeError(f"Cannot parse {type(obj).__name__} as {cls.__name__}")

    @classmethod
    def parse_str(cls, s: str) -> "Currency":
        if not isinstance(s, str):
            raise TypeError(f"Expected str, got {type(s).__name__}")
        raw = s.strip()
        if not raw:
            raise ValueError("Currency string cannot be empty")
        upper = raw.upper()
        canonical = _CURRENCY_ALIASES.get(upper, upper)
        return cls(canonical)

    @classmethod
    def polars_normalize(
        cls,
        col: "polars.Series | polars.Expr",
        *,
        lazy: bool = True,
        return_value: Literal["code"] = "code",
    ) -> "polars.Series | polars.Expr":
        import polars as pl

        if return_value != "code":
            raise ValueError(f"Unsupported return_value: {return_value!r}")
        if not isinstance(col, (pl.Series, pl.Expr)):
            raise TypeError(f"Expected polars.Series | polars.Expr, got {type(col).__name__}")

        normalized = (
            col.cast(pl.Utf8)
            .str.strip_chars()
            .str.to_uppercase()
        )
        return normalized.replace_strict(_CURRENCY_ALIASES, default=normalized)


Currency.USD = Currency("USD")
Currency.EUR = Currency("EUR")
Currency.GBP = Currency("GBP")
Currency.CHF = Currency("CHF")
Currency.JPY = Currency("JPY")
