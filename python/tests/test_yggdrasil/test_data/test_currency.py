from __future__ import annotations

import polars as pl
import pytest

from yggdrasil.data.enums.currency import Currency


def test_parse_aliases() -> None:
    assert Currency.parse_str("$") == Currency.USD
    assert Currency.parse_str("euro") == Currency.EUR
    assert Currency.parse(None) == Currency.USD


def test_invalid_code_raises() -> None:
    with pytest.raises(ValueError, match="alpha-3"):
        Currency("USDX")


def test_polars_normalize() -> None:
    s = pl.Series("c", [" usd ", "€", "gbp"])
    out = Currency.polars_normalize(s)
    assert out.to_list() == ["USD", "EUR", "GBP"]
