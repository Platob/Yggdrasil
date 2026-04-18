"""ISO-coded string data types.

This package provides first-class data types that parse free-form text
into canonical ISO codes (countries, subdivisions, cities, continents,
currencies).  All types support the full ``cast_*`` matrix used
elsewhere in ``yggdrasil.data.types``: arrow arrays, polars
series/expressions, pandas series, and pyspark columns.

``safe=False`` (the default) nulls out values that can't be resolved.
``safe=True`` raises on eager inputs (Arrow/Polars Series/Pandas) and
is a no-op on lazy inputs (polars.Expr, pyspark.Column) because values
aren't observable at call time.
"""
from .base import ISOType, normalize_iso_token, normalize_iso_token_keep_hyphen
from .city import ISOCityType
from .continent import ISOContinentType
from .country import ISOCountryType
from .currency import ISOCurrencyType
from .subdivision import ISOSubdivisionType
from .timezone import TimezoneType

__all__ = [
    "ISOType",
    "ISOCityType",
    "ISOContinentType",
    "ISOCountryType",
    "ISOCurrencyType",
    "ISOSubdivisionType",
    "TimezoneType",
    "normalize_iso_token",
    "normalize_iso_token_keep_hyphen",
]
