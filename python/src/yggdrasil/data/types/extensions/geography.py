"""GeographyType — first-class data type for geographic zone data.

Stores geography identifiers as strings (ISO codes, country names, region
codes, EIC codes, coordinates) and normalizes them through the GeoZone
catalog on cast.

Databricks DDL follows the native ``GEOGRAPHY`` type spec::

    GEOGRAPHY          -> GEOGRAPHY(4326)   -- WGS 84 default
    GEOGRAPHY(4326)    -> explicit SRID
    GEOGRAPHY(ANY)     -> accepts any SRID

Cast behavior:
- Input strings are resolved through GeoZoneCatalog (codes, aliases,
  fuzzy name matching, coordinate strings).
- Resolved zones produce a canonical ``code`` (region_iso or country_iso
  or key).
- When ``safe=True``: unparseable values raise ValueError.
- When ``safe=False`` (default): unparseable values become null.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.types.id import DataTypeId
from yggdrasil.io import SaveMode
from ..base import DataType
from ..support import get_polars, get_spark_sql

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst
    from yggdrasil.data.cast.options import CastOptions
    from yggdrasil.data.data_field import Field
    from yggdrasil.data.enums.geozone.base import GeoZone
    from yggdrasil.data.enums.geozone.catalog import GeoZoneCatalog

__all__ = ["GeographyType"]

LOGGER = logging.getLogger(__name__)

# WGS 84 — the default SRID for Databricks GEOGRAPHY columns.
DEFAULT_SRID: int = 4326


def _get_catalog(catalog: "GeoZoneCatalog | None" = None) -> "GeoZoneCatalog":
    if catalog is not None:
        return catalog
    from yggdrasil.data.enums.geozone.load import load_geozones

    return load_geozones()


def _resolve_zone(
    value: str | None,
    catalog: "GeoZoneCatalog",
) -> "GeoZone | None":
    """Parse a single string value into a GeoZone, or None if unresolvable."""
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    return catalog.parse_str(stripped)


# ---------------------------------------------------------------------------
# SRID normalization
# ---------------------------------------------------------------------------


def _normalize_srid(value: int | str | None) -> int | str:
    """Normalize an SRID value to either an int or the string ``"ANY"``.

    Accepts ``None`` (-> default 4326), ints, numeric strings, and the
    literal ``"ANY"`` (case-insensitive).
    """
    if value is None:
        return DEFAULT_SRID

    if isinstance(value, int):
        return value

    text = str(value).strip().upper()
    if text == "ANY":
        return "ANY"

    try:
        return int(text)
    except (ValueError, TypeError):
        raise ValueError(
            f"Invalid SRID value {value!r}. "
            f"Pass an integer SRID (e.g. 4326), 'ANY', or None for the "
            f"default (WGS 84 / 4326)."
        )


# ---------------------------------------------------------------------------
# Bulk parsing utilities
# ---------------------------------------------------------------------------


def parse_geography_arrow(
    array: pa.Array | pa.ChunkedArray,
    *,
    catalog: "GeoZoneCatalog | None" = None,
    safe: bool = False,
    output: str = "code",
) -> pa.Array | pa.ChunkedArray:
    """Parse an Arrow string array into normalized geography codes.

    Parameters
    ----------
    array
        Arrow array of strings to parse (country names, ISO codes, aliases,
        coordinate strings, etc.).
    catalog
        GeoZone catalog to resolve against.  Uses the default catalog when
        ``None``.
    safe
        When ``True``, raise ``ValueError`` on the first unparseable value.
        When ``False`` (default), unparseable values become null.
    output
        What to extract from resolved zones.  One of:
        ``"code"`` (default), ``"name"``, ``"country_iso"``,
        ``"region_iso"``, ``"key"``, ``"ccy"``, ``"gtype"``.

    Returns
    -------
    pa.Array or pa.ChunkedArray
        String array of the same length with resolved values (or nulls).
    """
    cat = _get_catalog(catalog)
    is_chunked = isinstance(array, pa.ChunkedArray)

    if is_chunked:
        chunks = [
            _parse_geography_flat(chunk, cat, safe=safe, output=output)
            for chunk in array.chunks
        ]
        return pa.chunked_array(chunks, type=pa.string())

    return _parse_geography_flat(array, cat, safe=safe, output=output)


def _parse_geography_flat(
    array: pa.Array,
    catalog: "GeoZoneCatalog",
    *,
    safe: bool,
    output: str,
) -> pa.Array:
    """Parse a flat (non-chunked) Arrow string array."""
    results: list[str | None] = []

    for i in range(len(array)):
        if array[i].as_py() is None:
            results.append(None)
            continue

        raw = str(array[i].as_py())
        zone = _resolve_zone(raw, catalog)

        if zone is None:
            if safe:
                raise ValueError(
                    f"Cannot resolve geography value {raw!r} at index {i}. "
                    "No matching zone found in the catalog. "
                    "Pass safe=False to null out unparseable values, "
                    "or check your input data / catalog."
                )
            results.append(None)
            continue

        results.append(_extract_field(zone, output))

    return pa.array(results, type=pa.string())


def _extract_field(zone: "GeoZone", output: str) -> str | None:
    """Pull the requested field from a resolved GeoZone."""
    if output == "code":
        return zone.code
    if output == "name":
        return zone.name
    if output == "country_iso":
        return zone.country_iso
    if output == "region_iso":
        return zone.region_iso
    if output == "key":
        return zone.key
    if output == "ccy":
        return zone.ccy
    if output == "gtype":
        return zone.gtype.value
    raise ValueError(
        f"Unknown output field {output!r}. "
        f"Valid options: 'code', 'name', 'country_iso', 'region_iso', "
        f"'key', 'ccy', 'gtype'."
    )


def parse_geography_polars(
    series: "polars.Series",
    *,
    catalog: "GeoZoneCatalog | None" = None,
    safe: bool = False,
    output: str = "code",
) -> "polars.Series":
    """Parse a Polars string series into normalized geography codes.

    Same semantics as :func:`parse_geography_arrow` but operates on Polars
    Series.  Goes through Arrow internally for catalog resolution.
    """
    pl = get_polars()
    arrow = series.to_arrow()
    parsed = parse_geography_arrow(arrow, catalog=catalog, safe=safe, output=output)
    return pl.Series(name=series.name, values=parsed, dtype=pl.String)


# ---------------------------------------------------------------------------
# GeographyType
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeographyType(DataType):
    """First-class data type for geographic zone identifiers.

    Stores geography data as strings and normalizes on cast through
    the GeoZone catalog.  Resolved values are canonical codes (region_iso
    or country_iso or key).

    Parameters
    ----------
    srid : int | str
        Spatial Reference System Identifier.  Default is ``4326`` (WGS 84).
        Pass ``"ANY"`` to accept any SRID.  Controls the Databricks DDL:
        ``GEOGRAPHY(4326)``, ``GEOGRAPHY(ANY)``, etc.
    output : str
        What field to extract from resolved zones: ``"code"`` (default),
        ``"name"``, ``"country_iso"``, ``"region_iso"``, ``"key"``,
        ``"ccy"``, ``"gtype"``.

    Example::

        geo = GeographyType()               # GEOGRAPHY(4326), output=code
        geo = GeographyType(srid="ANY")      # GEOGRAPHY(ANY)
        geo = GeographyType(srid=4326, output="country_iso")
    """

    srid: int | str = DEFAULT_SRID
    output: str = "code"

    def __post_init__(self) -> None:
        object.__setattr__(self, "srid", _normalize_srid(self.srid))

    # ------------------------------------------------------------------
    # DataType protocol
    # ------------------------------------------------------------------
    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.GEOGRAPHY

    @property
    def children_fields(self) -> list[Field]:
        return []

    # ------------------------------------------------------------------
    # Arrow — geography stores as plain string
    # ------------------------------------------------------------------
    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        # Arrow has no native geography type.
        return False

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> GeographyType:
        raise TypeError(
            f"Cannot infer GeographyType from Arrow type {dtype!r}. "
            "Arrow has no native geography type. "
            "Use DataType.from_str('geography') or GeographyType() directly."
        )

    def to_arrow(self) -> pa.DataType:
        return pa.string()

    # ------------------------------------------------------------------
    # Polars — plain string, no native geography
    # ------------------------------------------------------------------
    @classmethod
    def handles_polars_type(cls, dtype: polars.DataType) -> bool:
        return False

    def to_polars(self) -> polars.DataType:
        pl = get_polars()
        return pl.String

    # ------------------------------------------------------------------
    # Spark — plain string, no native geography
    # ------------------------------------------------------------------
    @classmethod
    def handles_spark_type(cls, dtype: pst.DataType) -> bool:
        return False

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        return spark.types.StringType()

    # ------------------------------------------------------------------
    # Databricks DDL — native GEOGRAPHY type
    # ------------------------------------------------------------------
    def to_databricks_ddl(self) -> str:
        return f"GEOGRAPHY({self.srid})"

    # ------------------------------------------------------------------
    # Dict round-trip
    # ------------------------------------------------------------------
    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        type_id = value.get("id")
        if type_id == int(DataTypeId.GEOGRAPHY):
            return True
        name = str(value.get("name", "")).upper()
        return name == "GEOGRAPHY"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GeographyType:
        srid = _normalize_srid(value.get("srid"))
        output = value.get("output", "code")
        return cls(srid=srid, output=output)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": int(DataTypeId.GEOGRAPHY),
            "name": DataTypeId.GEOGRAPHY.name,
        }
        if self.srid != DEFAULT_SRID:
            d["srid"] = str(self.srid) if isinstance(self.srid, int) else self.srid
        if self.output != "code":
            d["output"] = self.output
        return d

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def _merge_with_same_id(
        self,
        other: DataType,
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> GeographyType:
        return self

    # ------------------------------------------------------------------
    # Cast — normalize through GeoZone catalog
    # ------------------------------------------------------------------
    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: CastOptions,
    ) -> pa.Array:
        """Cast an Arrow array into geography codes.

        Normalizes input strings through the GeoZone catalog.
        Unparseable values become null when safe=False (default),
        or raise ValueError when safe=True.
        """
        safe = getattr(options, "safe", False)

        src = array
        # Cast to string if not already.
        if not pa.types.is_string(src.type) and not pa.types.is_large_string(src.type):
            try:
                src = pc.cast(src, pa.string(), safe=False)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                if safe:
                    raise
                return pa.nulls(len(src), type=pa.string())

        parsed = parse_geography_arrow(src, safe=safe, output=self.output)

        if isinstance(parsed, pa.ChunkedArray):
            parsed = parsed.combine_chunks()

        return parsed

    # ------------------------------------------------------------------
    # Python object conversion
    # ------------------------------------------------------------------
    def convert_pyobj(
        self, value: Any, nullable: bool, safe: bool = False
    ) -> str | None:
        if value is None:
            if nullable:
                return None
            raise ValueError(
                "Got None for a non-nullable GeographyType field. "
                "Pass a valid geography string or set nullable=True."
            )

        zone = _resolve_zone(str(value), _get_catalog())
        code = _extract_field(zone, self.output) if zone is not None else None

        if code is not None:
            return code

        if safe:
            raise ValueError(
                f"Cannot resolve geography value {value!r}. "
                "No matching zone found in the catalog. "
                "Pass safe=False to allow null for unparseable values."
            )

        if not nullable:
            raise ValueError(
                f"Cannot resolve geography value {value!r} and field "
                "is not nullable. Either fix the input or make the field nullable."
            )
        return None

    def _convert_pyobj(self, value: Any, safe: bool = False) -> str | None:
        zone = _resolve_zone(str(value), _get_catalog())
        result = _extract_field(zone, self.output) if zone is not None else None
        if result is None and safe:
            raise ValueError(
                f"Cannot resolve geography value {value!r}. "
                "No matching zone found in the catalog."
            )
        return result

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    def default_pyobj(self, nullable: bool) -> str | None:
        if nullable:
            return None
        return "WORLD"

    def default_arrow_scalar(self, nullable: bool = True) -> pa.Scalar:
        if nullable:
            return pa.scalar(None, type=pa.string())
        return pa.scalar("WORLD", type=pa.string())

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        parts: list[str] = []
        if self.srid != DEFAULT_SRID:
            parts.append(f"srid={self.srid!r}")
        if self.output != "code":
            parts.append(f"output={self.output!r}")
        return f"GeographyType({', '.join(parts)})"

    def __str__(self) -> str:
        srid_str = str(self.srid)
        if self.output != "code":
            return f"geography({srid_str})[{self.output}]"
        if self.srid != DEFAULT_SRID:
            return f"geography({srid_str})"
        return "geography"
