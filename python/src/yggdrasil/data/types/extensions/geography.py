"""GeographyType — first-class data type for geographic zone data.

Stores geography as ``struct<lat: float64 not null, lon: float64 not null>``.
On cast, input strings (ISO codes, country names, aliases, coordinate strings)
are resolved through the GeoZone catalog into lat/lon coordinates.

Databricks DDL follows the native ``GEOGRAPHY`` type spec::

    GEOGRAPHY              -> GEOGRAPHY(4326)
    GEOGRAPHY(4326)        -> explicit SRID
    GEOGRAPHY(ANY)         -> accepts any SRID
    GEOGRAPHY(OGC:CRS84, SPHERICAL) -> named CRS with model

Cast behavior:
- Input strings are resolved through GeoZoneCatalog.
- Resolved zones produce ``{lat, lon}`` struct values.
- When ``safe=True``: unparseable values raise ValueError.
- When ``safe=False`` (default): unparseable values become null structs.
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

# The physical Arrow struct that stores geography coordinates.
GEOGRAPHY_ARROW_TYPE: pa.DataType = pa.struct(
    [
        pa.field("lat", pa.float64(), nullable=True),
        pa.field("lon", pa.float64(), nullable=True),
    ]
)


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
    """Normalize an SRID value.

    Returns an int for numeric SRIDs, or a string for named references.

    Accepted inputs:
    - ``None`` -> default 4326
    - int -> kept as-is
    - ``"ANY"`` (case-insensitive) -> ``"ANY"``
    - ``"4326"`` (numeric string) -> ``4326``
    - ``"OGC:CRS84"`` or other named CRS references -> kept as uppercase string
    """
    if value is None:
        return DEFAULT_SRID

    if isinstance(value, int):
        return value

    text = str(value).strip()
    upper = text.upper()

    if upper == "ANY":
        return "ANY"

    try:
        return int(text)
    except (ValueError, TypeError):
        pass

    # Named CRS references like OGC:CRS84 — keep as uppercase string.
    if text:
        return upper

    raise ValueError(
        f"Invalid SRID value {value!r}. "
        f"Pass an integer SRID (e.g. 4326), 'ANY', a named CRS "
        f"(e.g. 'OGC:CRS84'), or None for the default (WGS 84 / 4326)."
    )


# ---------------------------------------------------------------------------
# Bulk parsing utilities — resolve to lat/lon struct arrays
# ---------------------------------------------------------------------------


def parse_geography_arrow(
    array: pa.Array | pa.ChunkedArray,
    *,
    catalog: "GeoZoneCatalog | None" = None,
    safe: bool = False,
) -> pa.Array | pa.ChunkedArray:
    """Parse an Arrow string array into a struct<lat, lon> array.

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

    Returns
    -------
    pa.Array or pa.ChunkedArray
        StructArray with ``lat`` and ``lon`` float64 fields.
    """
    cat = _get_catalog(catalog)
    is_chunked = isinstance(array, pa.ChunkedArray)

    if is_chunked:
        chunks = [
            _parse_geography_flat(chunk, cat, safe=safe) for chunk in array.chunks
        ]
        return pa.chunked_array(chunks, type=GEOGRAPHY_ARROW_TYPE)

    return _parse_geography_flat(array, cat, safe=safe)


def _parse_geography_flat(
    array: pa.Array,
    catalog: "GeoZoneCatalog",
    *,
    safe: bool,
) -> pa.StructArray:
    """Parse a flat (non-chunked) Arrow string array into struct<lat, lon>."""
    lats: list[float | None] = []
    lons: list[float | None] = []
    valid: list[bool] = []

    for i in range(len(array)):
        if array[i].as_py() is None:
            lats.append(None)
            lons.append(None)
            valid.append(False)
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
            lats.append(None)
            lons.append(None)
            valid.append(False)
            continue

        lats.append(zone.lat)
        lons.append(zone.lon)
        valid.append(True)

    lat_arr = pa.array(lats, type=pa.float64())
    lon_arr = pa.array(lons, type=pa.float64())
    mask = pa.array([not v for v in valid], type=pa.bool_())

    return pa.StructArray.from_arrays(
        [lat_arr, lon_arr],
        names=["lat", "lon"],
        mask=mask,
    )


def parse_geography_polars(
    series: "polars.Series",
    *,
    catalog: "GeoZoneCatalog | None" = None,
    safe: bool = False,
) -> "polars.Series":
    """Parse a Polars string series into a struct<lat, lon> series.

    Same semantics as :func:`parse_geography_arrow` but operates on Polars
    Series.  Goes through Arrow internally for catalog resolution.
    """
    pl = get_polars()
    arrow = series.to_arrow()
    parsed = parse_geography_arrow(arrow, catalog=catalog, safe=safe)
    if isinstance(parsed, pa.ChunkedArray):
        parsed = parsed.combine_chunks()
    return pl.Series(name=series.name, values=parsed)


# ---------------------------------------------------------------------------
# GeographyType
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeographyType(DataType):
    """First-class data type for geographic coordinates.

    Inner type: ``struct<lat: float64 not null, lon: float64 not null>``.

    On cast, input strings are resolved through the GeoZone catalog into
    lat/lon coordinate pairs.  The struct representation makes geographic
    data directly usable in Arrow analytics and Databricks dashboards.

    Parameters
    ----------
    srid : int | str
        Spatial Reference System Identifier.  Default is ``4326`` (WGS 84).
        Pass ``"ANY"`` to accept any SRID, or a named CRS like
        ``"OGC:CRS84"``.  Controls the Databricks DDL output.
    model : str | None
        Coordinate model.  ``None`` (default) omits it from DDL.
        ``"SPHERICAL"`` produces ``GEOGRAPHY(srid, SPHERICAL)``.

    Example::

        geo = GeographyType()
        geo = GeographyType(srid="OGC:CRS84", model="SPHERICAL")

    Casting a string array through this type resolves to lat/lon::

        arr = pa.array(["France", "CH-ZH", "bogus"])
        result = geo._cast_arrow_array(arr, options)
        # -> [{lat: 46.2276, lon: 2.2137}, {lat: 47.3769, lon: 8.5417}, null]
    """

    srid: int | str = DEFAULT_SRID
    model: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "srid", _normalize_srid(self.srid))
        if self.model is not None:
            object.__setattr__(self, "model", str(self.model).strip().upper())

    # ------------------------------------------------------------------
    # DataType protocol
    # ------------------------------------------------------------------
    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.GEOGRAPHY

    @property
    def children_fields(self) -> list[Field]:
        field_cls = self.get_data_field_class()
        from ..primitive import FloatingPointType

        f64 = FloatingPointType(byte_size=8)
        return [
            field_cls(name="lat", dtype=f64, nullable=True),
            field_cls(name="lon", dtype=f64, nullable=True),
        ]

    # ------------------------------------------------------------------
    # Arrow — struct<lat: float64 not null, lon: float64 not null>
    # ------------------------------------------------------------------
    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        # We don't claim arbitrary structs — GeographyType must be created
        # explicitly or via from_str / from_dict / from_parsed.
        return False

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> GeographyType:
        raise TypeError(
            f"Cannot infer GeographyType from Arrow type {dtype!r}. "
            "Use DataType.from_str('geography') or GeographyType() directly."
        )

    def to_arrow(self) -> pa.DataType:
        return GEOGRAPHY_ARROW_TYPE

    # ------------------------------------------------------------------
    # Polars — struct with lat/lon
    # ------------------------------------------------------------------
    @classmethod
    def handles_polars_type(cls, dtype: polars.DataType) -> bool:
        return False

    def to_polars(self) -> polars.DataType:
        pl = get_polars()
        return pl.Struct(
            [
                pl.Field("lat", pl.Float64),
                pl.Field("lon", pl.Float64),
            ]
        )

    # ------------------------------------------------------------------
    # Spark — struct with lat/lon
    # ------------------------------------------------------------------
    @classmethod
    def handles_spark_type(cls, dtype: pst.DataType) -> bool:
        return False

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        t = spark.types
        return t.StructType(
            [
                t.StructField("lat", t.DoubleType(), nullable=True),
                t.StructField("lon", t.DoubleType(), nullable=True),
            ]
        )

    # ------------------------------------------------------------------
    # Databricks DDL — native GEOGRAPHY type
    # ------------------------------------------------------------------
    def to_databricks_ddl(self) -> str:
        if self.model:
            return f"GEOGRAPHY({self.srid}, {self.model})"
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
        model = value.get("model")
        return cls(srid=srid, model=model)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": int(DataTypeId.GEOGRAPHY),
            "name": DataTypeId.GEOGRAPHY.name,
        }
        if self.srid != DEFAULT_SRID:
            d["srid"] = str(self.srid) if isinstance(self.srid, int) else self.srid
        if self.model:
            d["model"] = self.model
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
    # Cast — resolve strings to lat/lon struct via GeoZone catalog
    # ------------------------------------------------------------------
    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: CastOptions,
    ) -> pa.Array:
        """Cast an Arrow array into geography struct<lat, lon>.

        Accepts string arrays (ISO codes, names, coordinate strings) and
        struct arrays that already have lat/lon fields.  Unparseable values
        become null when safe=False (default), or raise ValueError when
        safe=True.
        """
        safe = getattr(options, "safe", False)

        # Already a struct with lat/lon — pass through.
        if pa.types.is_struct(array.type):
            names = {f.name for f in array.type}
            if "lat" in names and "lon" in names:
                return array

        src = array
        # Cast to string if not already, so we can resolve through catalog.
        if not pa.types.is_string(src.type) and not pa.types.is_large_string(src.type):
            try:
                src = pc.cast(src, pa.string(), safe=False)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                if safe:
                    raise
                return pa.nulls(len(src), type=GEOGRAPHY_ARROW_TYPE)

        parsed = parse_geography_arrow(src, safe=safe)

        if isinstance(parsed, pa.ChunkedArray):
            parsed = parsed.combine_chunks()

        return parsed

    # ------------------------------------------------------------------
    # Python object conversion
    # ------------------------------------------------------------------
    def convert_pyobj(
        self, value: Any, nullable: bool, safe: bool = False
    ) -> dict[str, float] | None:
        if value is None:
            if nullable:
                return None
            raise ValueError(
                "Got None for a non-nullable GeographyType field. "
                "Pass a valid geography string or set nullable=True."
            )

        # Accept dicts with lat/lon already.
        if isinstance(value, dict) and "lat" in value and "lon" in value:
            return {"lat": float(value["lat"]), "lon": float(value["lon"])}

        zone = _resolve_zone(str(value), _get_catalog())

        if zone is not None:
            return {"lat": zone.lat, "lon": zone.lon}

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

    def _convert_pyobj(self, value: Any, safe: bool = False) -> dict[str, float] | None:
        if isinstance(value, dict) and "lat" in value and "lon" in value:
            return {"lat": float(value["lat"]), "lon": float(value["lon"])}

        zone = _resolve_zone(str(value), _get_catalog())
        if zone is not None:
            return {"lat": zone.lat, "lon": zone.lon}

        if safe:
            raise ValueError(
                f"Cannot resolve geography value {value!r}. "
                "No matching zone found in the catalog."
            )
        return None

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    def default_pyobj(self, nullable: bool) -> dict[str, float] | None:
        if nullable:
            return None
        # WORLD at 0, 0 — always safe.
        return {"lat": 0.0, "lon": 0.0}

    def default_arrow_scalar(self, nullable: bool = True) -> pa.Scalar:
        if nullable:
            return pa.scalar(None, type=GEOGRAPHY_ARROW_TYPE)
        return pa.scalar({"lat": 0.0, "lon": 0.0}, type=GEOGRAPHY_ARROW_TYPE)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        parts: list[str] = []
        if self.srid != DEFAULT_SRID:
            parts.append(f"srid={self.srid!r}")
        if self.model:
            parts.append(f"model={self.model!r}")
        return f"GeographyType({', '.join(parts)})"

    def __str__(self) -> str:
        srid_str = str(self.srid)
        if self.model:
            return f"geography({srid_str}, {self.model})"
        if self.srid != DEFAULT_SRID:
            return f"geography({srid_str})"
        return "geography"
