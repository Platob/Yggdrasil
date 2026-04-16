"""GeographyType — first-class data type for geographic coordinates.

Inner type: ``struct<lat: float64, lon: float64>``.

Handles flexible input formats on cast:
- struct arrays with lat/lon (or latitude/longitude) fields → passthrough
- string arrays with coordinate pairs: ``"48.8566, 2.3522"``,
  ``"48.8566 2.3522"``, ``"48.8566|2.3522"``, ``"48.8566;2.3522"``
- dict-like rows: ``{"lat": 48.8, "lon": 2.35}``
- Python tuples/lists: ``(48.8, 2.35)``

When ``safe=False`` (default), unparseable values become null.
When ``safe=True``, unparseable values raise ValueError.

Databricks DDL follows the native ``GEOGRAPHY`` type spec::

    GEOGRAPHY(4326)                  -- WGS 84 default
    GEOGRAPHY(ANY)                   -- accepts any SRID
    GEOGRAPHY(OGC:CRS84, SPHERICAL) -- named CRS with model
"""

from __future__ import annotations

import logging
import re
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

# Matches "lat, lon" / "lat lon" / "lat|lon" / "lat;lon" coordinate strings.
_COORD_RE = re.compile(
    r"^\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"\s*[,;|\s]\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"\s*$"
)

# Common field name aliases → canonical (lat, lon).
_LAT_ALIASES = frozenset({"lat", "latitude", "LAT", "LATITUDE", "Lat", "Latitude"})
_LON_ALIASES = frozenset(
    {
        "lon",
        "lng",
        "long",
        "longitude",
        "LON",
        "LNG",
        "LONG",
        "LONGITUDE",
        "Lon",
        "Lng",
        "Long",
        "Longitude",
    }
)


# ---------------------------------------------------------------------------
# SRID normalization
# ---------------------------------------------------------------------------


def _normalize_srid(value: int | str | None) -> int | str:
    """Normalize an SRID value.

    Accepts None (→ 4326), ints, numeric strings, ``"ANY"``,
    or named CRS references like ``"OGC:CRS84"``.
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
    if text:
        return upper
    raise ValueError(
        f"Invalid SRID value {value!r}. "
        f"Pass an integer SRID (e.g. 4326), 'ANY', a named CRS "
        f"(e.g. 'OGC:CRS84'), or None for the default (WGS 84 / 4326)."
    )


# ---------------------------------------------------------------------------
# Coordinate parsing helpers
# ---------------------------------------------------------------------------


def _parse_coord_string(s: str) -> tuple[float, float] | None:
    """Try to parse ``"lat, lon"`` from a string. Returns None on failure."""
    m = _COORD_RE.match(s)
    if m is None:
        return None
    lat = float(m.group(1))
    lon = float(m.group(2))
    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return (lat, lon)
    return None


def _resolve_struct_field_names(
    dtype: pa.StructType,
) -> tuple[str | None, str | None]:
    """Find the lat and lon field names in a struct, tolerating aliases."""
    lat_name: str | None = None
    lon_name: str | None = None
    for i in range(dtype.num_fields):
        name = dtype.field(i).name
        if name in _LAT_ALIASES and lat_name is None:
            lat_name = name
        elif name in _LON_ALIASES and lon_name is None:
            lon_name = name
    return lat_name, lon_name


# ---------------------------------------------------------------------------
# Bulk Arrow parsing — strings → struct<lat, lon>
# ---------------------------------------------------------------------------


def parse_geography_arrow(
    array: pa.Array | pa.ChunkedArray,
    *,
    safe: bool = False,
) -> pa.Array | pa.ChunkedArray:
    """Parse an Arrow string array of coordinate pairs into struct<lat, lon>.

    Accepted string formats: ``"48.8, 2.3"``, ``"48.8 2.3"``,
    ``"48.8|2.3"``, ``"48.8;2.3"``.

    Parameters
    ----------
    array
        Arrow string array to parse.
    safe
        When True, raise on unparseable values.
        When False (default), unparseable values become null.
    """
    if isinstance(array, pa.ChunkedArray):
        chunks = [_parse_coords_flat(c, safe=safe) for c in array.chunks]
        return pa.chunked_array(chunks, type=GEOGRAPHY_ARROW_TYPE)
    return _parse_coords_flat(array, safe=safe)


def _parse_coords_flat(array: pa.Array, *, safe: bool) -> pa.StructArray:
    lats: list[float | None] = []
    lons: list[float | None] = []
    valid: list[bool] = []

    for i in range(len(array)):
        py = array[i].as_py()
        if py is None:
            lats.append(None)
            lons.append(None)
            valid.append(False)
            continue

        coord = _parse_coord_string(str(py))
        if coord is not None:
            lats.append(coord[0])
            lons.append(coord[1])
            valid.append(True)
        else:
            if safe:
                raise ValueError(
                    f"Cannot parse coordinate from {py!r} at index {i}. "
                    f"Expected 'lat, lon' format (e.g. '48.8566, 2.3522'). "
                    f"Pass safe=False to null out unparseable values."
                )
            lats.append(None)
            lons.append(None)
            valid.append(False)

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
    safe: bool = False,
) -> "polars.Series":
    """Parse a Polars string series of coordinate pairs into struct<lat, lon>."""
    pl = get_polars()
    arrow = series.to_arrow()
    parsed = parse_geography_arrow(arrow, safe=safe)
    if isinstance(parsed, pa.ChunkedArray):
        parsed = parsed.combine_chunks()
    return pl.Series(name=series.name, values=parsed)


# ---------------------------------------------------------------------------
# GeographyType
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeographyType(DataType):
    """First-class data type for geographic coordinates.

    Inner type: ``struct<lat: float64, lon: float64>``.

    Handles flexible input on cast:

    - **struct** with lat/lon (or latitude/longitude) fields → passthrough
    - **string** ``"48.8566, 2.3522"`` → parsed into lat/lon
    - **float pairs** in struct form → kept as-is
    - **dicts** ``{"lat": 48.8, "lon": 2.3}`` via convert_pyobj
    - **tuples** ``(48.8, 2.3)`` via convert_pyobj

    Parameters
    ----------
    srid : int | str
        Spatial Reference System Identifier.  Default ``4326`` (WGS 84).
    model : str | None
        Coordinate model (e.g. ``"SPHERICAL"``).  None omits from DDL.
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
    # Arrow — struct<lat: float64, lon: float64>
    # ------------------------------------------------------------------
    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
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
    # Cast — flexible input → struct<lat, lon>
    # ------------------------------------------------------------------
    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: CastOptions,
    ) -> pa.Array:
        """Cast an Arrow array into struct<lat, lon>.

        Handles:
        - struct with lat/lon fields → passthrough (rename if needed)
        - struct with latitude/longitude → rename to lat/lon
        - string → parse "lat, lon" coordinate pairs
        - numeric → not meaningful for a point, null out (safe=False)
        """
        safe = getattr(options, "safe", False)

        # -- struct passthrough / rename --
        if pa.types.is_struct(array.type):
            lat_name, lon_name = _resolve_struct_field_names(array.type)
            if lat_name is not None and lon_name is not None:
                # Already has lat/lon fields — extract and rebuild canonical.
                if (
                    lat_name == "lat"
                    and lon_name == "lon"
                    and array.type == GEOGRAPHY_ARROW_TYPE
                ):
                    return array
                lat_col = pc.struct_field(array, lat_name)
                lon_col = pc.struct_field(array, lon_name)
                return pa.StructArray.from_arrays(
                    [pc.cast(lat_col, pa.float64()), pc.cast(lon_col, pa.float64())],
                    names=["lat", "lon"],
                )

        # -- string → parse coordinate pairs --
        if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
            return parse_geography_arrow(array, safe=safe)

        # -- try casting to string first --
        try:
            as_str = pc.cast(array, pa.string(), safe=False)
            return parse_geography_arrow(as_str, safe=safe)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            pass

        if safe:
            raise ValueError(
                f"Cannot cast Arrow array of type {array.type!r} to geography. "
                f"Expected struct<lat, lon>, string coordinate pairs, or similar."
            )
        return pa.nulls(len(array), type=GEOGRAPHY_ARROW_TYPE)

    # ------------------------------------------------------------------
    # Python object conversion — flexible input
    # ------------------------------------------------------------------
    def convert_pyobj(
        self,
        value: Any,
        nullable: bool,
        safe: bool = False,
    ) -> dict[str, float] | None:
        if value is None:
            if nullable:
                return None
            raise ValueError(
                "Got None for a non-nullable GeographyType field. "
                "Pass a coordinate value or set nullable=True."
            )

        result = _pyobj_to_latlon(value)
        if result is not None:
            return result

        if safe:
            raise ValueError(
                f"Cannot parse geography coordinate from {value!r}. "
                f"Expected a dict with lat/lon, a (lat, lon) tuple, "
                f"or a 'lat, lon' string."
            )
        if not nullable:
            raise ValueError(
                f"Cannot parse geography coordinate from {value!r} and field "
                "is not nullable. Fix the input or make the field nullable."
            )
        return None

    def _convert_pyobj(self, value: Any, safe: bool = False) -> dict[str, float] | None:
        result = _pyobj_to_latlon(value)
        if result is not None:
            return result
        if safe:
            raise ValueError(f"Cannot parse geography coordinate from {value!r}.")
        return None

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    def default_pyobj(self, nullable: bool) -> dict[str, float] | None:
        if nullable:
            return None
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


# ---------------------------------------------------------------------------
# Python object → {lat, lon} dict
# ---------------------------------------------------------------------------


def _pyobj_to_latlon(value: Any) -> dict[str, float] | None:
    """Try every reasonable format to extract lat/lon from a Python value.

    Returns ``{"lat": ..., "lon": ...}`` or None if nothing works.
    """
    # dict with lat/lon keys
    if isinstance(value, dict):
        lat = _dict_get_lat(value)
        lon = _dict_get_lon(value)
        if lat is not None and lon is not None:
            return {"lat": float(lat), "lon": float(lon)}

    # tuple / list of two numbers
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            lat, lon = float(value[0]), float(value[1])
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return {"lat": lat, "lon": lon}
        except (TypeError, ValueError):
            pass

    # string coordinate pair
    if isinstance(value, str):
        coord = _parse_coord_string(value)
        if coord is not None:
            return {"lat": coord[0], "lon": coord[1]}

    # object with .lat / .lon attributes
    lat = getattr(value, "lat", None)
    lon = getattr(value, "lon", None) or getattr(value, "lng", None)
    if lat is not None and lon is not None:
        try:
            return {"lat": float(lat), "lon": float(lon)}
        except (TypeError, ValueError):
            pass

    return None


def _dict_get_lat(d: dict) -> float | None:
    for key in ("lat", "latitude", "LAT", "LATITUDE", "Lat", "Latitude"):
        v = d.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _dict_get_lon(d: dict) -> float | None:
    for key in (
        "lon",
        "lng",
        "long",
        "longitude",
        "LON",
        "LNG",
        "LONG",
        "LONGITUDE",
        "Lon",
        "Lng",
    ):
        v = d.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None
