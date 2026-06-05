""":class:`GeoPointType` — canonical struct of (lat, lon) ``float64``.

A two-column nested struct used by curated tables that carry a
location reference. The contract:

* ``lat: float64`` in WGS84 degrees, range ``[-90, 90]``.
* ``lon: float64`` in WGS84 degrees, range ``[-180, 180]``.

Both columns are non-nullable inside the struct — when a row has
no location, mark the **whole struct** nullable via the parent
:class:`Field` (``geo_point("position", nullable=True)``). Mixing
``NULL`` with one valid coordinate is meaningless.

Why a struct (not two flat columns)
-----------------------------------
Two flat columns (``lat``, ``lon``) are fine when the location is a
top-level attribute of the row. Switch to ``GeoPointType`` when:

* the row carries *multiple* locations (``origin``, ``destination``)
  and flat naming would proliferate (``origin_lat`` / ``origin_lon``
  / ``destination_lat`` / ``destination_lon`` reads worse than two
  ``GeoPoint`` columns),
* a downstream sink (Arrow, Parquet, JSON, GeoJSON encoder, frontend
  map plugin) speaks the ``{"lat": …, "lon": …}`` shape natively,
* a list-of-points column (``points: list<struct<lat, lon, …>>``)
  is the shape — see :func:`geo_point` below.

For flat-column curated rows the column-naming convention from
``ygg-curated-views`` (`lat: float64`, `lon: float64` at the top
level) still applies — they're equivalent payloads, pick whichever
reads better at the call site.

Implementation
--------------
:data:`GEO_POINT_TYPE` is a pre-built :class:`StructType` instance
the rest of the codebase imports as a singleton; building a fresh
``StructType`` per call would defeat the cast-registry's exact-match
fast path. :func:`geo_point` is the corresponding ``Field`` factory
so callers don't have to repeat the field args (``nullable``,
``metadata``, ``tags``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from yggdrasil.data.types.nested.struct import StructType
from yggdrasil.data.types.primitive.numeric.floating_point import Float64Type
from yggdrasil.lazy_imports import field_class

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.data.data_field import Field


__all__ = [
    "GEO_POINT_TYPE",
    "GEO_POINT_FIELDS",
    "geo_point",
    "is_geo_point_type",
]


def _build_geo_point_struct() -> StructType:
    Field = field_class()
    return StructType(fields=(
        Field("lat", Float64Type(), nullable=False),
        Field("lon", Float64Type(), nullable=False),
    ))


#: Canonical :class:`StructType` for a WGS84 point. Singleton — same
#: instance reused across every :func:`geo_point` call so the cast
#: registry hits its exact-match fast path on every read.
GEO_POINT_TYPE: StructType = _build_geo_point_struct()


#: The two child field names, in canonical order. Exposed so callers
#: that have to fan out the struct (write_arrow row builders, GeoJSON
#: encoders) don't hand-roll the strings.
GEO_POINT_FIELDS: tuple[str, str] = ("lat", "lon")


def _merge_comment(
    metadata: Optional[Mapping[Union[bytes, str], Any]],
    comment: Optional[str],
) -> Optional[dict[Union[bytes, str], Any]]:
    """Splice ``comment`` into ``metadata`` under the ``b"comment"`` key.

    Mirrors how the rest of the codebase stores column comments
    (``Field.metadata[b"comment"]`` → bytes-keyed UTF-8). Returning
    ``None`` when both inputs are empty lets ``Field.__init__`` keep
    the metadata slot clean.
    """
    if comment is None:
        return dict(metadata) if metadata else None
    merged: dict[Union[bytes, str], Any] = dict(metadata) if metadata else {}
    merged.setdefault(b"comment", comment.encode("utf-8"))
    return merged


def geo_point(
    name: str,
    *,
    nullable: bool = True,
    comment: Optional[str] = None,
    metadata: Optional[Mapping[Union[bytes, str], Any]] = None,
    tags: Optional[Mapping[Union[bytes, str], Any]] = None,
) -> "Field":
    """Build a :class:`Field` of the canonical :data:`GEO_POINT_TYPE`.

    ``nullable`` flips on the *struct* — the inner ``lat`` / ``lon``
    children stay non-nullable so partial-coordinate rows (one of
    the two NULL) can't slip through.

    Example::

        from yggdrasil.data import Field, Schema, geo_point

        ORIGIN_DESTINATION_SCHEMA = Schema.from_fields([
            Field("trip_id", DataType.string(), nullable=False),
            geo_point("origin",      comment="Pickup point, WGS84."),
            geo_point("destination", comment="Drop-off point, WGS84."),
        ])
    """
    Field = field_class()
    return Field(
        name,
        GEO_POINT_TYPE,
        nullable=nullable,
        metadata=_merge_comment(metadata, comment),
        tags=dict(tags) if tags else None,
    )


def is_geo_point_type(value: Any) -> bool:
    """Return ``True`` when *value* is the canonical GeoPoint shape.

    Matches a :class:`StructType` with exactly the two children
    ``lat: float64`` + ``lon: float64`` (any nullability, any extra
    metadata). Useful for downstream code that wants to emit a
    GeoJSON / map-plugin representation for these and only these
    struct columns.
    """
    if not isinstance(value, StructType):
        return False
    if len(value.fields) != 2:
        return False
    by_name = {f.name: f for f in value.fields}
    lat = by_name.get("lat")
    lon = by_name.get("lon")
    if lat is None or lon is None:
        return False
    return (
        isinstance(lat.dtype, Float64Type)
        and isinstance(lon.dtype, Float64Type)
    )
