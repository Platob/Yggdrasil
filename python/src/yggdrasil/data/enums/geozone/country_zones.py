from __future__ import annotations

from .builders import _validate_unique_attrs, _from_coordinates_with_optional_metadata
from .geozone import GeoZone, GeoZoneType

__all__ = ["load_country_zones"]

_SRC = "seed: module (country-level geographic/political zone representative point)"


def _country_zone(
    key: str,
    name: str,
    lat: float,
    lon: float,
    *,
    aliases: tuple[str, ...] = (),
    country_iso: str | None = None,
    country_name: str | None = None,
    tz: str | None = None,
    confidence: str = "medium",
    coord_kind: str = "representative_point",
) -> GeoZone:
    return GeoZone.put(
        _from_coordinates_with_optional_metadata(
            gtype=GeoZoneType.ZONE,
            lat=lat,
            lon=lon,
            key=key,
            aliases=aliases,
            name=name,
            country_iso=country_iso,
            country_name=country_name,
            tz=tz,
            ccy=None,
            coord_source=_SRC,
            coord_kind=coord_kind,
            confidence=confidence,
        )
    )


COUNTRY_ZONES = [
    (
        "GREAT_BRITAIN",
        "GREAT_BRITAIN",
        "Great Britain",
        54.8,
        -4.6,
        {
            "aliases": (
                "BRITAIN",
                "GB_GREAT_BRITAIN",
                "GREAT_BRITAIN_XGB",
            ),
            "country_iso": "GB",
            "country_name": "United Kingdom",
            "tz": "Europe/London",
            "confidence": "high",
        },
    ),
    (
        "NORTHERN_IRELAND",
        "NORTHERN_IRELAND",
        "Northern Ireland",
        54.7877,
        -6.4923,
        {
            "aliases": (
                "NI",
                "NIR",
                "NORTH_IRELAND",
                "NORTHERN_IRELAND_XNI",
            ),
            "country_iso": "GB",
            "country_name": "United Kingdom",
            "tz": "Europe/London",
            "confidence": "high",
        },
    ),
]


def load_country_zones() -> None:
    _validate_unique_attrs(COUNTRY_ZONES, "COUNTRY_ZONES")
    for attr, key, name, lat, lon, kwargs in COUNTRY_ZONES:
        setattr(GeoZone, attr, _country_zone(key, name, lat, lon, **kwargs))