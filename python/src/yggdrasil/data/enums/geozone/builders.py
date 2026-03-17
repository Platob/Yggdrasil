from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from .constants import _METADATA_KEYS, _country_defaults
from .geozone import GeoZone, GeoZoneType

__all__ = [
    "_from_coordinates_with_optional_metadata",
    "_country",
    "_city",
    "_zone",
    "_de_tso_zone",
    "_validate_unique_attrs",
]


def _from_coordinates_with_optional_metadata(**kwargs):
    try:
        return GeoZone.from_coordinates(**kwargs)
    except TypeError:
        slim = {k: v for k, v in kwargs.items() if k not in _METADATA_KEYS}
        return GeoZone.from_coordinates(**slim)


def _country(
    iso: str,
    name: str,
    lat: float,
    lon: float,
    tz: Optional[str] = None,
    ccy: Optional[str] = None,
    *,
    aliases: tuple[str, ...] = (),
    eic: Optional[str] = None,
    srid: int = 4326,
    coord_source: Optional[str] = None,
    coord_kind: str = "country_centroid",
    confidence: str = "high",
    valid_from: Optional[str] = None,
    valid_to: Optional[str] = None,
) -> GeoZone:
    default_tz, default_ccy = _country_defaults(iso)
    return GeoZone.put(
        _from_coordinates_with_optional_metadata(
            gtype=GeoZoneType.COUNTRY,
            lat=lat,
            lon=lon,
            srid=srid,
            country_iso=iso,
            country_name=name,
            key=iso,
            aliases=aliases,
            name=name,
            eic=eic,
            tz=tz if tz is not None else default_tz,
            ccy=ccy if ccy is not None else default_ccy,
            coord_source=coord_source,
            coord_kind=coord_kind,
            confidence=confidence,
            valid_from=valid_from,
            valid_to=valid_to,
        )
    )


def _city(
    iso: str,
    name: str,
    lat: float,
    lon: float,
    country_iso: Optional[str] = None,
    country_name: Optional[str] = None,
    tz: Optional[str] = None,
    ccy: Optional[str] = None,
    *,
    aliases: tuple[str, ...] = (),
    eic: Optional[str] = None,
    srid: int = 4326,
    coord_source: Optional[str] = None,
    coord_kind: str = "city_center",
    confidence: str = "medium",
    valid_from: Optional[str] = None,
    valid_to: Optional[str] = None,
) -> GeoZone:
    # Fill country_iso / country_name from the cached country zone when omitted.
    if country_iso is None or country_name is None:
        # If country_iso is not given, try to infer it from the first two
        # characters of the city iso (e.g. "ZRH" → "ZR" won't match, but
        # callers that pass "CH" as country_iso will). When that also fails,
        # fall back to a two-letter prefix of the city iso so that patterns
        # like _city("SE_STHLM", ...) can resolve to "SE".
        candidate_iso = country_iso or (iso[:2] if len(iso) >= 2 else None)
        country_zone = GeoZone.get_by_key(candidate_iso) if candidate_iso else None
        if country_zone is not None:
            if country_iso is None:
                country_iso = country_zone.country_iso or candidate_iso
            if country_name is None:
                country_name = country_zone.country_name or country_zone.name
    default_tz, default_ccy = _country_defaults(country_iso)
    return GeoZone.put(
        _from_coordinates_with_optional_metadata(
            gtype=GeoZoneType.CITY,
            lat=lat,
            lon=lon,
            srid=srid,
            country_iso=country_iso,
            country_name=country_name,
            city_iso=iso,
            city_name=name,
            key=iso,
            aliases=aliases,
            name=name,
            eic=eic,
            tz=tz if tz is not None else default_tz,
            ccy=ccy if ccy is not None else default_ccy,
            coord_source=coord_source or "seed: module (city_center representative point)",
            coord_kind=coord_kind,
            confidence=confidence,
            valid_from=valid_from,
            valid_to=valid_to,
        )
    )


def _zone(
    key: str,
    name: str,
    lat: float,
    lon: float,
    tz: Optional[str] = None,
    ccy: Optional[str] = None,
    *,
    aliases: tuple[str, ...] = (),
    eic: Optional[str] = None,
    country_iso: Optional[str] = None,
    country_name: Optional[str] = None,
    city_iso: Optional[str] = None,
    city_name: Optional[str] = None,
    srid: int = 4326,
    coord_source: Optional[str] = None,
    coord_kind: str = "representative_point",
    confidence: str = "medium",
    valid_from: Optional[str] = None,
    valid_to: Optional[str] = None,
) -> GeoZone:
    default_tz, default_ccy = _country_defaults(country_iso)
    return GeoZone.put(
        _from_coordinates_with_optional_metadata(
            gtype=GeoZoneType.ZONE,
            lat=lat,
            lon=lon,
            srid=srid,
            country_iso=country_iso,
            country_name=country_name,
            city_iso=city_iso,
            city_name=city_name,
            key=key,
            aliases=aliases,
            name=name,
            eic=eic,
            tz=tz if tz is not None else default_tz,
            ccy=ccy if ccy is not None else default_ccy,
            coord_source=coord_source or "seed: module (representative_point; not a polygon centroid)",
            coord_kind=coord_kind,
            confidence=confidence,
            valid_from=valid_from,
            valid_to=valid_to,
        )
    )


def _de_tso_zone(
    key: str,
    name: str,
    lat: float,
    lon: float,
    eic: str,
    *,
    aliases: tuple[str, ...] = (),
    coord_source: Optional[str] = None,
    coord_kind: str = "representative_point",
    confidence: str = "medium",
    valid_from: Optional[str] = None,
    valid_to: Optional[str] = None,
) -> GeoZone:
    return _zone(
        key,
        name,
        lat,
        lon,
        "Europe/Berlin",
        "EUR",
        country_iso="DE",
        country_name="Germany",
        eic=eic,
        aliases=aliases,
        coord_source=coord_source or "seed: module (representative point for German TSO area)",
        coord_kind=coord_kind,
        confidence=confidence,
        valid_from=valid_from,
        valid_to=valid_to,
    )


def _validate_unique_attrs(rows: Iterable[tuple[str, ...]], label: str) -> None:
    seen: set[str] = set()
    for row in rows:
        attr = row[0]
        if attr in seen:
            raise ValueError(f"Duplicate GeoZone attr in {label}: {attr}")
        seen.add(attr)