from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .base import GeoZone, GeoZoneType, normalize_geozone_name, normalize_geozone_token

__all__ = ["GeoZoneCatalog"]

_COORDINATE_RE = re.compile(
    r"^\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"\s*[,;|\s]\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"\s*$"
)


@dataclass(frozen=True, slots=True)
class GeoZoneCatalog:
    zones: tuple[GeoZone, ...]
    _token_index: dict[str, GeoZone] = field(init=False, repr=False)
    _coordinate_index: dict[tuple[float, float], GeoZone] = field(init=False, repr=False)
    _search_rows: tuple[tuple[str, GeoZone], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        token_index: dict[str, GeoZone] = {}
        coordinate_index: dict[tuple[float, float], GeoZone] = {}
        search_rows: list[tuple[str, GeoZone]] = []

        for zone in self.zones:
            for token in self._zone_tokens(zone):
                token_index[token] = zone
                search_rows.append((token, zone))
            coordinate_index[self._coordinate_key(zone.lat, zone.lon)] = zone

        object.__setattr__(self, "_token_index", token_index)
        object.__setattr__(self, "_coordinate_index", coordinate_index)
        object.__setattr__(self, "_search_rows", tuple(search_rows))

    @classmethod
    def empty(cls) -> "GeoZoneCatalog":
        return cls(zones=())

    @classmethod
    def from_values(cls, values: Iterable[GeoZone | Mapping[str, Any]]) -> "GeoZoneCatalog":
        zones = tuple(
            value if isinstance(value, GeoZone) else GeoZone.from_mapping(value)
            for value in values
        )
        return cls(zones=zones)

    @staticmethod
    def _coordinate_key(lat: float, lon: float) -> tuple[float, float]:
        return round(float(lat), 6), round(float(lon), 6)

    @staticmethod
    def _zone_tokens(zone: GeoZone) -> tuple[str, ...]:
        tokens = {
            token
            for token in (
                zone.key,
                zone.country_iso,
                zone.region_iso,
                zone.sub_iso,
                zone.ccy,
                zone.eic,
                normalize_geozone_token(zone.name),
                *zone.aliases,
            )
            if token is not None
        }
        return tuple(sorted(tokens))

    def extend(self, values: Iterable[GeoZone | Mapping[str, Any]]) -> "GeoZoneCatalog":
        return type(self).from_values((*self.zones, *values))

    def with_country_geozones(self) -> "GeoZoneCatalog":
        from .countries import fetch_country_geozones

        return self.extend(fetch_country_geozones())

    def with_entsoe_bidding_zones(self) -> "GeoZoneCatalog":
        from .entsoe import fetch_entsoe_bidding_zones

        return self.extend(fetch_entsoe_bidding_zones())

    def lookup(self, token: str | None) -> GeoZone | None:
        normalized = normalize_geozone_token(token)
        if normalized is None:
            return None
        return self._token_index.get(normalized)

    def lookup_coordinates(self, lat: float, lon: float) -> GeoZone | None:
        return self._coordinate_index.get(self._coordinate_key(lat, lon))

    def find_by_str(self, value: str | None) -> GeoZone | None:
        normalized = normalize_geozone_token(value)
        if normalized is None:
            return None

        exact = self.lookup(normalized)
        if exact is not None:
            return exact

        candidates: list[tuple[float, int, GeoZone]] = []
        for token, zone in self._search_rows:
            score = difflib.SequenceMatcher(a=normalized, b=token).ratio()
            if normalized in token or token in normalized:
                score += 0.2
            candidates.append((score, len(token), zone))

        if not candidates:
            return None

        best_score, _, best_zone = max(candidates, key=lambda item: (item[0], -item[1]))
        if best_score < 0.6:
            return None
        return best_zone

    def parse_str(self, value: str | None) -> GeoZone | None:
        normalized = normalize_geozone_name(value)
        if normalized is None:
            return None

        zone = self.lookup(normalized)
        if zone is not None:
            return zone

        match = _COORDINATE_RE.match(normalized)
        if match is None:
            return self.find_by_str(normalized)

        lat = float(match.group(1))
        lon = float(match.group(2))
        return self.lookup_coordinates(lat, lon) or GeoZone(gtype=GeoZoneType.CUSTOM, lat=lat, lon=lon)

    def parse(self, value: Any) -> GeoZone | None:
        if value is None:
            return None
        if isinstance(value, GeoZone):
            return value
        if isinstance(value, str):
            return self.parse_str(value)
        if isinstance(value, Mapping):
            token = value.get("region_iso") or value.get("country_iso") or value.get("key") or value.get("name")
            if token is not None:
                zone = self.lookup(str(token))
                if zone is not None:
                    return zone
            if "lat" in value and "lon" in value:
                zone = self.lookup_coordinates(value["lat"], value["lon"])
                if zone is not None:
                    return zone
                return GeoZone.from_mapping(value)
            return None
        if isinstance(value, tuple) and len(value) == 2:
            lat, lon = value
            zone = self.lookup_coordinates(lat, lon)
            if zone is not None:
                return zone
            return GeoZone(gtype=GeoZoneType.CUSTOM, lat=lat, lon=lon)
        return None

    def _matches_filter(self, zone: GeoZone, *, field_name: str, value: str | None) -> bool:
        if value is None:
            return True

        target = getattr(zone, field_name)
        if target is None:
            return False
        return normalize_geozone_token(target) == normalize_geozone_token(value)

    def filter_zones(
        self,
        *,
        key: str | None = None,
        country_iso: str | None = None,
        region_iso: str | None = None,
        sub_iso: str | None = None,
        ccy: str | None = None,
        eic: str | None = None,
        text: str | None = None,
    ) -> tuple[GeoZone, ...]:
        zones = tuple(
            zone
            for zone in self.zones
            if self._matches_filter(zone, field_name="key", value=key)
            and self._matches_filter(zone, field_name="country_iso", value=country_iso)
            and self._matches_filter(zone, field_name="region_iso", value=region_iso)
            and self._matches_filter(zone, field_name="sub_iso", value=sub_iso)
            and self._matches_filter(zone, field_name="ccy", value=ccy)
            and self._matches_filter(zone, field_name="eic", value=eic)
        )

        if text is None:
            return zones

        matched = self.find_by_str(text)
        if matched is None:
            return ()
        return tuple(zone for zone in zones if zone == matched)

    def to_rows(self) -> list[dict[str, Any]]:
        return [zone.to_dict() for zone in self.zones]

    def to_polars(
        self,
        *,
        key: str | None = None,
        country_iso: str | None = None,
        region_iso: str | None = None,
        sub_iso: str | None = None,
        ccy: str | None = None,
        eic: str | None = None,
        text: str | None = None,
    ):
        from yggdrasil.polars.lib import polars

        zones = self.filter_zones(
            key=key,
            country_iso=country_iso,
            region_iso=region_iso,
            sub_iso=sub_iso,
            ccy=ccy,
            eic=eic,
            text=text,
        )
        return polars.DataFrame(zone.to_dict() for zone in zones)

    def lookup_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for zone in self.zones:
            for token in self._zone_tokens(zone):
                rows.append(
                    {
                        "lookup_token": token,
                        "gtype": zone.gtype.value,
                        "key": zone.key,
                        "name": zone.name,
                        "country_iso": zone.country_iso,
                        "region_iso": zone.region_iso,
                        "sub_iso": zone.sub_iso,
                        "ccy": zone.ccy,
                        "eic": zone.eic,
                        "lat": zone.lat,
                        "lon": zone.lon,
                    }
                )
        return rows
