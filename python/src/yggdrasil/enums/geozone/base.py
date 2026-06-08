from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

__all__ = ["GeoZone", "GeoZoneType", "normalize_geozone_name", "normalize_geozone_token"]


def normalize_geozone_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().split())
    return normalized or None


def normalize_geozone_token(value: str | None) -> str | None:
    normalized = normalize_geozone_name(value)
    if normalized is None:
        return None
    return normalized.replace("-", " ").replace("_", " ").upper()


class GeoZoneType(str, Enum):
    EARTH = "EARTH"
    CONTINENT = "CONTINENT"
    COUNTRY = "COUNTRY"
    REGION = "REGION"
    CITY = "CITY"
    EIC = "EIC"
    CUSTOM = "CUSTOM"


@dataclass(frozen=True, slots=True)
class GeoZone:
    gtype: GeoZoneType
    lat: float
    lon: float
    key: str | None = None
    name: str | None = None
    country_iso: str | None = None
    region_iso: str | None = None
    sub_iso: str | None = None
    ccy: str | None = None
    eic: str | None = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        lat = float(self.lat)
        lon = float(self.lon)
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"latitude must be between -90 and 90, got {lat}")
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"longitude must be between -180 and 180, got {lon}")

        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)
        object.__setattr__(self, "gtype", GeoZoneType(self.gtype))
        object.__setattr__(self, "key", normalize_geozone_token(self.key))
        object.__setattr__(self, "name", normalize_geozone_name(self.name))
        object.__setattr__(self, "country_iso", normalize_geozone_token(self.country_iso))
        object.__setattr__(self, "region_iso", normalize_geozone_token(self.region_iso))
        object.__setattr__(self, "sub_iso", normalize_geozone_token(self.sub_iso))
        object.__setattr__(self, "ccy", normalize_geozone_token(self.ccy))
        object.__setattr__(self, "eic", normalize_geozone_token(self.eic))
        object.__setattr__(
            self,
            "aliases",
            tuple(
                alias
                for alias in (
                    normalize_geozone_token(value)
                    for value in self.aliases
                )
                if alias is not None
            ),
        )

    @property
    def code(self) -> str | None:
        return self.region_iso or self.country_iso or self.key

    def to_dict(self) -> dict[str, Any]:
        return {
            "gtype": self.gtype.value,
            "key": self.key,
            "name": self.name,
            "country_iso": self.country_iso,
            "region_iso": self.region_iso,
            "sub_iso": self.sub_iso,
            "ccy": self.ccy,
            "eic": self.eic,
            "lat": self.lat,
            "lon": self.lon,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GeoZone":
        return cls(
            gtype=value.get("gtype", GeoZoneType.CUSTOM),
            lat=value["lat"],
            lon=value["lon"],
            key=value.get("key"),
            name=value.get("name"),
            country_iso=value.get("country_iso"),
            region_iso=value.get("region_iso"),
            sub_iso=value.get("sub_iso"),
            ccy=value.get("ccy"),
            eic=value.get("eic"),
            aliases=tuple(value.get("aliases", ()) or ()),
        )
