from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from yggdrasil.io.http_ import HTTPSession

from .base import GeoZone, GeoZoneType, normalize_geozone_name, normalize_geozone_token
from .catalog import GeoZoneCatalog
from .countries import fetch_country_geozones

__all__ = [
    "ENTSOE_EIC_BIDDING_ZONES_URL",
    "EntsoeBiddingZoneRecord",
    "fetch_entsoe_bidding_zones",
]


ENTSOE_EIC_BIDDING_ZONES_URL = "https://eepublicdownloads.entsoe.eu/eic-codes-csv/Y_eiccodes.csv"


@dataclass(frozen=True, slots=True)
class EntsoeBiddingZoneRecord:
    eic: str
    display_name: str
    long_name: str | None
    parent: str | None
    responsible_party: str | None
    status: str | None
    postal_code: str | None
    country_iso: str | None
    vat_code: str | None
    function: str | None
    type_code: str | None

    @classmethod
    def from_csv_row(cls, line: str) -> "EntsoeBiddingZoneRecord | None":
        row = line.strip()
        if not row:
            return None

        parts = [part.strip() for part in row.split(";")]
        if len(parts) < 11:
            return None

        return cls(
            eic=parts[0],
            display_name=parts[1],
            long_name=";".join(parts[2:-8]).strip() or None,
            parent=parts[-8] or None,
            responsible_party=parts[-7] or None,
            status=parts[-6] or None,
            postal_code=parts[-5] or None,
            country_iso=parts[-4] or None,
            vat_code=parts[-3] or None,
            function=parts[-2] or None,
            type_code=parts[-1] or None,
        )


def _infer_country_iso(display_name: str, country_iso: str | None) -> str | None:
    if country_iso:
        return normalize_geozone_token(country_iso)

    normalized = normalize_geozone_token(display_name)
    if normalized is None:
        return None

    prefix = normalized.split(" ", 1)[0]
    if len(prefix) == 2 and prefix.isalpha():
        return prefix
    return None


def _country_lookup() -> dict[str, GeoZone]:
    return {
        zone.country_iso: zone
        for zone in fetch_country_geozones()
        if zone.country_iso is not None
    }


def _record_to_zone(record: EntsoeBiddingZoneRecord, countries: Mapping[str, GeoZone]) -> GeoZone | None:
    if record.function != "Bidding Zone":
        return None

    display_name = normalize_geozone_name(record.display_name)
    if display_name is None:
        return None

    country_iso = _infer_country_iso(record.display_name, record.country_iso)
    country_zone = countries.get(country_iso) if country_iso is not None else None

    lat = country_zone.lat if country_zone is not None else 0.0
    lon = country_zone.lon if country_zone is not None else 0.0

    aliases = [record.eic]
    if record.long_name:
        aliases.append(record.long_name)

    return GeoZone(
        gtype=GeoZoneType.EIC,
        key=display_name,
        name=record.long_name or display_name,
        country_iso=country_iso,
        region_iso=display_name,
        sub_iso=None,
        ccy=country_zone.ccy if country_zone is not None else None,
        eic=record.eic,
        lat=lat,
        lon=lon,
        aliases=tuple(aliases),
    )


def _parse_entsoe_csv(text: str) -> tuple[EntsoeBiddingZoneRecord, ...]:
    records: list[EntsoeBiddingZoneRecord] = []
    for index, line in enumerate(text.splitlines()):
        if index == 0 and line.lower().startswith("eiccode;"):
            continue
        record = EntsoeBiddingZoneRecord.from_csv_row(line)
        if record is not None:
            records.append(record)
    return tuple(records)


def fetch_entsoe_bidding_zones(
    *,
    session: HTTPSession | None = None,
    url: str = ENTSOE_EIC_BIDDING_ZONES_URL,
    countries: Mapping[str, GeoZone] | None = None,
) -> tuple[GeoZone, ...]:
    owns_session = session is None
    http = session or HTTPSession()

    try:
        response = http.get(url, headers={"Accept": "text/csv"})
        records = _parse_entsoe_csv(response.text)
    finally:
        if owns_session:
            http.__exit__(None, None, None)

    country_map = dict(countries) if countries is not None else _country_lookup()
    zones: list[GeoZone] = []
    seen: set[str] = set()

    for record in records:
        zone = _record_to_zone(record, country_map)
        if zone is None:
            continue
        if zone.key in seen:
            continue
        seen.add(zone.key)
        zones.append(zone)

    zones.sort(key=lambda zone: (zone.country_iso or "", zone.key or ""))
    return tuple(zones)
