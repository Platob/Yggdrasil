"""Geography enrichment — attach a country GeoZone to each currency.

A :class:`~yggdrasil.enums.geozone.GeoZoneCatalog` built from the REST
countries dataset carries a currency code (``ccy``) per zone. We index it
once (currency → representative zone), cache the index process-wide, and
join it onto an FX frame's ``source`` / ``target`` columns. The first call
pays the one-time HTTP fetch + index build; every call after reuses the
cached index.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from yggdrasil.enums.geozone import GeoZone

__all__ = ["currency_geozones", "enrich_frame"]

_INDEX: dict[str, "GeoZone"] | None = None


def currency_geozones() -> dict[str, "GeoZone"]:
    """``currency-code → GeoZone`` index (cached; builds on first use)."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    from yggdrasil.enums.geozone import GeoZoneCatalog

    catalog = GeoZoneCatalog.empty().with_country_geozones()
    index: dict[str, GeoZone] = {}
    for zone in catalog.zones:
        if zone.ccy and zone.ccy not in index:
            index[zone.ccy] = zone
    _INDEX = index
    return index


def enrich_frame(frame: "pl.DataFrame") -> "pl.DataFrame":
    """Join country/lat/lon onto the ``source`` and ``target`` columns."""
    import polars as pl

    index = currency_geozones()
    geo = pl.DataFrame(
        {
            "_ccy": list(index.keys()),
            "_country": [z.country_iso for z in index.values()],
            "_lat": [z.lat for z in index.values()],
            "_lon": [z.lon for z in index.values()],
        }
    )
    out = frame
    for side in ("source", "target"):
        renamed = geo.rename({
            "_ccy": side,
            "_country": f"{side}_country",
            "_lat": f"{side}_lat",
            "_lon": f"{side}_lon",
        })
        out = out.join(renamed, on=side, how="left")
    return out
