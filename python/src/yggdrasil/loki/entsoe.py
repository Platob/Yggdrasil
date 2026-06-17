"""ENTSO-E Transparency Platform — European power-market data as frames.

Loki's energy-data path: pull day-ahead **prices**, actual **load**, and
**generation** for a European bidding zone from the
`ENTSO-E Transparency Platform <https://transparency.entsoe.eu>`_ REST API and
parse the publication XML straight into a tidy polars frame (one row per
timestamp). Everything rides the project's own machinery — the request goes
through :class:`~yggdrasil.http_.HTTPSession`, the frame is cached/stored
through the io handlers (:class:`~yggdrasil.io.holder.IO`) by
:class:`~yggdrasil.loki.skills.EntsoeSkill`.

The API is token-gated: set ``ENTSOE_API_TOKEN`` (or ``ENTSOE_SECURITY_TOKEN``)
to the free security token from your Transparency Platform account. Bidding
zones are addressed by their EIC code; common zones have short aliases
(``DE_LU``, ``FR``, ``NL`` …) and the full EIC registry is available through
:mod:`yggdrasil.enums.geozone.entsoe`.

This module is pure + offline-friendly: :func:`build_query` and
:func:`parse_timeseries_xml` never touch the network, so the parsing is unit
tested against fixture XML; only :func:`fetch_frame` makes the call.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Optional

__all__ = [
    "ENTSOE_API", "DOCUMENT_TYPES", "ZONES",
    "resolve_zone", "build_query", "parse_timeseries_xml", "to_frame",
    "fetch_frame", "token", "infer_query",
]

#: The Transparency Platform REST endpoint (XML).
ENTSOE_API = "https://web-api.tp.entsoe.eu/api"

#: Friendly series name → ENTSO-E request shape. ``domain`` names which
#: parameter carries the zone EIC; ``process`` is the (optional) processType.
#: Data over code — add a row here, not a branch.
DOCUMENT_TYPES: dict[str, dict[str, str]] = {
    # Day-ahead spot prices (EUR/MWh) — in/out domain are the same zone.
    "day_ahead_prices": {"documentType": "A44", "domain": "in_out"},
    # Actual total load (MW), realised.
    "load": {"documentType": "A65", "process": "A16", "domain": "outBiddingZone"},
    # Actual aggregated generation (MW), realised.
    "generation": {"documentType": "A75", "process": "A16", "domain": "in"},
}

#: Short aliases for the most-used European bidding zones → EIC code. The full
#: registry (every EIC) is in :func:`yggdrasil.enums.geozone.entsoe.fetch_entsoe_bidding_zones`.
ZONES: dict[str, str] = {
    "DE_LU": "10Y1001A1001A82H", "DE_AT_LU": "10Y1001A1001A63L",
    "FR": "10YFR-RTE------C", "NL": "10YNL----------L", "BE": "10YBE----------2",
    "AT": "10YAT-APG------L", "CH": "10YCH-SWISSGRIDZ", "ES": "10YES-REE------0",
    "PT": "10YPT-REN------W", "IT_NORD": "10Y1001A1001A73I", "IT_CNOR": "10Y1001A1001A70O",
    "PL": "10YPL-AREA-----S", "CZ": "10YCZ-CEPS-----N", "DK_1": "10YDK-1--------W",
    "DK_2": "10YDK-2--------M", "NO_2": "10YNO-2--------T", "SE_3": "10Y1001A1001A46L",
    "FI": "10YFI-1--------U", "GB": "10YGB----------A", "IE_SEM": "10Y1001A1001A59C",
    "HU": "10YHU-MAVIR----U", "RO": "10YRO-TEL------P", "GR": "10YGR-HTSO-----Y",
}

#: ISO-8601 duration → minutes per step, for the resolutions ENTSO-E emits.
_RESOLUTION_MIN: dict[str, int] = {
    "PT15M": 15, "PT30M": 30, "PT60M": 60, "PT1H": 60,
    "P1D": 1440, "P7D": 10080,
}


def token() -> Optional[str]:
    """The Transparency Platform security token, or ``None`` (offline-safe)."""
    return os.getenv("ENTSOE_API_TOKEN") or os.getenv("ENTSOE_SECURITY_TOKEN")


#: Country / region words (noun + adjective forms) → bidding-zone alias, for
#: autonomous NL routing — all matched on word boundaries (so "es" in "prices"
#: never resolves to Spain).
_COUNTRY_ZONES: dict[str, str] = {
    "germany": "DE_LU", "german": "DE_LU", "deutschland": "DE_LU",
    "france": "FR", "french": "FR", "netherlands": "NL", "dutch": "NL", "holland": "NL",
    "belgium": "BE", "belgian": "BE", "spain": "ES", "spanish": "ES",
    "portugal": "PT", "portuguese": "PT", "austria": "AT", "austrian": "AT",
    "switzerland": "CH", "swiss": "CH", "poland": "PL", "polish": "PL",
    "czech": "CZ", "denmark": "DK_1", "danish": "DK_1", "norway": "NO_2",
    "norwegian": "NO_2", "sweden": "SE_3", "swedish": "SE_3", "finland": "FI",
    "finnish": "FI", "britain": "GB", "british": "GB", "uk": "GB", "england": "GB",
    "italy": "IT_NORD", "italian": "IT_NORD", "ireland": "IE_SEM", "irish": "IE_SEM",
    "hungary": "HU", "hungarian": "HU", "romania": "RO", "romanian": "RO",
    "greece": "GR", "greek": "GR",
}


def infer_query(text: str) -> dict[str, str]:
    """Infer ``{series, zone}`` from a free-text energy request (for NL routing).

    Maps demand words → ``load``, production words → ``generation``, else
    ``day_ahead_prices``; and a country/zone mention → its EIC alias (default
    ``DE_LU``). Country names and zone aliases are matched on **word boundaries**
    so a 2-letter alias never matches a substring of an ordinary word.
    """
    low = text.lower()
    if any(w in low for w in ("load", "demand", "consumption")):
        series = "load"
    elif any(w in low for w in ("generation", "production", "generation mix")):
        series = "generation"
    else:
        series = "day_ahead_prices"
    zone = "DE_LU"
    for word, z in _COUNTRY_ZONES.items():
        if re.search(rf"\b{word}\b", low):
            return {"series": series, "zone": z}
    for alias in ZONES:                                   # explicit zone code / alias
        if re.search(rf"\b{re.escape(alias.lower())}\b", low):
            return {"series": series, "zone": alias}
    return {"series": series, "zone": zone}


def resolve_zone(zone: str) -> str:
    """A zone *alias or EIC* → its 16-char EIC code.

    Accepts a short alias (``"DE_LU"``, case/sep-insensitive — ``"de-lu"`` works),
    or an EIC code passed straight through. Raises for an unknown alias.
    """
    z = zone.strip()
    if re.fullmatch(r"[0-9A-Za-z\-]{16}", z) and not z.isalpha():
        return z                                   # already an EIC code
    key = re.sub(r"[\s\-]+", "_", z).upper()
    if key in ZONES:
        return ZONES[key]
    raise KeyError(
        f"unknown bidding zone {zone!r}; known aliases: {', '.join(sorted(ZONES))} "
        f"(or pass a 16-char EIC code directly)"
    )


def _stamp(when: dt.datetime | dt.date | str) -> str:
    """A datetime/date/ISO-string → ENTSO-E ``yyyyMMddHHmm`` (UTC)."""
    if isinstance(when, str):
        when = dt.datetime.fromisoformat(when.replace("Z", "+00:00"))
    if isinstance(when, dt.datetime):
        if when.tzinfo is not None:
            when = when.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return when.strftime("%Y%m%d%H%M")
    return dt.datetime(when.year, when.month, when.day).strftime("%Y%m%d%H%M")


def build_query(
    series: str,
    zone: str,
    start: dt.datetime | dt.date | str,
    end: dt.datetime | dt.date | str,
    *,
    security_token: Optional[str] = None,
) -> dict[str, str]:
    """Build the Transparency Platform query params for *series* over a period.

    *series* is a key of :data:`DOCUMENT_TYPES` (``"day_ahead_prices"`` /
    ``"load"`` / ``"generation"``); *zone* an alias or EIC. Pure — no network,
    no token required (pass ``security_token`` to embed one). The zone lands on
    the right parameter for the document type (both in/out for prices, the
    bidding-zone domain for load, the in-domain for generation).
    """
    if series not in DOCUMENT_TYPES:
        raise KeyError(f"unknown series {series!r}; known: {', '.join(DOCUMENT_TYPES)}")
    spec = DOCUMENT_TYPES[series]
    eic = resolve_zone(zone)
    params: dict[str, str] = {
        "documentType": spec["documentType"],
        "periodStart": _stamp(start),
        "periodEnd": _stamp(end),
    }
    if spec.get("process"):
        params["processType"] = spec["process"]
    domain = spec["domain"]
    if domain == "in_out":
        params["in_Domain"] = params["out_Domain"] = eic
    elif domain == "outBiddingZone":
        params["outBiddingZone_Domain"] = eic
    else:
        params["in_Domain"] = eic
    if security_token:
        params["securityToken"] = security_token
    return params


def _local(tag: str) -> str:
    """An XML tag without its namespace (``{ns}Point`` → ``Point``)."""
    return tag.rsplit("}", 1)[-1]


def _find(elem: ET.Element, name: str) -> "ET.Element | None":
    for child in elem:
        if _local(child.tag) == name:
            return child
    return None


def _findall(elem: ET.Element, name: str) -> "list[ET.Element]":
    return [c for c in elem if _local(c.tag) == name]


def _text(elem: "ET.Element | None", name: str) -> "str | None":
    child = _find(elem, name) if elem is not None else None
    return child.text.strip() if (child is not None and child.text) else None


# Cached tag-stripping: ENTSO-E documents reuse the same ~10 tag names across
# thousands of elements, so memoizing the namespace-strip pays off immediately.
_local_cache: dict[str, str] = {}


def _lc(tag: str) -> str:
    """Fast cached variant of _local — for the hot parse loop only."""
    v = _local_cache.get(tag)
    if v is None:
        v = tag.rsplit("}", 1)[-1]
        _local_cache[tag] = v
    return v


def parse_timeseries_xml(xml: str) -> list[dict[str, Any]]:
    """Parse an ENTSO-E publication document into tidy rows.

    Returns one dict per point — ``{timestamp, value, unit, currency,
    resolution, position}`` — with the UTC ``timestamp`` reconstructed from each
    Period's interval start + ``(position-1) * resolution``. Namespace-agnostic
    (the publication schema's namespace changes between document types/versions),
    and tolerant of the two value spellings — ``price.amount`` (prices) and
    ``quantity`` (load/generation). An ``Acknowledgement_MarketDocument`` (the
    API's "no matching data" reply) yields an empty list rather than raising.
    """
    root = ET.fromstring(xml.encode() if isinstance(xml, str) else xml)
    if _lc(root.tag).startswith("Acknowledgement"):
        return []                                   # API "no data" envelope

    rows: list[dict[str, Any]] = []
    for child0 in root:
        if _lc(child0.tag) != "TimeSeries":
            continue
        # Single pass over TimeSeries children — collect currency, unit, and
        # Period elements without re-iterating for each metadata field.
        currency: str | None = None
        unit: str | None = None
        periods: list[ET.Element] = []
        for child1 in child0:
            lname = _lc(child1.tag)
            if lname == "currency_Unit.name":
                currency = child1.text.strip() if child1.text else None
            elif lname == "price_Measure_Unit.name":
                unit = child1.text.strip() if child1.text else None
            elif lname == "quantity_Measure_Unit.name" and unit is None:
                unit = child1.text.strip() if child1.text else "MAW"
            elif lname == "Period":
                periods.append(child1)
        if unit is None:
            unit = "MAW"

        for period in periods:
            # Single pass over Period children — collect timeInterval, resolution,
            # and Point elements.
            start_txt: str | None = None
            resolution: str | None = None
            points: list[ET.Element] = []
            for child2 in period:
                lname = _lc(child2.tag)
                if lname == "timeInterval":
                    for child3 in child2:
                        if _lc(child3.tag) == "start":
                            start_txt = child3.text.strip() if child3.text else None
                            break
                elif lname == "resolution":
                    resolution = child2.text.strip() if child2.text else "PT60M"
                elif lname == "Point":
                    points.append(child2)
            if not start_txt:
                continue
            period_start = dt.datetime.fromisoformat(start_txt.replace("Z", "+00:00"))
            step = _RESOLUTION_MIN.get(resolution or "PT60M", 60)
            # Pre-compute the timedelta for one step — avoids constructing it
            # for every point in the inner loop.
            step_delta = dt.timedelta(minutes=step)

            for point in points:
                pos: str | None = None
                val: str | None = None
                for child3 in point:
                    lname = _lc(child3.tag)
                    if lname == "position":
                        pos = child3.text
                    elif lname in ("price.amount", "quantity"):
                        val = child3.text
                if pos is None or val is None:
                    continue
                position = int(pos)
                rows.append({
                    "timestamp": period_start + step_delta * (position - 1),
                    "value": float(val),
                    "unit": unit,
                    "currency": currency,
                    "resolution": resolution,
                    "position": position,
                })
    return rows


def to_frame(xml: str, *, zone: Optional[str] = None, series: Optional[str] = None):
    """Parse a publication document into a polars frame (timestamp-sorted).

    Columns: ``timestamp`` (UTC datetime), ``value`` (float), ``unit``,
    ``currency`` — plus constant ``zone`` / ``series`` columns when provided, so
    multi-zone fetches concat into one tidy long frame.
    """
    import polars as pl

    rows = parse_timeseries_xml(xml)
    if not rows:
        return pl.DataFrame(
            schema={"timestamp": pl.Datetime, "value": pl.Float64, "unit": pl.Utf8,
                    "currency": pl.Utf8, "resolution": pl.Utf8, "position": pl.Int64})
    frame = pl.DataFrame(rows, schema_overrides={"value": pl.Float64})
    if zone is not None:
        frame = frame.with_columns(pl.lit(zone).alias("zone"))
    if series is not None:
        frame = frame.with_columns(pl.lit(series).alias("series"))
    return frame.sort("timestamp")


def fetch_frame(
    series: str,
    zone: str,
    start: dt.datetime | dt.date | str,
    end: dt.datetime | dt.date | str,
    *,
    security_token: Optional[str] = None,
):
    """Fetch *series* for *zone* over a period → a polars frame.

    Rides :class:`~yggdrasil.http_.HTTPSession` (its pooling + retry). The token
    comes from the argument, else ``ENTSOE_API_TOKEN`` / ``ENTSOE_SECURITY_TOKEN``;
    a missing token raises a clear :class:`ValueError` (the skill turns that into
    an offline-safe message) rather than calling without auth.
    """
    from yggdrasil.http_ import HTTPSession

    tok = security_token or token()
    if not tok:
        raise ValueError(
            "no ENTSO-E token — set ENTSOE_API_TOKEN to the free security token "
            "from your Transparency Platform account (transparency.entsoe.eu)."
        )
    params = build_query(series, zone, start, end, security_token=tok)
    resp = HTTPSession().get(ENTSOE_API, params=params, raise_error=True)
    return to_frame(resp.text, zone=zone, series=series)
