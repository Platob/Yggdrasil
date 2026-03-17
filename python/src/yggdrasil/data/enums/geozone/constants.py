from __future__ import annotations

from typing import Optional

__all__ = [
    "_SRC_GOOGLE_COUNTRIES_CSV",
    "_SRC_IANA_ZONE1970_TAB",
    "_SRC_ENTSOE_TP_AREA_LIST",
    "_SRC_ENTSOE_MARKET_AREAS_V21",
    "_SRC_ECB_BG_EURO_INTRO",
    "_SRC_EU_COUNCIL_BG_EURO_FINAL",
    "_SRC_DE_AT_GO_LIVE_2018",
    "_GEOZONE_PATCH_SUMMARY",
    "_COUNTRY_DEFAULTS",
    "_country_defaults",
    "_METADATA_KEYS",
]

_SRC_GOOGLE_COUNTRIES_CSV = "https://developers.google.com/public-data/docs/canonical/countries_csv"
_SRC_IANA_ZONE1970_TAB = "https://data.iana.org/time-zones/data/zone1970.tab"
_SRC_ENTSOE_TP_AREA_LIST = (
    "https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC"
)
_SRC_ENTSOE_MARKET_AREAS_V21 = (
    "https://eepublicdownloads.entsoe.eu/clean-documents/EDI/Library/Market_Areas_v2.1.pdf"
)
_SRC_ECB_BG_EURO_INTRO = "https://www.ecb.europa.eu/press/pr/date/2026/html/ecb.pr260101~c830245e42.en.html"
_SRC_EU_COUNCIL_BG_EURO_FINAL = (
    "https://www.consilium.europa.eu/en/press/press-releases/2025/07/08/"
    "bulgaria-ready-to-use-the-euro-from-1-january-2026-council-takes-final-steps/"
)
_SRC_DE_AT_GO_LIVE_2018 = (
    "https://www.transnetbw.de/en/newsroom/press-releases/"
    "go-live-of-congestion-management-on-the-german-austrian-bidding-zone-border-de-at-bzb-on-1st-of-october-2018"
)

_GEOZONE_PATCH_SUMMARY = """
key | field_changed | old_value | new_value | source
BG (country defaults) | ccy | BGN | EUR (effective 2026-01-01) | https://www.ecb.europa.eu/press/pr/date/2026/html/ecb.pr260101~c830245e42.en.html
IS (country) | eic | 10Y1001A1001A39I | None | https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC
DK (country) | eic | None | 10Y1001A1001A65H | https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC
BY (country) | eic | None | 10Y1001A1001A51S | https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC
DE_AT_LU (zone) | valid_to | None | 2018-09-30 | https://www.transnetbw.de/en/newsroom/press-releases/go-live-of-congestion-management-on-the-german-austrian-bidding-zone-border-de-at-bzb-on-1st-of-october-2018
DE_LU (zone) | valid_from | None | 2018-10-01 | https://www.transnetbw.de/en/newsroom/press-releases/go-live-of-congestion-management-on-the-german-austrian-bidding-zone-border-de-at-bzb-on-1st-of-october-2018
DE_AMPRION (zone) | new_record | None | 10YDE-RWENET---I | https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC
DE_50HERTZ (zone) | new_record | None | 10YDE-VE-------2 | https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC
DE_TENNET (zone) | new_record | None | 10YDE-EON------1 | https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC
DE_TRANSNETBW (zone) | new_record | None | 10YDE-ENBW-----N | https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC
DE_AMPRION_LU (zone) | new_record | None | 10Y1001C--00002H | https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC
DE_DK1_LU (zone) | new_record | None | 10YCB-GERMANY--8 | https://transparencyplatform.zendesk.com/hc/en-us/articles/15885757676308-Area-List-with-Energy-Identification-Code-EIC
""".strip()

_METADATA_KEYS = {
    "coord_source",
    "coord_kind",
    "confidence",
    "valid_from",
    "valid_to",
}

_COUNTRY_DEFAULTS: dict[str, tuple[Optional[str], Optional[str]]] = {
    # Europe
    "FR": ("Europe/Paris", "EUR"),
    "CH": ("Europe/Zurich", "CHF"),
    "GB": ("Europe/London", "GBP"),
    "AT": ("Europe/Vienna", "EUR"),
    "BE": ("Europe/Brussels", "EUR"),
    "BG": ("Europe/Sofia", "EUR"),
    "HR": ("Europe/Zagreb", "EUR"),
    "CZ": ("Europe/Prague", "CZK"),
    "DK": ("Europe/Copenhagen", "DKK"),
    "EE": ("Europe/Tallinn", "EUR"),
    "FI": ("Europe/Helsinki", "EUR"),
    "DE": ("Europe/Berlin", "EUR"),
    "GR": ("Europe/Athens", "EUR"),
    "HU": ("Europe/Budapest", "HUF"),
    "IT": ("Europe/Rome", "EUR"),
    "LV": ("Europe/Riga", "EUR"),
    "LT": ("Europe/Vilnius", "EUR"),
    "LU": ("Europe/Luxembourg", "EUR"),
    "ME": ("Europe/Podgorica", "EUR"),
    "NL": ("Europe/Amsterdam", "EUR"),
    "NO": ("Europe/Oslo", "NOK"),
    "PL": ("Europe/Warsaw", "PLN"),
    "PT": ("Europe/Lisbon", "EUR"),
    "RO": ("Europe/Bucharest", "RON"),
    "RS": ("Europe/Belgrade", "RSD"),
    "SK": ("Europe/Bratislava", "EUR"),
    "SI": ("Europe/Ljubljana", "EUR"),
    "ES": ("Europe/Madrid", "EUR"),
    "SE": ("Europe/Stockholm", "SEK"),
    "UA": ("Europe/Kyiv", "UAH"),
    "XK": ("Europe/Belgrade", "EUR"),
    "MD": ("Europe/Chisinau", "MDL"),
    "MT": ("Europe/Malta", "EUR"),
    "IE": ("Europe/Dublin", "EUR"),
    "CY": ("Asia/Nicosia", "EUR"),
    "AL": ("Europe/Tirane", "ALL"),
    "BA": ("Europe/Sarajevo", "BAM"),
    "MK": ("Europe/Skopje", "MKD"),
    "BY": ("Europe/Minsk", "BYN"),
    "GE": ("Asia/Tbilisi", "GEL"),
    "AM": ("Asia/Yerevan", "AMD"),
    "AZ": ("Asia/Baku", "AZN"),
    "TR": ("Europe/Istanbul", "TRY"),
    "RU": ("Europe/Moscow", "RUB"),
    "IS": ("Atlantic/Reykjavik", "ISK"),
    "LI": ("Europe/Zurich", "CHF"),

    # Americas
    "US": ("America/New_York", "USD"),
    "CA": ("America/Toronto", "CAD"),
    "MX": ("America/Mexico_City", "MXN"),
    "BR": ("America/Sao_Paulo", "BRL"),
    "AR": ("America/Argentina/Buenos_Aires", "ARS"),
    "CL": ("America/Santiago", "CLP"),
    "CO": ("America/Bogota", "COP"),

    # Asia-Pacific
    "JP": ("Asia/Tokyo", "JPY"),
    "CN": ("Asia/Shanghai", "CNY"),
    "IN": ("Asia/Kolkata", "INR"),
    "KR": ("Asia/Seoul", "KRW"),
    "AU": ("Australia/Sydney", "AUD"),
    "NZ": ("Pacific/Auckland", "NZD"),
    "SG": ("Asia/Singapore", "SGD"),

    # Middle-East & Africa
    "SA": ("Asia/Riyadh", "SAR"),
    "AE": ("Asia/Dubai", "AED"),
    "ZA": ("Africa/Johannesburg", "ZAR"),
}


def _country_defaults(country_iso: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not country_iso:
        return None, None
    return _COUNTRY_DEFAULTS.get(country_iso, (None, None))