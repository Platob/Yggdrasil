from __future__ import annotations

from .builders import _country, _validate_unique_attrs
from .constants import _SRC_GOOGLE_COUNTRIES_CSV
from .geozone import GeoZone

__all__ = ["load_countries"]

COUNTRIES = [
    ("FRANCE", "FR", "France", 46.2276, 2.2137, {"eic": "10YFR-RTE------C", "aliases": ("BZN|FR", "FRANCE_FR"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("SWITZERLAND", "CH", "Switzerland", 46.8182, 8.2275, {"eic": "10YCH-SWISSGRIDZ", "aliases": ("BZN|CH", "SWITZERLAND_CH"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("UNITED_KINGDOM", "GB", "United Kingdom", 55.3781, -3.4360, {"eic": "10Y1001A1001A92E", "aliases": ("UK", "GBR", "BZN|GB", "UNITED_KINGDOM_GB"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("AUSTRIA", "AT", "Austria", 47.5162, 14.5501, {"eic": "10YAT-APG------L", "aliases": ("BZN|AT",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("BELGIUM", "BE", "Belgium", 50.5039, 4.4699, {"eic": "10YBE----------2", "aliases": ("BZN|BE",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("BULGARIA", "BG", "Bulgaria", 42.7339, 25.4858, {"eic": "10YCA-BULGARIA-R", "aliases": ("BZN|BG",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("CROATIA", "HR", "Croatia", 45.1000, 15.2000, {"eic": "10YHR-HEP------M", "aliases": ("BZN|HR",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("CZECHIA", "CZ", "Czechia", 49.8175, 15.4730, {"eic": "10YCZ-CEPS-----N", "aliases": ("CZ", "CZECH_REPUBLIC", "BZN|CZ"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("DENMARK", "DK", "Denmark", 56.2639, 9.5018, {"eic": "10Y1001A1001A65H", "aliases": ("BZN|DK",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("ESTONIA", "EE", "Estonia", 58.5953, 25.0136, {"eic": "10Y1001A1001A39I", "aliases": ("BZN|EE",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("FINLAND", "FI", "Finland", 61.9241, 25.7482, {"eic": "10YFI-1--------U", "aliases": ("BZN|FI",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("GERMANY", "DE", "Germany", 51.1657, 10.4515, {"eic": "10Y1001A1001A83F", "aliases": ("GERMANY_DE", "IPA|DE"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("GREECE", "GR", "Greece", 39.0742, 21.8243, {"eic": "10YGR-HTSO-----Y", "aliases": ("BZN|GR",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("HUNGARY", "HU", "Hungary", 47.1625, 19.5033, {"eic": "10YHU-MAVIR----U", "aliases": ("BZN|HU",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("ITALY", "IT", "Italy", 41.8719, 12.5674, {"eic": "10YIT-GRTN-----B", "aliases": ("BZN|IT",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("LATVIA", "LV", "Latvia", 56.8796, 24.6032, {"eic": "10YLV-1001A00074", "aliases": ("BZN|LV",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("LITHUANIA", "LT", "Lithuania", 55.1694, 23.8813, {"eic": "10YLT-1001A0008Q", "aliases": ("BZN|LT",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("LUXEMBOURG", "LU", "Luxembourg", 49.8153, 6.1296, {"eic": "10YLU-CEGEDEL-NQ", "aliases": ("BZN|LU", "LUX"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("MONTENEGRO", "ME", "Montenegro", 42.7087, 19.3744, {"eic": "10YCS-CG-TSO---S", "aliases": ("BZN|ME",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("NETHERLANDS", "NL", "Netherlands", 52.1326, 5.2913, {"eic": "10YNL----------L", "aliases": ("BZN|NL", "HOLLAND"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("NORWAY", "NO", "Norway", 60.4720, 8.4689, {"eic": "10YNO-0--------C", "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("POLAND", "PL", "Poland", 51.9194, 19.1451, {"eic": "10YPL-AREA-----S", "aliases": ("BZN|PL",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("PORTUGAL", "PT", "Portugal", 39.3999, -8.2245, {"eic": "10YPT-REN------W", "aliases": ("BZN|PT",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("ROMANIA", "RO", "Romania", 45.9432, 24.9668, {"eic": "10YRO-TEL------P", "aliases": ("BZN|RO",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("SERBIA", "RS", "Serbia", 44.0165, 21.0059, {"eic": "10YCS-SERBIATSOV", "aliases": ("BZN|RS",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("SLOVAKIA", "SK", "Slovakia", 48.6690, 19.6990, {"eic": "10YSK-SEPS-----K", "aliases": ("BZN|SK",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("SLOVENIA", "SI", "Slovenia", 46.1512, 14.9955, {"eic": "10YSI-ELES-----O", "aliases": ("BZN|SI",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("SPAIN", "ES", "Spain", 40.4637, -3.7492, {"eic": "10YES-REE------0", "aliases": ("BZN|ES",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("SWEDEN", "SE", "Sweden", 60.1282, 18.6435, {"eic": "10YSE-1--------K", "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("UKRAINE", "UA", "Ukraine", 48.3794, 31.1656, {"eic": "10Y1001C--00003F", "aliases": ("BZN|UA",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("KOSOVO", "XK", "Kosovo", 42.6026, 20.9030, {"eic": "10Y1001C--00100H", "aliases": ("BZN|XK",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("MOLDOVA", "MD", "Moldova", 47.4116, 28.3699, {"eic": "10Y1001A1001A990", "aliases": ("BZN|MD",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("MALTA", "MT", "Malta", 35.9375, 14.3754, {"eic": "10Y1001A1001A93C", "aliases": ("BZN|MT",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("IRELAND", "IE", "Ireland", 53.4, -7.7, {"eic": "10Y1001A1001A59C", "aliases": ("IE_SEM", "SEM", "BZN|IE_SEM"), "coord_source": "seed: module (rounded/approx; consider aligning to Google countries.csv)", "confidence": "medium"}),
    ("CYPRUS", "CY", "Cyprus", 35.1264, 33.4299, {"eic": "10YCY-1001A0003J", "aliases": ("BZN|CY",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("ALBANIA", "AL", "Albania", 41.1533, 20.1683, {"eic": "10YAL-KESH-----5", "aliases": ("BZN|AL",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("BOSNIA", "BA", "Bosnia and Herzegovina", 43.9159, 17.6791, {"eic": "10YBA-JPCC-----D", "aliases": ("BZN|BA", "BOSNIA_HERZEGOVINA"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("NORTH_MACEDONIA", "MK", "North Macedonia", 41.6086, 21.7453, {"eic": "10YMK-MEPSO----8", "aliases": ("BZN|MK", "MACEDONIA"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("BELARUS", "BY", "Belarus", 53.7098, 27.9534, {"eic": "10Y1001A1001A51S", "aliases": ("BYELORUSSIA",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("GEORGIA", "GE", "Georgia", 42.3154, 43.3569, {"aliases": ("BZN|GE",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("ARMENIA", "AM", "Armenia", 40.0691, 45.0382, {"aliases": ("BZN|AM",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("AZERBAIJAN", "AZ", "Azerbaijan", 40.1431, 47.5769, {"aliases": ("BZN|AZ",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("TURKEY", "TR", "Turkey", 38.9637, 35.2433, {"eic": "10YTR-TEIAS----W", "aliases": ("BZN|TR", "TURKIYE"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("RUSSIA", "RU", "Russia", 61.5240, 105.3188, {"aliases": ("RUS",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("ICELAND", "IS", "Iceland", 64.9631, -19.0208, {"eic": None, "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("LIECHTENSTEIN", "LI", "Liechtenstein", 47.1660, 9.5554, {"coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("UNITED_STATES", "US", "United States", 39.8283, -98.5795, {"aliases": ("USA",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("CANADA", "CA", "Canada", 56.1304, -106.3468, {"aliases": ("CAN",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("MEXICO", "MX", "Mexico", 23.6345, -102.5528, {"aliases": ("MEX",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("BRAZIL", "BR", "Brazil", -14.2350, -51.9253, {"aliases": ("BRA",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("ARGENTINA", "AR", "Argentina", -38.4161, -63.6167, {"aliases": ("ARG",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("CHILE", "CL", "Chile", -35.6751, -71.5430, {"aliases": ("CHL",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("COLOMBIA", "CO", "Colombia", 4.5709, -74.2973, {"aliases": ("COL",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("JAPAN", "JP", "Japan", 36.2048, 138.2529, {"aliases": ("JPN",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("CHINA", "CN", "China", 35.8617, 104.1954, {"aliases": ("CHN",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("INDIA", "IN", "India", 20.5937, 78.9629, {"aliases": ("IND",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("SOUTH_KOREA", "KR", "South Korea", 35.9078, 127.7669, {"aliases": ("KOR", "KOREA"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("AUSTRALIA", "AU", "Australia", -25.2744, 133.7751, {"aliases": ("AUS",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("NEW_ZEALAND", "NZ", "New Zealand", -40.9006, 174.8860, {"aliases": ("NZL",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("SINGAPORE", "SG", "Singapore", 1.3521, 103.8198, {"aliases": ("SGP",), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("SAUDI_ARABIA", "SA", "Saudi Arabia", 23.8859, 45.0792, {"aliases": ("SAU", "KSA"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("UAE", "AE", "United Arab Emirates", 23.4241, 53.8478, {"aliases": ("AED", "EMIRATES"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
    ("SOUTH_AFRICA", "ZA", "South Africa", -30.5595, 22.9375, {"aliases": ("ZAF", "RSA"), "coord_source": _SRC_GOOGLE_COUNTRIES_CSV}),
]


def load_countries() -> None:
    _validate_unique_attrs(COUNTRIES, "COUNTRIES")
    for attr, iso, name, lat, lon, kwargs in COUNTRIES:
        setattr(GeoZone, attr, _country(iso, name, lat, lon, **kwargs))