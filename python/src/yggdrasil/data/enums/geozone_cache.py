from __future__ import annotations

from typing import Optional

from .geozone import GeoZone, GeoZoneType


__all__ = [
    "load_geozones",
]


_COUNTRY_DEFAULTS: dict[str, tuple[Optional[str], Optional[str]]] = {
    # iso -> (tz, ccy)

    # Europe
    "FR": ("Europe/Paris", "EUR"),
    "CH": ("Europe/Zurich", "CHF"),
    "GB": ("Europe/London", "GBP"),
    "AT": ("Europe/Vienna", "EUR"),
    "BE": ("Europe/Brussels", "EUR"),
    "BG": ("Europe/Sofia", "BGN"),
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
) -> GeoZone:
    default_tz, default_ccy = _country_defaults(iso)
    return GeoZone.put(
        GeoZone.from_coordinates(
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
        )
    )


def _city(
    iso: str,
    name: str,
    lat: float,
    lon: float,
    country_iso: str,
    country_name: str,
    tz: Optional[str] = None,
    ccy: Optional[str] = None,
    *,
    aliases: tuple[str, ...] = (),
    eic: Optional[str] = None,
    srid: int = 4326,
) -> GeoZone:
    default_tz, default_ccy = _country_defaults(country_iso)
    return GeoZone.put(
        GeoZone.from_coordinates(
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
) -> GeoZone:
    default_tz, default_ccy = _country_defaults(country_iso)
    return GeoZone.put(
        GeoZone.from_coordinates(
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
        )
    )


def load_geozones() -> None:
    GeoZone.clear_cache()

    # ------------------------------------------------------------------
    # World / continents
    # ------------------------------------------------------------------
    GeoZone.WORLD = GeoZone.put(
        GeoZone.from_coordinates(
            gtype=GeoZoneType.WORLD,
            lat=0.0, lon=0.0,
            key="WORLD", aliases=("EARTH",),
            name="World", tz="UTC", ccy=None,
        )
    )

    GeoZone.EUROPE = GeoZone.put(
        GeoZone.from_coordinates(
            gtype=GeoZoneType.CONTINENT,
            lat=54.5260, lon=15.2551,
            key="EU", aliases=("EUROPE",),
            name="Europe", tz="Europe/Brussels", ccy="EUR",
        )
    )
    GeoZone.ASIA = GeoZone.put(
        GeoZone.from_coordinates(
            gtype=GeoZoneType.CONTINENT,
            lat=34.0479, lon=100.6197,
            key="AS", aliases=("ASIA",),
            name="Asia", tz=None, ccy=None,
        )
    )
    GeoZone.NORTH_AMERICA = GeoZone.put(
        GeoZone.from_coordinates(
            gtype=GeoZoneType.CONTINENT,
            lat=54.5260, lon=-105.2551,
            key="NA", aliases=("NORTH_AMERICA", "NORTHAMERICA"),
            name="North America", tz=None, ccy=None,
        )
    )
    GeoZone.SOUTH_AMERICA = GeoZone.put(
        GeoZone.from_coordinates(
            gtype=GeoZoneType.CONTINENT,
            lat=-8.7832, lon=-55.4915,
            key="SA", aliases=("SOUTH_AMERICA", "SOUTHAMERICA"),
            name="South America", tz=None, ccy=None,
        )
    )
    GeoZone.AFRICA = GeoZone.put(
        GeoZone.from_coordinates(
            gtype=GeoZoneType.CONTINENT,
            lat=-8.7832, lon=34.5085,
            key="AF", aliases=("AFRICA",),
            name="Africa", tz=None, ccy=None,
        )
    )
    GeoZone.OCEANIA = GeoZone.put(
        GeoZone.from_coordinates(
            gtype=GeoZoneType.CONTINENT,
            lat=-22.7359, lon=140.0188,
            key="OC", aliases=("OCEANIA", "AUSTRALASIA"),
            name="Oceania", tz=None, ccy=None,
        )
    )

    # ------------------------------------------------------------------
    # European countries (ENTSO-E + neighbours)
    # ------------------------------------------------------------------
    GeoZone.FRANCE = _country(
        "FR", "France", 46.2276, 2.2137,
        eic="10YFR-RTE------C", aliases=("BZN|FR", "FRANCE_FR"),
    )
    GeoZone.SWITZERLAND = _country(
        "CH", "Switzerland", 46.8182, 8.2275,
        eic="10YCH-SWISSGRIDZ", aliases=("BZN|CH", "SWITZERLAND_CH"),
    )
    GeoZone.UNITED_KINGDOM = _country(
        "GB", "United Kingdom", 55.3781, -3.4360,
        eic="10Y1001A1001A92E", aliases=("UK", "GBR", "BZN|GB", "UNITED_KINGDOM_GB"),
    )
    GeoZone.AUSTRIA = _country(
        "AT", "Austria", 47.5162, 14.5501,
        eic="10YAT-APG------L", aliases=("BZN|AT",),
    )
    GeoZone.BELGIUM = _country(
        "BE", "Belgium", 50.5039, 4.4699,
        eic="10YBE----------2", aliases=("BZN|BE",),
    )
    GeoZone.BULGARIA = _country(
        "BG", "Bulgaria", 42.7339, 25.4858,
        eic="10YCA-BULGARIA-R", aliases=("BZN|BG",),
    )
    GeoZone.CROATIA = _country(
        "HR", "Croatia", 45.1000, 15.2000,
        eic="10YHR-HEP------M", aliases=("BZN|HR",),
    )
    GeoZone.CZECHIA = _country(
        "CZ", "Czechia", 49.8175, 15.4730,
        eic="10YCZ-CEPS-----N", aliases=("CZ", "CZECH_REPUBLIC", "BZN|CZ"),
    )
    GeoZone.DENMARK = _country(
        "DK", "Denmark", 56.2639, 9.5018,
        aliases=("BZN|DK",),
    )
    GeoZone.ESTONIA = _country(
        "EE", "Estonia", 58.5953, 25.0136,
        eic="10Y1001A1001A39I", aliases=("BZN|EE",),
    )
    GeoZone.FINLAND = _country(
        "FI", "Finland", 61.9241, 25.7482,
        eic="10YFI-1--------U", aliases=("BZN|FI",),
    )
    GeoZone.GERMANY = _country(
        "DE", "Germany", 51.1657, 10.4515,
        eic="10Y1001A1001A83F",
    )
    GeoZone.GREECE = _country(
        "GR", "Greece", 39.0742, 21.8243,
        eic="10YGR-HTSO-----Y", aliases=("BZN|GR",),
    )
    GeoZone.HUNGARY = _country(
        "HU", "Hungary", 47.1625, 19.5033,
        eic="10YHU-MAVIR----U", aliases=("BZN|HU",),
    )
    GeoZone.ITALY = _country(
        "IT", "Italy", 41.8719, 12.5674,
        eic="10YIT-GRTN-----B", aliases=("BZN|IT",),
    )
    GeoZone.LATVIA = _country(
        "LV", "Latvia", 56.8796, 24.6032,
        eic="10YLV-1001A00074", aliases=("BZN|LV",),
    )
    GeoZone.LITHUANIA = _country(
        "LT", "Lithuania", 55.1694, 23.8813,
        eic="10YLT-1001A0008Q", aliases=("BZN|LT",),
    )
    GeoZone.LUXEMBOURG = _country(
        "LU", "Luxembourg", 49.8153, 6.1296,
        eic="10YLU-CEGEDEL-NQ", aliases=("BZN|LU", "LUX"),
    )
    GeoZone.MONTENEGRO = _country(
        "ME", "Montenegro", 42.7087, 19.3744,
        eic="10YCS-CG-TSO---S", aliases=("BZN|ME",),
    )
    GeoZone.NETHERLANDS = _country(
        "NL", "Netherlands", 52.1326, 5.2913,
        eic="10YNL----------L", aliases=("BZN|NL", "HOLLAND"),
    )
    GeoZone.NORWAY = _country(
        "NO", "Norway", 60.4720, 8.4689,
        eic="10YNO-0--------C",
    )
    GeoZone.POLAND = _country(
        "PL", "Poland", 51.9194, 19.1451,
        eic="10YPL-AREA-----S", aliases=("BZN|PL",),
    )
    GeoZone.PORTUGAL = _country(
        "PT", "Portugal", 39.3999, -8.2245,
        eic="10YPT-REN------W", aliases=("BZN|PT",),
    )
    GeoZone.ROMANIA = _country(
        "RO", "Romania", 45.9432, 24.9668,
        eic="10YRO-TEL------P", aliases=("BZN|RO",),
    )
    GeoZone.SERBIA = _country(
        "RS", "Serbia", 44.0165, 21.0059,
        eic="10YCS-SERBIATSOV", aliases=("BZN|RS",),
    )
    GeoZone.SLOVAKIA = _country(
        "SK", "Slovakia", 48.6690, 19.6990,
        eic="10YSK-SEPS-----K", aliases=("BZN|SK",),
    )
    GeoZone.SLOVENIA = _country(
        "SI", "Slovenia", 46.1512, 14.9955,
        eic="10YSI-ELES-----O", aliases=("BZN|SI",),
    )
    GeoZone.SPAIN = _country(
        "ES", "Spain", 40.4637, -3.7492,
        eic="10YES-REE------0", aliases=("BZN|ES",),
    )
    GeoZone.SWEDEN = _country(
        "SE", "Sweden", 60.1282, 18.6435,
        eic="10YSE-1--------K",
    )
    GeoZone.UKRAINE = _country(
        "UA", "Ukraine", 48.3794, 31.1656,
        eic="10Y1001C--00003F", aliases=("BZN|UA",),
    )
    GeoZone.KOSOVO = _country(
        "XK", "Kosovo", 42.6026, 20.9030,
        eic="10Y1001C--00100H", aliases=("BZN|XK",),
    )
    GeoZone.MOLDOVA = _country(
        "MD", "Moldova", 47.4116, 28.3699,
        eic="10Y1001A1001A990", aliases=("BZN|MD",),
    )
    GeoZone.MALTA = _country(
        "MT", "Malta", 35.9375, 14.3754,
        eic="10Y1001A1001A93C", aliases=("BZN|MT",),
    )
    GeoZone.IRELAND = _country(
        "IE", "Ireland", 53.4, -7.7,
        eic="10Y1001A1001A59C", aliases=("IE_SEM", "SEM", "BZN|IE_SEM"),
    )
    GeoZone.CYPRUS = _country(
        "CY", "Cyprus", 35.1264, 33.4299,
        eic="10YCY-1001A0003J", aliases=("BZN|CY",),
    )
    GeoZone.ALBANIA = _country(
        "AL", "Albania", 41.1533, 20.1683,
        eic="10YAL-KESH-----5", aliases=("BZN|AL",),
    )
    GeoZone.BOSNIA = _country(
        "BA", "Bosnia and Herzegovina", 43.9159, 17.6791,
        eic="10YBA-JPCC-----D", aliases=("BZN|BA", "BOSNIA_HERZEGOVINA"),
    )
    GeoZone.NORTH_MACEDONIA = _country(
        "MK", "North Macedonia", 41.6086, 21.7453,
        eic="10YMK-MEPSO----8", aliases=("BZN|MK", "MACEDONIA"),
    )
    GeoZone.BELARUS = _country(
        "BY", "Belarus", 53.7098, 27.9534,
        aliases=("BYELORUSSIA",),
    )
    GeoZone.GEORGIA = _country(
        "GE", "Georgia", 42.3154, 43.3569,
        aliases=("BZN|GE",),
    )
    GeoZone.ARMENIA = _country(
        "AM", "Armenia", 40.0691, 45.0382,
        aliases=("BZN|AM",),
    )
    GeoZone.AZERBAIJAN = _country(
        "AZ", "Azerbaijan", 40.1431, 47.5769,
        aliases=("BZN|AZ",),
    )
    GeoZone.TURKEY = _country(
        "TR", "Turkey", 38.9637, 35.2433,
        eic="10YTR-TEIAS----W", aliases=("BZN|TR", "TURKIYE"),
    )
    GeoZone.RUSSIA = _country(
        "RU", "Russia", 61.5240, 105.3188,
        aliases=("RUS",),
    )
    GeoZone.ICELAND = _country(
        "IS", "Iceland", 64.9631, -19.0208,
        eic="10Y1001A1001A39I",
    )
    GeoZone.LIECHTENSTEIN = _country(
        "LI", "Liechtenstein", 47.1660, 9.5554,
    )

    # ------------------------------------------------------------------
    # Americas – countries
    # ------------------------------------------------------------------
    GeoZone.UNITED_STATES = _country("US", "United States", 39.8283, -98.5795, aliases=("USA",))
    GeoZone.CANADA = _country("CA", "Canada", 56.1304, -106.3468, aliases=("CAN",))
    GeoZone.MEXICO = _country("MX", "Mexico", 23.6345, -102.5528, aliases=("MEX",))
    GeoZone.BRAZIL = _country("BR", "Brazil", -14.2350, -51.9253, aliases=("BRA",))
    GeoZone.ARGENTINA = _country("AR", "Argentina", -38.4161, -63.6167, aliases=("ARG",))
    GeoZone.CHILE = _country("CL", "Chile", -35.6751, -71.5430, aliases=("CHL",))
    GeoZone.COLOMBIA = _country("CO", "Colombia", 4.5709, -74.2973, aliases=("COL",))

    # ------------------------------------------------------------------
    # Asia-Pacific – countries
    # ------------------------------------------------------------------
    GeoZone.JAPAN = _country("JP", "Japan", 36.2048, 138.2529, aliases=("JPN",))
    GeoZone.CHINA = _country("CN", "China", 35.8617, 104.1954, aliases=("CHN",))
    GeoZone.INDIA = _country("IN", "India", 20.5937, 78.9629, aliases=("IND",))
    GeoZone.SOUTH_KOREA = _country("KR", "South Korea", 35.9078, 127.7669, aliases=("KOR", "KOREA"))
    GeoZone.AUSTRALIA = _country("AU", "Australia", -25.2744, 133.7751, aliases=("AUS",))
    GeoZone.NEW_ZEALAND = _country("NZ", "New Zealand", -40.9006, 174.8860, aliases=("NZL",))
    GeoZone.SINGAPORE = _country("SG", "Singapore", 1.3521, 103.8198, aliases=("SGP",))

    # ------------------------------------------------------------------
    # Middle-East & Africa – countries
    # ------------------------------------------------------------------
    GeoZone.SAUDI_ARABIA = _country("SA", "Saudi Arabia", 23.8859, 45.0792, aliases=("SAU", "KSA"))
    GeoZone.UAE = _country("AE", "United Arab Emirates", 23.4241, 53.8478, aliases=("AED", "EMIRATES"))
    GeoZone.SOUTH_AFRICA = _country("ZA", "South Africa", -30.5595, 22.9375, aliases=("ZAF", "RSA"))

    # ------------------------------------------------------------------
    # European cities
    # ------------------------------------------------------------------
    GeoZone.PARIS = _city("PAR", "Paris", 48.8566, 2.3522, "FR", "France")
    GeoZone.ZURICH = _city("ZRH", "Zurich", 47.3769, 8.5417, "CH", "Switzerland")
    GeoZone.GENEVA = _city("GVA", "Geneva", 46.2044, 6.1432, "CH", "Switzerland")
    GeoZone.LONDON = _city("LON", "London", 51.5074, -0.1278, "GB", "United Kingdom")
    GeoZone.BERLIN = _city("BER", "Berlin", 52.5200, 13.4050, "DE", "Germany")
    GeoZone.MADRID = _city("MAD", "Madrid", 40.4168, -3.7038, "ES", "Spain")
    GeoZone.ROME = _city("ROM", "Rome", 41.9028, 12.4964, "IT", "Italy")
    GeoZone.AMSTERDAM = _city("AMS", "Amsterdam", 52.3676, 4.9041, "NL", "Netherlands")
    GeoZone.BRUSSELS = _city("BRU", "Brussels", 50.8503, 4.3517, "BE", "Belgium")
    GeoZone.VIENNA = _city("VIE", "Vienna", 48.2082, 16.3738, "AT", "Austria")
    GeoZone.WARSAW = _city("WAW", "Warsaw", 52.2297, 21.0122, "PL", "Poland")
    GeoZone.STOCKHOLM = _city("STO", "Stockholm", 59.3293, 18.0686, "SE", "Sweden")
    GeoZone.OSLO = _city("OSL", "Oslo", 59.9139, 10.7522, "NO", "Norway")
    GeoZone.COPENHAGEN = _city("CPH", "Copenhagen", 55.6761, 12.5683, "DK", "Denmark")
    GeoZone.HELSINKI = _city("HEL", "Helsinki", 60.1699, 24.9384, "FI", "Finland")
    GeoZone.LISBON = _city("LIS", "Lisbon", 38.7169, -9.1395, "PT", "Portugal")
    GeoZone.ATHENS = _city("ATH", "Athens", 37.9838, 23.7275, "GR", "Greece")
    GeoZone.BUDAPEST = _city("BUD", "Budapest", 47.4979, 19.0402, "HU", "Hungary")
    GeoZone.PRAGUE = _city("PRG", "Prague", 50.0755, 14.4378, "CZ", "Czechia")
    GeoZone.BUCHAREST = _city("BUH", "Bucharest", 44.4268, 26.1025, "RO", "Romania")
    GeoZone.BRATISLAVA = _city("BTS", "Bratislava", 48.1486, 17.1077, "SK", "Slovakia")
    GeoZone.LJUBLJANA = _city("LJU", "Ljubljana", 46.0569, 14.5058, "SI", "Slovenia")
    GeoZone.ZAGREB = _city("ZAG", "Zagreb", 45.8150, 15.9819, "HR", "Croatia")
    GeoZone.BELGRADE = _city("BEG", "Belgrade", 44.8176, 20.4633, "RS", "Serbia")
    GeoZone.RIGA = _city("RIX", "Riga", 56.9460, 24.1059, "LV", "Latvia")
    GeoZone.VILNIUS = _city("VNO", "Vilnius", 54.6872, 25.2797, "LT", "Lithuania")
    GeoZone.TALLINN = _city("TLL", "Tallinn", 59.4370, 24.7536, "EE", "Estonia")
    GeoZone.DUBLIN = _city("DUB", "Dublin", 53.3498, -6.2603, "IE", "Ireland")

    # ------------------------------------------------------------------
    # Americas cities
    # ------------------------------------------------------------------
    GeoZone.NEW_YORK = _city("NYC", "New York", 40.7128, -74.0060, "US", "United States")
    GeoZone.LOS_ANGELES = _city(
        "LAX", "Los Angeles", 34.0522, -118.2437, "US", "United States",
        tz="America/Los_Angeles",
    )
    GeoZone.CHICAGO = _city(
        "CHI", "Chicago", 41.8781, -87.6298, "US", "United States",
        tz="America/Chicago",
    )
    GeoZone.HOUSTON = _city(
        "HOU", "Houston", 29.7604, -95.3698, "US", "United States",
        tz="America/Chicago",
    )
    GeoZone.TORONTO = _city("YYZ", "Toronto", 43.6532, -79.3832, "CA", "Canada")
    GeoZone.SAO_PAULO = _city("GRU", "Sao Paulo", -23.5505, -46.6333, "BR", "Brazil")
    GeoZone.BUENOS_AIRES = _city("EZE", "Buenos Aires", -34.6037, -58.3816, "AR", "Argentina")

    # ------------------------------------------------------------------
    # Asia-Pacific cities
    # ------------------------------------------------------------------
    GeoZone.TOKYO = _city("TYO", "Tokyo", 35.6762, 139.6503, "JP", "Japan")
    GeoZone.BEIJING = _city("PEK", "Beijing", 39.9042, 116.4074, "CN", "China")
    GeoZone.SHANGHAI = _city("PVG", "Shanghai", 31.2304, 121.4737, "CN", "China")
    GeoZone.SEOUL = _city("ICN", "Seoul", 37.5665, 126.9780, "KR", "South Korea")
    GeoZone.MUMBAI = _city("BOM", "Mumbai", 19.0760, 72.8777, "IN", "India")
    GeoZone.DELHI = _city("DEL", "Delhi", 28.6139, 77.2090, "IN", "India")
    GeoZone.SYDNEY = _city("SYD", "Sydney", -33.8688, 151.2093, "AU", "Australia")
    GeoZone.SINGAPORE_CITY = _city("SIN", "Singapore", 1.3521, 103.8198, "SG", "Singapore")
    GeoZone.DUBAI = _city("DXB", "Dubai", 25.2048, 55.2708, "AE", "United Arab Emirates")

    # ------------------------------------------------------------------
    # European bidding zones (ENTSO-E)
    # ------------------------------------------------------------------

    GeoZone.DE_LU = _zone(
        "DE_LU", "Germany-Luxembourg", 50.9, 8.7, "Europe/Berlin", "EUR",
        country_iso="DE", country_name="Germany-Luxembourg",
        eic="10Y1001A1001A82H",
        aliases=("DE-LU", "BZN|DE-LU", "GERMANY_LUXEMBOURG"),
    )
    GeoZone.DE_AT_LU = _zone(
        "DE_AT_LU", "Germany-Austria-Luxembourg", 48.5, 11.0, "Europe/Berlin", "EUR",
        country_iso="DE", country_name="Germany-Austria-Luxembourg",
        eic="10Y1001A1001A63L",
        aliases=("DE-AT-LU", "BZN|DE-AT-LU"),
    )

    GeoZone.DK1 = _zone(
        "DK1", "Denmark DK1", 56.2639, 9.5018,
        country_iso="DK", country_name="Denmark",
        eic="10YDK-1--------W", aliases=("DK-1", "BZN|DK1", "DENMARK_DK1"),
    )
    GeoZone.DK2 = _zone(
        "DK2", "Denmark DK2", 55.6761, 12.5683,
        country_iso="DK", country_name="Denmark",
        eic="10YDK-2--------M", aliases=("DK-2", "BZN|DK2", "DENMARK_DK2"),
    )

    GeoZone.NO1 = _zone(
        "NO1", "Norway NO1", 59.9139, 10.7522,
        country_iso="NO", country_name="Norway",
        eic="10YNO-1--------2", aliases=("BZN|NO1", "NORWAY_NO1"),
    )
    GeoZone.NO2 = _zone(
        "NO2", "Norway NO2", 58.1467, 7.9956,
        country_iso="NO", country_name="Norway",
        eic="10YNO-2--------T", aliases=("BZN|NO2", "NORWAY_NO2"),
    )
    GeoZone.NO3 = _zone(
        "NO3", "Norway NO3", 63.4305, 10.3951,
        country_iso="NO", country_name="Norway",
        eic="10YNO-3--------J", aliases=("BZN|NO3", "NORWAY_NO3"),
    )
    GeoZone.NO4 = _zone(
        "NO4", "Norway NO4", 69.6492, 18.9553,
        country_iso="NO", country_name="Norway",
        eic="10YNO-4--------9", aliases=("BZN|NO4", "NORWAY_NO4"),
    )
    GeoZone.NO5 = _zone(
        "NO5", "Norway NO5", 60.3930, 5.3242,
        country_iso="NO", country_name="Norway",
        eic="10Y1001A1001A48H", aliases=("BZN|NO5", "NORWAY_NO5"),
    )
    GeoZone.NO1A = _zone(
        "NO1A", "Norway NO1A", 59.9, 10.7,
        country_iso="NO", country_name="Norway",
        eic="10Y1001A1001A64J", aliases=("BZN|NO1A",),
    )
    GeoZone.NO2A = _zone(
        "NO2A", "Norway NO2A", 58.1, 8.0,
        country_iso="NO", country_name="Norway",
        eic="10Y1001C--001219", aliases=("BZN|NO2A",),
    )

    GeoZone.SE1 = _zone(
        "SE1", "Sweden SE1", 67.8558, 20.2253,
        country_iso="SE", country_name="Sweden",
        eic="10Y1001A1001A44P", aliases=("BZN|SE1", "SWEDEN_SE1"),
    )
    GeoZone.SE2 = _zone(
        "SE2", "Sweden SE2", 63.8258, 20.2630,
        country_iso="SE", country_name="Sweden",
        eic="10Y1001A1001A45N", aliases=("BZN|SE2", "SWEDEN_SE2"),
    )
    GeoZone.SE3 = _zone(
        "SE3", "Sweden SE3", 59.3293, 18.0686,
        country_iso="SE", country_name="Sweden",
        eic="10Y1001A1001A46L", aliases=("BZN|SE3", "SWEDEN_SE3"),
    )
    GeoZone.SE4 = _zone(
        "SE4", "Sweden SE4", 55.6050, 13.0038,
        country_iso="SE", country_name="Sweden",
        eic="10Y1001A1001A47J", aliases=("BZN|SE4", "SWEDEN_SE4"),
    )

    GeoZone.FI_BZ = _zone(
        "FI_BZ", "Finland", 61.9241, 25.7482,
        country_iso="FI", country_name="Finland",
        eic="10YFI-1--------U", aliases=("BZN|FI",),
    )

    GeoZone.EE_BZ = _zone(
        "EE_BZ", "Estonia", 58.5953, 25.0136,
        country_iso="EE", country_name="Estonia",
        eic="10Y1001A1001A39I", aliases=("BZN|EE",),
    )
    GeoZone.LV_BZ = _zone(
        "LV_BZ", "Latvia", 56.8796, 24.6032,
        country_iso="LV", country_name="Latvia",
        eic="10YLV-1001A00074", aliases=("BZN|LV",),
    )
    GeoZone.LT_BZ = _zone(
        "LT_BZ", "Lithuania", 55.1694, 23.8813,
        country_iso="LT", country_name="Lithuania",
        eic="10YLT-1001A0008Q", aliases=("BZN|LT",),
    )

    GeoZone.PL_BZ = _zone(
        "PL_BZ", "Poland", 51.9194, 19.1451,
        country_iso="PL", country_name="Poland",
        eic="10YPL-AREA-----S", aliases=("BZN|PL",),
    )

    GeoZone.PT_BZ = _zone(
        "PT_BZ", "Portugal", 39.3999, -8.2245,
        country_iso="PT", country_name="Portugal",
        eic="10YPT-REN------W", aliases=("BZN|PT",),
    )
    GeoZone.ES_BZ = _zone(
        "ES_BZ", "Spain", 40.4637, -3.7492,
        country_iso="ES", country_name="Spain",
        eic="10YES-REE------0", aliases=("BZN|ES",),
    )

    GeoZone.RO_BZ = _zone(
        "RO_BZ", "Romania", 45.9432, 24.9668,
        country_iso="RO", country_name="Romania",
        eic="10YRO-TEL------P", aliases=("BZN|RO",),
    )
    GeoZone.BG_BZ = _zone(
        "BG_BZ", "Bulgaria", 42.7339, 25.4858,
        country_iso="BG", country_name="Bulgaria",
        eic="10YCA-BULGARIA-R", aliases=("BZN|BG",),
    )
    GeoZone.HU_BZ = _zone(
        "HU_BZ", "Hungary", 47.1625, 19.5033,
        country_iso="HU", country_name="Hungary",
        eic="10YHU-MAVIR----U", aliases=("BZN|HU",),
    )
    GeoZone.CZ_BZ = _zone(
        "CZ_BZ", "Czechia", 49.8175, 15.4730,
        country_iso="CZ", country_name="Czechia",
        eic="10YCZ-CEPS-----N", aliases=("BZN|CZ",),
    )
    GeoZone.SK_BZ = _zone(
        "SK_BZ", "Slovakia", 48.6690, 19.6990,
        country_iso="SK", country_name="Slovakia",
        eic="10YSK-SEPS-----K", aliases=("BZN|SK",),
    )
    GeoZone.SI_BZ = _zone(
        "SI_BZ", "Slovenia", 46.1512, 14.9955,
        country_iso="SI", country_name="Slovenia",
        eic="10YSI-ELES-----O", aliases=("BZN|SI",),
    )
    GeoZone.HR_BZ = _zone(
        "HR_BZ", "Croatia", 45.1000, 15.2000,
        country_iso="HR", country_name="Croatia",
        eic="10YHR-HEP------M", aliases=("BZN|HR",),
    )
    GeoZone.AT_BZ = _zone(
        "AT_BZ", "Austria", 47.5162, 14.5501,
        country_iso="AT", country_name="Austria",
        eic="10YAT-APG------L", aliases=("BZN|AT",),
    )
    GeoZone.BE_BZ = _zone(
        "BE_BZ", "Belgium", 50.5039, 4.4699,
        country_iso="BE", country_name="Belgium",
        eic="10YBE----------2", aliases=("BZN|BE",),
    )
    GeoZone.NL_BZ = _zone(
        "NL_BZ", "Netherlands", 52.1326, 5.2913,
        country_iso="NL", country_name="Netherlands",
        eic="10YNL----------L", aliases=("BZN|NL",),
    )
    GeoZone.RS_BZ = _zone(
        "RS_BZ", "Serbia", 44.0165, 21.0059,
        country_iso="RS", country_name="Serbia",
        eic="10YCS-SERBIATSOV", aliases=("BZN|RS",),
    )
    GeoZone.ME_BZ = _zone(
        "ME_BZ", "Montenegro", 42.7087, 19.3744,
        country_iso="ME", country_name="Montenegro",
        eic="10YCS-CG-TSO---S", aliases=("BZN|ME",),
    )
    GeoZone.AL_BZ = _zone(
        "AL_BZ", "Albania", 41.1533, 20.1683,
        country_iso="AL", country_name="Albania",
        eic="10YAL-KESH-----5", aliases=("BZN|AL",),
    )
    GeoZone.BA_BZ = _zone(
        "BA_BZ", "Bosnia and Herzegovina", 43.9159, 17.6791,
        country_iso="BA", country_name="Bosnia and Herzegovina",
        eic="10YBA-JPCC-----D", aliases=("BZN|BA",),
    )
    GeoZone.MK_BZ = _zone(
        "MK_BZ", "North Macedonia", 41.6086, 21.7453,
        country_iso="MK", country_name="North Macedonia",
        eic="10YMK-MEPSO----8", aliases=("BZN|MK",),
    )
    GeoZone.GR_BZ = _zone(
        "GR_BZ", "Greece", 39.0742, 21.8243,
        country_iso="GR", country_name="Greece",
        eic="10YGR-HTSO-----Y", aliases=("BZN|GR",),
    )
    GeoZone.TR_BZ = _zone(
        "TR_BZ", "Turkey", 38.9637, 35.2433,
        country_iso="TR", country_name="Turkey",
        eic="10YTR-TEIAS----W", aliases=("BZN|TR",),
    )
    GeoZone.CY_BZ = _zone(
        "CY_BZ", "Cyprus", 35.1264, 33.4299,
        country_iso="CY", country_name="Cyprus",
        eic="10YCY-1001A0003J", aliases=("BZN|CY",),
    )
    GeoZone.XK_BZ = _zone(
        "XK_BZ", "Kosovo", 42.6026, 20.9030,
        country_iso="XK", country_name="Kosovo",
        eic="10Y1001C--00100H", aliases=("BZN|XK",),
    )
    GeoZone.MD_BZ = _zone(
        "MD_BZ", "Moldova", 47.4116, 28.3699,
        country_iso="MD", country_name="Moldova",
        eic="10Y1001A1001A990", aliases=("BZN|MD",),
    )
    GeoZone.MT_BZ = _zone(
        "MT_BZ", "Malta", 35.9375, 14.3754,
        country_iso="MT", country_name="Malta",
        eic="10Y1001A1001A93C", aliases=("BZN|MT",),
    )
    GeoZone.GB_BZ = _zone(
        "GB_BZ", "Great Britain", 55.3781, -3.4360,
        country_iso="GB", country_name="United Kingdom",
        eic="10Y1001A1001A92E", aliases=("BZN|GB",),
    )
    GeoZone.IE_SEM = _zone(
        "IE_SEM", "Ireland SEM", 53.4, -7.7,
        country_iso="IE", country_name="Ireland",
        eic="10Y1001A1001A59C", aliases=("SEM", "BZN|IE_SEM"),
    )
    GeoZone.UA_IPS = _zone(
        "UA_IPS", "Ukraine UA-IPS", 48.7, 31.2,
        country_iso="UA", country_name="Ukraine",
        eic="10Y1001C--000182", aliases=("UA-IPS", "UKRAINE_IPS", "BZN|UA-IPS"),
    )
    GeoZone.UA_DOBAS = _zone(
        "UA_DOBAS", "Ukraine Donbas", 48.0, 37.8,
        country_iso="UA", country_name="Ukraine",
        eic="10Y1001C--000244", aliases=("UA-DOBAS", "BZN|UA-DOBAS"),
    )
    GeoZone.UA_BEI = _zone(
        "UA_BEI", "Ukraine BEI", 50.8, 29.4,
        country_iso="UA", country_name="Ukraine",
        eic="10Y1001C--00025I", aliases=("UA-BEI", "BZN|UA-BEI"),
    )

    GeoZone.RU_KGD = _zone(
        "RU_KGD", "Russia Kaliningrad", 54.71, 20.51, "Europe/Kaliningrad", "RUB",
        country_iso="RU", country_name="Russia",
        eic="10Y1001A1001A50U", aliases=("RUSSIA_KALININGRAD",),
    )
    GeoZone.GB_IFA = _zone(
        "GB_IFA", "Great Britain IFA", 50.95, 1.85,
        country_iso="GB", country_name="United Kingdom",
        eic="10Y1001C--00098F", aliases=("GREAT_BRITAIN_IFA",),
    )
    GeoZone.GB_IFA2 = _zone(
        "GB_IFA2", "Great Britain IFA2", 50.95, 1.85,
        country_iso="GB", country_name="United Kingdom",
        eic="17Y0000009369493", aliases=("GREAT_BRITAIN_IFA2",),
    )
    GeoZone.GB_ELECLINK = _zone(
        "GB_ELECLINK", "Great Britain ElecLink", 51.1, 1.3,
        country_iso="GB", country_name="United Kingdom",
        eic="11Y0-0000-0265-K", aliases=("GB-ELECLINK",),
    )
    GeoZone.GB_NEMO = _zone(
        "GB_NEMO", "Great Britain Nemo Link", 51.1, 1.6,
        country_iso="GB", country_name="United Kingdom",
        eic="11Y0-0000-0265-L", aliases=("GB-NEMO",),
    )

    # Italian zones keep explicit overrides removed where same as country default
    GeoZone.IT_NORTH = _zone(
        "IT_NORTH", "Italy North", 45.3, 10.5,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A73I",
        aliases=("IT-NORTH", "ITALY_NORTH", "BZN|IT-NORTH"),
    )
    GeoZone.IT_SOUTH = _zone(
        "IT_SOUTH", "Italy South", 40.8, 16.5,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A788",
        aliases=("IT-SOUTH", "ITALY_SOUTH", "BZN|IT-SOUTH"),
    )
    GeoZone.IT_CENTRE_NORTH = _zone(
        "IT_CENTRE_NORTH", "Italy Centre-North", 43.5, 11.5,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A70O",
        aliases=("IT-CENTRE-NORTH", "IT-CENTER-NORTH", "ITALY_CENTRE_NORTH", "ITALY_CENTER_NORTH"),
    )
    GeoZone.IT_CENTRE_SOUTH = _zone(
        "IT_CENTRE_SOUTH", "Italy Centre-South", 41.8, 14.0,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A71M",
        aliases=("IT-CENTRE-SOUTH", "IT-CENTER-SOUTH", "ITALY_CENTRE_SOUTH", "ITALY_CENTER_SOUTH"),
    )
    GeoZone.IT_SARDINIA = _zone(
        "IT_SARDINIA", "Italy Sardinia", 40.1209, 9.0129,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A74G", aliases=("IT-SARDINIA", "ITALY_SARDINIA"),
    )
    GeoZone.IT_SICILY = _zone(
        "IT_SICILY", "Italy Sicily", 37.5999, 14.0154,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A75E", aliases=("IT-SICILY", "ITALY_SICILY"),
    )
    GeoZone.IT_CALABRIA = _zone(
        "IT_CALABRIA", "Italy Calabria", 38.9, 16.6,
        country_iso="IT", country_name="Italy",
        eic="10Y1001C--00096J", aliases=("IT-CALABRIA", "ITALY_CALABRIA"),
    )
    GeoZone.IT_FOGGIA = _zone(
        "IT_FOGGIA", "Italy Foggia", 41.46, 15.54,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A72K", aliases=("IT-FOGGIA", "ITALY_FOGGIA"),
    )
    GeoZone.IT_BRINDISI = _zone(
        "IT_BRINDISI", "Italy Brindisi", 40.64, 17.94,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A699", aliases=("IT-BRINDISI", "ITALY_BRINDISI"),
    )
    GeoZone.IT_PRIOLO = _zone(
        "IT_PRIOLO", "Italy Priolo", 37.15, 15.18,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A76C", aliases=("IT-PRIOLO", "ITALY_PRIOLO"),
    )
    GeoZone.IT_ROSSANO = _zone(
        "IT_ROSSANO", "Italy Rossano", 39.58, 16.64,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A77A", aliases=("IT-ROSSANO", "ITALY_ROSSANO"),
    )
    GeoZone.IT_GR = _zone(
        "IT_GR", "Italy-Greece", 39.0, 19.0,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A66F", aliases=("IT-GR", "ITALY_GREECE"),
    )
    GeoZone.IT_NORTH_SI = _zone(
        "IT_NORTH_SI", "Italy North-Slovenia", 45.6, 13.8,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A67D", aliases=("IT-NORTH-SI", "ITALY_NORTH_SLOVENIA"),
    )
    GeoZone.IT_NORTH_CH = _zone(
        "IT_NORTH_CH", "Italy North-Switzerland", 46.2, 8.8,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A68B", aliases=("IT-NORTH-CH", "ITALY_NORTH_SWITZERLAND"),
    )
    GeoZone.IT_NORTH_AT = _zone(
        "IT_NORTH_AT", "Italy North-Austria", 46.6, 12.0,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A80L", aliases=("IT-NORTH-AT", "ITALY_NORTH_AUSTRIA"),
    )
    GeoZone.IT_NORTH_FR = _zone(
        "IT_NORTH_FR", "Italy North-France", 45.1, 6.8,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A81J", aliases=("IT-NORTH-FR", "ITALY_NORTH_FRANCE"),
    )
    GeoZone.IT_MALTA = _zone(
        "IT_MALTA", "Italy-Malta", 36.1, 14.4,
        country_iso="IT", country_name="Italy",
        eic="10Y1001A1001A877", aliases=("IT-MALTA", "ITALY_MALTA"),
    )

    # ------------------------------------------------------------------
    # US power market zones / hubs
    # ------------------------------------------------------------------
    GeoZone.PJM_AEP = _zone(
        "PJM_AEP", "PJM AEP", 39.9612, -82.9988,
        country_iso="US", country_name="United States",
        aliases=("PJM-AEP",),
    )
    GeoZone.PJM_ATSI = _zone(
        "PJM_ATSI", "PJM ATSI", 40.7, -81.0,
        country_iso="US", country_name="United States",
        aliases=("PJM-ATSI",),
    )
    GeoZone.PJM_COMED = _zone(
        "PJM_COMED", "PJM ComEd", 41.8, -88.0, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("PJM-COMED",),
    )
    GeoZone.PJM_DOM = _zone(
        "PJM_DOM", "PJM Dominion", 37.5, -77.5,
        country_iso="US", country_name="United States",
        aliases=("PJM-DOM", "PJM_DOMINION"),
    )
    GeoZone.PJM_DUQ = _zone(
        "PJM_DUQ", "PJM Duquesne", 40.4, -80.0,
        country_iso="US", country_name="United States",
        aliases=("PJM-DUQ",),
    )
    GeoZone.PJM_EKPC = _zone(
        "PJM_EKPC", "PJM EKPC", 37.8, -84.3,
        country_iso="US", country_name="United States",
        aliases=("PJM-EKPC",),
    )
    GeoZone.PJM_JCPL = _zone(
        "PJM_JCPL", "PJM JCP&L", 40.2, -74.5,
        country_iso="US", country_name="United States",
        aliases=("PJM-JCPL",),
    )
    GeoZone.PJM_METED = _zone(
        "PJM_METED", "PJM Met-Ed", 40.3, -76.0,
        country_iso="US", country_name="United States",
        aliases=("PJM-METED",),
    )
    GeoZone.PJM_PECO = _zone(
        "PJM_PECO", "PJM PECO", 39.9, -75.2,
        country_iso="US", country_name="United States",
        aliases=("PJM-PECO",),
    )
    GeoZone.PJM_PENELEC = _zone(
        "PJM_PENELEC", "PJM Penelec", 41.2, -78.0,
        country_iso="US", country_name="United States",
        aliases=("PJM-PENELEC",),
    )
    GeoZone.PJM_PPL = _zone(
        "PJM_PPL", "PJM PPL", 40.6, -75.5,
        country_iso="US", country_name="United States",
        aliases=("PJM-PPL",),
    )
    GeoZone.PJM_PSEG = _zone(
        "PJM_PSEG", "PJM PSEG", 40.7, -74.2,
        country_iso="US", country_name="United States",
        aliases=("PJM-PSEG",),
    )
    GeoZone.PJM_RECO = _zone(
        "PJM_RECO", "PJM RECO", 41.1, -74.6,
        country_iso="US", country_name="United States",
        aliases=("PJM-RECO",),
    )
    GeoZone.PJM_WEST = _zone(
        "PJM_WEST", "PJM West Hub", 40.5, -80.5,
        country_iso="US", country_name="United States",
        aliases=("PJM-WEST", "PJM_WEST_HUB"),
    )
    GeoZone.PJM_EAST = _zone(
        "PJM_EAST", "PJM East Hub", 40.0, -74.8,
        country_iso="US", country_name="United States",
        aliases=("PJM-EAST", "PJM_EAST_HUB"),
    )

    GeoZone.MISO_NORTH = _zone(
        "MISO_NORTH", "MISO North", 46.0, -93.0, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("MISO-NORTH",),
    )
    GeoZone.MISO_CENTRAL = _zone(
        "MISO_CENTRAL", "MISO Central", 40.5, -89.0, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("MISO-CENTRAL",),
    )
    GeoZone.MISO_SOUTH = _zone(
        "MISO_SOUTH", "MISO South", 33.0, -91.0, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("MISO-SOUTH",),
    )
    GeoZone.MISO_ILLINOIS = _zone(
        "MISO_ILLINOIS", "MISO Illinois Hub", 41.8, -87.6, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("MISO-ILLINOIS", "MISO_IL_HUB"),
    )
    GeoZone.MISO_INDIANA = _zone(
        "MISO_INDIANA", "MISO Indiana Hub", 39.8, -86.2,
        country_iso="US", country_name="United States",
        aliases=("MISO-INDIANA", "MISO_IN_HUB"),
    )
    GeoZone.MISO_MINNESOTA = _zone(
        "MISO_MINNESOTA", "MISO Minnesota Hub", 44.9, -93.3, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("MISO-MINNESOTA", "MISO_MN_HUB"),
    )

    GeoZone.NP15 = _zone(
        "NP15", "CAISO NP15", 38.5816, -121.4944, "America/Los_Angeles", "USD",
        country_iso="US", country_name="United States",
        aliases=("CAISO_NP15",),
    )
    GeoZone.SP15 = _zone(
        "SP15", "CAISO SP15", 34.0, -118.3, "America/Los_Angeles", "USD",
        country_iso="US", country_name="United States",
        aliases=("CAISO_SP15",),
    )
    GeoZone.ZP26 = _zone(
        "ZP26", "CAISO ZP26", 37.3, -120.5, "America/Los_Angeles", "USD",
        country_iso="US", country_name="United States",
        aliases=("CAISO_ZP26",),
    )

    GeoZone.ERCOT_NORTH = _zone(
        "ERCOT_NORTH", "ERCOT North", 32.7767, -96.7970, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("ERCOT-NORTH",),
    )
    GeoZone.ERCOT_SOUTH = _zone(
        "ERCOT_SOUTH", "ERCOT South", 29.7604, -95.3698, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("ERCOT-SOUTH",),
    )
    GeoZone.ERCOT_WEST = _zone(
        "ERCOT_WEST", "ERCOT West", 31.8, -102.0, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("ERCOT-WEST",),
    )
    GeoZone.ERCOT_HOUSTON = _zone(
        "ERCOT_HOUSTON", "ERCOT Houston", 29.7604, -95.3698, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("ERCOT-HOUSTON", "ERCOT_HB_HOUSTON"),
    )
    GeoZone.ERCOT_BUSAVG = _zone(
        "ERCOT_BUSAVG", "ERCOT Bus Average", 31.0, -99.0, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("ERCOT-BUSAVG", "ERCOT_BUS_AVG"),
    )

    GeoZone.ISONE_HUB = _zone(
        "ISONE_HUB", "ISO-NE Hub", 42.4, -71.1,
        country_iso="US", country_name="United States",
        aliases=("ISONE-HUB", "ISO_NE_HUB", "NEPOOL_HUB"),
    )
    GeoZone.ISONE_CT = _zone(
        "ISONE_CT", "ISO-NE Connecticut", 41.6, -72.7,
        country_iso="US", country_name="United States",
        aliases=("ISONE-CT", "ISO_NE_CT"),
    )
    GeoZone.ISONE_ME = _zone(
        "ISONE_ME", "ISO-NE Maine", 45.3, -69.4,
        country_iso="US", country_name="United States",
        aliases=("ISONE-ME", "ISO_NE_ME"),
    )
    GeoZone.ISONE_NH = _zone(
        "ISONE_NH", "ISO-NE New Hampshire", 43.2, -71.5,
        country_iso="US", country_name="United States",
        aliases=("ISONE-NH", "ISO_NE_NH"),
    )
    GeoZone.ISONE_RI = _zone(
        "ISONE_RI", "ISO-NE Rhode Island", 41.7, -71.5,
        country_iso="US", country_name="United States",
        aliases=("ISONE-RI", "ISO_NE_RI"),
    )
    GeoZone.ISONE_VT = _zone(
        "ISONE_VT", "ISO-NE Vermont", 44.0, -72.7,
        country_iso="US", country_name="United States",
        aliases=("ISONE-VT", "ISO_NE_VT"),
    )
    GeoZone.ISONE_SEMASS = _zone(
        "ISONE_SEMASS", "ISO-NE SE Massachusetts", 41.9, -70.9,
        country_iso="US", country_name="United States",
        aliases=("ISONE-SEMASS",),
    )
    GeoZone.ISONE_WCMASS = _zone(
        "ISONE_WCMASS", "ISO-NE WC Massachusetts", 42.2, -71.8,
        country_iso="US", country_name="United States",
        aliases=("ISONE-WCMASS",),
    )
    GeoZone.ISONE_NEMASS = _zone(
        "ISONE_NEMASS", "ISO-NE NE Massachusetts", 42.4, -71.0,
        country_iso="US", country_name="United States",
        aliases=("ISONE-NEMASS",),
    )

    GeoZone.NYISO_NYC = _zone(
        "NYISO_NYC", "NYISO New York City", 40.7128, -74.0060,
        country_iso="US", country_name="United States",
        aliases=("NYISO-NYC", "NYISO_ZONE_J"),
    )
    GeoZone.NYISO_WEST = _zone(
        "NYISO_WEST", "NYISO West", 42.9, -78.7,
        country_iso="US", country_name="United States",
        aliases=("NYISO-WEST", "NYISO_ZONE_A"),
    )
    GeoZone.NYISO_GENESE = _zone(
        "NYISO_GENESE", "NYISO Genesee", 43.1, -77.6,
        country_iso="US", country_name="United States",
        aliases=("NYISO-GENESE", "NYISO_ZONE_B"),
    )
    GeoZone.NYISO_CENTRL = _zone(
        "NYISO_CENTRL", "NYISO Central", 43.0, -76.2,
        country_iso="US", country_name="United States",
        aliases=("NYISO-CENTRL", "NYISO_ZONE_C"),
    )
    GeoZone.NYISO_NORTH = _zone(
        "NYISO_NORTH", "NYISO North", 44.5, -73.5,
        country_iso="US", country_name="United States",
        aliases=("NYISO-NORTH", "NYISO_ZONE_D"),
    )
    GeoZone.NYISO_MHK_VL = _zone(
        "NYISO_MHK_VL", "NYISO Mohawk Valley", 43.0, -75.3,
        country_iso="US", country_name="United States",
        aliases=("NYISO-MHK-VL", "NYISO_ZONE_E"),
    )
    GeoZone.NYISO_CAPITAL = _zone(
        "NYISO_CAPITAL", "NYISO Capital", 42.7, -73.8,
        country_iso="US", country_name="United States",
        aliases=("NYISO-CAPITAL", "NYISO_ZONE_F"),
    )
    GeoZone.NYISO_HUD_VL = _zone(
        "NYISO_HUD_VL", "NYISO Hudson Valley", 41.7, -74.0,
        country_iso="US", country_name="United States",
        aliases=("NYISO-HUD-VL", "NYISO_ZONE_G"),
    )
    GeoZone.NYISO_MILLWD = _zone(
        "NYISO_MILLWD", "NYISO Millwood", 41.2, -73.8,
        country_iso="US", country_name="United States",
        aliases=("NYISO-MILLWD", "NYISO_ZONE_H"),
    )
    GeoZone.NYISO_DUNWOD = _zone(
        "NYISO_DUNWOD", "NYISO Dunwoodie", 40.9, -73.9,
        country_iso="US", country_name="United States",
        aliases=("NYISO-DUNWOD", "NYISO_ZONE_I"),
    )
    GeoZone.NYISO_LI = _zone(
        "NYISO_LI", "NYISO Long Island", 40.8, -73.2,
        country_iso="US", country_name="United States",
        aliases=("NYISO-LI", "NYISO_ZONE_K"),
    )

    GeoZone.SPP_NORTH = _zone(
        "SPP_NORTH", "SPP North", 42.0, -98.0, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("SPP-NORTH",),
    )
    GeoZone.SPP_SOUTH = _zone(
        "SPP_SOUTH", "SPP South", 35.0, -97.0, "America/Chicago", "USD",
        country_iso="US", country_name="United States",
        aliases=("SPP-SOUTH",),
    )

    # ------------------------------------------------------------------
    # Japan JEPX regions
    # ------------------------------------------------------------------
    GeoZone.JEPX_HOKKAIDO = _zone(
        "JEPX_HOKKAIDO", "JEPX Hokkaido", 43.0642, 141.3469,
        country_iso="JP", country_name="Japan",
        aliases=("JEPX-HOKKAIDO", "JAPAN_HOKKAIDO"),
    )
    GeoZone.JEPX_TOHOKU = _zone(
        "JEPX_TOHOKU", "JEPX Tohoku", 38.2688, 140.8721,
        country_iso="JP", country_name="Japan",
        aliases=("JEPX-TOHOKU", "JAPAN_TOHOKU"),
    )
    GeoZone.JEPX_TOKYO = _zone(
        "JEPX_TOKYO", "JEPX Tokyo", 35.6762, 139.6503,
        country_iso="JP", country_name="Japan",
        aliases=("JEPX-TOKYO", "JAPAN_TOKYO"),
    )
    GeoZone.JEPX_CHUBU = _zone(
        "JEPX_CHUBU", "JEPX Chubu", 35.1802, 136.9066,
        country_iso="JP", country_name="Japan",
        aliases=("JEPX-CHUBU", "JAPAN_CHUBU"),
    )
    GeoZone.JEPX_HOKURIKU = _zone(
        "JEPX_HOKURIKU", "JEPX Hokuriku", 36.5947, 136.6256,
        country_iso="JP", country_name="Japan",
        aliases=("JEPX-HOKURIKU", "JAPAN_HOKURIKU"),
    )
    GeoZone.JEPX_KANSAI = _zone(
        "JEPX_KANSAI", "JEPX Kansai", 34.6937, 135.5023,
        country_iso="JP", country_name="Japan",
        aliases=("JEPX-KANSAI", "JAPAN_KANSAI"),
    )
    GeoZone.JEPX_CHUGOKU = _zone(
        "JEPX_CHUGOKU", "JEPX Chugoku", 34.3963, 132.4596,
        country_iso="JP", country_name="Japan",
        aliases=("JEPX-CHUGOKU", "JAPAN_CHUGOKU"),
    )
    GeoZone.JEPX_SHIKOKU = _zone(
        "JEPX_SHIKOKU", "JEPX Shikoku", 33.8416, 132.7658,
        country_iso="JP", country_name="Japan",
        aliases=("JEPX-SHIKOKU", "JAPAN_SHIKOKU"),
    )
    GeoZone.JEPX_KYUSHU = _zone(
        "JEPX_KYUSHU", "JEPX Kyushu", 33.5902, 130.4017,
        country_iso="JP", country_name="Japan",
        aliases=("JEPX-KYUSHU", "JAPAN_KYUSHU"),
    )
    GeoZone.JEPX_OKINAWA = _zone(
        "JEPX_OKINAWA", "JEPX Okinawa", 26.2124, 127.6809,
        country_iso="JP", country_name="Japan",
        aliases=("JEPX-OKINAWA", "JAPAN_OKINAWA"),
    )
    GeoZone.JEPX_SYSTEM = _zone(
        "JEPX_SYSTEM", "JEPX System", 35.6762, 139.6503,
        country_iso="JP", country_name="Japan",
        aliases=("JEPX-SYSTEM", "JAPAN_SYSTEM"),
    )

    # ------------------------------------------------------------------
    # Australian AEMO regions (NEM)
    # ------------------------------------------------------------------
    GeoZone.AEMO_NSW = _zone(
        "AEMO_NSW", "AEMO New South Wales", -33.8688, 151.2093,
        country_iso="AU", country_name="Australia",
        aliases=("AEMO-NSW", "NEM_NSW", "NSW1"),
    )
    GeoZone.AEMO_VIC = _zone(
        "AEMO_VIC", "AEMO Victoria", -37.8136, 144.9631, "Australia/Melbourne", "AUD",
        country_iso="AU", country_name="Australia",
        aliases=("AEMO-VIC", "NEM_VIC", "VIC1"),
    )
    GeoZone.AEMO_QLD = _zone(
        "AEMO_QLD", "AEMO Queensland", -27.4698, 153.0251, "Australia/Brisbane", "AUD",
        country_iso="AU", country_name="Australia",
        aliases=("AEMO-QLD", "NEM_QLD", "QLD1"),
    )
    GeoZone.AEMO_SA = _zone(
        "AEMO_SA", "AEMO South Australia", -34.9285, 138.6007, "Australia/Adelaide", "AUD",
        country_iso="AU", country_name="Australia",
        aliases=("AEMO-SA", "NEM_SA", "SA1"),
    )
    GeoZone.AEMO_TAS = _zone(
        "AEMO_TAS", "AEMO Tasmania", -42.8821, 147.3272, "Australia/Hobart", "AUD",
        country_iso="AU", country_name="Australia",
        aliases=("AEMO-TAS", "NEM_TAS", "TAS1"),
    )

    # ------------------------------------------------------------------
    # Indian energy exchange regions (IEX)
    # ------------------------------------------------------------------
    GeoZone.IEX_NORTHERN = _zone(
        "IEX_NORTHERN", "IEX Northern Region", 28.6139, 77.2090,
        country_iso="IN", country_name="India",
        aliases=("IEX-NORTHERN",),
    )
    GeoZone.IEX_WESTERN = _zone(
        "IEX_WESTERN", "IEX Western Region", 19.0760, 72.8777,
        country_iso="IN", country_name="India",
        aliases=("IEX-WESTERN",),
    )
    GeoZone.IEX_SOUTHERN = _zone(
        "IEX_SOUTHERN", "IEX Southern Region", 13.0827, 80.2707,
        country_iso="IN", country_name="India",
        aliases=("IEX-SOUTHERN",),
    )
    GeoZone.IEX_EASTERN = _zone(
        "IEX_EASTERN", "IEX Eastern Region", 22.5726, 88.3639,
        country_iso="IN", country_name="India",
        aliases=("IEX-EASTERN",),
    )
    GeoZone.IEX_NORTHEASTERN = _zone(
        "IEX_NORTHEASTERN", "IEX North-Eastern Region", 26.1445, 91.7362,
        country_iso="IN", country_name="India",
        aliases=("IEX-NORTHEASTERN",),
    )

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------
    GeoZone.UK = GeoZone.UNITED_KINGDOM
    GeoZone.USA = GeoZone.UNITED_STATES
    GeoZone.US = GeoZone.UNITED_STATES