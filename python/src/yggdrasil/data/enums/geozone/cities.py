from __future__ import annotations

from .builders import _city, _validate_unique_attrs
from .geozone import GeoZone

__all__ = ["load_cities"]

CITIES = [
    ("PARIS", "PAR", "Paris", 48.8566, 2.3522, "FR", "France", {}),
    ("ZURICH", "ZRH", "Zurich", 47.3769, 8.5417, "CH", "Switzerland", {}),
    ("GENEVA", "GVA", "Geneva", 46.2044, 6.1432, "CH", "Switzerland", {}),
    ("LONDON", "LON", "London", 51.5074, -0.1278, "GB", "United Kingdom", {}),
    ("BERLIN", "BER", "Berlin", 52.5200, 13.4050, "DE", "Germany", {}),
    ("MADRID", "MAD", "Madrid", 40.4168, -3.7038, "ES", "Spain", {}),
    ("ROME", "ROM", "Rome", 41.9028, 12.4964, "IT", "Italy", {}),
    ("AMSTERDAM", "AMS", "Amsterdam", 52.3676, 4.9041, "NL", "Netherlands", {}),
    ("BRUSSELS", "BRU", "Brussels", 50.8503, 4.3517, "BE", "Belgium", {}),
    ("VIENNA", "VIE", "Vienna", 48.2082, 16.3738, "AT", "Austria", {}),
    ("WARSAW", "WAW", "Warsaw", 52.2297, 21.0122, "PL", "Poland", {}),
    ("STOCKHOLM", "STO", "Stockholm", 59.3293, 18.0686, "SE", "Sweden", {}),
    ("OSLO", "OSL", "Oslo", 59.9139, 10.7522, "NO", "Norway", {}),
    ("COPENHAGEN", "CPH", "Copenhagen", 55.6761, 12.5683, "DK", "Denmark", {}),
    ("HELSINKI", "HEL", "Helsinki", 60.1699, 24.9384, "FI", "Finland", {}),
    ("LISBON", "LIS", "Lisbon", 38.7169, -9.1395, "PT", "Portugal", {}),
    ("ATHENS", "ATH", "Athens", 37.9838, 23.7275, "GR", "Greece", {}),
    ("BUDAPEST", "BUD", "Budapest", 47.4979, 19.0402, "HU", "Hungary", {}),
    ("PRAGUE", "PRG", "Prague", 50.0755, 14.4378, "CZ", "Czechia", {}),
    ("BUCHAREST", "BUH", "Bucharest", 44.4268, 26.1025, "RO", "Romania", {}),
    ("BRATISLAVA", "BTS", "Bratislava", 48.1486, 17.1077, "SK", "Slovakia", {}),
    ("LJUBLJANA", "LJU", "Ljubljana", 46.0569, 14.5058, "SI", "Slovenia", {}),
    ("ZAGREB", "ZAG", "Zagreb", 45.8150, 15.9819, "HR", "Croatia", {}),
    ("BELGRADE", "BEG", "Belgrade", 44.8176, 20.4633, "RS", "Serbia", {}),
    ("RIGA", "RIX", "Riga", 56.9460, 24.1059, "LV", "Latvia", {}),
    ("VILNIUS", "VNO", "Vilnius", 54.6872, 25.2797, "LT", "Lithuania", {}),
    ("TALLINN", "TLL", "Tallinn", 59.4370, 24.7536, "EE", "Estonia", {}),
    ("DUBLIN", "DUB", "Dublin", 53.3498, -6.2603, "IE", "Ireland", {}),
    ("NEW_YORK", "NYC", "New York", 40.7128, -74.0060, "US", "United States", {}),
    ("LOS_ANGELES", "LAX", "Los Angeles", 34.0522, -118.2437, "US", "United States", {"tz": "America/Los_Angeles"}),
    ("CHICAGO", "CHI", "Chicago", 41.8781, -87.6298, "US", "United States", {"tz": "America/Chicago"}),
    ("HOUSTON", "HOU", "Houston", 29.7604, -95.3698, "US", "United States", {"tz": "America/Chicago"}),
    ("TORONTO", "YYZ", "Toronto", 43.6532, -79.3832, "CA", "Canada", {}),
    ("SAO_PAULO", "GRU", "Sao Paulo", -23.5505, -46.6333, "BR", "Brazil", {}),
    ("BUENOS_AIRES", "EZE", "Buenos Aires", -34.6037, -58.3816, "AR", "Argentina", {}),
    ("TOKYO", "TYO", "Tokyo", 35.6762, 139.6503, "JP", "Japan", {}),
    ("BEIJING", "PEK", "Beijing", 39.9042, 116.4074, "CN", "China", {}),
    ("SHANGHAI", "PVG", "Shanghai", 31.2304, 121.4737, "CN", "China", {}),
    ("SEOUL", "ICN", "Seoul", 37.5665, 126.9780, "KR", "South Korea", {}),
    ("MUMBAI", "BOM", "Mumbai", 19.0760, 72.8777, "IN", "India", {}),
    ("DELHI", "DEL", "Delhi", 28.6139, 77.2090, "IN", "India", {}),
    ("SYDNEY", "SYD", "Sydney", -33.8688, 151.2093, "AU", "Australia", {}),
    ("SINGAPORE_CITY", "SIN", "Singapore", 1.3521, 103.8198, "SG", "Singapore", {}),
    ("DUBAI", "DXB", "Dubai", 25.2048, 55.2708, "AE", "United Arab Emirates", {}),
]


def load_cities() -> None:
    _validate_unique_attrs(CITIES, "CITIES")
    for attr, iso, name, lat, lon, country_iso, country_name, kwargs in CITIES:
        setattr(GeoZone, attr, _city(iso, name, lat, lon, country_iso, country_name, **kwargs))