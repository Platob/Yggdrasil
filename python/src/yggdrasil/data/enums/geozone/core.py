from __future__ import annotations

from .builders import _from_coordinates_with_optional_metadata
from .geozone import GeoZone, GeoZoneType

__all__ = ['load_core_geozones']

_CONTINENTS = [
    ('EUROPE', 'EU', ('EUROPE',), 'Europe', 54.5260, 15.2551, 'Europe/Brussels', 'EUR'),
    ('ASIA', 'AS', ('ASIA',), 'Asia', 34.0479, 100.6197, None, None),
    ('NORTH_AMERICA', 'NA', ('NORTH_AMERICA', 'NORTHAMERICA'), 'North America', 54.5260, -105.2551, None, None),
    ('SOUTH_AMERICA', 'SA', ('SOUTH_AMERICA', 'SOUTHAMERICA'), 'South America', -8.7832, -55.4915, None, None),
    ('AFRICA', 'AF', ('AFRICA',), 'Africa', -8.7832, 34.5085, None, None),
    ('OCEANIA', 'OC', ('OCEANIA', 'AUSTRALASIA'), 'Oceania', -22.7359, 140.0188, None, None),
]


def _register(attr: str, **kwargs) -> None:
    setattr(GeoZone, attr, GeoZone.put(_from_coordinates_with_optional_metadata(**kwargs)))


def load_core_geozones() -> None:
    _register(
        'UNKNOWN',
        gtype=GeoZoneType.UNKNOWN,
        lat=0.0,
        lon=0.0,
        key='UNKNOWN',
        aliases=('UNKN', 'N/A', 'NA', 'NONE', 'NULL', 'UNDEFINED'),
        name='Unknown',
        tz=None,
        ccy=None,
        coord_source='seed: module (sentinel – unknown / unresolved zone)',
        coord_kind='sentinel',
        confidence='low',
    )

    _register(
        'WORLD',
        gtype=GeoZoneType.WORLD,
        lat=0.0,
        lon=0.0,
        key='WORLD',
        aliases=('EARTH',),
        name='World',
        tz='UTC',
        ccy=None,
        coord_source='seed: module (0,0 placeholder)',
        coord_kind='representative_point',
        confidence='medium',
    )

    for attr, key, aliases, name, lat, lon, tz, ccy in _CONTINENTS:
        _register(
            attr,
            gtype=GeoZoneType.CONTINENT,
            lat=lat,
            lon=lon,
            key=key,
            aliases=aliases,
            name=name,
            tz=tz,
            ccy=ccy,
            coord_source='seed: module (continent representative point)',
            coord_kind='representative_point',
            confidence='medium',
        )
