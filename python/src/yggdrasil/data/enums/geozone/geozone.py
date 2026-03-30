from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Optional, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import polars as pl

__all__ = ["GeoZoneType", "GeoZone"]


class GeoZoneType:
    UNKNOWN: int = -1
    WORLD: int = 0
    CONTINENT: int = 1
    COUNTRY: int = 2
    CITY: int = 3
    ZONE: int = 4


_COORDINATE_RE = re.compile(
    r"^\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"\s*[,;|\s]\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"\s*$"
)
_DEFAULT_SRID = 4326


@dataclass(frozen=True, slots=True)
class _ZoneMeta:
    gtype: int
    wkb: bytes
    srid: int
    country_iso: Optional[str]
    country_name: Optional[str]
    city_iso: Optional[str]
    city_name: Optional[str]
    key: Optional[str]
    name: Optional[str]
    eic: Optional[str]
    tz: Optional[str]
    ccy: Optional[str]
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class GeoZone:
    gtype: int
    wkb: bytes
    srid: int = _DEFAULT_SRID

    country_iso: Optional[str] = None
    country_name: Optional[str] = None
    city_iso: Optional[str] = None
    city_name: Optional[str] = None

    key: Optional[str] = None
    aliases: tuple[str, ...] = ()

    name: Optional[str] = None
    eic: Optional[str] = None
    tz: Optional[str] = None
    ccy: Optional[str] = None

    lat: float = 0.0
    lon: float = 0.0

    CACHE_BY_GEOM: ClassVar[dict[bytes, "GeoZone"]] = {}
    CACHE_BY_KEY: ClassVar[dict[str, "GeoZone"]] = {}
    CACHE_BY_NAME: ClassVar[dict[str, "GeoZone"]] = {}
    CACHE_ALL: ClassVar[list["GeoZone"]] = []

    _TEXT_PRIORITY: ClassVar[dict[int, int]] = {
        GeoZoneType.CONTINENT: 0,
        GeoZoneType.COUNTRY: 1,
        GeoZoneType.CITY: 2,
        GeoZoneType.ZONE: 3,
        GeoZoneType.WORLD: 4,
        GeoZoneType.UNKNOWN: 5,
    }

    def __post_init__(self) -> None:
        wkb = bytes(self.wkb)
        if not wkb:
            raise ValueError("wkb must not be empty")

        srid = int(self.srid)
        if srid < 0:
            raise ValueError(f"srid must be >= 0, got {srid}")

        if self.lat == 0.0 and self.lon == 0.0:
            try:
                lat, lon = _parse_point_wkb(wkb)
            except ValueError:
                lat, lon = 0.0, 0.0
        else:
            lat = float(self.lat)
            lon = float(self.lon)

        aliases = tuple(sorted({
            token
            for raw in self.aliases
            for token in self._expand_lookup_tokens(raw)
            if token
        }))

        object.__setattr__(self, 'wkb', wkb)
        object.__setattr__(self, 'srid', srid)
        object.__setattr__(self, 'country_iso', self._norm_upper(self.country_iso))
        object.__setattr__(self, 'country_name', self._norm_text(self.country_name))
        object.__setattr__(self, 'city_iso', self._norm_upper(self.city_iso))
        object.__setattr__(self, 'city_name', self._norm_text(self.city_name))
        object.__setattr__(self, 'key', self._norm_upper(self.key))
        object.__setattr__(self, 'aliases', aliases)
        object.__setattr__(self, 'name', self._norm_text(self.name))
        object.__setattr__(self, 'eic', self._norm_upper(self.eic))
        object.__setattr__(self, 'tz', self._norm_text(self.tz))
        object.__setattr__(self, 'ccy', self._norm_upper(self.ccy))
        object.__setattr__(self, 'lat', lat)
        object.__setattr__(self, 'lon', lon)

    @property
    def geom_key(self) -> bytes:
        return _geom_cache_key(self.wkb, self.srid)

    @property
    def point(self) -> tuple[float, float]:
        return _parse_point_wkb(self.wkb)

    @classmethod
    def from_coordinates(
        cls,
        *,
        gtype: int,
        lat: float,
        lon: float,
        srid: int = _DEFAULT_SRID,
        country_iso: Optional[str] = None,
        country_name: Optional[str] = None,
        city_iso: Optional[str] = None,
        city_name: Optional[str] = None,
        key: Optional[str] = None,
        aliases: tuple[str, ...] = (),
        name: Optional[str] = None,
        eic: Optional[str] = None,
        tz: Optional[str] = None,
        ccy: Optional[str] = None,
    ) -> "GeoZone":
        lat = float(lat)
        lon = float(lon)
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"latitude must be between -90 and 90, got {lat}")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"longitude must be between -180 and 180, got {lon}")
        return cls(
            gtype=int(gtype),
            wkb=_point_wkb(lat, lon),
            srid=int(srid),
            country_iso=country_iso,
            country_name=country_name,
            city_iso=city_iso,
            city_name=city_name,
            key=key,
            aliases=aliases,
            name=name,
            eic=eic,
            tz=tz,
            ccy=ccy,
            lat=lat,
            lon=lon,
        )

    @classmethod
    def put(cls, zone: "GeoZone") -> "GeoZone":
        cls.CACHE_ALL.append(zone)
        cls.CACHE_BY_GEOM[zone.geom_key] = zone

        for token in cls._exact_lookup_tokens(zone):
            cls.CACHE_BY_KEY[token] = zone

        for name in cls._name_lookup_tokens(zone):
            cls.CACHE_BY_NAME[name] = zone

        cls._resolver_cache.cache_clear()
        cls._build_bidding_zone_regex_cache.cache_clear()
        cls._build_bin_lookup_cache.cache_clear()
        return zone

    @classmethod
    def clear_cache(cls) -> None:
        cls.CACHE_ALL.clear()
        cls.CACHE_BY_GEOM.clear()
        cls.CACHE_BY_KEY.clear()
        cls.CACHE_BY_NAME.clear()
        cls._resolver_cache.cache_clear()
        cls._build_bidding_zone_regex_cache.cache_clear()
        cls._build_bin_lookup_cache.cache_clear()

    @classmethod
    def get_by_geom(cls, wkb: bytes, srid: int = _DEFAULT_SRID) -> Optional["GeoZone"]:
        return cls.CACHE_BY_GEOM.get(_geom_cache_key(bytes(wkb), int(srid)))

    @classmethod
    def get_by_coordinates(cls, lat: float, lon: float, srid: int = _DEFAULT_SRID) -> Optional["GeoZone"]:
        return cls.CACHE_BY_GEOM.get(_geom_cache_key(_point_wkb(float(lat), float(lon)), int(srid)))

    @classmethod
    def get_by_key(cls, key: str) -> Optional["GeoZone"]:
        return cls.CACHE_BY_KEY.get(cls._norm_upper(key))

    @classmethod
    def get_by_name(cls, name: str) -> Optional["GeoZone"]:
        return cls.CACHE_BY_NAME.get(name.strip().casefold())

    @classmethod
    def get_by_eic(cls, eic: str) -> Optional["GeoZone"]:
        return cls.CACHE_BY_KEY.get(cls._norm_upper(eic))

    @staticmethod
    def _norm_text(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @staticmethod
    def _norm_upper(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip().upper()
        return value or None

    @staticmethod
    def _normalize_zone_token_py(value: str) -> str:
        return re.sub(r"_+", "_", re.sub(r"[\s\-/|]+", "_", value.strip().upper())).strip("_")

    @classmethod
    def _expand_lookup_tokens(cls, value: Optional[str]) -> tuple[str, ...]:
        if value is None:
            return ()
        raw = cls._norm_upper(value)
        if not raw:
            return ()
        norm = cls._normalize_zone_token_py(raw)
        if norm and norm != raw:
            return (raw, norm)
        return (raw,)

    @classmethod
    def _exact_lookup_tokens(cls, zone: "GeoZone") -> tuple[str, ...]:
        values: list[str] = []
        for value in [zone.key, zone.eic, zone.country_iso if zone.gtype == GeoZoneType.COUNTRY else None, zone.city_iso if zone.gtype == GeoZoneType.CITY else None]:
            values.extend(cls._expand_lookup_tokens(value))
        for alias in zone.aliases:
            values.extend(cls._expand_lookup_tokens(alias))
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                ordered.append(value)
        return tuple(ordered)

    @classmethod
    def _name_lookup_tokens(cls, zone: "GeoZone") -> tuple[str, ...]:
        values = [zone.name]
        if zone.gtype == GeoZoneType.COUNTRY:
            values.append(zone.country_name)
        if zone.gtype == GeoZoneType.CITY:
            values.append(zone.city_name)
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value is None:
                continue
            key = value.strip().casefold()
            if key and key not in seen:
                seen.add(key)
                ordered.append(key)
        return tuple(ordered)

    @classmethod
    def _priority_key(cls, zone: "GeoZone") -> tuple[int, int, str]:
        return (
            cls._TEXT_PRIORITY.get(zone.gtype, 99),
            -len(zone.key or zone.name or ""),
            zone.key or zone.name or "",
        )

    @classmethod
    def _pick_best(cls, zones: list["GeoZone"] | tuple["GeoZone", ...]) -> Optional["GeoZone"]:
        if not zones:
            return None
        return min(zones, key=cls._priority_key)

    @classmethod
    @lru_cache(maxsize=1)
    def _resolver_cache(cls) -> dict[str, tuple["GeoZone", ...]]:
        buckets: dict[str, list["GeoZone"]] = {}
        for zone in cls.CACHE_ALL:
            tokens: set[str] = set()
            for value in [zone.key, zone.name, zone.eic, zone.country_iso, zone.country_name, zone.city_iso, zone.city_name]:
                tokens.update(cls._expand_lookup_tokens(value))
            for alias in zone.aliases:
                tokens.update(cls._expand_lookup_tokens(alias))
            for token in tokens:
                norm = cls._normalize_zone_token_py(token)
                if norm:
                    buckets.setdefault(norm, []).append(zone)
        return {key: tuple(sorted(values, key=cls._priority_key)) for key, values in buckets.items()}

    @classmethod
    def parse_coordinates(cls, obj: Any) -> Optional[tuple[float, float]]:
        if obj is None:
            return None
        if isinstance(obj, str):
            match = _COORDINATE_RE.match(obj)
            if match is None:
                return None
            return float(match.group(1)), float(match.group(2))
        if isinstance(obj, (tuple, list)):
            if len(obj) != 2:
                raise ValueError(f"coordinates must have length 2, got {len(obj)}")
            return float(obj[0]), float(obj[1])
        if isinstance(obj, dict):
            if 'lat' in obj and 'lon' in obj:
                return float(obj['lat']), float(obj['lon'])
            if 'latitude' in obj and 'longitude' in obj:
                return float(obj['latitude']), float(obj['longitude'])
            return None
        lat = getattr(obj, 'lat', None)
        lon = getattr(obj, 'lon', None)
        if lat is not None and lon is not None:
            return float(lat), float(lon)
        return None

    @classmethod
    def _parse_exact_str(cls, value: str) -> Optional["GeoZone"]:
        zone = cls.CACHE_BY_KEY.get(cls._norm_upper(value))
        if zone is not None:
            return zone
        return cls.CACHE_BY_NAME.get(value.strip().casefold())

    @classmethod
    def _parse_normalized_str(cls, value: str) -> Optional["GeoZone"]:
        norm = cls._normalize_zone_token_py(value)
        if not norm:
            return None
        return cls._pick_best(cls._resolver_cache().get(norm, ()))

    @classmethod
    def _parse_token_windows(cls, value: str) -> Optional["GeoZone"]:
        norm = cls._normalize_zone_token_py(value)
        if not norm:
            return None
        tokens = norm.split('_')
        matches: list["GeoZone"] = []
        resolver = cls._resolver_cache()
        for length in range(len(tokens), 0, -1):
            for start in range(len(tokens) - length + 1):
                candidate = '_'.join(tokens[start:start + length])
                zones = resolver.get(candidate)
                if zones:
                    matches.extend(zones)
            if matches:
                break
        return cls._pick_best(matches)

    @classmethod
    def parse_str(cls, s: str) -> Optional["GeoZone"]:
        s = s.strip()
        if not s:
            return None
        zone = cls._parse_exact_str(s)
        if zone is not None:
            return zone
        zone = cls._parse_normalized_str(s)
        if zone is not None:
            return zone
        zone = cls._parse_token_windows(s)
        if zone is not None:
            return zone
        coords = cls.parse_coordinates(s)
        if coords is not None:
            return cls.get_by_coordinates(*coords)
        return None

    @classmethod
    def parse(cls, obj: Any) -> Optional["GeoZone"]:
        if obj is None:
            return None
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            return cls.parse_str(obj)
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return cls.get_by_geom(bytes(obj))
        coords = cls.parse_coordinates(obj)
        if coords is not None:
            zone = cls.get_by_coordinates(*coords)
            if zone is not None:
                return zone
        if isinstance(obj, dict):
            for field, getter in (
                ('key', cls.get_by_key),
                ('eic', cls.get_by_eic),
                ('name', cls.get_by_name),
                ('country_iso', cls.get_by_key),
                ('city_iso', cls.get_by_key),
            ):
                value = obj.get(field)
                if value is not None:
                    zone = getter(str(value))
                    if zone is not None:
                        return zone
            wkb = obj.get('wkb')
            if wkb is not None:
                return cls.get_by_geom(bytes(wkb), srid=int(obj.get('srid', _DEFAULT_SRID)))
            return None
        for field, getter in (
            ('key', cls.get_by_key),
            ('eic', cls.get_by_eic),
            ('name', cls.get_by_name),
        ):
            value = getattr(obj, field, None)
            if value is not None:
                zone = getter(str(value))
                if zone is not None:
                    return zone
        wkb = getattr(obj, 'wkb', None)
        if wkb is not None:
            return cls.get_by_geom(bytes(wkb), srid=int(getattr(obj, 'srid', _DEFAULT_SRID)))
        return None

    @classmethod
    def _null_point(cls) -> dict[str, None]:
        return {'lat': None, 'lon': None}

    @classmethod
    def _null_struct(cls) -> dict[str, None]:
        return {
            'gtype': None, 'wkb': None, 'srid': None,
            'country_iso': None, 'country_name': None,
            'city_iso': None, 'city_name': None,
            'key': None, 'name': None, 'eic': None,
            'tz': None, 'ccy': None, 'lat': None, 'lon': None,
        }

    @classmethod
    def _to_meta(cls, zone: "GeoZone") -> _ZoneMeta:
        return _ZoneMeta(
            gtype=zone.gtype,
            wkb=zone.wkb,
            srid=zone.srid,
            country_iso=zone.country_iso,
            country_name=zone.country_name,
            city_iso=zone.city_iso,
            city_name=zone.city_name,
            key=zone.key,
            name=zone.name,
            eic=zone.eic,
            tz=zone.tz,
            ccy=zone.ccy,
            lat=zone.lat,
            lon=zone.lon,
        )

    @classmethod
    def _meta_to_struct(cls, meta: _ZoneMeta) -> dict[str, Any]:
        return {
            'gtype': meta.gtype,
            'wkb': meta.wkb,
            'srid': meta.srid,
            'country_iso': meta.country_iso,
            'country_name': meta.country_name,
            'city_iso': meta.city_iso,
            'city_name': meta.city_name,
            'key': meta.key,
            'name': meta.name,
            'eic': meta.eic,
            'tz': meta.tz,
            'ccy': meta.ccy,
            'lat': meta.lat,
            'lon': meta.lon,
        }

    @classmethod
    def _zone_field(cls, zone: Optional["GeoZone"], return_value: Literal['wkb', 'country_iso', 'city_iso', 'point', 'struct', 'dataclass']) -> Any:
        if zone is None:
            if return_value == 'point':
                return cls._null_point()
            if return_value == 'struct':
                return cls._null_struct()
            return None
        if return_value == 'dataclass':
            return zone
        if return_value == 'wkb':
            return zone.wkb
        if return_value == 'country_iso':
            return zone.country_iso
        if return_value == 'city_iso':
            return zone.city_iso
        if return_value == 'point':
            return {'lat': zone.lat, 'lon': zone.lon}
        if return_value == 'struct':
            return cls._meta_to_struct(cls._to_meta(zone))
        raise ValueError(f'unsupported return_value: {return_value!r}')

    @classmethod
    def py_parse_str(
        cls,
        value: str | None,
        *,
        return_value: Literal['wkb', 'country_iso', 'city_iso', 'point', 'struct', 'dataclass'] = 'wkb',
    ) -> Any:
        return cls._zone_field(cls.parse_str(value) if value is not None else None, return_value)

    @classmethod
    def py_parse_bin(
        cls,
        value: bytes | bytearray | memoryview | None,
        *,
        return_value: Literal['wkb', 'country_iso', 'city_iso', 'point', 'struct', 'dataclass'] = 'wkb',
    ) -> Any:
        zone = None if value is None else cls.get_by_geom(bytes(value))
        return cls._zone_field(zone, return_value)

    @classmethod
    @lru_cache(maxsize=1)
    def _build_bidding_zone_regex_cache(
        cls,
    ) -> tuple[list[str], dict[str, str], dict[str, _ZoneMeta]]:
        alias_to_key: dict[str, str] = {}
        key_to_meta: dict[str, _ZoneMeta] = {}
        resolver = cls._resolver_cache()

        for token, zones in resolver.items():
            zone = cls._pick_best(zones)
            if zone is None or not zone.key:
                continue
            alias_to_key[token] = zone.key
            key_to_meta.setdefault(zone.key, cls._to_meta(zone))

        sorted_aliases = sorted(alias_to_key, key=lambda value: (-len(value), value))
        chunk_patterns = [
            r'(?:^|_)(' + '|'.join(re.escape(alias) for alias in sorted_aliases[i:i + 200]) + r')(?:_|$)'
            for i in range(0, len(sorted_aliases), 200)
            if sorted_aliases[i:i + 200]
        ]
        return chunk_patterns, alias_to_key, key_to_meta

    @classmethod
    @lru_cache(maxsize=1)
    def _build_bin_lookup_cache(cls) -> tuple[dict[bytes, _ZoneMeta], frozenset[bytes]]:
        wkb_to_meta: dict[bytes, _ZoneMeta] = {}
        for zone in cls.CACHE_ALL:
            wkb_to_meta.setdefault(zone.wkb, cls._to_meta(zone))
        return wkb_to_meta, frozenset(wkb_to_meta)

    @classmethod
    def polars_parse_str(
        cls,
        col: 'pl.Series | pl.Expr',
        *,
        return_value: Literal['wkb', 'country_iso', 'city_iso', 'point', 'struct'] = 'wkb',
        lazy: bool = False,
    ) -> 'pl.Series | pl.Expr':
        import polars as pl

        chunk_patterns, alias_to_key, key_to_meta = cls._build_bidding_zone_regex_cache()

        def _normalize_expr(expr: pl.Expr) -> pl.Expr:
            return (
                expr.cast(pl.Utf8)
                .str.strip_chars()
                .str.to_uppercase()
                .str.replace_all(r'[\s\-/|]+', '_')
                .str.replace_all(r'_+', '_')
                .str.strip_chars('_')
            )

        def _canonical_key_expr(expr: pl.Expr) -> pl.Expr:
            norm = _normalize_expr(expr)
            exact = norm.replace_strict(alias_to_key, default=None)
            if not chunk_patterns:
                return exact
            token_alias = pl.coalesce([norm.str.extract(pattern, group_index=1) for pattern in chunk_patterns])
            token_key = token_alias.replace_strict(alias_to_key, default=None)
            return pl.coalesce([exact, token_key])

        def _output_from_key(key_expr: pl.Expr) -> pl.Expr:
            if return_value == 'wkb':
                return key_expr.replace_strict({k: m.wkb for k, m in key_to_meta.items()}, default=None).cast(pl.Binary)
            if return_value == 'country_iso':
                return key_expr.replace_strict({k: m.country_iso for k, m in key_to_meta.items()}, default=None).cast(pl.Utf8)
            if return_value == 'city_iso':
                return key_expr.replace_strict({k: m.city_iso for k, m in key_to_meta.items()}, default=None).cast(pl.Utf8)
            if return_value == 'point':
                return pl.struct(
                    lat=key_expr.replace_strict({k: m.lat for k, m in key_to_meta.items()}, default=None).cast(pl.Float64),
                    lon=key_expr.replace_strict({k: m.lon for k, m in key_to_meta.items()}, default=None).cast(pl.Float64),
                )
            if return_value == 'struct':
                return pl.struct(
                    gtype=key_expr.replace_strict({k: m.gtype for k, m in key_to_meta.items()}, default=None).cast(pl.Int32),
                    wkb=key_expr.replace_strict({k: m.wkb for k, m in key_to_meta.items()}, default=None).cast(pl.Binary),
                    srid=key_expr.replace_strict({k: m.srid for k, m in key_to_meta.items()}, default=None).cast(pl.Int32),
                    country_iso=key_expr.replace_strict({k: m.country_iso for k, m in key_to_meta.items()}, default=None).cast(pl.Utf8),
                    country_name=key_expr.replace_strict({k: m.country_name for k, m in key_to_meta.items()}, default=None).cast(pl.Utf8),
                    city_iso=key_expr.replace_strict({k: m.city_iso for k, m in key_to_meta.items()}, default=None).cast(pl.Utf8),
                    city_name=key_expr.replace_strict({k: m.city_name for k, m in key_to_meta.items()}, default=None).cast(pl.Utf8),
                    key=key_expr.cast(pl.Utf8),
                    name=key_expr.replace_strict({k: m.name for k, m in key_to_meta.items()}, default=None).cast(pl.Utf8),
                    eic=key_expr.replace_strict({k: m.eic for k, m in key_to_meta.items()}, default=None).cast(pl.Utf8),
                    tz=key_expr.replace_strict({k: m.tz for k, m in key_to_meta.items()}, default=None).cast(pl.Utf8),
                    ccy=key_expr.replace_strict({k: m.ccy for k, m in key_to_meta.items()}, default=None).cast(pl.Utf8),
                    lat=key_expr.replace_strict({k: m.lat for k, m in key_to_meta.items()}, default=None).cast(pl.Float64),
                    lon=key_expr.replace_strict({k: m.lon for k, m in key_to_meta.items()}, default=None).cast(pl.Float64),
                )
            raise ValueError(f'unsupported return_value: {return_value!r}')

        expr = _output_from_key(_canonical_key_expr(col if isinstance(col, pl.Expr) else pl.col(col.name)))
        if isinstance(col, pl.Expr):
            return expr
        if isinstance(col, pl.Series):
            expr = expr.alias(col.name)
            return expr if lazy else col.to_frame().select(expr).to_series()
        raise TypeError(f'expected pl.Series | pl.Expr, got {type(col)!r}')

    @classmethod
    def polars_parse_bin(
        cls,
        col: 'pl.Series | pl.Expr',
        *,
        return_value: Literal['wkb', 'country_iso', 'city_iso', 'point', 'struct'] = 'wkb',
        lazy: bool = False,
    ) -> 'pl.Series | pl.Expr':
        import polars as pl

        wkb_to_meta, known_wkb = cls._build_bin_lookup_cache()
        binary_expr = col.cast(pl.Binary) if isinstance(col, pl.Expr) else pl.col(col.name).cast(pl.Binary)

        if return_value == 'wkb':
            mapping = {wkb: wkb for wkb in known_wkb}
            expr = binary_expr.replace_strict(mapping, default=None).cast(pl.Binary)
        elif return_value == 'country_iso':
            expr = binary_expr.replace_strict({w: m.country_iso for w, m in wkb_to_meta.items()}, default=None).cast(pl.Utf8)
        elif return_value == 'city_iso':
            expr = binary_expr.replace_strict({w: m.city_iso for w, m in wkb_to_meta.items()}, default=None).cast(pl.Utf8)
        elif return_value == 'point':
            expr = pl.struct(
                lat=binary_expr.replace_strict({w: m.lat for w, m in wkb_to_meta.items()}, default=None).cast(pl.Float64),
                lon=binary_expr.replace_strict({w: m.lon for w, m in wkb_to_meta.items()}, default=None).cast(pl.Float64),
            )
        elif return_value == 'struct':
            expr = pl.struct(
                gtype=binary_expr.replace_strict({w: m.gtype for w, m in wkb_to_meta.items()}, default=None).cast(pl.Int32),
                wkb=binary_expr.replace_strict({wkb: wkb for wkb in known_wkb}, default=None).cast(pl.Binary),
                srid=binary_expr.replace_strict({w: m.srid for w, m in wkb_to_meta.items()}, default=None).cast(pl.Int32),
                country_iso=binary_expr.replace_strict({w: m.country_iso for w, m in wkb_to_meta.items()}, default=None).cast(pl.Utf8),
                country_name=binary_expr.replace_strict({w: m.country_name for w, m in wkb_to_meta.items()}, default=None).cast(pl.Utf8),
                city_iso=binary_expr.replace_strict({w: m.city_iso for w, m in wkb_to_meta.items()}, default=None).cast(pl.Utf8),
                city_name=binary_expr.replace_strict({w: m.city_name for w, m in wkb_to_meta.items()}, default=None).cast(pl.Utf8),
                key=binary_expr.replace_strict({w: m.key for w, m in wkb_to_meta.items()}, default=None).cast(pl.Utf8),
                name=binary_expr.replace_strict({w: m.name for w, m in wkb_to_meta.items()}, default=None).cast(pl.Utf8),
                eic=binary_expr.replace_strict({w: m.eic for w, m in wkb_to_meta.items()}, default=None).cast(pl.Utf8),
                tz=binary_expr.replace_strict({w: m.tz for w, m in wkb_to_meta.items()}, default=None).cast(pl.Utf8),
                ccy=binary_expr.replace_strict({w: m.ccy for w, m in wkb_to_meta.items()}, default=None).cast(pl.Utf8),
                lat=binary_expr.replace_strict({w: m.lat for w, m in wkb_to_meta.items()}, default=None).cast(pl.Float64),
                lon=binary_expr.replace_strict({w: m.lon for w, m in wkb_to_meta.items()}, default=None).cast(pl.Float64),
            )
        else:
            raise ValueError(f'unsupported return_value: {return_value!r}')

        if isinstance(col, pl.Expr):
            return expr
        if isinstance(col, pl.Series):
            expr = expr.alias(col.name)
            return expr if lazy else col.to_frame().select(expr).to_series()
        raise TypeError(f'expected pl.Series | pl.Expr, got {type(col)!r}')


def _geom_cache_key(wkb: bytes, srid: int) -> bytes:
    if srid == _DEFAULT_SRID:
        return wkb
    return b'\x00' + struct.pack('>I', srid) + wkb


def _point_wkb(lat: float, lon: float) -> bytes:
    return struct.pack('<BIdd', 1, 1, float(lon), float(lat))


def _parse_point_wkb(wkb: bytes) -> tuple[float, float]:
    if len(wkb) != 21:
        raise ValueError(f'unsupported WKB length for POINT: {len(wkb)}')
    byte_order = wkb[0]
    if byte_order == 1:
        endian = '<'
    elif byte_order == 0:
        endian = '>'
    else:
        raise ValueError(f'invalid WKB byte order: {byte_order}')
    gtype = struct.unpack(f'{endian}I', wkb[1:5])[0]
    if gtype != 1:
        raise ValueError(f'unsupported WKB geometry type: {gtype}')
    lon, lat = struct.unpack(f'{endian}dd', wkb[5:21])
    return float(lat), float(lon)
