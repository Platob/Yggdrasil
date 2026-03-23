from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Optional, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import polars as pl


__all__ = [
    "GeoZoneType",
    "GeoZone",
]


class GeoZoneType:
    UNKNOWN: int = -1   # sentinel – "not resolved / not applicable"
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

_DEFAULT_SRID: int = 4326
# Prefix prepended to wkb bytes when srid != _DEFAULT_SRID to form the geom
# cache key.  Format: b"\x00" + srid as 4-byte big-endian unsigned int.
_SRID_PREFIX_LEN: int = 5


def _geom_cache_key(wkb: bytes, srid: int) -> bytes:
    """Return the key used in CACHE_BY_GEOM.

    For the common case (srid == 4326) the key is the raw WKB bytes, saving
    a tuple allocation per entry.  For other SRIDs a compact 5-byte prefix is
    prepended so there is no collision with standard 21-byte WKB POINT values.
    """
    if srid == _DEFAULT_SRID:
        return wkb
    return b"\x00" + struct.pack(">I", srid) + wkb


def _point_wkb(lat: float, lon: float) -> bytes:
    # WKB POINT, little-endian, geometry type 1
    # WKB point coordinate order is (x, y) -> (lon, lat)
    return struct.pack("<BIdd", 1, 1, float(lon), float(lat))


def _parse_point_wkb(wkb: bytes) -> tuple[float, float]:
    if len(wkb) != 21:
        raise ValueError(f"unsupported WKB length for POINT: {len(wkb)}")

    byte_order = wkb[0]
    if byte_order == 1:
        endian = "<"
    elif byte_order == 0:
        endian = ">"
    else:
        raise ValueError(f"invalid WKB byte order: {byte_order}")

    gtype = struct.unpack(f"{endian}I", wkb[1:5])[0]
    if gtype != 1:
        raise ValueError(f"unsupported WKB geometry type: {gtype}")

    lon, lat = struct.unpack(f"{endian}dd", wkb[5:21])
    return float(lat), float(lon)


class _ZoneMeta:
    """Compact per-zone metadata record used in lookup caches.

    Using explicit ``__slots__`` avoids the per-instance ``__dict__`` overhead
    that a plain dataclass or namedtuple would carry.  One ``_ZoneMeta`` object
    consolidates what was previously spread across 12–14 separate dicts,
    reducing total allocation count dramatically.
    """
    __slots__ = (
        "wkb", "srid", "gtype",
        "key", "name", "eic",
        "country_iso", "country_name",
        "city_iso", "city_name",
        "tz", "ccy",
    )

    def __init__(
        self,
        wkb: bytes,
        srid: int,
        gtype: int,
        key: Optional[str],
        name: Optional[str],
        eic: Optional[str],
        country_iso: Optional[str],
        country_name: Optional[str],
        city_iso: Optional[str],
        city_name: Optional[str],
        tz: Optional[str],
        ccy: Optional[str],
    ) -> None:
        self.wkb          = wkb
        self.srid         = srid
        self.gtype        = gtype
        self.key          = key
        self.name         = name
        self.eic          = eic
        self.country_iso  = country_iso
        self.country_name = country_name
        self.city_iso     = city_iso
        self.city_name    = city_name
        self.tz           = tz
        self.ccy          = ccy

    @property
    def lat(self) -> float:
        return _parse_point_wkb(self.wkb)[0]

    @property
    def lon(self) -> float:
        return _parse_point_wkb(self.wkb)[1]


@dataclass(frozen=True, slots=True)
class GeoZone:
    gtype: int
    wkb: bytes
    srid: int = 4326

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

    # Derived from wkb when not explicitly provided (default sentinel = 0.0).
    lat: float = 0.0
    lon: float = 0.0

    # CACHE_BY_GEOM key: raw wkb bytes for srid==4326; prefixed bytes otherwise.
    # CACHE_BY_EIC removed – EICs are already stored in CACHE_BY_KEY.
    CACHE_BY_KEY:  ClassVar[dict[str, "GeoZone"]]   = {}
    CACHE_BY_NAME: ClassVar[dict[str, "GeoZone"]]   = {}
    CACHE_BY_GEOM: ClassVar[dict[bytes, "GeoZone"]] = {}

    def __post_init__(self) -> None:
        if not isinstance(self.wkb, (bytes, bytearray, memoryview)):
            raise TypeError("wkb must be bytes-like")

        wkb = bytes(self.wkb)
        if not wkb:
            raise ValueError("wkb must not be empty")

        srid = int(self.srid)
        if srid < 0:
            raise ValueError(f"srid must be >= 0, got {srid}")

        aliases = tuple(
            alias
            for alias in (self._norm_upper(x) for x in self.aliases)
            if alias
        )

        # Derive lat/lon from WKB when both are at the default sentinel (0.0).
        # If the caller passed explicit non-zero values, trust them; if only one
        # is non-zero we still re-derive both from WKB for consistency.
        if self.lat == 0.0 and self.lon == 0.0:
            try:
                parsed_lat, parsed_lon = _parse_point_wkb(wkb)
            except ValueError:
                parsed_lat, parsed_lon = 0.0, 0.0
        else:
            parsed_lat = float(self.lat)
            parsed_lon = float(self.lon)

        object.__setattr__(self, "wkb", wkb)
        object.__setattr__(self, "srid", srid)
        object.__setattr__(self, "country_iso", self._norm_upper(self.country_iso))
        object.__setattr__(self, "country_name", self._norm_text(self.country_name))
        object.__setattr__(self, "city_iso", self._norm_upper(self.city_iso))
        object.__setattr__(self, "city_name", self._norm_text(self.city_name))
        object.__setattr__(self, "key", self._norm_upper(self.key))
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "name", self._norm_text(self.name))
        object.__setattr__(self, "eic", self._norm_upper(self.eic))
        object.__setattr__(self, "tz", self._norm_text(self.tz))
        object.__setattr__(self, "ccy", self._norm_upper(self.ccy))
        object.__setattr__(self, "lat", parsed_lat)
        object.__setattr__(self, "lon", parsed_lon)

    @property
    def geom_key(self) -> bytes:
        """Cache key for CACHE_BY_GEOM (flat bytes; no tuple allocation)."""
        return _geom_cache_key(self.wkb, self.srid)

    @property
    def point(self) -> tuple[float, float]:
        """Re-parse lat/lon strictly from WKB, raising on invalid data."""
        return _parse_point_wkb(self.wkb)

    @classmethod
    def from_coordinates(
        cls,
        *,
        gtype: int,
        lat: float,
        lon: float,
        srid: int = 4326,
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
            gtype=gtype,
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
        )

    @classmethod
    def put(cls, zone: "GeoZone") -> "GeoZone":
        cls.CACHE_BY_GEOM[zone.geom_key] = zone

        keys: set[str] = set(zone.aliases)

        if zone.key:
            keys.add(zone.key)

        if zone.eic:
            # EIC is registered in CACHE_BY_KEY (no separate CACHE_BY_EIC).
            keys.add(zone.eic)

        if zone.gtype == GeoZoneType.COUNTRY and zone.country_iso:
            keys.add(zone.country_iso)

        if zone.gtype == GeoZoneType.CITY and zone.city_iso:
            keys.add(zone.city_iso)

        for key in keys:
            cls.CACHE_BY_KEY[key] = zone

        if zone.name:
            cls.CACHE_BY_NAME[zone.name.casefold()] = zone

        if zone.gtype == GeoZoneType.COUNTRY and zone.country_name:
            cls.CACHE_BY_NAME[zone.country_name.casefold()] = zone

        if zone.gtype == GeoZoneType.CITY and zone.city_name:
            cls.CACHE_BY_NAME[zone.city_name.casefold()] = zone

        cls._build_bidding_zone_regex_cache.cache_clear()
        cls._build_bin_lookup_cache.cache_clear()
        return zone

    @classmethod
    def get_by_geom(cls, wkb: bytes, srid: int = 4326) -> Optional["GeoZone"]:
        return cls.CACHE_BY_GEOM.get(_geom_cache_key(bytes(wkb), int(srid)))

    @classmethod
    def get_by_coordinates(cls, lat: float, lon: float, srid: int = 4326) -> Optional["GeoZone"]:
        return cls.CACHE_BY_GEOM.get(_geom_cache_key(_point_wkb(float(lat), float(lon)), int(srid)))

    @classmethod
    def get_by_key(cls, key: str) -> Optional["GeoZone"]:
        return cls.CACHE_BY_KEY.get(key.strip().upper())

    @classmethod
    def get_by_name(cls, name: str) -> Optional["GeoZone"]:
        return cls.CACHE_BY_NAME.get(name.strip().casefold())

    @classmethod
    def get_by_eic(cls, eic: str) -> Optional["GeoZone"]:
        # EIC keys live in CACHE_BY_KEY; no separate dict needed.
        return cls.CACHE_BY_KEY.get(eic.strip().upper())

    @classmethod
    def clear_cache(cls) -> None:
        cls.CACHE_BY_KEY.clear()
        cls.CACHE_BY_NAME.clear()
        cls.CACHE_BY_GEOM.clear()
        cls._build_bidding_zone_regex_cache.cache_clear()
        cls._build_bin_lookup_cache.cache_clear()

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
            if "lat" in obj and "lon" in obj:
                return float(obj["lat"]), float(obj["lon"])
            if "latitude" in obj and "longitude" in obj:
                return float(obj["latitude"]), float(obj["longitude"])
            return None

        lat = getattr(obj, "lat", None)
        lon = getattr(obj, "lon", None)
        if lat is not None and lon is not None:
            return float(lat), float(lon)

        return None

    @classmethod
    def parse_str(cls, s: str) -> Optional["GeoZone"]:
        s = s.strip()
        if not s:
            return None

        s_upper = s.upper()

        zone = cls.CACHE_BY_KEY.get(s_upper)
        if zone is not None:
            return zone

        zone = cls.CACHE_BY_NAME.get(s.casefold())
        if zone is not None:
            return zone

        if 2 <= len(s_upper) <= 64:
            compact_upper = s_upper.replace("-", "_").replace(" ", "_")
            zone = cls.CACHE_BY_KEY.get(compact_upper)
            if zone is not None:
                return zone

        match = _COORDINATE_RE.match(s)
        if match is not None:
            return cls.get_by_coordinates(float(match.group(1)), float(match.group(2)))

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
            key = obj.get("key")
            if key is not None:
                zone = cls.get_by_key(str(key))
                if zone is not None:
                    return zone

            eic = obj.get("eic")
            if eic is not None:
                zone = cls.get_by_eic(str(eic))
                if zone is not None:
                    return zone

            name = obj.get("name")
            if name is not None:
                zone = cls.get_by_name(str(name))
                if zone is not None:
                    return zone

            country_iso = obj.get("country_iso")
            if country_iso is not None:
                zone = cls.get_by_key(str(country_iso))
                if zone is not None:
                    return zone

            city_iso = obj.get("city_iso")
            if city_iso is not None:
                zone = cls.get_by_key(str(city_iso))
                if zone is not None:
                    return zone

            wkb = obj.get("wkb")
            if wkb is not None:
                srid = int(obj.get("srid", 4326))
                zone = cls.get_by_geom(bytes(wkb), srid=srid)
                if zone is not None:
                    return zone

            return None

        key = getattr(obj, "key", None)
        if key is not None:
            zone = cls.get_by_key(str(key))
            if zone is not None:
                return zone

        eic = getattr(obj, "eic", None)
        if eic is not None:
            zone = cls.get_by_eic(str(eic))
            if zone is not None:
                return zone

        name = getattr(obj, "name", None)
        if name is not None:
            zone = cls.get_by_name(str(name))
            if zone is not None:
                return zone

        wkb = getattr(obj, "wkb", None)
        if wkb is not None:
            srid = int(getattr(obj, "srid", 4326))
            zone = cls.get_by_geom(bytes(wkb), srid=srid)
            if zone is not None:
                return zone

        return None

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
    @lru_cache(maxsize=1)
    def _build_bidding_zone_regex_cache(
        cls,
    ) -> tuple[
        list[str],                # chunk_patterns (one per ≤200-alias group)
        dict[str, str],           # alias_to_key
        dict[str, "_ZoneMeta"],   # key_to_meta  (replaces 11 separate field dicts)
    ]:
        """Build string-key lookup structures used by polars_parse_str.

        Returns a compact triple instead of the previous 13-element tuple:
        - ``chunk_patterns`` – pre-built regex strings for token-scan pass 2.
        - ``alias_to_key`` – normalised alias → canonical key.
        - ``key_to_meta`` – canonical key → :class:`_ZoneMeta` with all fields.
        """
        alias_to_key: dict[str, str] = {}
        key_to_meta:  dict[str, _ZoneMeta] = {}

        def _make_meta(zone: "GeoZone", canonical_key: str) -> _ZoneMeta:
            return _ZoneMeta(
                wkb=zone.wkb,
                srid=zone.srid,
                gtype=zone.gtype,
                key=canonical_key,
                name=zone.name,
                eic=zone.eic,
                country_iso=zone.country_iso,
                country_name=zone.country_name,
                city_iso=zone.city_iso,
                city_name=zone.city_name,
                tz=zone.tz,
                ccy=zone.ccy,
            )

        def _index_zone(zone: "GeoZone") -> None:
            canonical_key = zone.key
            if not canonical_key:
                return

            canonical_key = cls._normalize_zone_token_py(canonical_key)
            if canonical_key not in key_to_meta:
                key_to_meta[canonical_key] = _make_meta(zone, canonical_key)

            raw_aliases: set[str] = {
                canonical_key,
                canonical_key.replace("_", ""),
                canonical_key.replace("_", "-"),
                canonical_key.replace("_", " "),
            }
            raw_aliases.update(zone.aliases)

            if zone.name:
                raw_aliases.add(zone.name)
            if zone.country_name:
                raw_aliases.add(zone.country_name)
                raw_aliases.add(f"{zone.country_name} {canonical_key}")
            if zone.city_name:
                raw_aliases.add(zone.city_name)
                raw_aliases.add(f"{zone.city_name} {canonical_key}")
            if zone.country_iso and zone.country_iso != canonical_key:
                raw_aliases.add(zone.country_iso)
                raw_aliases.add(f"{zone.country_iso} {canonical_key}")
            if zone.city_iso and zone.city_iso != canonical_key:
                raw_aliases.add(zone.city_iso)
            if zone.eic:
                raw_aliases.add(zone.eic)

            for alias in raw_aliases:
                norm_alias = cls._normalize_zone_token_py(alias)
                if norm_alias:
                    alias_to_key[norm_alias] = canonical_key

        for _, zone in cls.CACHE_BY_KEY.items():
            if zone is not None:
                _index_zone(zone)

        for alias, zone in cls.CACHE_BY_NAME.items():
            if zone is None or not zone.key:
                continue

            canonical_key = cls._normalize_zone_token_py(zone.key)
            if canonical_key not in key_to_meta:
                key_to_meta[canonical_key] = _make_meta(zone, canonical_key)

            raw_aliases = {
                alias,
                alias.replace("_", " "),
                canonical_key,
                canonical_key.replace("_", ""),
                canonical_key.replace("_", "-"),
                canonical_key.replace("_", " "),
            }
            raw_aliases.update(zone.aliases)

            if zone.country_name:
                raw_aliases.add(zone.country_name)
            if zone.city_name:
                raw_aliases.add(zone.city_name)

            for raw_alias in raw_aliases:
                norm_alias = cls._normalize_zone_token_py(raw_alias)
                if norm_alias:
                    alias_to_key[norm_alias] = canonical_key

        # Build lookup structures for polars_parse_str.
        #
        # Strategy (two-pass, O(1) expression depth):
        #
        # Pass 1 – exact replace: normalise the whole input string and look it
        #   up directly in alias_to_key.  This handles ~99 % of real-world
        #   inputs (exact keys, EICs, country names, aliases, etc.).
        #
        # Pass 2 – token scan: for free-text strings like "Sweden SE1 wind
        #   power" that don't have a whole-string match, scan each
        #   underscore-delimited token left-to-right and return the first token
        #   that is a known alias.  We implement this as a sequence of
        #   str.extract() calls, one per chunk of ≤ CHUNK aliases, combined
        #   with pl.coalesce() so only one result is kept.  Each individual
        #   regex is small and safe for the regex engine.
        sorted_aliases = sorted(alias_to_key.keys(), key=lambda a: (-len(a), a))

        # Chunk the sorted aliases into groups to keep each regex small.
        _CHUNK = 200
        alias_chunks: list[list[str]] = [
            sorted_aliases[i: i + _CHUNK]
            for i in range(0, len(sorted_aliases), _CHUNK)
        ]
        # Pre-build the per-chunk pattern strings (token-boundary match).
        chunk_patterns: list[str] = [
            r"(?:^|_)(" + "|".join(re.escape(a) for a in chunk) + r")(?:_|$)"
            for chunk in alias_chunks
        ]

        return (
            chunk_patterns,
            alias_to_key,
            key_to_meta,
        )

    # ------------------------------------------------------------------
    # Pure-Python parse helpers (mirror of polars_parse_str / polars_parse_bin)
    # ------------------------------------------------------------------

    @classmethod
    def _canonical_key_for_str(cls, s: str) -> Optional[str]:
        """Normalise *s* and resolve it to a canonical key (or *None*).

        Mirrors the two-pass logic used in :meth:`polars_parse_str`:

        * **Pass 1** – normalise the whole string and look it up directly in
          ``alias_to_key``.  Covers exact keys, EICs, names, etc.
        * **Pass 2** – token scan: split the normalised string on ``_`` and
          check each token (longest first) against ``alias_to_key``.  This
          handles free-text strings like ``"Sweden SE1 wind power"``.
        """
        _chunk_patterns, alias_to_key, _key_to_meta = cls._build_bidding_zone_regex_cache()

        norm = re.sub(r"_+", "_", re.sub(r"[\s\-/|]+", "_", s.strip().upper())).strip("_")
        if not norm:
            return None

        # Pass 1 – exact whole-string match.
        ck = alias_to_key.get(norm)
        if ck is not None:
            return ck

        # Pass 2 – token scan (sorted longest first so more-specific wins).
        tokens = norm.split("_")
        # Build candidate multi-token windows (longest first) then single tokens.
        candidates: list[str] = []
        for length in range(len(tokens), 0, -1):
            for start in range(len(tokens) - length + 1):
                candidates.append("_".join(tokens[start: start + length]))
        for candidate in candidates:
            ck = alias_to_key.get(candidate)
            if ck is not None:
                return ck

        return None

    @classmethod
    def _zone_field_from_key(
        cls,
        canonical_key: Optional[str],
        return_value: "Literal['wkb', 'country_iso', 'city_iso', 'point', 'struct', 'dataclass']",
        *,
        key_to_meta: "dict[str, _ZoneMeta]",
    ) -> Any:
        """Return the requested field value for *canonical_key*, or *None*."""
        _NULL_POINT  = {"lat": None, "lon": None}
        _NULL_STRUCT = {
            "gtype": None, "wkb": None, "srid": None,
            "country_iso": None, "country_name": None,
            "city_iso": None, "city_name": None,
            "key": None, "name": None, "eic": None,
            "tz": None, "ccy": None, "lat": None, "lon": None,
        }

        if canonical_key is None:
            if return_value == "point":
                return _NULL_POINT
            if return_value == "struct":
                return _NULL_STRUCT
            return None

        if return_value == "dataclass":
            return cls.get_by_key(canonical_key)

        meta = key_to_meta.get(canonical_key)
        if meta is None:
            if return_value == "point":
                return _NULL_POINT
            if return_value == "struct":
                return _NULL_STRUCT
            return None

        if return_value == "wkb":
            return meta.wkb

        if return_value == "country_iso":
            return meta.country_iso

        if return_value == "city_iso":
            return meta.city_iso

        if return_value == "point":
            lat, lon = _parse_point_wkb(meta.wkb)
            return {"lat": lat, "lon": lon}

        if return_value == "struct":
            lat, lon = _parse_point_wkb(meta.wkb)
            return {
                "gtype": meta.gtype,
                "wkb": meta.wkb,
                "srid": meta.srid,
                "country_iso": meta.country_iso,
                "country_name": meta.country_name,
                "city_iso": meta.city_iso,
                "city_name": meta.city_name,
                "key": canonical_key,
                "name": meta.name,
                "eic": meta.eic,
                "tz": meta.tz,
                "ccy": meta.ccy,
                "lat": lat,
                "lon": lon,
            }

        raise ValueError(
            f"return_value must be one of 'wkb', 'country_iso', 'city_iso', 'point', 'struct', 'dataclass';"
            f" got {return_value!r}"
        )

    @classmethod
    def py_parse_str(
        cls,
        value: "str | None",
        *,
        return_value: Literal["wkb", "country_iso", "city_iso", "point", "struct", "dataclass"] = "wkb",
    ) -> Any:
        """Pure-Python equivalent of :meth:`polars_parse_str` for scalar values.

        Accepts a single string (or *None*) and returns the zone field
        indicated by *return_value*:

        * ``"wkb"`` (default) – ``bytes | None``
        * ``"country_iso"`` – ``str | None``
        * ``"city_iso"`` – ``str | None``
        * ``"point"`` – ``{"lat": float | None, "lon": float | None}``
        * ``"struct"`` – ``dict`` with all GeoZone fields (field order matches
          the dataclass declaration; missing values are ``None``).
        * ``"dataclass"`` – the cached :class:`GeoZone` singleton (or *None*).

        The lookup uses the same two-pass strategy as the Polars version:
        whole-string normalisation first, then a longest-token scan for
        free-text inputs like ``"Sweden SE1 wind power"``.
        """
        if value is None:
            if return_value == "point":
                return {"lat": None, "lon": None}
            if return_value == "struct":
                return {
                    "gtype": None, "wkb": None, "srid": None,
                    "country_iso": None, "country_name": None,
                    "city_iso": None, "city_name": None,
                    "key": None, "name": None, "eic": None,
                    "tz": None, "ccy": None, "lat": None, "lon": None,
                }
            return None

        _chunk_patterns, _alias_to_key, key_to_meta = cls._build_bidding_zone_regex_cache()

        canonical_key = cls._canonical_key_for_str(value)
        return cls._zone_field_from_key(canonical_key, return_value, key_to_meta=key_to_meta)

    @classmethod
    def py_parse_bin(
        cls,
        value: "bytes | bytearray | memoryview | None",
        *,
        return_value: Literal["wkb", "country_iso", "city_iso", "point", "struct", "dataclass"] = "wkb",
    ) -> Any:
        """Pure-Python equivalent of :meth:`polars_parse_bin` for scalar values.

        Accepts a single WKB bytes value (or *None*) and returns the zone
        field indicated by *return_value*:

        * ``"wkb"`` (default) – ``bytes | None`` (pass-through canonical WKB)
        * ``"country_iso"`` – ``str | None``
        * ``"city_iso"`` – ``str | None``
        * ``"point"`` – ``{"lat": float | None, "lon": float | None}``
        * ``"struct"`` – ``dict`` with all GeoZone fields (field order matches
          the dataclass declaration; missing values are ``None``).
        * ``"dataclass"`` – the cached :class:`GeoZone` singleton (or *None*).

        Unknown / unregistered WKB values map to ``None`` (or a null-filled
        dict for ``"point"`` and ``"struct"``).
        """
        if return_value not in ("wkb", "country_iso", "city_iso", "point", "struct", "dataclass"):
            raise ValueError(
                f"return_value must be one of 'wkb', 'country_iso', 'city_iso', 'point', 'struct', 'dataclass';"
                f" got {return_value!r}"
            )

        if value is None:
            if return_value == "point":
                return {"lat": None, "lon": None}
            if return_value == "struct":
                return {
                    "gtype": None, "wkb": None, "srid": None,
                    "country_iso": None, "country_name": None,
                    "city_iso": None, "city_name": None,
                    "key": None, "name": None, "eic": None,
                    "tz": None, "ccy": None, "lat": None, "lon": None,
                }
            return None

        wkb_to_meta, known_wkb = cls._build_bin_lookup_cache()

        b = bytes(value)

        if return_value == "dataclass":
            meta = wkb_to_meta.get(b)
            if meta is None:
                return None
            return cls.get_by_geom(b, srid=meta.srid)

        if return_value == "wkb":
            return b if b in known_wkb else None

        meta = wkb_to_meta.get(b)

        if return_value == "country_iso":
            return meta.country_iso if meta is not None else None

        if return_value == "city_iso":
            return meta.city_iso if meta is not None else None

        if return_value == "point":
            if meta is None:
                return {"lat": None, "lon": None}
            lat, lon = _parse_point_wkb(meta.wkb)
            return {"lat": lat, "lon": lon}

        # return_value == "struct"
        if meta is None:
            return {
                "gtype": None, "wkb": None, "srid": None,
                "country_iso": None, "country_name": None,
                "city_iso": None, "city_name": None,
                "key": None, "name": None, "eic": None,
                "tz": None, "ccy": None, "lat": None, "lon": None,
            }
        lat, lon = _parse_point_wkb(meta.wkb)
        return {
            "gtype": meta.gtype,
            "wkb": meta.wkb,
            "srid": meta.srid,
            "country_iso": meta.country_iso,
            "country_name": meta.country_name,
            "city_iso": meta.city_iso,
            "city_name": meta.city_name,
            "key": meta.key,
            "name": meta.name,
            "eic": meta.eic,
            "tz": meta.tz,
            "ccy": meta.ccy,
            "lat": lat,
            "lon": lon,
        }

    @classmethod
    def _build_output_expr_from_key(
        cls,
        ck: "pl.Expr",
        return_value: "Literal['wkb', 'country_iso', 'city_iso', 'point', 'struct']",
        key_to_meta: "dict[str, _ZoneMeta]",
    ) -> "pl.Expr":
        """Shared expression builder: canonical-key Expr → output Expr.

        Derives flat ``dict[str, V]`` projections from *key_to_meta* on demand;
        these small per-field dicts are only materialised when needed for the
        chosen *return_value* and are not cached themselves.
        """
        import polars as pl

        if return_value == "wkb":
            key_to_wkb = {k: m.wkb for k, m in key_to_meta.items()}
            return ck.replace_strict(key_to_wkb, default=None).cast(pl.Binary)

        if return_value == "country_iso":
            key_to_country_iso = {k: m.country_iso for k, m in key_to_meta.items()}
            return ck.replace_strict(key_to_country_iso, default=None).cast(pl.Utf8)

        if return_value == "city_iso":
            key_to_city_iso = {k: m.city_iso for k, m in key_to_meta.items()}
            return ck.replace_strict(key_to_city_iso, default=None).cast(pl.Utf8)

        if return_value == "point":
            key_to_lat = {k: _parse_point_wkb(m.wkb)[0] for k, m in key_to_meta.items()}
            key_to_lon = {k: _parse_point_wkb(m.wkb)[1] for k, m in key_to_meta.items()}
            lat_expr = ck.replace_strict(key_to_lat, default=None).cast(pl.Float64)
            lon_expr = ck.replace_strict(key_to_lon, default=None).cast(pl.Float64)
            return pl.struct(lat=lat_expr, lon=lon_expr)

        if return_value == "struct":
            key_to_wkb         = {k: m.wkb          for k, m in key_to_meta.items()}
            key_to_gtype       = {k: m.gtype         for k, m in key_to_meta.items()}
            key_to_srid        = {k: m.srid          for k, m in key_to_meta.items()}
            key_to_country_iso = {k: m.country_iso   for k, m in key_to_meta.items()}
            key_to_country_name= {k: m.country_name  for k, m in key_to_meta.items()}
            key_to_city_iso    = {k: m.city_iso      for k, m in key_to_meta.items()}
            key_to_city_name   = {k: m.city_name     for k, m in key_to_meta.items()}
            key_to_name        = {k: m.name          for k, m in key_to_meta.items()}
            key_to_eic         = {k: m.eic           for k, m in key_to_meta.items()}
            key_to_tz          = {k: m.tz            for k, m in key_to_meta.items()}
            key_to_ccy         = {k: m.ccy           for k, m in key_to_meta.items()}
            key_to_lat         = {k: m.lat for k, m in key_to_meta.items()}
            key_to_lon         = {k: m.lon for k, m in key_to_meta.items()}
            return pl.struct(
                # Field order mirrors GeoZone dataclass declaration exactly.
                # aliases is a tuple[str, ...] and has no scalar Polars equivalent
                # so it is intentionally omitted.
                gtype=ck.replace_strict(key_to_gtype, default=None).cast(pl.Int32),
                wkb=ck.replace_strict(key_to_wkb, default=None).cast(pl.Binary),
                srid=ck.replace_strict(key_to_srid, default=None).cast(pl.Int32),
                country_iso=ck.replace_strict(key_to_country_iso, default=None).cast(pl.Utf8),
                country_name=ck.replace_strict(key_to_country_name, default=None).cast(pl.Utf8),
                city_iso=ck.replace_strict(key_to_city_iso, default=None).cast(pl.Utf8),
                city_name=ck.replace_strict(key_to_city_name, default=None).cast(pl.Utf8),
                key=ck.cast(pl.Utf8),
                name=ck.replace_strict(key_to_name, default=None).cast(pl.Utf8),
                eic=ck.replace_strict(key_to_eic, default=None).cast(pl.Utf8),
                tz=ck.replace_strict(key_to_tz, default=None).cast(pl.Utf8),
                ccy=ck.replace_strict(key_to_ccy, default=None).cast(pl.Utf8),
                lat=ck.replace_strict(key_to_lat, default=None).cast(pl.Float64),
                lon=ck.replace_strict(key_to_lon, default=None).cast(pl.Float64),
            )

        raise ValueError(
            f"return_value must be one of 'wkb', 'country_iso', 'city_iso', 'point', 'struct'; got {return_value!r}"
        )

    @classmethod
    def polars_parse_str(
        cls,
        col: "pl.Series | pl.Expr",
        *,
        return_value: Literal["wkb", "country_iso", "city_iso", "point", "struct"] = "wkb",
        lazy: bool = False,
    ) -> "pl.Series | pl.Expr":
        """Parse a string column into a zone field.

        Parameters
        ----------
        col:
            A ``pl.Series`` (Utf8) or a ``pl.Expr``.
        return_value:
            ``"wkb"`` (default) · ``"country_iso"`` · ``"city_iso"`` · ``"point"`` ·
            ``"struct"`` (all GeoZone fields as a Polars Struct).
        lazy:
            When *True* and *col* is a ``pl.Series``, return a ``pl.Expr``
            instead of evaluating immediately.  When *False* (default) a
            ``pl.Series`` input always produces a ``pl.Series``.
        """
        import polars as pl

        chunk_patterns, alias_to_key, key_to_meta = cls._build_bidding_zone_regex_cache()

        def _normalize_expr(expr: pl.Expr) -> pl.Expr:
            return (
                expr.cast(pl.Utf8)
                .str.strip_chars()
                .str.to_uppercase()
                .str.replace_all(r"[\s\-/|]+", "_")
                .str.replace_all(r"_+", "_")
                .str.strip_chars("_")
            )

        def _canonical_key_expr(expr: pl.Expr) -> pl.Expr:
            norm = _normalize_expr(expr)

            # Pass 1: direct alias-map replace on the whole normalised string.
            exact = norm.replace_strict(alias_to_key, default=None)

            # Pass 2: token-boundary scan for free-text strings like
            # "Sweden SE1 wind power" where no whole-string alias matches.
            # We run one small str.extract() per chunk and coalesce the results.
            # Each chunk regex is ≤200 alternatives → safe regex engine depth.
            if chunk_patterns:
                token_candidates = [
                    norm.str.extract(pat, group_index=1)
                    for pat in chunk_patterns
                ]
                token_alias = pl.coalesce(token_candidates)
                token_key = token_alias.replace_strict(alias_to_key, default=None)
                return pl.coalesce([exact, token_key])

            return exact

        def _build_expr(expr: pl.Expr) -> pl.Expr:
            return cls._build_output_expr_from_key(
                _canonical_key_expr(expr),
                return_value,
                key_to_meta,
            )

        if isinstance(col, pl.Expr):
            return _build_expr(col)

        if isinstance(col, pl.Series):
            expr = _build_expr(pl.col(col.name)).alias(col.name)
            if lazy:
                return expr
            return col.to_frame().select(expr).to_series()

        raise TypeError(f"expected pl.Series | pl.Expr, got {type(col)!r}")

    @classmethod
    @lru_cache(maxsize=1)
    def _build_bin_lookup_cache(
        cls,
    ) -> tuple[
        dict[bytes, "_ZoneMeta"],   # wkb → _ZoneMeta  (replaces 12 separate field dicts)
        frozenset[bytes],           # set of known canonical WKB values (replaces identity wkb→wkb dict)
    ]:
        """Build direct wkb→field maps.

        WKB is the unique key for every registered zone, so all field lookups
        for polars_parse_bin are a single replace_strict on the Binary column —
        no intermediate string canonical-key step required.

        Returns a compact pair instead of the previous 14-element tuple:
        - ``wkb_to_meta`` – wkb bytes → :class:`_ZoneMeta`.
        - ``known_wkb`` – frozenset of all registered WKB values (for the
          pass-through ``"wkb"`` return mode without an identity dict).
        """
        _chunk_patterns, _alias_to_key, key_to_meta = cls._build_bidding_zone_regex_cache()

        wkb_to_meta: dict[bytes, _ZoneMeta] = {}

        for _k, meta in key_to_meta.items():
            wkb = meta.wkb
            if wkb not in wkb_to_meta:
                wkb_to_meta[wkb] = meta

        return wkb_to_meta, frozenset(wkb_to_meta)

    @classmethod
    def polars_parse_bin(
        cls,
        col: "pl.Series | pl.Expr",
        *,
        return_value: Literal["wkb", "country_iso", "city_iso", "point", "struct"] = "wkb",
        lazy: bool = False,
    ) -> "pl.Series | pl.Expr":
        """Parse a WKB binary column into a zone field.

        Parameters
        ----------
        col:
            A ``pl.Series`` (Binary) or a ``pl.Expr`` that resolves to Binary.
            Each value must be a little-endian WKB POINT as produced by
            ``GeoZone.wkb`` (21 bytes).  Unknown / unregistered WKB values
            map to ``None``.
        return_value:
            ``"wkb"`` (default, pass-through canonical WKB) · ``"country_iso"`` ·
            ``"city_iso"`` · ``"point"`` · ``"struct"`` (all GeoZone fields).
        lazy:
            When *True* and *col* is a ``pl.Series``, return a ``pl.Expr``
            instead of evaluating immediately.
        """
        import polars as pl

        wkb_to_meta, known_wkb = cls._build_bin_lookup_cache()

        def _build_expr(expr: pl.Expr) -> pl.Expr:
            # Cast once; every branch does a single replace_strict on raw WKB.
            b = expr.cast(pl.Binary)

            if return_value == "wkb":
                # Pass-through: keep value when known, else null.
                wkb_identity = {w: w for w in known_wkb}
                return b.replace_strict(wkb_identity, default=None).cast(pl.Binary)

            if return_value == "country_iso":
                wkb_to_country_iso = {w: m.country_iso for w, m in wkb_to_meta.items()}
                return b.replace_strict(wkb_to_country_iso, default=None).cast(pl.Utf8)

            if return_value == "city_iso":
                wkb_to_city_iso = {w: m.city_iso for w, m in wkb_to_meta.items()}
                return b.replace_strict(wkb_to_city_iso, default=None).cast(pl.Utf8)

            if return_value == "point":
                wkb_to_lat = {w: _parse_point_wkb(w)[0] for w in known_wkb}
                wkb_to_lon = {w: _parse_point_wkb(w)[1] for w in known_wkb}
                lat_expr = b.replace_strict(wkb_to_lat, default=None).cast(pl.Float64)
                lon_expr = b.replace_strict(wkb_to_lon, default=None).cast(pl.Float64)
                return pl.struct(lat=lat_expr, lon=lon_expr)

            if return_value == "struct":
                wkb_to_gtype        = {w: m.gtype        for w, m in wkb_to_meta.items()}
                wkb_identity        = {w: w               for w in known_wkb}
                wkb_to_srid         = {w: m.srid         for w, m in wkb_to_meta.items()}
                wkb_to_country_iso  = {w: m.country_iso  for w, m in wkb_to_meta.items()}
                wkb_to_country_name = {w: m.country_name for w, m in wkb_to_meta.items()}
                wkb_to_city_iso     = {w: m.city_iso     for w, m in wkb_to_meta.items()}
                wkb_to_city_name    = {w: m.city_name    for w, m in wkb_to_meta.items()}
                wkb_to_key          = {w: m.key          for w, m in wkb_to_meta.items()}
                wkb_to_name         = {w: m.name         for w, m in wkb_to_meta.items()}
                wkb_to_eic          = {w: m.eic          for w, m in wkb_to_meta.items()}
                wkb_to_tz           = {w: m.tz           for w, m in wkb_to_meta.items()}
                wkb_to_ccy          = {w: m.ccy          for w, m in wkb_to_meta.items()}
                wkb_to_lat          = {w: _parse_point_wkb(w)[0] for w in known_wkb}
                wkb_to_lon          = {w: _parse_point_wkb(w)[1] for w in known_wkb}
                return pl.struct(
                    # Field order mirrors GeoZone dataclass declaration exactly.
                    gtype=b.replace_strict(wkb_to_gtype, default=None).cast(pl.Int32),
                    wkb=b.replace_strict(wkb_identity, default=None).cast(pl.Binary),
                    srid=b.replace_strict(wkb_to_srid, default=None).cast(pl.Int32),
                    country_iso=b.replace_strict(wkb_to_country_iso, default=None).cast(pl.Utf8),
                    country_name=b.replace_strict(wkb_to_country_name, default=None).cast(pl.Utf8),
                    city_iso=b.replace_strict(wkb_to_city_iso, default=None).cast(pl.Utf8),
                    city_name=b.replace_strict(wkb_to_city_name, default=None).cast(pl.Utf8),
                    key=b.replace_strict(wkb_to_key, default=None).cast(pl.Utf8),
                    name=b.replace_strict(wkb_to_name, default=None).cast(pl.Utf8),
                    eic=b.replace_strict(wkb_to_eic, default=None).cast(pl.Utf8),
                    tz=b.replace_strict(wkb_to_tz, default=None).cast(pl.Utf8),
                    ccy=b.replace_strict(wkb_to_ccy, default=None).cast(pl.Utf8),
                    lat=b.replace_strict(wkb_to_lat, default=None).cast(pl.Float64),
                    lon=b.replace_strict(wkb_to_lon, default=None).cast(pl.Float64),
                )

            raise ValueError(
                f"return_value must be one of 'wkb', 'country_iso', 'city_iso', 'point', 'struct';"
                f" got {return_value!r}"
            )

        if isinstance(col, pl.Expr):
            return _build_expr(col)

        if isinstance(col, pl.Series):
            expr = _build_expr(pl.col(col.name)).alias(col.name)
            if lazy:
                return expr
            return col.to_frame().select(expr).to_series()

        raise TypeError(f"expected pl.Series | pl.Expr, got {type(col)!r}")
