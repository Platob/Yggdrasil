"""ISO-style continent code type (2-letter continent code)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from .base import ISOType
from .data import CONTINENTS, CONTINENT_ALIASES
from .data.continents import CONTINENT_ALIASES as _ALIASES

__all__ = ["ISOContinentType"]


def _build_continent_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for code, name in CONTINENTS:
        mapping[code] = code
        mapping[name.upper()] = code
        mapping[name.upper().replace(" ", "")] = code
    for alias, code in CONTINENT_ALIASES.items():
        mapping[alias] = code
    return mapping


_CONTINENT_MAP: dict[str, str] = _build_continent_map()
_VALID_CODES: frozenset[str] = frozenset(code for code, _ in CONTINENTS)


@dataclass(frozen=True)
class ISOContinentType(ISOType):
    """Continent code in ISO-style 2-letter form (e.g. ``EU``, ``NA``, ``AS``).

    Accepts common aliases (``EUROPE``, ``NORTH AMERICA``, ``EU``, …)
    and normalizes to a canonical 2-letter code.
    """

    iso_name: ClassVar[str] = "iso_continent"

    def _resolve_token(self, token: str) -> str | None:
        if token in _VALID_CODES:
            return token
        direct = _CONTINENT_MAP.get(token)
        if direct is not None:
            return direct
        # Drop spaces and retry (e.g. "NORTH AMERICA" -> "NORTHAMERICA").
        squashed = token.replace(" ", "")
        return _CONTINENT_MAP.get(squashed)

    @classmethod
    def _build_lookup_map(cls) -> Mapping[str, str]:
        return _CONTINENT_MAP

    # ------------------------------------------------------------------
    # Dict
    # ------------------------------------------------------------------
    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        name = str(value.get("name", "")).upper()
        iso = str(value.get("iso", "")).lower()
        return name in {"ISOCONTINENTTYPE", "ISO_CONTINENT"} or iso == cls.iso_name
