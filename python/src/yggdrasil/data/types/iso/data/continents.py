"""ISO-style continent codes (not a formal ISO standard, but widely used)."""
from __future__ import annotations

__all__ = ["CONTINENTS", "CONTINENT_ALIASES"]


# (code_2, name)
CONTINENTS: tuple[tuple[str, str], ...] = (
    ("AF", "Africa"),
    ("AN", "Antarctica"),
    ("AS", "Asia"),
    ("EU", "Europe"),
    ("NA", "North America"),
    ("OC", "Oceania"),
    ("SA", "South America"),
)


# common aliases -> code_2
CONTINENT_ALIASES: dict[str, str] = {
    "AFRICA": "AF",
    "ANTARCTICA": "AN",
    "ASIA": "AS",
    "EUROPE": "EU",
    "NORTH AMERICA": "NA",
    "NORTHAMERICA": "NA",
    "NORTH_AMERICA": "NA",
    "SOUTH AMERICA": "SA",
    "SOUTHAMERICA": "SA",
    "SOUTH_AMERICA": "SA",
    "OCEANIA": "OC",
    "AUSTRALIA": "OC",
    "AMERICAS": "NA",
}
