"""Geographic feature zones: mountain ranges, massifs, passes, and rivers.

Coordinates are representative points (approximate centroid or highest-peak
location) in WGS-84 (EPSG:4326).  All zones use ``GeoZoneType.ZONE``.

Sources
-------
* Wikipedia – individual mountain / range / pass / river articles
* OpenStreetMap nominatim – cross-checked centroids
"""
from __future__ import annotations

from typing import Optional

from .builders import _validate_unique_attrs, _from_coordinates_with_optional_metadata
from .geozone import GeoZone, GeoZoneType

__all__ = ["load_mountains"]

_SRC = "seed: module (geographic feature representative point; Wikipedia/OSM cross-check)"


def _feature(
    key: str,
    name: str,
    lat: float,
    lon: float,
    *,
    aliases: tuple[str, ...] = (),
    country_iso: Optional[str] = None,
    country_name: Optional[str] = None,
    tz: Optional[str] = None,
    confidence: str = "medium",
    coord_kind: str = "representative_point",
) -> GeoZone:
    """Register a geographic feature as a ``GeoZoneType.ZONE``."""
    return GeoZone.put(
        _from_coordinates_with_optional_metadata(
            gtype=GeoZoneType.ZONE,
            lat=lat,
            lon=lon,
            key=key,
            aliases=aliases,
            name=name,
            country_iso=country_iso,
            country_name=country_name,
            tz=tz,
            ccy=None,
            coord_source=_SRC,
            coord_kind=coord_kind,
            confidence=confidence,
        )
    )


# ---------------------------------------------------------------------------
# Table columns: attr, key, name, lat, lon, kwargs
# ---------------------------------------------------------------------------

# ── Mountain ranges ─────────────────────────────────────────────────────────

MOUNTAIN_RANGES = [
    # Alps – the main arc (CH/AT/IT/FR/DE/LI/SI)
    ("ALPS", "ALPS", "Alps",
     46.5000, 10.5000,
     {"aliases": ("THE_ALPS", "ALPINE"), "confidence": "high"}),

    # Western Alps
    ("WESTERN_ALPS", "WESTERN_ALPS", "Western Alps",
     44.8000, 7.0000,
     {"aliases": ("ALPS_WEST",)}),

    # Central Alps
    ("CENTRAL_ALPS", "CENTRAL_ALPS", "Central Alps",
     46.6000, 9.5000,
     {"aliases": ("ALPS_CENTRAL",)}),

    # Eastern Alps
    ("EASTERN_ALPS", "EASTERN_ALPS", "Eastern Alps",
     47.2000, 13.5000,
     {"aliases": ("ALPS_EAST",)}),

    # Swiss Alps (centroid of the Swiss portion)
    ("SWISS_ALPS", "SWISS_ALPS", "Swiss Alps",
     46.5500, 8.1000,
     {"aliases": ("SCHWEIZER_ALPEN",), "country_iso": "CH", "country_name": "Switzerland",
      "tz": "Europe/Zurich"}),

    # French Alps
    ("FRENCH_ALPS", "FRENCH_ALPS", "French Alps",
     45.3000, 6.4000,
     {"aliases": ("ALPES_FRANCAISES",), "country_iso": "FR", "country_name": "France",
      "tz": "Europe/Paris"}),

    # Italian Alps
    ("ITALIAN_ALPS", "ITALIAN_ALPS", "Italian Alps",
     46.0000, 11.5000,
     {"aliases": ("ALPI_ITALIANE",), "country_iso": "IT", "country_name": "Italy",
      "tz": "Europe/Rome"}),

    # Austrian Alps
    ("AUSTRIAN_ALPS", "AUSTRIAN_ALPS", "Austrian Alps",
     47.2000, 13.0000,
     {"aliases": ("OESTERREICHISCHE_ALPEN",), "country_iso": "AT", "country_name": "Austria",
      "tz": "Europe/Vienna"}),

    # Pennine Alps (CH/IT; highest massif of the Alps)
    ("PENNINE_ALPS", "PENNINE_ALPS", "Pennine Alps",
     45.9766, 7.6586,
     {"aliases": ("VALAIS_ALPS", "WALLIS_ALPEN")}),

    # Bernese Alps (CH)
    ("BERNESE_ALPS", "BERNESE_ALPS", "Bernese Alps",
     46.5782, 7.9800,
     {"aliases": ("BERNESE_OBERLAND", "BERNER_ALPEN"), "country_iso": "CH",
      "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Lepontine Alps (CH/IT)
    ("LEPONTINE_ALPS", "LEPONTINE_ALPS", "Lepontine Alps",
     46.4500, 8.9000,
     {"aliases": ("LEPONTIC_ALPS",)}),

    # Graian Alps (FR/IT)
    ("GRAIAN_ALPS", "GRAIAN_ALPS", "Graian Alps",
     45.5000, 7.1000,
     {"aliases": ("ALPES_GRAIENNES",)}),

    # Cottian Alps (FR/IT)
    ("COTTIAN_ALPS", "COTTIAN_ALPS", "Cottian Alps",
     44.6500, 6.8000,
     {"aliases": ("ALPES_COTTIENNES",)}),

    # Maritime Alps (FR/IT)
    ("MARITIME_ALPS", "MARITIME_ALPS", "Maritime Alps",
     44.1500, 7.1000,
     {"aliases": ("ALPES_MARITIMES_RANGE",)}),

    # Rhaetian Alps (CH/AT/IT)
    ("RHAETIAN_ALPS", "RHAETIAN_ALPS", "Rhaetian Alps",
     46.5000, 10.0000,
     {"aliases": ("RAETISCHE_ALPEN",)}),

    # Glarus Alps (CH)
    ("GLARUS_ALPS", "GLARUS_ALPS", "Glarus Alps",
     46.9000, 9.0000,
     {"country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Ötztaler Alps (AT/IT)
    ("OTZTAL_ALPS", "OTZTAL_ALPS", "Ötztal Alps",
     46.8300, 10.8500,
     {"aliases": ("OTZTALER_ALPEN",)}),

    # Stubai Alps (AT)
    ("STUBAI_ALPS", "STUBAI_ALPS", "Stubai Alps",
     47.0000, 11.2000,
     {"aliases": ("STUBAIER_ALPEN",), "country_iso": "AT", "country_name": "Austria",
      "tz": "Europe/Vienna"}),

    # Tauern (High Tauern + Low Tauern, AT)
    ("HIGH_TAUERN", "HIGH_TAUERN", "High Tauern",
     47.0742, 12.6950,
     {"aliases": ("HOHE_TAUERN",), "country_iso": "AT", "country_name": "Austria",
      "tz": "Europe/Vienna"}),

    ("LOW_TAUERN", "LOW_TAUERN", "Low Tauern",
     47.3000, 14.2000,
     {"aliases": ("NIEDERE_TAUERN",), "country_iso": "AT", "country_name": "Austria",
      "tz": "Europe/Vienna"}),

    # Dolomites (IT)
    ("DOLOMITES", "DOLOMITES", "Dolomites",
     46.4102, 11.8440,
     {"aliases": ("DOLOMITI",), "country_iso": "IT", "country_name": "Italy",
      "tz": "Europe/Rome", "confidence": "high"}),

    # Apennines (IT)
    ("APENNINES", "APENNINES", "Apennine Mountains",
     43.1500, 13.1000,
     {"aliases": ("APENNINI",), "country_iso": "IT", "country_name": "Italy",
      "tz": "Europe/Rome"}),

    # Pyrenees (FR/ES/AD)
    ("PYRENEES", "PYRENEES", "Pyrenees",
     42.7500, 1.0000,
     {"aliases": ("PIRINEOS", "PIRENEES"), "confidence": "high"}),

    # Carpathians (PL/SK/CZ/UA/RO/RS/AT)
    ("CARPATHIANS", "CARPATHIANS", "Carpathian Mountains",
     49.0000, 23.0000,
     {"aliases": ("CARPATES", "KARPATEN"), "confidence": "high"}),

    ("TATRA", "TATRA", "Tatra Mountains",
     49.2292, 19.9808,
     {"aliases": ("TATRY", "HIGH_TATRA")}),

    # Scandinavian Mountains
    ("SCANDINAVIAN_MOUNTAINS", "SCANDINAVIAN_MOUNTAINS", "Scandinavian Mountains",
     63.0000, 13.5000,
     {"aliases": ("SCANDES", "KJOLEN")}),

    # Black Forest (DE)
    ("BLACK_FOREST", "BLACK_FOREST", "Black Forest",
     48.2000, 8.2000,
     {"aliases": ("SCHWARZWALD",), "country_iso": "DE", "country_name": "Germany",
      "tz": "Europe/Berlin"}),

    # Vosges (FR)
    ("VOSGES", "VOSGES", "Vosges Mountains",
     48.1500, 7.1000,
     {"country_iso": "FR", "country_name": "France", "tz": "Europe/Paris"}),

    # Jura (FR/CH)
    ("JURA", "JURA", "Jura Mountains",
     47.0000, 6.5000,
     {"aliases": ("JURA_MOUNTAINS",)}),

    # Massif Central (FR)
    ("MASSIF_CENTRAL", "MASSIF_CENTRAL", "Massif Central",
     45.5000, 3.0000,
     {"country_iso": "FR", "country_name": "France", "tz": "Europe/Paris"}),

    # Caucasus
    ("CAUCASUS", "CAUCASUS", "Caucasus Mountains",
     42.6000, 44.0000,
     {"aliases": ("KAVKAZ",), "confidence": "high"}),

    ("GREATER_CAUCASUS", "GREATER_CAUCASUS", "Greater Caucasus",
     43.1000, 44.0000,
     {"aliases": ("BOLSHOY_KAVKAZ",)}),

    ("LESSER_CAUCASUS", "LESSER_CAUCASUS", "Lesser Caucasus",
     40.5000, 44.5000,
     {"aliases": ("MALYI_KAVKAZ",)}),

    # Ural (RU)
    ("URAL", "URAL", "Ural Mountains",
     60.0000, 59.5000,
     {"aliases": ("URALS",), "country_iso": "RU", "country_name": "Russia"}),

    # Balkans (BGR/SRB/MK)
    ("BALKAN_MOUNTAINS", "BALKAN_MOUNTAINS", "Balkan Mountains",
     42.8500, 25.3000,
     {"aliases": ("STARA_PLANINA",)}),

    # Rhodopes (BG/GR)
    ("RHODOPES", "RHODOPES", "Rhodope Mountains",
     41.8000, 24.5000,
     {"aliases": ("RHODOPI",)}),

    # Atlas (MA/DZ/TN)
    ("ATLAS", "ATLAS", "Atlas Mountains",
     31.0000, -5.0000,
     {"aliases": ("ATLAS_MOUNTAINS",), "confidence": "high"}),

    # Himalayas
    ("HIMALAYAS", "HIMALAYAS", "Himalayas",
     28.0000, 84.0000,
     {"aliases": ("HIMALAYA",), "confidence": "high"}),

    # Andes
    ("ANDES", "ANDES", "Andes",
     -20.0000, -68.0000,
     {"aliases": ("CORDILLERA_DE_LOS_ANDES",), "confidence": "high"}),

    # Rocky Mountains
    ("ROCKY_MOUNTAINS", "ROCKY_MOUNTAINS", "Rocky Mountains",
     43.0000, -110.0000,
     {"aliases": ("ROCKIES",), "country_iso": "US", "country_name": "United States",
      "tz": "America/Denver", "confidence": "high"}),
]


# ── Individual peaks ─────────────────────────────────────────────────────────

PEAKS = [
    # Mont Blanc – highest peak in the Alps and Western Europe (FR/IT border)
    ("MONT_BLANC", "MONT_BLANC", "Mont Blanc",
     45.8326, 6.8652,
     {"aliases": ("MONTE_BIANCO",), "coord_kind": "summit", "confidence": "high"}),

    # Monte Rosa – highest entirely in Switzerland
    ("MONTE_ROSA", "MONTE_ROSA", "Monte Rosa",
     45.9374, 7.8679,
     {"aliases": ("DUFOURSPITZE",), "coord_kind": "summit", "confidence": "high"}),

    # Matterhorn (CH/IT)
    ("MATTERHORN", "MATTERHORN", "Matterhorn",
     45.9766, 7.6586,
     {"aliases": ("CERVINO", "CERVIN"), "coord_kind": "summit", "confidence": "high",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Jungfrau (CH)
    ("JUNGFRAU", "JUNGFRAU", "Jungfrau",
     46.5371, 7.9626,
     {"coord_kind": "summit", "confidence": "high",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Eiger (CH)
    ("EIGER", "EIGER", "Eiger",
     46.5775, 8.0054,
     {"coord_kind": "summit", "confidence": "high",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Mönch (CH)
    ("MONCH", "MONCH", "Mönch",
     46.5589, 7.9986,
     {"aliases": ("MOENCH",), "coord_kind": "summit", "confidence": "high",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Piz Bernina – highest peak in the Rhaetian Alps (CH)
    ("PIZ_BERNINA", "PIZ_BERNINA", "Piz Bernina",
     46.3783, 9.9088,
     {"aliases": ("BERNINA_PEAK",), "coord_kind": "summit", "confidence": "high",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Grossglockner – highest peak in Austria
    ("GROSSGLOCKNER", "GROSSGLOCKNER", "Grossglockner",
     47.0742, 12.6950,
     {"aliases": ("GROSS_GLOCKNER",), "coord_kind": "summit", "confidence": "high",
      "country_iso": "AT", "country_name": "Austria", "tz": "Europe/Vienna"}),

    # Gran Paradiso – highest entirely in Italy
    ("GRAN_PARADISO", "GRAN_PARADISO", "Gran Paradiso",
     45.5173, 7.2699,
     {"coord_kind": "summit", "confidence": "high",
      "country_iso": "IT", "country_name": "Italy", "tz": "Europe/Rome"}),

    # Ortler (IT)
    ("ORTLER", "ORTLER", "Ortler",
     46.5075, 10.5436,
     {"aliases": ("ORTLES",), "coord_kind": "summit",
      "country_iso": "IT", "country_name": "Italy", "tz": "Europe/Rome"}),

    # Zugspitze – highest peak in Germany
    ("ZUGSPITZE", "ZUGSPITZE", "Zugspitze",
     47.4211, 10.9850,
     {"coord_kind": "summit", "confidence": "high",
      "country_iso": "DE", "country_name": "Germany", "tz": "Europe/Berlin"}),

    # Pic du Midi d'Ossau (FR/ES, Pyrenees)
    ("PIC_DU_MIDI", "PIC_DU_MIDI", "Pic du Midi d'Ossau",
     42.8417, -0.4386,
     {"aliases": ("PIC_DU_MIDI_OSSAU",), "coord_kind": "summit",
      "country_iso": "FR", "country_name": "France", "tz": "Europe/Paris"}),

    # Elbrus – highest peak in Europe (RU, Caucasus)
    ("ELBRUS", "ELBRUS", "Mount Elbrus",
     43.3499, 42.4453,
     {"aliases": ("MOUNT_ELBRUS",), "coord_kind": "summit", "confidence": "high",
      "country_iso": "RU", "country_name": "Russia"}),

    # Everest
    ("EVEREST", "EVEREST", "Mount Everest",
     27.9881, 86.9250,
     {"aliases": ("MT_EVEREST", "SAGARMATHA", "CHOMOLUNGMA"),
      "coord_kind": "summit", "confidence": "high"}),

    # Aconcagua
    ("ACONCAGUA", "ACONCAGUA", "Aconcagua",
     -32.6532, -70.0109,
     {"coord_kind": "summit", "confidence": "high",
      "country_iso": "AR", "country_name": "Argentina"}),
]


# ── Alpine passes ─────────────────────────────────────────────────────────────

PASSES = [
    # Great St Bernard Pass (CH/IT)
    ("GREAT_ST_BERNARD", "GREAT_ST_BERNARD", "Great St Bernard Pass",
     45.8695, 7.1706,
     {"aliases": ("GRAN_SAN_BERNARDO", "GRAND_SAINT_BERNARD"),
      "coord_kind": "pass", "confidence": "high"}),

    # Little St Bernard Pass (FR/IT)
    ("LITTLE_ST_BERNARD", "LITTLE_ST_BERNARD", "Little St Bernard Pass",
     45.6847, 6.8844,
     {"aliases": ("PETIT_SAINT_BERNARD",), "coord_kind": "pass"}),

    # Simplon Pass (CH/IT)
    ("SIMPLON", "SIMPLON", "Simplon Pass",
     46.2535, 8.0331,
     {"aliases": ("SEMPIONE", "SIMPLON_PASS"), "coord_kind": "pass", "confidence": "high",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Gotthard Pass (CH)
    ("GOTTHARD", "GOTTHARD", "Gotthard Pass",
     46.5568, 8.5647,
     {"aliases": ("ST_GOTTHARD", "SAN_GOTTARDO", "SAINT_GOTTHARD"),
      "coord_kind": "pass", "confidence": "high",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Furka Pass (CH)
    ("FURKA", "FURKA", "Furka Pass",
     46.5736, 8.4154,
     {"aliases": ("FURKA_PASS",), "coord_kind": "pass",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Grimsel Pass (CH)
    ("GRIMSEL", "GRIMSEL", "Grimsel Pass",
     46.5609, 8.3350,
     {"aliases": ("GRIMSEL_PASS",), "coord_kind": "pass",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Susten Pass (CH)
    ("SUSTEN", "SUSTEN", "Susten Pass",
     46.7287, 8.4523,
     {"aliases": ("SUSTEN_PASS",), "coord_kind": "pass",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Nufenen Pass (CH)
    ("NUFENEN", "NUFENEN", "Nufenen Pass",
     46.4786, 8.3836,
     {"aliases": ("NOVENA",), "coord_kind": "pass",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Maloja Pass (CH)
    ("MALOJA", "MALOJA", "Maloja Pass",
     46.4026, 9.6956,
     {"aliases": ("MALOJA_PASS",), "coord_kind": "pass",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Bernina Pass (CH)
    ("BERNINA", "BERNINA", "Bernina Pass",
     46.4118, 10.0297,
     {"aliases": ("BERNINA_PASS",), "coord_kind": "pass",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Julier Pass (CH)
    ("JULIER", "JULIER", "Julier Pass",
     46.4722, 9.7354,
     {"aliases": ("JULIER_PASS", "PASSO_DEL_GIULIA"), "coord_kind": "pass",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Splügen Pass (CH/IT)
    ("SPLUGEN", "SPLUGEN", "Splügen Pass",
     46.5044, 9.3272,
     {"aliases": ("SPLUGEN_PASS", "PASSO_DELLO_SPLUGA"), "coord_kind": "pass"}),

    # Brenner Pass (AT/IT) – major trans-Alpine route
    ("BRENNER", "BRENNER", "Brenner Pass",
     47.0067, 11.5069,
     {"aliases": ("BRENNER_PASS", "PASSO_DEL_BRENNERO"),
      "coord_kind": "pass", "confidence": "high"}),

    # Reschen Pass (AT/IT)
    ("RESCHEN", "RESCHEN", "Reschen Pass",
     46.8517, 10.5358,
     {"aliases": ("PASSO_DI_RESIA",), "coord_kind": "pass"}),

    # Stelvio Pass (IT) – highest paved road in the Alps (2757 m)
    ("STELVIO", "STELVIO", "Stelvio Pass",
     46.5284, 10.4537,
     {"aliases": ("PASSO_DELLO_STELVIO",), "coord_kind": "pass",
      "country_iso": "IT", "country_name": "Italy", "tz": "Europe/Rome"}),

    # Col du Mont Cenis (FR/IT)
    ("MONT_CENIS", "MONT_CENIS", "Col du Mont Cenis",
     45.2614, 6.9028,
     {"aliases": ("COL_DU_MONT_CENIS",), "coord_kind": "pass"}),

    # Col de la Faucille (FR)
    ("FAUCILLE", "FAUCILLE", "Col de la Faucille",
     46.3667, 6.0269,
     {"aliases": ("COL_DE_LA_FAUCILLE",), "coord_kind": "pass",
      "country_iso": "FR", "country_name": "France", "tz": "Europe/Paris"}),
]


# ── Rivers ───────────────────────────────────────────────────────────────────

RIVERS = [
    # Rhone – source in Valais (CH), flows through Lake Geneva, south through France to Med
    ("RHONE", "RHONE", "Rhône",
     45.0000, 5.5000,
     {"aliases": ("RHONE_RIVER", "FLEUVE_RHONE"),
      "coord_kind": "river_centroid", "confidence": "high"}),

    # Rhone source (Rhône Glacier, CH)
    ("RHONE_SOURCE", "RHONE_SOURCE", "Rhône Source",
     46.5733, 8.3733,
     {"aliases": ("RHONE_GLACIER",), "coord_kind": "source",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Rhine – source in CH (Graubünden), flows via Germany/France to North Sea
    ("RHINE", "RHINE", "Rhine",
     49.0000, 8.0000,
     {"aliases": ("RHEIN", "RHIN", "RHINE_RIVER"),
      "coord_kind": "river_centroid", "confidence": "high"}),

    # Rhine source (Lake Toma, CH)
    ("RHINE_SOURCE", "RHINE_SOURCE", "Rhine Source",
     46.6389, 8.6833,
     {"aliases": ("LAKE_TOMA",), "coord_kind": "source",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Aare (CH)
    ("AARE", "AARE", "Aare",
     46.8000, 7.8000,
     {"aliases": ("AAR",), "coord_kind": "river_centroid",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Inn (CH/AT/DE) – tributary of the Danube
    ("INN", "INN", "Inn",
     47.0000, 10.9000,
     {"aliases": ("INN_RIVER",), "coord_kind": "river_centroid"}),

    # Danube – longest EU river
    ("DANUBE", "DANUBE", "Danube",
     47.0000, 20.0000,
     {"aliases": ("DONAU", "DUNA", "DUNAV", "DUNAREA"),
      "coord_kind": "river_centroid", "confidence": "high"}),

    # Po (IT) – longest river entirely in Italy
    ("PO", "PO", "Po",
     45.0000, 10.5000,
     {"aliases": ("PO_RIVER",), "coord_kind": "river_centroid",
      "country_iso": "IT", "country_name": "Italy", "tz": "Europe/Rome"}),

    # Adige (IT)
    ("ADIGE", "ADIGE", "Adige",
     46.0000, 11.3000,
     {"aliases": ("ETSCH",), "coord_kind": "river_centroid",
      "country_iso": "IT", "country_name": "Italy", "tz": "Europe/Rome"}),

    # Isère (FR) – Alpine tributary of the Rhône
    ("ISERE", "ISERE", "Isère",
     45.2000, 5.7000,
     {"aliases": ("ISERE_RIVER",), "coord_kind": "river_centroid",
      "country_iso": "FR", "country_name": "France", "tz": "Europe/Paris"}),

    # Durance (FR) – right-bank tributary of the Rhône
    ("DURANCE", "DURANCE", "Durance",
     43.9000, 5.4000,
     {"coord_kind": "river_centroid",
      "country_iso": "FR", "country_name": "France", "tz": "Europe/Paris"}),

    # Saône (FR) – main tributary of the Rhône
    ("SAONE", "SAONE", "Saône",
     46.5000, 5.0000,
     {"aliases": ("SAONE_RIVER",), "coord_kind": "river_centroid",
      "country_iso": "FR", "country_name": "France", "tz": "Europe/Paris"}),

    # Ticino (CH/IT)
    ("TICINO", "TICINO", "Ticino",
     46.0000, 8.9000,
     {"aliases": ("TESSIN",), "coord_kind": "river_centroid"}),

    # Reuss (CH) – drains Gotthard area into Lake Lucerne → Aare
    ("REUSS", "REUSS", "Reuss",
     47.0000, 8.3000,
     {"coord_kind": "river_centroid",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Valais / Vispa (CH) – drains Matterhorn/Monte Rosa area into Rhône
    ("VISPA", "VISPA", "Vispa",
     46.1200, 7.8000,
     {"aliases": ("VISP", "VIEGE"), "coord_kind": "river_centroid",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Lütschine (CH) – drains Jungfrau area
    ("LUTSCHINE", "LUTSCHINE", "Lütschine",
     46.6500, 7.9000,
     {"aliases": ("LUTSCHINE",), "coord_kind": "river_centroid",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),
]


# ── Alpine lakes ──────────────────────────────────────────────────────────────

LAKES = [
    # Lake Geneva / Lac Léman (CH/FR)
    ("LAKE_GENEVA", "LAKE_GENEVA", "Lake Geneva",
     46.4500, 6.5833,
     {"aliases": ("LAC_LEMAN", "LEMAN", "GENFERSEE"),
      "coord_kind": "lake_centroid", "confidence": "high"}),

    # Lake Constance (DE/AT/CH)
    ("LAKE_CONSTANCE", "LAKE_CONSTANCE", "Lake Constance",
     47.6500, 9.3000,
     {"aliases": ("BODENSEE",), "coord_kind": "lake_centroid", "confidence": "high"}),

    # Lake Lucerne (CH)
    ("LAKE_LUCERNE", "LAKE_LUCERNE", "Lake Lucerne",
     47.0000, 8.4333,
     {"aliases": ("VIERWALDSTATTERSEE",), "coord_kind": "lake_centroid",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Lake Zurich (CH)
    ("LAKE_ZURICH", "LAKE_ZURICH", "Lake Zurich",
     47.2394, 8.7184,
     {"aliases": ("ZUERICHSEE",), "coord_kind": "lake_centroid",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Lake Maggiore (IT/CH)
    ("LAKE_MAGGIORE", "LAKE_MAGGIORE", "Lake Maggiore",
     45.8908, 8.5726,
     {"aliases": ("LAGO_MAGGIORE", "VERBANO"), "coord_kind": "lake_centroid",
      "confidence": "high"}),

    # Lake Como (IT)
    ("LAKE_COMO", "LAKE_COMO", "Lake Como",
     46.0069, 9.2685,
     {"aliases": ("LAGO_DI_COMO", "LARIO"), "coord_kind": "lake_centroid",
      "country_iso": "IT", "country_name": "Italy", "tz": "Europe/Rome",
      "confidence": "high"}),

    # Lake Garda (IT) – largest lake in Italy
    ("LAKE_GARDA", "LAKE_GARDA", "Lake Garda",
     45.6389, 10.6556,
     {"aliases": ("LAGO_DI_GARDA", "BENACO"), "coord_kind": "lake_centroid",
      "country_iso": "IT", "country_name": "Italy", "tz": "Europe/Rome",
      "confidence": "high"}),

    # Lake Brienz (CH)
    ("LAKE_BRIENZ", "LAKE_BRIENZ", "Lake Brienz",
     46.7200, 7.9700,
     {"aliases": ("BRIENZERSEE",), "coord_kind": "lake_centroid",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Lake Thun (CH)
    ("LAKE_THUN", "LAKE_THUN", "Lake Thun",
     46.6900, 7.7400,
     {"aliases": ("THUNERSEE",), "coord_kind": "lake_centroid",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Lake Lugano (CH/IT)
    ("LAKE_LUGANO", "LAKE_LUGANO", "Lake Lugano",
     45.9667, 8.9333,
     {"aliases": ("LAGO_DI_LUGANO", "CERESIO"), "coord_kind": "lake_centroid"}),

    # Lake Neuchâtel (CH)
    ("LAKE_NEUCHATEL", "LAKE_NEUCHATEL", "Lake Neuchâtel",
     46.9667, 6.9167,
     {"aliases": ("NEUENBURGERSEE",), "coord_kind": "lake_centroid",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),
]


# ── Glaciers ──────────────────────────────────────────────────────────────────

GLACIERS = [
    # Aletsch – largest glacier in the Alps (CH)
    ("ALETSCH", "ALETSCH", "Aletsch Glacier",
     46.5083, 8.0361,
     {"aliases": ("GROSSER_ALETSCHGLETSCHER",), "coord_kind": "glacier_centroid",
      "confidence": "high",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Rhône Glacier (CH) – source of the Rhône
    ("RHONE_GLACIER", "RHONE_GLACIER", "Rhône Glacier",
     46.5733, 8.3733,
     {"aliases": ("RHONEGLETSCHER",), "coord_kind": "glacier_centroid",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Gorner Glacier (CH) – near Zermatt, second-largest in the Alps
    ("GORNER", "GORNER", "Gorner Glacier",
     45.9800, 7.7900,
     {"aliases": ("GORNERGLETSCHER",), "coord_kind": "glacier_centroid",
      "country_iso": "CH", "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Mer de Glace (FR) – largest glacier in France
    ("MER_DE_GLACE", "MER_DE_GLACE", "Mer de Glace",
     45.8900, 6.9300,
     {"coord_kind": "glacier_centroid",
      "country_iso": "FR", "country_name": "France", "tz": "Europe/Paris"}),

    # Pasterze (AT) – longest glacier in the Eastern Alps
    ("PASTERZE", "PASTERZE", "Pasterze Glacier",
     47.0742, 12.6500,
     {"coord_kind": "glacier_centroid",
      "country_iso": "AT", "country_name": "Austria", "tz": "Europe/Vienna"}),
]


# ── Alpine valleys & regions ──────────────────────────────────────────────────

VALLEYS = [
    # Valais / Wallis (CH) – canton of Switzerland, home of Rhône and Matterhorn
    ("VALAIS", "VALAIS", "Valais",
     46.2000, 7.5000,
     {"aliases": ("WALLIS",), "country_iso": "CH", "country_name": "Switzerland",
      "tz": "Europe/Zurich", "confidence": "high"}),

    # Rhône Valley (CH) – Haut-Valais portion
    ("RHONE_VALLEY", "RHONE_VALLEY", "Rhône Valley",
     46.3000, 7.8000,
     {"aliases": ("RHONETAL",), "country_iso": "CH", "country_name": "Switzerland",
      "tz": "Europe/Zurich"}),

    # Engadin / Engadine (CH) – high Alpine valley, Inn headwaters
    ("ENGADIN", "ENGADIN", "Engadin",
     46.5000, 9.8500,
     {"aliases": ("ENGADINE",), "country_iso": "CH", "country_name": "Switzerland",
      "tz": "Europe/Zurich", "confidence": "high"}),

    # Ticino (CH) – canton south of Gotthard
    ("CANTON_TICINO", "CANTON_TICINO", "Canton Ticino",
     46.1500, 8.9000,
     {"aliases": ("TESSIN",), "country_iso": "CH", "country_name": "Switzerland",
      "tz": "Europe/Zurich"}),

    # Graubünden (CH) – largest canton, Rhine/Inn source
    ("GRAUBUENDEN", "GRAUBUENDEN", "Graubünden",
     46.6500, 9.5000,
     {"aliases": ("GRISONS", "GRIGIONI"), "country_iso": "CH",
      "country_name": "Switzerland", "tz": "Europe/Zurich"}),

    # Savoy (FR) – Alpine region
    ("SAVOIE", "SAVOIE", "Savoie",
     45.5667, 6.5000,
     {"aliases": ("SAVOY", "HAUTE_SAVOIE"), "country_iso": "FR",
      "country_name": "France", "tz": "Europe/Paris"}),

    # Valle d'Aosta (IT) – autonomous region, Mont Blanc side
    ("VALLE_DAOSTA", "VALLE_DAOSTA", "Valle d'Aosta",
     45.7375, 7.4262,
     {"aliases": ("AOSTA_VALLEY",), "country_iso": "IT",
      "country_name": "Italy", "tz": "Europe/Rome"}),

    # South Tyrol / Alto Adige (IT)
    ("SOUTH_TYROL", "SOUTH_TYROL", "South Tyrol",
     46.7000, 11.3000,
     {"aliases": ("ALTO_ADIGE", "SUEDTIROL"), "country_iso": "IT",
      "country_name": "Italy", "tz": "Europe/Rome"}),

    # Tyrol (AT)
    ("TYROL", "TYROL", "Tyrol",
     47.2500, 11.4000,
     {"aliases": ("TIROL",), "country_iso": "AT", "country_name": "Austria",
      "tz": "Europe/Vienna"}),

    # Vorarlberg (AT)
    ("VORARLBERG", "VORARLBERG", "Vorarlberg",
     47.2500, 9.9000,
     {"country_iso": "AT", "country_name": "Austria", "tz": "Europe/Vienna"}),

    # Carinthia (AT)
    ("CARINTHIA", "CARINTHIA", "Carinthia",
     46.7000, 13.9000,
     {"aliases": ("KAERNTEN",), "country_iso": "AT", "country_name": "Austria",
      "tz": "Europe/Vienna"}),

    # Dauphiné (FR)
    ("DAUPHINE", "DAUPHINE", "Dauphiné",
     45.0000, 5.7000,
     {"country_iso": "FR", "country_name": "France", "tz": "Europe/Paris"}),
]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_mountains() -> None:
    all_tables = [
        (MOUNTAIN_RANGES, "MOUNTAIN_RANGES"),
        (PEAKS, "PEAKS"),
        (PASSES, "PASSES"),
        (RIVERS, "RIVERS"),
        (LAKES, "LAKES"),
        (GLACIERS, "GLACIERS"),
        (VALLEYS, "VALLEYS"),
    ]

    for table, label in all_tables:
        _validate_unique_attrs(table, label)

    for table, _label in all_tables:
        for attr, key, name, lat, lon, kwargs in table:
            setattr(
                GeoZone,
                attr,
                _feature(key, name, lat, lon, **kwargs),
            )

