from __future__ import annotations

from .builders import _de_tso_zone, _validate_unique_attrs, _zone
from .geozone import GeoZone

__all__ = ["load_entsoe_zones"]

ENTSOE_ZONES = [
    ("DE_LU", "Germany-Luxembourg", 50.9, 8.7, {"tz": "Europe/Berlin", "ccy": "EUR", "country_iso": "DE", "country_name": "Germany-Luxembourg", "eic": "10Y1001A1001A82H", "aliases": ("DE-LU", "BZN|DE-LU", "IPA|DE-LU", "MBA|DE-LU", "SCA|DE-LU", "GERMANY_LUXEMBOURG"), "valid_from": "2018-10-01"}),
    ("DE_AT_LU", "Germany-Austria-Luxembourg", 48.5, 11.0, {"tz": "Europe/Berlin", "ccy": "EUR", "country_iso": "DE", "country_name": "Germany-Austria-Luxembourg", "eic": "10Y1001A1001A63L", "aliases": ("DE-AT-LU", "BZN|DE-AT-LU"), "valid_to": "2018-09-30"}),
    ("DK1", "Denmark DK1", 56.2639, 9.5018, {"country_iso": "DK", "country_name": "Denmark", "eic": "10YDK-1--------W", "aliases": ("DK-1", "BZN|DK1", "DENMARK_DK1")}),
    ("DK2", "Denmark DK2", 55.6761, 12.5683, {"country_iso": "DK", "country_name": "Denmark", "eic": "10YDK-2--------M", "aliases": ("DK-2", "BZN|DK2", "DENMARK_DK2")}),
    ("NO1", "Norway NO1", 59.9139, 10.7522, {"country_iso": "NO", "country_name": "Norway", "eic": "10YNO-1--------2", "aliases": ("BZN|NO1", "NORWAY_NO1")}),
    ("NO2", "Norway NO2", 58.1467, 7.9956, {"country_iso": "NO", "country_name": "Norway", "eic": "10YNO-2--------T", "aliases": ("BZN|NO2", "NORWAY_NO2")}),
    ("NO3", "Norway NO3", 63.4305, 10.3951, {"country_iso": "NO", "country_name": "Norway", "eic": "10YNO-3--------J", "aliases": ("BZN|NO3", "NORWAY_NO3")}),
    ("NO4", "Norway NO4", 69.6492, 18.9553, {"country_iso": "NO", "country_name": "Norway", "eic": "10YNO-4--------9", "aliases": ("BZN|NO4", "NORWAY_NO4")}),
    ("NO5", "Norway NO5", 60.3930, 5.3242, {"country_iso": "NO", "country_name": "Norway", "eic": "10Y1001A1001A48H", "aliases": ("BZN|NO5", "NORWAY_NO5")}),
    ("NO1A", "Norway NO1A", 59.9, 10.7, {"country_iso": "NO", "country_name": "Norway", "eic": "10Y1001A1001A64J", "aliases": ("BZN|NO1A",)}),
    ("NO2A", "Norway NO2A", 58.1, 8.0, {"country_iso": "NO", "country_name": "Norway", "eic": "10Y1001C--001219", "aliases": ("BZN|NO2A",)}),
    ("SE1", "Sweden SE1", 67.8558, 20.2253, {"country_iso": "SE", "country_name": "Sweden", "eic": "10Y1001A1001A44P", "aliases": ("BZN|SE1", "SWEDEN_SE1")}),
    ("SE2", "Sweden SE2", 63.8258, 20.2630, {"country_iso": "SE", "country_name": "Sweden", "eic": "10Y1001A1001A45N", "aliases": ("BZN|SE2", "SWEDEN_SE2")}),
    ("SE3", "Sweden SE3", 59.3293, 18.0686, {"country_iso": "SE", "country_name": "Sweden", "eic": "10Y1001A1001A46L", "aliases": ("BZN|SE3", "SWEDEN_SE3")}),
    ("SE4", "Sweden SE4", 55.6050, 13.0038, {"country_iso": "SE", "country_name": "Sweden", "eic": "10Y1001A1001A47J", "aliases": ("BZN|SE4", "SWEDEN_SE4")}),
    ("FI_BZ", "Finland", 61.9241, 25.7482, {"country_iso": "FI", "country_name": "Finland", "eic": "10YFI-1--------U", "aliases": ("BZN|FI",)}),
    ("EE_BZ", "Estonia", 58.5953, 25.0136, {"country_iso": "EE", "country_name": "Estonia", "eic": "10Y1001A1001A39I", "aliases": ("BZN|EE",)}),
    ("LV_BZ", "Latvia", 56.8796, 24.6032, {"country_iso": "LV", "country_name": "Latvia", "eic": "10YLV-1001A00074", "aliases": ("BZN|LV",)}),
    ("LT_BZ", "Lithuania", 55.1694, 23.8813, {"country_iso": "LT", "country_name": "Lithuania", "eic": "10YLT-1001A0008Q", "aliases": ("BZN|LT",)}),
    ("PL_BZ", "Poland", 51.9194, 19.1451, {"country_iso": "PL", "country_name": "Poland", "eic": "10YPL-AREA-----S", "aliases": ("BZN|PL",)}),
    ("PT_BZ", "Portugal", 39.3999, -8.2245, {"country_iso": "PT", "country_name": "Portugal", "eic": "10YPT-REN------W", "aliases": ("BZN|PT",)}),
    ("ES_BZ", "Spain", 40.4637, -3.7492, {"country_iso": "ES", "country_name": "Spain", "eic": "10YES-REE------0", "aliases": ("BZN|ES",)}),
    ("RO_BZ", "Romania", 45.9432, 24.9668, {"country_iso": "RO", "country_name": "Romania", "eic": "10YRO-TEL------P", "aliases": ("BZN|RO",)}),
    ("BG_BZ", "Bulgaria", 42.7339, 25.4858, {"country_iso": "BG", "country_name": "Bulgaria", "eic": "10YCA-BULGARIA-R", "aliases": ("BZN|BG",)}),
    ("HU_BZ", "Hungary", 47.1625, 19.5033, {"country_iso": "HU", "country_name": "Hungary", "eic": "10YHU-MAVIR----U", "aliases": ("BZN|HU",)}),
    ("CZ_BZ", "Czechia", 49.8175, 15.4730, {"country_iso": "CZ", "country_name": "Czechia", "eic": "10YCZ-CEPS-----N", "aliases": ("BZN|CZ",)}),
    ("SK_BZ", "Slovakia", 48.6690, 19.6990, {"country_iso": "SK", "country_name": "Slovakia", "eic": "10YSK-SEPS-----K", "aliases": ("BZN|SK",)}),
    ("SI_BZ", "Slovenia", 46.1512, 14.9955, {"country_iso": "SI", "country_name": "Slovenia", "eic": "10YSI-ELES-----O", "aliases": ("BZN|SI",)}),
    ("HR_BZ", "Croatia", 45.1000, 15.2000, {"country_iso": "HR", "country_name": "Croatia", "eic": "10YHR-HEP------M", "aliases": ("BZN|HR",)}),
    ("AT_BZ", "Austria", 47.5162, 14.5501, {"country_iso": "AT", "country_name": "Austria", "eic": "10YAT-APG------L", "aliases": ("BZN|AT",)}),
    ("BE_BZ", "Belgium", 50.5039, 4.4699, {"country_iso": "BE", "country_name": "Belgium", "eic": "10YBE----------2", "aliases": ("BZN|BE",)}),
    ("NL_BZ", "Netherlands", 52.1326, 5.2913, {"country_iso": "NL", "country_name": "Netherlands", "eic": "10YNL----------L", "aliases": ("BZN|NL",)}),
    ("RS_BZ", "Serbia", 44.0165, 21.0059, {"country_iso": "RS", "country_name": "Serbia", "eic": "10YCS-SERBIATSOV", "aliases": ("BZN|RS",)}),
    ("ME_BZ", "Montenegro", 42.7087, 19.3744, {"country_iso": "ME", "country_name": "Montenegro", "eic": "10YCS-CG-TSO---S", "aliases": ("BZN|ME",)}),
    ("AL_BZ", "Albania", 41.1533, 20.1683, {"country_iso": "AL", "country_name": "Albania", "eic": "10YAL-KESH-----5", "aliases": ("BZN|AL",)}),
    ("BA_BZ", "Bosnia and Herzegovina", 43.9159, 17.6791, {"country_iso": "BA", "country_name": "Bosnia and Herzegovina", "eic": "10YBA-JPCC-----D", "aliases": ("BZN|BA",)}),
    ("MK_BZ", "North Macedonia", 41.6086, 21.7453, {"country_iso": "MK", "country_name": "North Macedonia", "eic": "10YMK-MEPSO----8", "aliases": ("BZN|MK",)}),
    ("GR_BZ", "Greece", 39.0742, 21.8243, {"country_iso": "GR", "country_name": "Greece", "eic": "10YGR-HTSO-----Y", "aliases": ("BZN|GR",)}),
    ("TR_BZ", "Turkey", 38.9637, 35.2433, {"country_iso": "TR", "country_name": "Turkey", "eic": "10YTR-TEIAS----W", "aliases": ("BZN|TR",)}),
    ("CY_BZ", "Cyprus", 35.1264, 33.4299, {"country_iso": "CY", "country_name": "Cyprus", "eic": "10YCY-1001A0003J", "aliases": ("BZN|CY",)}),
    ("XK_BZ", "Kosovo", 42.6026, 20.9030, {"country_iso": "XK", "country_name": "Kosovo", "eic": "10Y1001C--00100H", "aliases": ("BZN|XK",)}),
    ("MD_BZ", "Moldova", 47.4116, 28.3699, {"country_iso": "MD", "country_name": "Moldova", "eic": "10Y1001A1001A990", "aliases": ("BZN|MD",)}),
    ("MT_BZ", "Malta", 35.9375, 14.3754, {"country_iso": "MT", "country_name": "Malta", "eic": "10Y1001A1001A93C", "aliases": ("BZN|MT",)}),
    ("GB_BZ", "Great Britain", 55.3781, -3.4360, {"country_iso": "GB", "country_name": "United Kingdom", "eic": "10Y1001A1001A92E", "aliases": ("BZN|GB",)}),
    ("IE_SEM", "Ireland SEM", 53.4, -7.7, {"country_iso": "IE", "country_name": "Ireland", "eic": "10Y1001A1001A59C", "aliases": ("SEM", "BZN|IE_SEM")}),
    ("UA_IPS", "Ukraine UA-IPS", 48.7, 31.2, {"country_iso": "UA", "country_name": "Ukraine", "eic": "10Y1001C--000182", "aliases": ("UA-IPS", "UKRAINE_IPS", "BZN|UA-IPS")}),
    ("UA_DOBAS", "Ukraine Donbas", 48.0, 37.8, {"country_iso": "UA", "country_name": "Ukraine", "eic": "10Y1001C--000244", "aliases": ("UA-DOBAS", "BZN|UA-DOBAS")}),
    ("UA_BEI", "Ukraine BEI", 50.8, 29.4, {"country_iso": "UA", "country_name": "Ukraine", "eic": "10Y1001C--00025I", "aliases": ("UA-BEI", "BZN|UA-BEI")}),
    ("RU_KGD", "Russia Kaliningrad", 54.71, 20.51, {"tz": "Europe/Kaliningrad", "ccy": "RUB", "country_iso": "RU", "country_name": "Russia", "eic": "10Y1001A1001A50U", "aliases": ("RUSSIA_KALININGRAD",)}),
    ("GB_IFA", "Great Britain IFA", 50.95, 1.85, {"country_iso": "GB", "country_name": "United Kingdom", "eic": "10Y1001C--00098F", "aliases": ("GREAT_BRITAIN_IFA",)}),
    ("GB_IFA2", "Great Britain IFA2", 50.95, 1.85, {"country_iso": "GB", "country_name": "United Kingdom", "eic": "17Y0000009369493", "aliases": ("GREAT_BRITAIN_IFA2",)}),
    ("GB_ELECLINK", "Great Britain ElecLink", 51.1, 1.3, {"country_iso": "GB", "country_name": "United Kingdom", "eic": "11Y0-0000-0265-K", "aliases": ("GB-ELECLINK",)}),
    ("GB_NEMO", "Great Britain Nemo Link", 51.1, 1.6, {"country_iso": "GB", "country_name": "United Kingdom", "eic": "11Y0-0000-0265-L", "aliases": ("GB-NEMO",)}),
    ("IT_NORTH", "Italy North", 45.3, 10.5, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A73I", "aliases": ("IT-NORTH", "ITALY_NORTH", "ITALY-NORD", "BZN|IT-NORTH")}),
    ("IT_SOUTH", "Italy South", 40.8, 16.5, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A788", "aliases": ("IT-SOUTH", "ITALY_SOUTH", "ITALY-SUD", "BZN|IT-SOUTH")}),
    ("IT_CENTRE_NORTH", "Italy Centre-North", 43.5, 11.5, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A70O", "aliases": ("IT-CENTRE-NORTH", "IT-CENTER-NORTH", "ITALY-CNOR", "ITALY_CENTRE_NORTH", "ITALY_CENTER_NORTH")}),
    ("IT_CENTRE_SOUTH", "Italy Centre-South", 41.8, 14.0, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A71M", "aliases": ("IT-CENTRE-SOUTH", "IT-CENTER-SOUTH", "ITALY-CSUD", "ITALY_CENTRE_SOUTH", "ITALY_CENTER_SOUTH")}),
    ("IT_SARDINIA", "Italy Sardinia", 40.1209, 9.0129, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A74G", "aliases": ("IT-SARDINIA", "ITALY_SARDINIA", "ITALY-SARDI")}),
    ("IT_SICILY", "Italy Sicily", 37.5999, 14.0154, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A75E", "aliases": ("IT-SICILY", "ITALY_SICILY", "ITALY-SICI")}),
    ("IT_CALABRIA", "Italy Calabria", 38.9, 16.6, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001C--00096J", "aliases": ("IT-CALABRIA", "ITALY_CALABRIA")}),
    ("IT_FOGGIA", "Italy Foggia", 41.46, 15.54, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A72K", "aliases": ("IT-FOGGIA", "ITALY_FOGGIA")}),
    ("IT_BRINDISI", "Italy Brindisi", 40.64, 17.94, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A699", "aliases": ("IT-BRINDISI", "ITALY_BRINDISI")}),
    ("IT_PRIOLO", "Italy Priolo", 37.15, 15.18, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A76C", "aliases": ("IT-PRIOLO", "ITALY_PRIOLO")}),
    ("IT_ROSSANO", "Italy Rossano", 39.58, 16.64, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A77A", "aliases": ("IT-ROSSANO", "ITALY_ROSSANO")}),
    ("IT_GR", "Italy-Greece", 39.0, 19.0, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A66F", "aliases": ("IT-GR", "ITALY_GREECE")}),
    ("IT_NORTH_SI", "Italy North-Slovenia", 45.6, 13.8, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A67D", "aliases": ("IT-NORTH-SI", "ITALY_NORTH_SLOVENIA")}),
    ("IT_NORTH_CH", "Italy North-Switzerland", 46.2, 8.8, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A68B", "aliases": ("IT-NORTH-CH", "ITALY_NORTH_SWITZERLAND")}),
    ("IT_NORTH_AT", "Italy North-Austria", 46.6, 12.0, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A80L", "aliases": ("IT-NORTH-AT", "ITALY_NORTH_AUSTRIA")}),
    ("IT_NORTH_FR", "Italy North-France", 45.1, 6.8, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A81J", "aliases": ("IT-NORTH-FR", "ITALY_NORTH_FRANCE")}),
    ("IT_MALTA", "Italy-Malta", 36.1, 14.4, {"country_iso": "IT", "country_name": "Italy", "eic": "10Y1001A1001A877", "aliases": ("IT-MALTA", "ITALY_MALTA")}),
]

DE_TSO_ZONES = [
    ("DE_AMPRION", "Germany Amprion", 51.48, 7.22, "10YDE-RWENET---I", ("AMPRION", "DE-AMPRION", "AMP", "CTA|DE(Amprion)", "LFA|DE(Amprion)", "SCA|DE(Amprion)")),
    ("DE_50HERTZ", "Germany 50Hertz", 52.52, 13.405, "10YDE-VE-------2", ("50HERTZ", "DE-50HERTZ", "50HZ", "CTA|DE(50Hertz)", "LFA|DE(50Hertz)", "SCA|DE(50Hertz)", "BZA|DE(50HzT)")),
    ("DE_TENNET", "Germany TenneT", 52.65, 10.20, "10YDE-EON------1", ("TENNET_DE", "TENNET_GER", "DE-TENNET", "TTG", "CTA|DE(TenneT GER)", "LFA|DE(TenneT GER)", "SCA|DE(TenneT GER)")),
    ("DE_TRANSNETBW", "Germany TransnetBW", 48.78, 9.18, "10YDE-ENBW-----N", ("TRANSNETBW", "DE-TRANSNETBW", "TNG", "CTA|DE(TransnetBW)", "LFA|DE(TransnetBW)", "SCA|DE(TransnetBW)")),
]

DE_TECHNICAL_ZONES = [
    ("DE_AMPRION_LU", "Germany Amprion-Luxembourg", 49.9, 6.4, {"tz": "Europe/Berlin", "ccy": "EUR", "country_iso": "DE", "country_name": "Germany-Luxembourg", "eic": "10Y1001C--00002H", "aliases": ("LFA|DE(Amprion)-LU", "SCA|DE(Amprion)-LU", "DE-AMP-LU"), "coord_source": "seed: module (representative point for Amprion-Luxembourg technical area)"}),
    ("DE_DK1_LU", "Germany-DK1-Luxembourg control block", 51.2, 9.9, {"tz": "Europe/Berlin", "ccy": "EUR", "country_iso": "DE", "country_name": "Germany-DK1-Luxembourg", "eic": "10YCB-GERMANY--8", "aliases": ("LFB|DE_DK1_LU", "SCA|DE_DK1_LU", "GERMANY_CONTROL_BLOCK"), "coord_source": "seed: module (representative point for control-block area)"}),
]


def load_entsoe_zones() -> None:
    _validate_unique_attrs(ENTSOE_ZONES, "ENTSOE_ZONES")
    _validate_unique_attrs(DE_TSO_ZONES, "DE_TSO_ZONES")
    _validate_unique_attrs(DE_TECHNICAL_ZONES, "DE_TECHNICAL_ZONES")

    for attr, name, lat, lon, kwargs in ENTSOE_ZONES:
        setattr(GeoZone, attr, _zone(attr, name, lat, lon, **kwargs))

    for attr, name, lat, lon, eic, aliases in DE_TSO_ZONES:
        setattr(GeoZone, attr, _de_tso_zone(attr, name, lat, lon, eic, aliases=aliases))

    for attr, name, lat, lon, kwargs in DE_TECHNICAL_ZONES:
        setattr(GeoZone, attr, _zone(attr, name, lat, lon, **kwargs))