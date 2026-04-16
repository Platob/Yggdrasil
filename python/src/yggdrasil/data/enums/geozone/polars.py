from __future__ import annotations

from typing import TYPE_CHECKING

from yggdrasil.polars.lib import polars

from .catalog import GeoZoneCatalog

if TYPE_CHECKING:
    import polars as pl

__all__ = ["geozone_lookup_frame", "join_geozones", "normalize_geozone_expr"]


def normalize_geozone_expr(value: str | "pl.Expr") -> "pl.Expr":
    expr = polars.col(value) if isinstance(value, str) else value
    return (
        expr.cast(polars.String)
        .str.strip_chars()
        .str.replace_all(r"[-_]+", " ")
        .str.to_uppercase()
    )


def geozone_lookup_frame(catalog: GeoZoneCatalog) -> "pl.DataFrame":
    return polars.DataFrame(catalog.lookup_rows())


def join_geozones(
    frame: "pl.DataFrame | pl.LazyFrame",
    on: str,
    *,
    catalog: GeoZoneCatalog,
    prefix: str = "geozone_",
) -> "pl.DataFrame | pl.LazyFrame":
    lookup = geozone_lookup_frame(catalog).lazy() if isinstance(frame, polars.LazyFrame) else geozone_lookup_frame(catalog)
    normalized = "__geozone_lookup_token"

    enriched = frame.with_columns(normalize_geozone_expr(on).alias(normalized)).join(
        lookup,
        left_on=normalized,
        right_on="lookup_token",
        how="left",
    )

    return enriched.rename(
        {
            "gtype": f"{prefix}gtype",
            "key": f"{prefix}key",
            "name": f"{prefix}name",
            "country_iso": f"{prefix}country_iso",
            "region_iso": f"{prefix}region_iso",
            "sub_iso": f"{prefix}sub_iso",
            "ccy": f"{prefix}ccy",
            "eic": f"{prefix}eic",
            "lat": f"{prefix}lat",
            "lon": f"{prefix}lon",
        }
    ).drop(normalized)
