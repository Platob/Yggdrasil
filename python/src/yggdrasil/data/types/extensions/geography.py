"""GeographyType — extension type for geographic zone data.

Stores geography identifiers as strings (ISO codes, country names, region
codes, EIC codes, coordinates) and normalizes them through the GeoZone
catalog on cast.

Cast behavior:
- Input strings are resolved through GeoZoneCatalog (codes, aliases,
  fuzzy name matching, coordinate strings).
- Resolved zones produce a canonical ``code`` (region_iso or country_iso
  or key).
- When ``safe=True``: unparseable values raise ValueError.
- When ``safe=False`` (default): unparseable values become null.

Databricks mapping: STRING — geography data lives as plain strings in SQL
tables and dashboards, which is the most portable representation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.types.extensions.base import ExtensionType

if TYPE_CHECKING:
    import polars
    from yggdrasil.data.enums.geozone.base import GeoZone
    from yggdrasil.data.enums.geozone.catalog import GeoZoneCatalog

__all__ = ["GeographyType"]

LOGGER = logging.getLogger(__name__)


def _get_catalog(catalog: "GeoZoneCatalog | None" = None) -> "GeoZoneCatalog":
    if catalog is not None:
        return catalog
    from yggdrasil.data.enums.geozone.load import load_geozones

    return load_geozones()


def _resolve_zone(
    value: str | None,
    catalog: "GeoZoneCatalog",
) -> "GeoZone | None":
    """Parse a single string value into a GeoZone, or None if unresolvable."""
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    return catalog.parse_str(stripped)


def _zone_to_code(zone: "GeoZone | None") -> str | None:
    """Extract the canonical code string from a resolved zone."""
    if zone is None:
        return None
    return zone.code


# ---------------------------------------------------------------------------
# Bulk parsing utilities
# ---------------------------------------------------------------------------


def parse_geography_arrow(
    array: pa.Array | pa.ChunkedArray,
    *,
    catalog: "GeoZoneCatalog | None" = None,
    safe: bool = False,
    output: str = "code",
) -> pa.Array | pa.ChunkedArray:
    """Parse an Arrow string array into normalized geography codes.

    Parameters
    ----------
    array
        Arrow array of strings to parse (country names, ISO codes, aliases,
        coordinate strings, etc.).
    catalog
        GeoZone catalog to resolve against.  Uses the default catalog when
        ``None``.
    safe
        When ``True``, raise ``ValueError`` on the first unparseable value.
        When ``False`` (default), unparseable values become null.
    output
        What to extract from resolved zones.  One of:
        ``"code"`` (default), ``"name"``, ``"country_iso"``,
        ``"region_iso"``, ``"key"``, ``"ccy"``, ``"gtype"``.

    Returns
    -------
    pa.Array or pa.ChunkedArray
        String array of the same length with resolved values (or nulls).
    """
    cat = _get_catalog(catalog)
    is_chunked = isinstance(array, pa.ChunkedArray)

    if is_chunked:
        chunks = [
            _parse_geography_flat(chunk, cat, safe=safe, output=output)
            for chunk in array.chunks
        ]
        return pa.chunked_array(chunks, type=pa.string())

    return _parse_geography_flat(array, cat, safe=safe, output=output)


def _parse_geography_flat(
    array: pa.Array,
    catalog: GeoZoneCatalog,
    *,
    safe: bool,
    output: str,
) -> pa.Array:
    """Parse a flat (non-chunked) Arrow string array."""
    results: list[str | None] = []

    for i in range(len(array)):
        if array[i].as_py() is None:
            results.append(None)
            continue

        raw = str(array[i].as_py())
        zone = _resolve_zone(raw, catalog)

        if zone is None:
            if safe:
                raise ValueError(
                    f"Cannot resolve geography value {raw!r} at index {i}. "
                    "No matching zone found in the catalog. "
                    "Pass safe=False to null out unparseable values, "
                    "or check your input data / catalog."
                )
            results.append(None)
            continue

        results.append(_extract_field(zone, output))

    return pa.array(results, type=pa.string())


def _extract_field(zone: GeoZone, output: str) -> str | None:
    """Pull the requested field from a resolved GeoZone."""
    if output == "code":
        return zone.code
    if output == "name":
        return zone.name
    if output == "country_iso":
        return zone.country_iso
    if output == "region_iso":
        return zone.region_iso
    if output == "key":
        return zone.key
    if output == "ccy":
        return zone.ccy
    if output == "gtype":
        return zone.gtype.value
    raise ValueError(
        f"Unknown output field {output!r}. "
        f"Valid options: 'code', 'name', 'country_iso', 'region_iso', "
        f"'key', 'ccy', 'gtype'."
    )


def parse_geography_polars(
    series: polars.Series,
    *,
    catalog: "GeoZoneCatalog | None" = None,
    safe: bool = False,
    output: str = "code",
) -> polars.Series:
    """Parse a Polars string series into normalized geography codes.

    Same semantics as :func:`parse_geography_arrow` but operates on Polars
    Series.  Goes through Arrow internally for catalog resolution.
    """
    from yggdrasil.data.types.support import get_polars

    pl = get_polars()

    arrow = series.to_arrow()
    parsed = parse_geography_arrow(arrow, catalog=catalog, safe=safe, output=output)

    # Rebuild as Polars Series, keep the original name.
    return pl.Series(name=series.name, values=parsed, dtype=pl.String)


# ---------------------------------------------------------------------------
# GeographyType
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeographyType(ExtensionType):
    """Extension type for geographic zone identifiers.

    Stores geography data as Arrow strings and normalizes on cast through
    the GeoZone catalog.  Resolved values are canonical codes (region_iso
    or country_iso or key).

    The ``output`` parameter controls what field is extracted from resolved
    zones: ``"code"`` (default), ``"name"``, ``"country_iso"``,
    ``"region_iso"``, ``"key"``, ``"ccy"``, ``"gtype"``.

    Example::

        geo = GeographyType()
        geo = GeographyType(output="country_iso")
        geo = GeographyType(output="name")

    Casting an Arrow array through this type normalizes every element::

        arr = pa.array(["France", "DE", "zuerich", "nope"])
        result = geo._cast_arrow_array(arr, options)
        # → ["FR IDF", "DE BE", "CH ZH", null]  (with safe=False)
    """

    extension_name: ClassVar[str] = "yggdrasil.geography"
    storage_type: ClassVar[pa.DataType] = pa.string()

    output: str = "code"

    # ------------------------------------------------------------------
    # Serialization — omit output field when it's the default "code"
    # ------------------------------------------------------------------
    def _own_field_values(self) -> dict[str, Any]:
        if self.output == "code":
            return {}
        return {"output": self.output}

    # ------------------------------------------------------------------
    # Databricks DDL — geography is just a string column in SQL
    # ------------------------------------------------------------------
    def to_databricks_ddl(self) -> str:
        return "STRING"

    # ------------------------------------------------------------------
    # Casting — the main show
    # ------------------------------------------------------------------
    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: Any,
    ) -> pa.Array:
        """Cast an Arrow array into geography codes.

        Normalizes input strings through the GeoZone catalog.
        Unparseable values become null when safe=False (default),
        or raise ValueError when safe=True.
        """
        safe = getattr(options, "safe", False)

        # If already our extension type, unwrap to storage first.
        src = array
        if isinstance(src.type, pa.ExtensionType):
            src = src.storage

        # Cast to string if not already — we accept ints, binaries, etc.
        if not pa.types.is_string(src.type) and not pa.types.is_large_string(src.type):
            try:
                src = pc.cast(src, pa.string(), safe=False)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                if safe:
                    raise
                # Can't even stringify — null everything out.
                return pa.ExtensionArray.from_storage(
                    self.to_arrow(),
                    pa.nulls(len(src), type=pa.string()),
                )

        parsed = parse_geography_arrow(src, safe=safe, output=self.output)

        # Flatten to plain array if chunked came back.
        if isinstance(parsed, pa.ChunkedArray):
            parsed = parsed.combine_chunks()

        # Wrap into extension type.
        return pa.ExtensionArray.from_storage(self.to_arrow(), parsed)

    # ------------------------------------------------------------------
    # Python object conversion — single value
    # ------------------------------------------------------------------
    def convert_pyobj(
        self, value: Any, nullable: bool, safe: bool = False
    ) -> str | None:
        if value is None:
            if nullable:
                return None
            raise ValueError(
                "Got None for a non-nullable GeographyType field. "
                "Pass a valid geography string or set nullable=True."
            )

        zone = _resolve_zone(str(value), _get_catalog())
        code = _extract_field(zone, self.output) if zone is not None else None

        if code is not None:
            return code

        if safe:
            raise ValueError(
                f"Cannot resolve geography value {value!r}. "
                "No matching zone found in the catalog. "
                "Pass safe=False to allow null for unparseable values."
            )

        if not nullable:
            raise ValueError(
                f"Cannot resolve geography value {value!r} and field "
                "is not nullable. Either fix the input or make the field nullable."
            )
        return None

    def _convert_pyobj(self, value: Any, safe: bool = False) -> str | None:
        zone = _resolve_zone(str(value), _get_catalog())
        result = _extract_field(zone, self.output) if zone is not None else None
        if result is None and safe:
            raise ValueError(
                f"Cannot resolve geography value {value!r}. "
                "No matching zone found in the catalog."
            )
        return result

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    def default_pyobj(self, nullable: bool) -> str | None:
        if nullable:
            return None
        # "WORLD" is a safe non-null default — it's always in the catalog.
        return "WORLD"

    def default_arrow_scalar(self, nullable: bool = True) -> pa.Scalar:
        if nullable:
            return pa.scalar(None, type=pa.string())
        return pa.scalar("WORLD", type=pa.string())

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if self.output != "code":
            return f"GeographyType(output={self.output!r})"
        return "GeographyType()"

    def __str__(self) -> str:
        if self.output != "code":
            return f"geography[{self.output}]"
        return "geography"
