"""Collection-level service for Unity Catalog volumes.

Provides ``__getitem__``, ``volume()``, ``find()``, and ``list()``
against the Databricks catalog API. Per-volume DDL / metadata /
credential vending lives in
:class:`~yggdrasil.databricks.volume.volume.Volume`.

Hierarchy
---------
::

    client.volumes                                # Volumes (root)
    client.volumes(catalog_name="main")           # Volumes scoped to "main"
    client.volumes["main.sales.uploads"]          # Volume (fully qualified)
    client.volumes["uploads"]                     # Volume (uses defaults)
    client.volumes.list()                         # Iterator[Volume]
    client.volumes.list(catalog_name="main")      # Iterator[Volume] in "main"

Caching strategy
----------------
:class:`Volume` instances are interned per
``(host, catalog, schema, name)`` via the
:attr:`Volume._INSTANCES` cache, so two callers asking for the
same UC volume collapse to the same live instance — and therefore
share the same TTL-bounded :class:`VolumeInfo` snapshot.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from databricks.sdk.errors import DatabricksError, ResourceDoesNotExist
from databricks.sdk.service.catalog import VolumeInfo
from yggdrasil.databricks.client import DatabricksService
from yggdrasil.url import URL

from .volume import Volume

__all__ = ["Volumes", "EXTERNAL_LOCATION_SCHEMES"]

logger = logging.getLogger(__name__)

#: Cloud object-store URI schemes that designate an **external** volume's
#: backing location. A first argument starting with one of these (or an explicit
#: ``storage_location``) routes :meth:`Volumes.get_or_create` to the external
#: path; anything else is treated as a managed-volume dotted name.
EXTERNAL_LOCATION_SCHEMES = (
    "s3://", "s3a://", "s3n://",
    "abfss://", "abfs://",
    "gs://", "gcs://",
    "wasbs://", "wasb://",
)


class Volumes(DatabricksService):
    """Collection-level service for Unity Catalog volumes."""

    def __init__(
        self,
        client=None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
    ):
        super().__init__(client=client)
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.volume_name = volume_name

    # ── context rebind ────────────────────────────────────────────────────────

    def __call__(
        self,
        catalog_name: Optional[str] = "",
        schema_name: Optional[str] = "",
        volume_name: Optional[str] = "",
        *args,
        **kwargs,
    ):
        if catalog_name == "":
            catalog_name = self.catalog_name
        if schema_name == "":
            schema_name = self.schema_name
        if volume_name == "":
            volume_name = self.volume_name

        if catalog_name and schema_name and volume_name:
            return self.volume(
                catalog_name=catalog_name,
                schema_name=schema_name,
                volume_name=volume_name,
            )

        return Volumes(
            client=self.client,
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name,
        )

    def __getstate__(self):
        state = super().__getstate__()
        state["catalog_name"] = self.catalog_name
        state["schema_name"] = self.schema_name
        state["volume_name"] = self.volume_name
        return state

    def __setstate__(self, state):
        self.catalog_name = state.get("catalog_name")
        self.schema_name = state.get("schema_name")
        self.volume_name = state.get("volume_name")
        super().__setstate__(state)

    # ── dict-like navigation ──────────────────────────────────────────────────

    def __getitem__(self, name: str) -> Volume:
        """Route a 1-, 2-, or 3-part dotted name to a :class:`Volume`.

        * ``volumes["uploads"]``                      → :class:`Volume` (needs catalog + schema defaults)
        * ``volumes["sales.uploads"]``                → :class:`Volume` (needs catalog default)
        * ``volumes["main.sales.uploads"]``           → :class:`Volume`
        """
        parts = [p.strip().strip("`") for p in name.split(".")]
        n = len(parts)
        if n == 1:
            if not (self.catalog_name and self.schema_name):
                raise ValueError(
                    f"Cannot resolve one-part volume name {name!r} without"
                    " default catalog_name + schema_name — set them on the"
                    " service or pass a fully-qualified name."
                )
            return self.volume(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                volume_name=parts[0],
            )
        if n == 2:
            if not self.catalog_name:
                raise ValueError(
                    f"Cannot resolve two-part volume name {name!r} without"
                    " a default catalog_name — set it on the service or"
                    " pass a fully-qualified name."
                )
            return self.volume(
                catalog_name=self.catalog_name,
                schema_name=parts[0],
                volume_name=parts[1],
            )
        if n == 3:
            return self.volume(
                catalog_name=parts[0],
                schema_name=parts[1],
                volume_name=parts[2],
            )
        raise KeyError(
            f"Expected a 1- to 3-part dotted name (volume / schema.volume /"
            f" catalog.schema.volume), got {name!r} with {len(parts)} parts."
        )

    def __setitem__(self, name: str, _new_name: str) -> None:
        raise NotImplementedError(
            "Unity Catalog does not support volume rename; create the new "
            "volume and migrate contents explicitly."
        )

    def __iter__(self) -> Iterator[Volume]:
        return self.list()

    # ── factory methods ───────────────────────────────────────────────────────

    def volume(
        self,
        location: "Volume | str | None" = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
    ) -> Volume:
        """Return a :class:`Volume` bound to this service.

        Routes through the singleton cache so repeated calls collapse
        to the same instance.
        """
        if location is not None:
            if isinstance(location, Volume):
                return location

        c, s, v = self._resolve_parts(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name,
        )
        if not (c and s and v):
            raise ValueError(
                f"Volumes.volume requires catalog + schema + volume names "
                f"(got {c!r}, {s!r}, {v!r}). Pass them explicitly or set "
                f"defaults on the service."
            )
        return Volume(
            service=self,
            catalog_name=c,
            schema_name=s,
            volume_name=v,
        )

    def create(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
        comment: str | None = None,
        storage_location: str | None = None,
        volume_type=None,
        missing_ok: bool = True,
    ) -> Volume:
        """Create a volume by name, auto-creating missing schema / catalog.

        Service-level entry point matching :meth:`Warehouses.create` —
        resolves the catalog / schema / volume parts (from *location*
        or keyword overrides, falling back to the service defaults),
        materialises the :class:`Volume` singleton, then delegates to
        :meth:`Volume.create` so the managed-volume-type default and
        the post-create ``_store_infos`` cache warm-up live in one
        place.
        """
        volume = self.volume(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name,
        )

        return volume.create(
            comment=comment,
            storage_location=storage_location,
            volume_type=volume_type,
            missing_ok=missing_ok
        )

    def get_or_create(
        self,
        location: "str | Volume | None" = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
        storage_location: str | None = None,
        comment: str | None = None,
        volume_type=None,
    ) -> Volume:
        """Get-or-create a volume — **external** by storage URI, else **managed**
        by dotted name.

        When the first argument (or *storage_location*) is a cloud object-store
        URI (``s3://…`` / ``abfss://…`` / ``gs://…`` — see
        :data:`EXTERNAL_LOCATION_SCHEMES`), returns or creates an **external**
        volume backed by that location. ``volume_name`` is required and names
        the volume explicitly; ``catalog_name`` / ``schema_name`` default to the
        service scope. (An external location + storage credential covering the
        URI must already exist in Unity Catalog.)

        Otherwise *location* is a 1-/2-/3-part dotted name (``cat.sch.vol``) and
        a **managed** volume is returned, created if absent. Mirrors the
        find-then-create shape of :meth:`Jobs.create_or_update`."""
        store = storage_location
        if (
            store is None
            and isinstance(location, str)
            and location.lower().startswith(EXTERNAL_LOCATION_SCHEMES)
        ):
            store = location

        if store:
            cat = catalog_name or self.catalog_name
            sch = schema_name or self.schema_name
            if not (cat and sch):
                raise ValueError(
                    "external volume needs a catalog + schema — pass "
                    "catalog_name/schema_name or set defaults on the service."
                )
            if not volume_name:
                raise ValueError(
                    "external volume needs an explicit volume_name — pass "
                    "volume_name=... (no name is derived from the storage URI)."
                )
            name = volume_name
            existing = self.find(
                catalog_name=cat, schema_name=sch, volume_name=name, raise_error=False,
            )
            if existing is not None:
                return existing
            logger.info("creating external volume %s.%s.%s → %s", cat, sch, name, store)
            return self.create(
                catalog_name=cat,
                schema_name=sch,
                volume_name=name,
                storage_location=store,
                volume_type=volume_type or "EXTERNAL",
                comment=comment,
                missing_ok=True,
            )

        c, s, v = self._resolve_parts(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name,
        )
        if not (c and s and v):
            raise ValueError(
                f"Volumes.get_or_create requires catalog + schema + volume names "
                f"(got {c!r}, {s!r}, {v!r}). Pass them explicitly, give a cloud "
                f"storage URI, or set defaults on the service."
            )
        existing = self.find(
            catalog_name=c, schema_name=s, volume_name=v, raise_error=False,
        )
        if existing is not None:
            return existing
        logger.info("creating managed volume %s.%s.%s", c, s, v)
        return self.create(
            catalog_name=c,
            schema_name=s,
            volume_name=v,
            comment=comment,
            volume_type=volume_type,
            missing_ok=True,
        )

    # ── remote fetch ──────────────────────────────────────────────────────────

    def find_remote(
        self,
        catalog_name: str,
        schema_name: str,
        volume_name: str,
        *,
        raise_error: bool = True,
    ) -> Optional[VolumeInfo]:
        """Raw API lookup — GET by fully-qualified name, no cache.

        Returns ``None`` on miss when ``raise_error=False``.
        """
        full_name = f"{catalog_name}.{schema_name}.{volume_name}"
        logger.debug("Fetching volume %s from remote", full_name)
        try:
            return self.client.workspace_client().volumes.read(full_name)
        except DatabricksError as exc:
            if raise_error:
                raise ResourceDoesNotExist(f"Volume {full_name} not found") from exc
            return None

    def find(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
        raise_error: bool = True,
    ) -> Optional[Volume]:
        """Resolve a volume by name.

        Caching lives on the :class:`Volume` singleton itself; this
        method drives a remote read when the cached info is stale (or
        missing) and seeds the resulting info onto the instance.
        """
        c, s, v = self._resolve_parts(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name,
        )
        if not (c and s and v):
            raise ValueError(
                f"Volumes.find requires catalog + schema + volume names "
                f"(got {c!r}, {s!r}, {v!r})."
            )
        info = self.find_remote(
            catalog_name=c, schema_name=s, volume_name=v, raise_error=raise_error,
        )
        if info is None:
            return None
        return Volume(
            service=self,
            catalog_name=info.catalog_name or c,
            schema_name=info.schema_name or s,
            volume_name=info.name or v,
            infos=info,
        )

    # ── listing ───────────────────────────────────────────────────────────────

    def list(
        self,
        name: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
    ) -> Iterator[Volume]:
        """Iterate over visible volumes in the resolved catalog / schema scope.

        ``catalog_name`` and ``schema_name`` default to this service's
        attached scope. When both are set the underlying SDK call is a
        single ``volumes.list`` against ``cat.sch``; otherwise the
        iteration fans out across the matching catalogs / schemas.
        """
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        if not catalog_name:
            for cat in self.client.catalogs.list_catalogs():
                yield from self.list(
                    name=name,
                    catalog_name=cat.catalog_name,
                    schema_name=schema_name,
                )
            return

        if not schema_name:
            for sch in self.client.schemas.list(catalog_name=catalog_name):
                yield from self.list(
                    name=name,
                    catalog_name=catalog_name,
                    schema_name=sch.schema_name,
                )
            return

        uc = self.client.workspace_client().volumes
        logger.debug(
            "Listing volumes in %s.%s (name_filter=%s)",
            catalog_name, schema_name, name,
        )
        for info in uc.list(catalog_name=catalog_name, schema_name=schema_name):
            if name is not None and info.name != name:
                continue
            yield Volume(
                service=self,
                catalog_name=info.catalog_name or catalog_name,
                schema_name=info.schema_name or schema_name,
                volume_name=info.name,
                infos=info,
            )

    # ── parse helper ──────────────────────────────────────────────────────────

    def parse_location(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse a 1- / 2- / 3-part dotted name into ``(catalog, schema, volume)``.

        Keyword overrides take precedence over parts extracted from
        *location*. Service defaults fill any remaining blanks.
        """
        if location:
            if isinstance(location, Volume):
                return location.catalog_name, location.schema_name, location.volume_name

            location = str(location)
            if "/" in location:
                parts = URL.from_(location).parts
            else:
                parts = [p.strip().strip("`") for p in location.split(".")]

            if len(parts) >= 3:
                catalog_name = catalog_name or parts[-3]
                schema_name = schema_name or parts[-2]
                volume_name = volume_name or parts[-1]
            elif len(parts) == 2:
                schema_name = schema_name or parts[0]
                volume_name = volume_name or parts[1]
            elif len(parts) == 1:
                volume_name = volume_name or parts[0]

        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name
        volume_name = volume_name or self.volume_name
        return catalog_name, schema_name, volume_name

    def _resolve_parts(
        self,
        location: str | None,
        catalog_name: str | None,
        schema_name: str | None,
        volume_name: str | None,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        return self.parse_location(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name,
        )

    # ── cache helpers ─────────────────────────────────────────────────────────

    def invalidate(
        self,
        volume: "Volume | str | None" = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> None:
        """Drop one volume's singleton instance.

        The next ``client.volumes[...]`` (or ``Volume(...)``) call will
        mint a fresh instance with an empty info cache — i.e. forces a
        re-read of :class:`VolumeInfo` on first access.
        """
        if isinstance(volume, Volume):
            c, s, v = volume.catalog_name, volume.schema_name, volume.volume_name
        else:
            c, s, v = self._resolve_parts(
                location=volume if isinstance(volume, str) else None,
                catalog_name=catalog_name,
                schema_name=schema_name,
                volume_name=volume_name,
            )
        if not (c and s and v):
            return
        # Singleton key is ``(cls, client, catalog, schema, name)`` —
        # mirror :meth:`Volume._singleton_key` exactly so the pop hits.
        # ``ExpiringDict.pop`` is GIL-atomic; no external mutex needed.
        key = Volume._singleton_key(
            self, catalog_name=c, schema_name=s, volume_name=v,
        )
        Volume._INSTANCES.pop(key, None)

    @classmethod
    def invalidate_all(cls) -> None:
        """Clear the entire singleton cache."""
        Volume._INSTANCES.clear()
