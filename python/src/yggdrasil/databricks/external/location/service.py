"""Collection-level service for Unity Catalog **external locations**.

Wraps the Databricks ``external_locations`` workspace API
(https://docs.databricks.com/api/workspace/externallocations) as a dict-like
:class:`~yggdrasil.databricks.securable.SecurableMapping`::

    client.external_locations["raw"]                          # fetch (KeyError if absent)
    client.external_locations["raw"] = {"url": u, "credential_name": c}  # create / update
    del client.external_locations["raw"]                      # delete
    "raw" in client.external_locations                        # exists
    list(client.external_locations)                           # names

External locations are identified by name (no id), so :meth:`location` /
:meth:`resolve` take a handle or a name.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

from databricks.sdk.service.catalog import ExternalLocationInfo

from yggdrasil.databricks.external.location.resource import ExternalLocation
from yggdrasil.databricks.securable import SecurableMapping

__all__ = ["ExternalLocations"]

logger = logging.getLogger(__name__)


class ExternalLocations(SecurableMapping):
    """Dict-like service over a workspace's Unity Catalog external locations."""

    @property
    def _api(self):
        return self.client.workspace_client().external_locations

    # -- flexible finder ------------------------------------------------
    def resolve(
        self,
        obj: "ExternalLocation | str | None" = None,
        *,
        name: Optional[str] = None,
    ) -> ExternalLocation:
        """Coerce to a lazy :class:`ExternalLocation` handle (a handle is
        returned as-is; a string is the name)."""
        if obj is not None:
            if isinstance(obj, ExternalLocation):
                return obj
            if isinstance(obj, str):
                name = obj
            else:
                raise TypeError(f"expected ExternalLocation | str | None, got {type(obj).__name__}")
        if name is None:
            raise ValueError("provide an ExternalLocation or a name")
        return ExternalLocation(name, service=self)

    location = resolve  # ergonomic alias

    # -- SecurableMapping hooks ----------------------------------------
    def _infos(self) -> Iterator[ExternalLocationInfo]:
        return self._api.list()

    def get_info(self, name: str, *, include_browse: Optional[bool] = None) -> ExternalLocationInfo:
        """Raw :class:`ExternalLocationInfo` for *name* (one GET)."""
        return self._api.get(name, include_browse=include_browse)

    def _resource(self, name: str, info: Any = None) -> ExternalLocation:
        return ExternalLocation(name, service=self, info=info)

    def delete(self, name: str, *, force: bool = False) -> None:
        self._api.delete(name, force=force)

    def _apply(self, name: str, spec: Any, *, exists: bool) -> ExternalLocation:
        spec = self._as_spec(spec)
        if exists:
            return self.update(name, **spec)
        if "url" not in spec or "credential_name" not in spec:
            raise ValueError(
                f"creating external location {name!r} needs 'url' and 'credential_name'"
            )
        return self.create(name, spec.pop("url"), spec.pop("credential_name"), **spec)

    # -- writes ---------------------------------------------------------
    def create(self, name: str, url: str, credential_name: str, **kwargs: Any) -> ExternalLocation:
        """Create an external location bound to *url* + *credential_name*.

        ``**kwargs`` forward to the SDK (``comment`` / ``read_only`` /
        ``skip_validation`` / ``fallback`` / ``enable_file_events`` / …)."""
        info = self._api.create(name=name, url=url, credential_name=credential_name, **kwargs)
        return ExternalLocation(name, service=self, info=info)

    def update(self, name: str, **changes: Any) -> ExternalLocation:
        """Patch an external location (``url`` / ``credential_name`` /
        ``comment`` / ``read_only`` / ``owner`` / …)."""
        info = self._api.update(name, **changes)
        return ExternalLocation(info.name or name, service=self, info=info)
