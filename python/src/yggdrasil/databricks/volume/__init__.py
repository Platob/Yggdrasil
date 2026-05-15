"""Unity Catalog volume resource + collection service.

:class:`Volume` is the per-volume resource — TTL-cached metadata,
credential vending, and storage-path resolution. Instances are
singletons per ``(host, catalog, schema, volume)`` so every caller
addressing the same UC volume in this process shares one cache and
one credentials refresher.

:class:`Volumes` is the collection-level service —
``client.volumes["cat.sch.vol"]``, ``client.volumes.list(...)``,
and friends. Sits next to :class:`Catalogs` / :class:`Schemas` /
:class:`Tables` in the Unity Catalog service hierarchy.
"""

from __future__ import annotations

from .volume import Volume
from .volumes import Volumes

__all__ = ["Volume", "Volumes"]
