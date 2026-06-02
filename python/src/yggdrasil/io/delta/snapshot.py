"""Snapshot — collapsed table state at a specific Delta version."""

from __future__ import annotations

import dataclasses
import json as _json
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Mapping, Optional

from yggdrasil.io.delta.log import DeltaLog, LogSegment
from yggdrasil.io.delta.protocol import (
    AddFile, DeltaAction, DomainMetadata, Metadata, Protocol, RemoveFile, Txn,
)

if TYPE_CHECKING:
    from yggdrasil.path import Path

__all__ = ["Snapshot"]


@dataclasses.dataclass(slots=True)
class Snapshot:
    version: int
    table_root: "Path"
    protocol: Protocol = dataclasses.field(default_factory=Protocol)
    metadata: Optional[Metadata] = None
    active_files: "Dict[str, AddFile]" = dataclasses.field(default_factory=dict)
    txns: "Dict[str, int]" = dataclasses.field(default_factory=dict)
    domain_metadata: "Dict[str, DomainMetadata]" = dataclasses.field(default_factory=dict)

    @classmethod
    def from_log(cls, log: DeltaLog, version: "Optional[int]" = None, *,
                 segment: "Optional[LogSegment]" = None) -> "Snapshot":
        seg = segment or log.segment(version)
        snap = cls(version=seg.version, table_root=log.table_root)
        for a in log.replay(seg):
            if isinstance(a, Protocol):    snap.protocol = a
            elif isinstance(a, Metadata):  snap.metadata = a
            elif isinstance(a, AddFile):   snap.active_files[a.path] = a
            elif isinstance(a, RemoveFile): snap.active_files.pop(a.path, None)
            elif isinstance(a, Txn):       snap.txns[a.app_id] = a.version
            elif isinstance(a, DomainMetadata):
                if a.removed: snap.domain_metadata.pop(a.domain, None)
                else:         snap.domain_metadata[a.domain] = a
        return snap

    @property
    def partition_columns(self) -> "List[str]":
        return list(self.metadata.partition_columns) if self.metadata else []

    @property
    def clustering_columns(self) -> "List[str]":
        """Liquid-clustering columns for this table, in cluster-key order.

        Databricks records liquid clustering in a ``domainMetadata``
        action under the ``delta.clustering`` domain — *not* in
        ``metaData.partitionColumns`` (a clustered table is unpartitioned).
        The domain config looks like::

            {"clusteringColumns": [["region"], ["id"]],
             "domainName": "delta.clustering"}

        where each entry is a *physical column path* (a list, to address
        nested struct fields). We flatten each path with ``.`` so a
        top-level column ``region`` surfaces as ``"region"`` and a nested
        ``addr.zip`` as ``"addr.zip"`` — the same dotted name the read
        predicate uses.

        Older / non-Databricks writers sometimes stamp the columns in
        ``metaData.configuration['delta.clusteringColumns']`` as a
        comma-joined string instead; we accept that shape as a fallback so
        a table written either way reports the same key. Returns ``[]`` when
        the table isn't clustered.
        """
        dm = self.domain_metadata.get("delta.clustering")
        if dm is not None and dm.configuration:
            try:
                cfg = _json.loads(dm.configuration)
            except Exception:
                cfg = None
            if isinstance(cfg, Mapping):
                cols = cfg.get("clusteringColumns")
                if isinstance(cols, list):
                    out: "List[str]" = []
                    for entry in cols:
                        if isinstance(entry, (list, tuple)):
                            out.append(".".join(str(p) for p in entry))
                        elif entry is not None:
                            out.append(str(entry))
                    if out:
                        return out
        if self.metadata is not None:
            raw = self.metadata.configuration.get("delta.clusteringColumns")
            if raw:
                return [c.strip() for c in str(raw).split(",") if c.strip()]
        return []

    @property
    def schema_string(self) -> str:
        return self.metadata.schema_string if self.metadata else ""

    @property
    def configuration(self) -> "Dict[str, str]":
        return dict(self.metadata.configuration) if self.metadata else {}

    @property
    def has_deletion_vectors(self) -> bool:
        return any(f.deletion_vector is not None for f in self.active_files.values())

    def num_active_files(self) -> int:
        return len(self.active_files)

    @property
    def num_rows_approx(self) -> int:
        total = 0
        for f in self.active_files.values():
            if f.stats:
                try: total += _json.loads(f.stats).get("numRecords", 0)
                except Exception: pass
        return total

    def prune_files(self, *, prune_values: "Optional[Mapping[str, Iterable]]" = None) -> "Iterator[AddFile]":
        if not prune_values:
            yield from self.active_files.values()
            return
        normalized = {}
        for col, values in prune_values.items():
            if isinstance(values, (str, bytes, int, float, bool)):
                values = (values,)
            normalized[col] = frozenset("" if v is None else str(v) for v in values)
        for f in self.active_files.values():
            if all(("" if f.partition_values.get(c) is None else str(f.partition_values.get(c))) in accepted
                   for c, accepted in normalized.items()):
                yield f

    def resolve(self, file: AddFile) -> "Path":
        return self.table_root / file.path
