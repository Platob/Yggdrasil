"""Snapshot — collapse a :class:`LogSegment`'s actions into table state.

A snapshot is the table-as-of-a-version:

- :attr:`protocol`        — minReader/Writer + named features.
- :attr:`metadata`        — schema, partition columns, configuration.
- :attr:`active_files`    — :class:`AddFile` for every file still alive.
- :attr:`txns`            — last committed application txn map.
- :attr:`domain_metadata` — last entry per domain.

Reduction: walk the action stream, apply every ``protocol`` /
``metaData`` / ``txn`` to the latest-wins slot, fold ``add`` into the
active-file map, fold ``remove`` to drop a path.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Mapping, Optional

from yggdrasil.io.nested.delta.log import DeltaLog, LogSegment
from yggdrasil.io.nested.delta.protocol import (
    AddFile,
    DeltaAction,
    DomainMetadata,
    Metadata,
    Protocol,
    RemoveFile,
    Txn,
)

if TYPE_CHECKING:
    from yggdrasil.path import Path


__all__ = ["Snapshot"]


@dataclasses.dataclass(slots=True)
class Snapshot:
    """Collapsed table state at a specific Delta version."""

    version: int
    table_root: "Path"
    protocol: Protocol = dataclasses.field(default_factory=Protocol)
    metadata: Optional[Metadata] = None
    active_files: "Dict[str, AddFile]" = dataclasses.field(default_factory=dict)
    txns: "Dict[str, int]" = dataclasses.field(default_factory=dict)
    domain_metadata: "Dict[str, DomainMetadata]" = dataclasses.field(
        default_factory=dict,
    )

    @classmethod
    def from_log(
        cls,
        log: DeltaLog,
        version: "Optional[int]" = None,
        *,
        segment: "Optional[LogSegment]" = None,
    ) -> "Snapshot":
        seg = segment or log.segment(version)
        snap = cls(version=seg.version, table_root=log.table_root)
        snap._apply(log.replay(seg))
        return snap

    # ==================================================================
    # Reduction
    # ==================================================================

    def _apply(self, actions: "Iterable[DeltaAction]") -> None:
        for a in actions:
            self._step(a)

    def _step(self, a: DeltaAction) -> None:
        if isinstance(a, Protocol):
            self.protocol = a
            return
        if isinstance(a, Metadata):
            self.metadata = a
            return
        if isinstance(a, AddFile):
            self.active_files[a.path] = a
            return
        if isinstance(a, RemoveFile):
            self.active_files.pop(a.path, None)
            return
        if isinstance(a, Txn):
            self.txns[a.app_id] = a.version
            return
        if isinstance(a, DomainMetadata):
            if a.removed:
                self.domain_metadata.pop(a.domain, None)
            else:
                self.domain_metadata[a.domain] = a
            return

    # ==================================================================
    # Convenience accessors
    # ==================================================================

    @property
    def partition_columns(self) -> "List[str]":
        return list(self.metadata.partition_columns) if self.metadata else []

    @property
    def schema_string(self) -> str:
        return self.metadata.schema_string if self.metadata else ""

    @property
    def configuration(self) -> "Dict[str, str]":
        return dict(self.metadata.configuration) if self.metadata else {}

    @property
    def has_deletion_vectors(self) -> bool:
        for f in self.active_files.values():
            if f.deletion_vector is not None:
                return True
        return False

    def num_active_files(self) -> int:
        return len(self.active_files)

    @property
    def num_rows_approx(self) -> int:
        """Approximate row count from file stats (when available)."""
        total = 0
        for f in self.active_files.values():
            if f.stats:
                import json
                try:
                    s = json.loads(f.stats)
                    total += s.get("numRecords", 0)
                except Exception:
                    pass
        return total

    # ==================================================================
    # Pruning helpers
    # ==================================================================

    def prune_files(
        self,
        *,
        prune_values: "Optional[Mapping[str, Iterable]]" = None,
    ) -> "Iterator[AddFile]":
        if not prune_values:
            yield from self.active_files.values()
            return

        normalized = {col: _to_str_set(values) for col, values in prune_values.items()}
        for f in self.active_files.values():
            if all(
                _matches(f.partition_values.get(col), accepted)
                for col, accepted in normalized.items()
            ):
                yield f

    # ==================================================================
    # Path resolution
    # ==================================================================

    def resolve(self, file: AddFile) -> "Path":
        return self.table_root / file.path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_str_set(values: "Iterable") -> "frozenset":
    out: list[str] = []
    for v in (
        values if not isinstance(values, (str, bytes, int, float, bool)) else (values,)
    ):
        if v is None:
            out.append("")
        else:
            out.append(str(v))
    return frozenset(out)


def _matches(stored: "Optional[str]", accepted: "frozenset") -> bool:
    if stored is None:
        return "" in accepted
    return str(stored) in accepted
