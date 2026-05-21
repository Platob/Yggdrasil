"""Snapshot — collapse a :class:`LogSegment`'s actions into table state.

A snapshot is the table-as-of-a-version:

- :attr:`protocol`        — minReader/Writer + named features.
- :attr:`metadata`        — schema, partition columns, configuration.
- :attr:`active_files`    — :class:`AddFile` for every file still alive.
- :attr:`txns`            — last committed application txn map.
- :attr:`domain_metadata` — last entry per domain (modern Delta extras).

Reduction is: walk the action stream, apply every ``protocol`` /
``metaData`` / ``txn`` to the latest-wins slot, fold ``add`` into the
active-file map, and fold ``remove`` to drop a path. The order
:meth:`DeltaLog.replay` delivers actions in (checkpoint first,
commits in version order, file order within each commit) is the
spec's required reduction order — there is exactly one consistent
state at any version.

We don't try to be clever: the active map keeps :class:`AddFile`
values keyed by ``path``, an ``add`` overwrites a same-path entry
(recommit-with-new-DV is a real wire shape — Delta updates a DV by
emitting a remove + add pair, but a same-path readd is also legal),
and a ``remove`` deletes the entry outright. ``RemoveFile`` actions
that name a path we never saw are silently dropped — the spec
explicitly tolerates this for vacuum-tail interleaving.
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
    from yggdrasil.io.path import Path


__all__ = ["Snapshot"]


@dataclasses.dataclass(slots=True)
class Snapshot:
    """Collapsed table state at a specific Delta version."""

    version: int
    table_root: "Path"
    protocol: Protocol = dataclasses.field(default_factory=Protocol)
    metadata: Optional[Metadata] = None
    #: Active files keyed by their *table-relative* path. Insertion
    #: order is preserved so a deterministic replay surface stays.
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
        """Build a snapshot. Pass *segment* to skip the resolution pass."""
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
        # CommitInfo & unknown actions: ignored on purpose.

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
        """True iff at least one active file carries a DV descriptor."""
        for f in self.active_files.values():
            if f.deletion_vector is not None:
                return True
        return False

    def num_active_files(self) -> int:
        return len(self.active_files)

    # ==================================================================
    # Pruning helpers
    # ==================================================================

    def prune_files(
        self,
        *,
        prune_values: "Optional[Mapping[str, Iterable]]" = None,
    ) -> "Iterator[AddFile]":
        """Yield active files whose partition values pass *prune_values*.

        Empty / ``None`` prune dict → yield everything. Per-column
        ``IN``-set semantics — same shape :class:`FolderPath` uses
        for its prune knob, so callers compose without re-mapping.
        """
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
        """Build the absolute :class:`Path` for an :class:`AddFile`.

        Delta paths are URL-quoted relative paths; :class:`Path`'s
        joinpath handles the unquote when constructing the URL.
        """
        return self.table_root / file.path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_str_set(values: "Iterable") -> "frozenset":
    """Normalize a prune ``IN``-set to strings.

    Delta stores partition values as strings on disk (``"42"`` for an
    int partition column, ``""`` for null) — comparing the prune set
    in string-space avoids a coerce-on-every-row overhead.
    """
    out: list[str] = []
    for v in (
        values if not isinstance(values, (str, bytes, int, float, bool)) else (values,)
    ):
        if v is None:
            out.append("")  # Hive default — Delta stores None as ""
        else:
            out.append(str(v))
    return frozenset(out)


def _matches(stored: "Optional[str]", accepted: "frozenset") -> bool:
    if stored is None:
        return "" in accepted
    return str(stored) in accepted
