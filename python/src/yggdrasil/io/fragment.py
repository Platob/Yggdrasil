"""A pointer to a chunk of data — URL-centric — plus the options
object that governs fragment iteration and persistence.

:class:`Fragment` is the data location primitive used throughout the
:class:`DataIO` fragment surface. The previous version carried both a
``path`` and a ``key``, which duplicated locating information and
forced callers to know which one to use when.

The current shape:

* **One field for "where"**: :attr:`url` — a :class:`URL`. URLs
  natively support ``#fragment`` for sub-locations, so a parquet
  row-group fragment renders as ``file:///tmp/data.parquet#0`` — one
  string captures both the parent file and the intra-file location.
* **Tree shape**: :attr:`parent` — points at the fragment that
  yielded this one during recursive enumeration. ``None`` for roots.
  Lets consumers walk back up to enclosing folders / files /
  partitions without re-running the discovery.
* **Derived**: :attr:`path` (drops the URL fragment), :attr:`key`
  (URL fragment if present, else basename), :attr:`ancestors`
  (iterator up the parent chain), :attr:`depth` (count of
  ancestors).
* **Metadata**: :attr:`schema`, :attr:`mtime`, :attr:`partition_values`
  as before plus partition values for Hive-style children.
* **Lifecycle**: :attr:`io` — optional live :class:`DataIO` attached
  when the caller wants a read-ready handle. ``None`` when the
  fragment is a pure location descriptor.

Draining a fragment's data is just ``frag.io.read_arrow_batches()``
once :attr:`io` is attached.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Iterator, Mapping

from yggdrasil.io.url import URL

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema
    from yggdrasil.io.buffer.primitive import PrimitiveIO

__all__ = [
    "Fragment",
    "FragmentInfos",
]


# ---------------------------------------------------------------------------
# Fragment
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentInfos:
    """Container for fragment information.

    :param url: canonical location of the fragment.
    :param mtime: source modification time in seconds since epoch.
        ``0.0`` when the source can't report a useful mtime.
    :param schema: :class:`Schema` of the fragment's rows. ``None``
        when the schema isn't known yet (cheap-enumeration mode).
    :param partition_values: Hive-style partition key/value pairs
        parsed from the path between the table root and the file —
        e.g. ``{"year": "2025", "month": "04"}`` for
        ``…/year=2025/month=04/part-0.parquet``. Always strings on
        the wire; partition-aware readers cast to the declared
        partition-column dtypes when injecting at read time.
        Empty mapping for non-partitioned children. ``None`` when
        partition discovery hasn't run (e.g. flat :class:`FolderIO`
        which doesn't model partitions at all). Distinguishing
        ``None`` from empty matters because Delta's ``AddFile``
        always carries a ``partitionValues`` map (possibly empty
        for unpartitioned tables) and round-trip identity matters.
    """
    url: URL
    mtime: float
    schema: "Schema | None"
    partition_values: "Mapping[str, str | None] | None" = None


@dataclasses.dataclass(frozen=True, slots=True)
class Fragment:
    """A URL-addressed pointer to a chunk of data, optionally tree-linked.

    The location of the fragment is fully captured by :attr:`url`,
    which can carry a ``#fragment`` selector to identify a sub-region
    of a larger object (e.g. ``file:///data.parquet#0`` = row group 0
    of the parquet file). :attr:`path` and :attr:`key` are derived
    views.

    During recursive enumeration via :meth:`DataIO.read_fragments`,
    each yielded fragment carries a :attr:`parent` reference to the
    fragment from which it was discovered — the root call yields
    fragments with ``parent=None``, descending into a nested IO
    yields fragments with ``parent`` set to the descended-into
    fragment, and so on. Use :attr:`ancestors` / :attr:`depth` for
    ergonomic traversal up the chain.
    """
    infos: FragmentInfos
    io: "PrimitiveIO"
    parent: "Fragment | None" = dataclasses.field(default=None, repr=False, compare=False)

    # ==================================================================
    # Derived views
    # ==================================================================

    @property
    def depth(self) -> int:
        n = 0
        node = self.parent
        while node is not None:
            n += 1
            node = node.parent
        return n

    @property
    def ancestors(self) -> Iterator["Fragment"]:
        node = self.parent
        while node is not None:
            yield node
            node = node.parent

    @property
    def root(self) -> "Fragment":
        node: Fragment = self
        while node.parent is not None:
            node = node.parent
        return node

    # ==================================================================
    # IO / parent attach helpers
    # ==================================================================

    def with_io(self, io: "PrimitiveIO | None") -> "Fragment":
        return dataclasses.replace(self, io=io)

    def without_io(self) -> "Fragment":
        return self.with_io(None)

    def with_parent(self, parent: "Fragment | None") -> "Fragment":
        return dataclasses.replace(self, parent=parent)