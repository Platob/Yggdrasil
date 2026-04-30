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
* **Metadata**: :attr:`schema`, :attr:`mtime` as before.
* **Lifecycle**: :attr:`io` — optional live :class:`DataIO` attached
  when the caller wants a read-ready handle. ``None`` when the
  fragment is a pure location descriptor.

Draining a fragment's data is just ``frag.io.read_arrow_batches()``
once :attr:`io` is attached.

Fragment iteration options
--------------------------

:class:`FragmentOptions` carries the knobs that govern enumeration
(``recursive``, ``key`` filter) and IO attachment (``open_io``).
Kept separate from :class:`CastOptions` because fragment-level
concerns are orthogonal to cast / read / write knobs — a single
``read_fragments`` call resolves both options objects, each from
its own kwarg subset.
"""

from __future__ import annotations

import dataclasses
import fnmatch
from typing import TYPE_CHECKING, Any, Iterator, Mapping, TypeVar, Union

from yggdrasil.io.url import URL

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema
    from yggdrasil.io.data import PrimitiveIO

__all__ = [
    "Fragment",
    "FragmentInfos",
]


# ---------------------------------------------------------------------------
# Fragment
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentInfos:
    """Container for fragment information."""
    url: URL
    mtime: float
    schema: Schema | None


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

    :param url: canonical location of the fragment. ``#N``-style URL
        fragments identify sub-locations within a parent file.
    :param schema: the :class:`Schema` of the fragment's rows.
        :class:`None` for fragments built before a schema is known;
        consumers that require a schema call
        :meth:`DataIO.collect_schema` on the attached :attr:`io`.
    :param mtime: source modification time in seconds since epoch,
        inherited from the parent object for sub-fragments. ``0.0``
        when the source can't report a useful mtime (in-memory,
        freshly-minted, transport-only path).
    :param io: live :class:`DataIO` bound to the fragment's location,
        or ``None`` when the fragment is a pure location descriptor.
        When attached, callers drain data via
        ``frag.io.read_arrow_batches()`` (or any other
        :class:`DataIO` surface).
    :param parent: the fragment from which this one was enumerated,
        or ``None`` for roots. Excluded from equality / repr to
        avoid surprising parent-chain comparisons in tests; copy
        helpers preserve it.
    """
    infos: FragmentInfos
    io: "PrimitiveIO"
    parent: "Fragment | None" = dataclasses.field(default=None, repr=False, compare=False)

    # ==================================================================
    # Derived views
    # ==========

    @property
    def depth(self) -> int:
        """Number of ancestors above this fragment.

        ``0`` for roots (fragments yielded directly from a top-level
        :meth:`DataIO.read_fragments`), ``1`` for one-level-deep
        descents, etc. Counts edges to the root, not the root
        itself.
        """
        n = 0
        node = self.parent
        while node is not None:
            n += 1
            node = node.parent
        return n

    @property
    def ancestors(self) -> Iterator["Fragment"]:
        """Walk up the parent chain, yielding each ancestor in turn.

        Yields nearest ancestor first (``self.parent``), then
        grandparent, etc., stopping at the first ``None``-parent
        (a root fragment). ``self`` is NOT yielded — use
        ``itertools.chain((self,), self.ancestors)`` if you want it.

        Cycles are impossible by construction: :attr:`parent` is
        frozen, so a fragment can't reference itself via lineage —
        the parent has to exist before this fragment is built.
        """
        node = self.parent
        while node is not None:
            yield node
            node = node.parent

    @property
    def root(self) -> "Fragment":
        """The top-most ancestor — ``self`` if this is already a root.

        Convenience for "give me the file this row group came from"
        when consumers descended through several layers of nesting.
        """
        node: Fragment = self
        while node.parent is not None:
            node = node.parent
        return node

    # ==================================================================
    # IO / parent attach helpers
    # ==================================================================

    def with_io(self, io: "PrimitiveIO | None") -> "Fragment":
        """Return a copy of this fragment with a different attached IO.

        Used for lifecycle handover — a consumer done with the
        fragment's data can call ``frag.without_io()`` to drop the
        reference; a producer assembling fragments can attach a live
        IO after building the location.

        Preserves :attr:`parent` — the lineage isn't owned by the
        IO, so swapping the IO doesn't change which tree node this
        fragment is.
        """
        return dataclasses.replace(self, io=io)

    def without_io(self) -> "Fragment":
        """Return a copy of this fragment with :attr:`io` set to ``None``."""
        return self.with_io(None)

    def with_parent(self, parent: "Fragment | None") -> "Fragment":
        """Return a copy of this fragment with a different parent link.

        Used by :meth:`DataIO.read_fragments` during recursive
        descent — each child fragment gets its parent stamped on
        as it's yielded out.

        Doesn't validate non-cyclicity (would require walking up
        *parent*'s chain to check ``self`` doesn't appear). The
        recursive walker only ever passes ancestors as parents, so
        cycles aren't reachable through normal use.
        """
        return dataclasses.replace(self, parent=parent)
