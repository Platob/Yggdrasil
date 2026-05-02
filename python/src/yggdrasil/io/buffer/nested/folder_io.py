"""Generic folder-of-tabular-files :class:`NestedIO` leaf.

:class:`FolderIO` is the canonical concrete :class:`NestedIO`: a
directory whose immediate children are tabular files (Parquet, IPC,
CSV, JSON, ...), and whose data is the union of those children's
data. No transaction log, no manifest, no partition discipline —
just a directory.

It registers against :data:`MimeTypes.FOLDER` (``inode/directory``)
so :meth:`TabularIO.from_path` lands on it whenever a path resolves
to a directory via :attr:`Path.is_dir_sink`.

What it does
------------

- :meth:`iter_fragments` lists the directory once, filters out
  ignored entries (hidden files by default), and yields one
  :class:`Fragment` per remaining child. Each fragment carries a
  fresh :class:`PrimitiveIO` bound to the child path; the IO's
  format is inferred from the child's extension via
  :meth:`TabularIO.from_path`.
- :meth:`_make_child_io` is the inverse: build a fresh child
  :class:`PrimitiveIO` under :attr:`path` for a given name.
- Read / write / save-mode resolution / staging — all inherited
  from :class:`NestedIO`.

What it doesn't do
------------------

- Schema homogeneity enforcement. Heterogeneous-children folders
  work but ``collect_schema`` returns the union (via
  ``Schema.merge_with``); writers assume the caller knows what
  they're doing.
- Recursive enumeration. ``FolderIO`` is one level deep —
  subfolders are *not* recursed into. A folder-of-folders
  arrangement is a different IO (Hive-partitioned), with a
  different enumeration discipline.
- Transaction semantics. APPEND adds a sibling, OVERWRITE clears
  and writes; there's no atomicity beyond the staging-rename
  pattern of a single child file.

Subclasses
----------

:class:`DeltaIO` (in ``yggdrasil.io.nested.delta``) inherits from
:class:`FolderIO` and overrides:

- :meth:`_is_ignored_path` to hide ``_delta_log/``;
- :meth:`_finalize_child` to write a commit-log entry instead of
  a plain rename;
- :meth:`iter_fragments` to read the commit log, not the
  directory.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterator

from yggdrasil.io.buffer.primitive import PrimitiveIO
from yggdrasil.io.enums import MimeType, MimeTypes
from yggdrasil.io.fragment import Fragment, FragmentInfos
from yggdrasil.io.fs import Path
from yggdrasil.io.tabular import TabularIO
from .base import NestedIO, NestedOptions

__all__ = ["FolderIO"]


class FolderIO(NestedIO[NestedOptions]):
    """A directory of homogeneous tabular files.

    Reads enumerate directory children, treating each as a
    :class:`PrimitiveIO`; writes mint a new child per call (or per
    ``options.child_row_size`` chunk) using staging + rename.

    Construction:

    >>> io = FolderIO(path="/tmp/parquet-store/")
    >>> for frag in io.iter_fragments():
    ...     print(frag.infos.url, frag.io.collect_schema())

    The folder is auto-created on first write. Reading a missing
    folder yields no fragments (``iter_fragments`` returns empty);
    explicit existence checks live on ``self.path.exists()``.
    """

    # Marker: this is a leaf, ``TabularIO.__new__`` skips dispatch.
    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_mime_type(cls) -> "MimeType | None":
        """Register against ``inode/directory``.

        :data:`MimeTypes.FOLDER` is what :class:`Path.infer_media_type`
        returns when ``is_dir_sink`` is True, so any directory path
        passed to :meth:`TabularIO.from_path` lands here.
        """
        return MimeTypes.FOLDER

    # ==================================================================
    # Fragment enumeration
    # ==================================================================

    def iter_fragments(
        self,
        options: "NestedOptions | None" = None,
        **kwargs: Any,
    ) -> Iterator[Fragment]:
        """Yield one :class:`Fragment` per non-ignored child file.

        Walks :attr:`path` once via :meth:`Path.iterdir`, filters
        out ignored entries, and for each survivor:

        1. Builds a child :class:`PrimitiveIO` via
           :meth:`TabularIO.from_path` — the format is inferred
           from the child's extension.
        2. Builds a :class:`FragmentInfos` carrying the child URL
           and (when ``populate_metadata`` is set) its mtime and
           schema.
        3. Yields a :class:`Fragment` linking the two, with
           ``parent=None`` (this is a root-level enumeration).

        Consumers that recursively walk into nested IOs are
        responsible for stamping ``parent`` themselves via
        :meth:`Fragment.with_parent` — see
        :meth:`NestedIO._read_arrow_batches` for an example that
        chains children's batches but doesn't preserve lineage
        (lineage isn't needed for batch reading).

        Missing folder is treated as empty: no fragments yielded,
        no error raised. This matches the rest of yggdrasil's
        "reading nothing yields nothing" convention.
        """
        opts = self.check_options(options, overrides=locals())

        if not self.path.exists():
            return

        populate = bool(opts.populate_metadata)

        for child in self.path.iterdir():
            if self._is_ignored_path(child):
                continue
            # Skip subdirectories. ``FolderIO`` is one level deep;
            # nested folders need a Hive-partitioned subclass.
            try:
                if child.is_dir():
                    continue
            except Exception:
                # Stat failure on a child mid-listing — skip rather
                # than abort. Listings on remote stores can race
                # with deletes.
                continue

            child_io = self._open_child_for_read(child)
            if child_io is None:
                # No registered tabular IO for this extension —
                # skip rather than crash. Lets folders containing
                # README files etc. enumerate cleanly.
                continue

            yield Fragment(
                infos=self._build_fragment_infos(child, child_io, populate),
                io=child_io,
            )

    def _open_child_for_read(self, child: Path) -> "PrimitiveIO | None":
        """Build a :class:`PrimitiveIO` for an existing child file.

        Goes through :meth:`TabularIO.from_path` so the registry
        picks the right format leaf. Returns ``None`` if no leaf
        is registered for the child's extension — caller skips.
        """
        try:
            io = TabularIO.from_path(child)
        except Exception:
            return None
        # ``from_path`` may return any TabularIO; folder children
        # should be primitives. Safety net: if a directory child
        # somehow resolved to another NestedIO, treat as unknown
        # to keep the recursion shape predictable.
        from yggdrasil.io.buffer.primitive import PrimitiveIO

        if not isinstance(io, PrimitiveIO):
            return None
        return io

    def _build_fragment_infos(
        self,
        child: Path,
        child_io: "PrimitiveIO",
        populate: bool,
    ) -> FragmentInfos:
        """Build a :class:`FragmentInfos` for a yielded child.

        When ``populate`` is True, eagerly collects schema and
        mtime. Each costs one extra round trip to the storage
        (mtime = stat, schema = footer/header read), so default
        is off; consumers that need them call back lazily.
        """
        if populate:
            try:
                mtime = child.mtime or 0.0
            except Exception:
                mtime = 0.0
            try:
                schema = child_io.collect_schema()
            except Exception:
                schema = None
        else:
            mtime = 0.0
            schema = None

        return FragmentInfos(
            url=child.url,
            mtime=mtime,
            schema=schema,
        )

    # ==================================================================
    # Cheap is_empty — avoid full iteration
    # ==================================================================

    def is_empty(self) -> bool:
        """True if no non-ignored children exist.

        Overrides the base which iterates fragments (each fragment
        builds a child IO — wasted work when we just need a yes/no).
        Walks ``iterdir`` directly and stops on the first non-
        ignored file.
        """
        if not self.path.exists():
            return True
        for child in self.path.iterdir():
            if self._is_ignored_path(child):
                continue
            try:
                if child.is_dir():
                    continue
            except Exception:
                continue
            return False
        return True

    # ==================================================================
    # Child IO factory — write side
    # ==================================================================

    def _make_child_io(
        self,
        name: str,
        media_type: Any = None,
    ) -> "PrimitiveIO":
        """Build a fresh write target under :attr:`path`.

        ``name`` may include forward-slash separators for nested
        layouts (Hive partitions, Delta-lake-style sub-prefixes);
        the parent directories are auto-created. Backslashes are
        rejected (Windows separator conflicts with URL semantics)
        and ``..`` segments are rejected (path traversal).

        The child's parent is materialized via :meth:`Path.mkdir`
        so the child writer's ``open_io`` doesn't need to do it.

        Returns a closed (un-acquired) :class:`PrimitiveIO`. The
        caller (:meth:`NestedIO._write_one_child`) opens it inside
        a ``with`` block; on close, the bound-path write-back
        flushes the bytes to the child file.
        """
        if "\\" in name:
            raise ValueError(
                f"Child name must not contain backslashes; got {name!r}. "
                "Use forward slashes for nested paths."
            )
        if name.startswith("/"):
            raise ValueError(
                f"Child name must be relative; got {name!r}."
            )
        # Reject '..' as a standalone segment to prevent path traversal.
        # We deliberately don't reject '..' substrings (a column called
        # 'foo..bar' is fine in a filename).
        segments = name.split("/")
        if any(s == ".." for s in segments):
            raise ValueError(
                f"Child name must not contain '..' segments; got {name!r}."
            )

        child_path = self.path.joinpath(*segments)
        # Ensure the parent exists before the child writer tries to
        # open. ``mkdir(parents=True, exist_ok=True)`` is idempotent;
        # it covers both the nested-partition case and the flat case.
        child_path.parent.mkdir(parents=True, exist_ok=True)

        io = TabularIO.from_path(child_path, media_type=media_type)

        if not isinstance(io, PrimitiveIO):
            raise TypeError(
                f"FolderIO child factory expected a PrimitiveIO for "
                f"name={name!r}, media_type={media_type!r}; got "
                f"{type(io).__name__}. Folders of nested IOs need a "
                "partition-aware subclass."
            )
        return io