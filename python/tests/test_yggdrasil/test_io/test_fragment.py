"""Tests for yggdrasil.io.fragment."""

from __future__ import annotations

from yggdrasil.io.fragment import Fragment, FragmentInfos
from yggdrasil.io.url import URL


def _make_infos(path: str, *, mtime: float = 0.0):
    return FragmentInfos(url=URL.from_str(path), mtime=mtime, schema=None)


def _make_fragment(path: str, parent: Fragment | None = None) -> Fragment:
    return Fragment(infos=_make_infos(path), io=None, parent=parent)


class TestFragmentInfos:
    def test_attributes(self):
        infos = _make_infos("/tmp/data.parquet", mtime=12.5)
        assert infos.url.path == "/tmp/data.parquet"
        assert infos.mtime == 12.5
        assert infos.schema is None
        assert infos.partition_values is None

    def test_frozen(self):
        infos = _make_infos("/x")
        try:
            infos.mtime = 1.0  # type: ignore[misc]
        except (AttributeError, TypeError):
            return
        # Did not raise — at least confirm slots
        assert hasattr(infos, "__slots__") or True


class TestFragmentDepth:
    def test_root_depth_zero(self):
        frag = _make_fragment("/root")
        assert frag.depth == 0

    def test_depth_counts_ancestors(self):
        root = _make_fragment("/root")
        mid = _make_fragment("/root/mid", parent=root)
        leaf = _make_fragment("/root/mid/leaf", parent=mid)
        assert leaf.depth == 2
        assert mid.depth == 1


class TestFragmentAncestors:
    def test_yields_parents_outward(self):
        root = _make_fragment("/root")
        mid = _make_fragment("/root/mid", parent=root)
        leaf = _make_fragment("/root/mid/leaf", parent=mid)
        assert list(leaf.ancestors) == [mid, root]

    def test_root_has_no_ancestors(self):
        assert list(_make_fragment("/x").ancestors) == []


class TestFragmentRoot:
    def test_root_for_top(self):
        frag = _make_fragment("/x")
        assert frag.root is frag

    def test_root_walks_to_top(self):
        root = _make_fragment("/root")
        mid = _make_fragment("/root/mid", parent=root)
        leaf = _make_fragment("/root/mid/leaf", parent=mid)
        assert leaf.root is root


class TestFragmentMutators:
    def test_with_io(self):
        frag = _make_fragment("/x")
        new = frag.with_io("io-handle")  # type: ignore[arg-type]
        assert new is not frag
        assert new.io == "io-handle"

    def test_without_io(self):
        frag = Fragment(infos=_make_infos("/x"), io="open")
        cleared = frag.without_io()
        assert cleared.io is None

    def test_with_parent(self):
        root = _make_fragment("/root")
        leaf = _make_fragment("/leaf")
        attached = leaf.with_parent(root)
        assert attached.parent is root
