"""``FolderIO`` child name minting — ``part-{epoch_ms}-{seed}.{ext}``.

The format pins are intentional: the millisecond prefix sorts a
plain ``ls`` chronologically, the seed disambiguates concurrent
writers landing in the same millisecond, and the extension flows
from :meth:`_extension_for` so engine-specific suffixes
(``.snappy.parquet``, ``.zstd.parquet``) round-trip unchanged.

Both :meth:`_next_child_name` and :meth:`_next_child_name_in` mint
through :meth:`_mint_part_name`, so non-partitioned and partitioned
write paths produce identically-shaped names.
"""
from __future__ import annotations

import re
import tempfile
import time
import unittest

from yggdrasil.io.nested.folder_io import FolderIO
from yggdrasil.data.enums import MimeTypes


_PART_NAME_RE = re.compile(
    r"^part-(?P<epoch_ms>\d+)-(?P<seed>[0-9a-f]{16})(?:\.(?P<ext>[\w.]+))?$"
)


class TestFolderIOChildNaming(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.io = FolderIO(path=self._tmp.name, auto_open=False)

    def test_next_child_name_matches_part_epoch_seed_form(self) -> None:
        before_ms = int(time.time() * 1000) - 1
        name = self.io._next_child_name(media_type=MimeTypes.PARQUET)
        after_ms = int(time.time() * 1000) + 1

        m = _PART_NAME_RE.match(name)
        self.assertIsNotNone(m, name)
        self.assertEqual(m.group("ext"), "parquet")

        epoch_ms = int(m.group("epoch_ms"))
        self.assertGreaterEqual(epoch_ms, before_ms)
        self.assertLessEqual(epoch_ms, after_ms)

    def test_next_child_name_in_uses_same_form(self) -> None:
        # Partitioned writes go through ``_next_child_name_in`` —
        # same shape as the non-partitioned path.
        name = self.io._next_child_name_in(
            self.io.path, media_type=MimeTypes.PARQUET,
        )
        self.assertIsNotNone(_PART_NAME_RE.match(name), name)

    def test_extension_reflects_media_type(self) -> None:
        for media, expected_ext in (
            (MimeTypes.PARQUET, "parquet"),
            (MimeTypes.CSV, "csv"),
            (MimeTypes.JSON, "json"),
        ):
            with self.subTest(media=media):
                name = self.io._next_child_name(media_type=media)
                m = _PART_NAME_RE.match(name)
                self.assertIsNotNone(m, name)
                self.assertEqual(m.group("ext"), expected_ext)

    def test_seeds_disambiguate_concurrent_calls(self) -> None:
        # 50 calls back-to-back — even if they share a millisecond,
        # the 64-bit seed should keep every name unique.
        names = {
            self.io._next_child_name(media_type=MimeTypes.PARQUET)
            for _ in range(50)
        }
        self.assertEqual(len(names), 50)

    def test_names_are_lexically_sortable_by_time(self) -> None:
        # Wait long enough for the millisecond to advance between
        # calls and confirm the lexical order matches the temporal
        # order.
        first = self.io._next_child_name(media_type=MimeTypes.PARQUET)
        time.sleep(0.005)
        second = self.io._next_child_name(media_type=MimeTypes.PARQUET)
        self.assertLess(first, second)
