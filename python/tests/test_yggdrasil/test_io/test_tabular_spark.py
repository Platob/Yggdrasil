"""Unit-level tests for :class:`yggdrasil.io.tabular.SparkTabular`.

These cover the merge of :class:`SparkTabular` with the former
:class:`yggdrasil.spark.frame.Dataset`:

* both ``frame=`` and ``df=`` constructor spellings, plus the ``df``
  property alias on the instance;
* :class:`Dataset` is an alias for :class:`SparkTabular`;
* the Spark executor-cache logic on :meth:`persist` / :meth:`unpersist`
  — skip when already cached, call ``df.persist`` /
  ``df.unpersist`` exactly once per round trip, accept
  storage-level strings.

Real Spark integration tests (transforms / collect / arrow round-trip)
live under :class:`yggdrasil.spark.tests.SparkTestCase` and skip when
pyspark is missing. This file stays pyspark-free so the merge surface
is checked even in the base install.
"""

from __future__ import annotations

import unittest
from unittest import mock

from yggdrasil.io.tabular import SparkTabular


class _FakeStorageLevel:
    """Stand-in for :class:`pyspark.StorageLevel` constants in tests."""

    MEMORY_AND_DISK = "MEMORY_AND_DISK"
    MEMORY_ONLY = "MEMORY_ONLY"
    DISK_ONLY = "DISK_ONLY"
    MEMORY_ONLY_SER = "MEMORY_ONLY_SER"
    MEMORY_AND_DISK_SER = "MEMORY_AND_DISK_SER"
    MEMORY_AND_DISK_2 = "MEMORY_AND_DISK_2"
    MEMORY_ONLY_2 = "MEMORY_ONLY_2"
    DISK_ONLY_2 = "DISK_ONLY_2"


def _fake_frame(*, is_cached: bool = False) -> mock.MagicMock:
    """Build a :class:`pyspark.sql.DataFrame` stand-in with a settable
    ``is_cached`` flag."""
    frame = mock.MagicMock(name="fake-spark-df")
    frame.is_cached = is_cached
    return frame


class TestSparkTabularConstruction(unittest.TestCase):
    """Constructor accepts both ``frame=`` and ``df=`` shapes."""

    def test_frame_keyword(self) -> None:
        df = _fake_frame()
        io = SparkTabular(frame=df)
        self.assertIs(io.frame, df)
        self.assertIs(io.df, df)

    def test_df_keyword_legacy_alias(self) -> None:
        df = _fake_frame()
        io = SparkTabular(df=df)
        self.assertIs(io.frame, df)
        self.assertIs(io.df, df)

    def test_positional_frame(self) -> None:
        df = _fake_frame()
        io = SparkTabular(df)
        self.assertIs(io.frame, df)

    def test_both_frame_and_df_raises(self) -> None:
        with self.assertRaises(TypeError):
            SparkTabular(frame=_fake_frame(), df=_fake_frame())

    def test_empty(self) -> None:
        io = SparkTabular()
        self.assertIsNone(io.frame)
        self.assertTrue(io.is_empty())
        self.assertFalse(bool(io))
        self.assertFalse(io.cached)

    def test_df_setter_routes_to_frame_slot(self) -> None:
        # Old Dataset code does ``self.df = ...``; the merged class
        # routes the setter to ``_frame`` so the legacy spelling keeps
        # working.
        df = _fake_frame()
        io = SparkTabular()
        io.df = df
        self.assertIs(io.frame, df)


def _pyspark_installed() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(
    _pyspark_installed(),
    "yggdrasil.spark.frame requires pyspark at import time",
)
class TestSparkTabularDatasetAlias(unittest.TestCase):
    """:class:`Dataset` resolves to the same class as :class:`SparkTabular`."""

    def test_dataset_is_spark_tabular(self) -> None:
        from yggdrasil.spark.frame import Dataset

        self.assertIs(Dataset, SparkTabular)

    def test_old_dataset_callsite_still_works(self) -> None:
        from yggdrasil.spark.frame import Dataset

        df = _fake_frame()
        ds = Dataset(df)
        # The old ``Dataset`` exposed ``.df`` and ``.schema`` and
        # ``.installed_modules`` — all still present on the merged class.
        self.assertIs(ds.df, df)
        self.assertIsNone(ds.schema)
        self.assertEqual(ds.installed_modules, set())


class TestSparkTabularIsCached(unittest.TestCase):
    """:attr:`is_cached` mirrors the underlying frame, defaults to ``False``."""

    def test_empty_holder_not_cached(self) -> None:
        self.assertFalse(SparkTabular().is_cached)

    def test_uncached_frame_not_cached(self) -> None:
        io = SparkTabular(_fake_frame(is_cached=False))
        self.assertFalse(io.is_cached)

    def test_cached_frame_is_cached(self) -> None:
        io = SparkTabular(_fake_frame(is_cached=True))
        self.assertTrue(io.is_cached)


class TestSparkTabularPersistCache(unittest.TestCase):
    """:meth:`persist` calls ``df.persist`` with skip-when-cached semantics."""

    def setUp(self) -> None:
        # Patch :mod:`pyspark` so :meth:`SparkTabular._resolve_storage_level`
        # finds a fake ``StorageLevel`` to dispatch against, even though
        # pyspark isn't installed in the test environment.
        self._pyspark_patch = mock.patch.dict(
            "sys.modules",
            {"pyspark": mock.MagicMock(StorageLevel=_FakeStorageLevel)},
        )
        self._pyspark_patch.start()

    def tearDown(self) -> None:
        self._pyspark_patch.stop()

    def test_persist_on_empty_holder_is_noop(self) -> None:
        io = SparkTabular()
        out = io.persist()
        self.assertIs(out, io)

    def test_persist_calls_df_persist_with_memory_and_disk_default(self) -> None:
        df = _fake_frame(is_cached=False)
        io = SparkTabular(df)
        io.persist()
        df.persist.assert_called_once_with(_FakeStorageLevel.MEMORY_AND_DISK)

    def test_persist_skipped_when_already_cached(self) -> None:
        df = _fake_frame(is_cached=True)
        io = SparkTabular(df)
        io.persist()
        df.persist.assert_not_called()

    def test_persist_storage_level_string(self) -> None:
        df = _fake_frame()
        io = SparkTabular(df)
        io.persist(storage_level="MEMORY_ONLY")
        df.persist.assert_called_once_with(_FakeStorageLevel.MEMORY_ONLY)

    def test_persist_storage_level_unknown_string_raises(self) -> None:
        df = _fake_frame()
        io = SparkTabular(df)
        with self.assertRaises(ValueError) as cm:
            io.persist(storage_level="NOT_A_REAL_LEVEL")
        self.assertIn("Unknown StorageLevel", str(cm.exception))
        # Frame was untouched on the failed dispatch.
        df.persist.assert_not_called()

    def test_persist_data_replaces_frame_before_caching(self) -> None:
        # Legacy ``Statement``-shim path: ``persist(data=...)`` swaps the
        # held frame, then the executor cache kicks in.
        new_df = _fake_frame()
        io = SparkTabular()
        # ``_coerce_frame`` would normally go through Spark; stub it for
        # this unit so we don't need the engine wired up.
        with mock.patch.object(io, "_coerce_frame", return_value=new_df):
            io.persist(data="fake-payload")
        self.assertIs(io.frame, new_df)
        new_df.persist.assert_called_once()

    def test_persist_swallows_persist_exceptions(self) -> None:
        # Spark Connect can reject ``persist`` on an unmaterialisable
        # logical plan — the cache call shouldn't crash the caller.
        df = _fake_frame(is_cached=False)
        df.persist.side_effect = RuntimeError("connect rejection")
        io = SparkTabular(df)
        out = io.persist()  # should not raise
        self.assertIs(out, io)


class TestSparkTabularUnpersist(unittest.TestCase):
    """:meth:`unpersist` evicts the executor cache then drops the local ref."""

    def test_unpersist_drops_local_reference(self) -> None:
        df = _fake_frame()
        io = SparkTabular(df)
        io.unpersist()
        self.assertIsNone(io.frame)
        self.assertFalse(io.cached)

    def test_unpersist_calls_df_unpersist_when_cached(self) -> None:
        df = _fake_frame(is_cached=True)
        io = SparkTabular(df)
        io.unpersist()
        df.unpersist.assert_called_once()
        self.assertIsNone(io.frame)

    def test_unpersist_skips_df_unpersist_when_not_cached(self) -> None:
        df = _fake_frame(is_cached=False)
        io = SparkTabular(df)
        io.unpersist()
        df.unpersist.assert_not_called()
        self.assertIsNone(io.frame)

    def test_unpersist_swallows_df_unpersist_exceptions(self) -> None:
        df = _fake_frame(is_cached=True)
        df.unpersist.side_effect = RuntimeError("connect rejection")
        io = SparkTabular(df)
        io.unpersist()
        self.assertIsNone(io.frame)


class TestSparkTabularProxyForwarding(unittest.TestCase):
    """``__getattr__`` forwards to the held DataFrame and wraps DF returns."""

    def test_attribute_proxies_to_frame(self) -> None:
        df = _fake_frame()
        df.something = "value"
        io = SparkTabular(df)
        self.assertEqual(io.something, "value")

    def test_callable_attribute_returns_proxy(self) -> None:
        df = _fake_frame()
        df.some_method = mock.MagicMock(return_value="non-df-result")
        io = SparkTabular(df)
        proxied = io.some_method
        # The proxy is callable and forwards to the underlying method.
        self.assertEqual(proxied("a", b=1), "non-df-result")
        df.some_method.assert_called_once_with("a", b=1)

    def test_attribute_lookup_on_empty_raises(self) -> None:
        io = SparkTabular()
        with self.assertRaises(AttributeError):
            io.something_random
