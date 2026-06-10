"""Unit tests for :mod:`yggdrasil.http_.cache_config`."""
from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.enums import Mode
from yggdrasil.http_.cache_config import (
    CacheConfig,
    DEFAULT_CACHE_CONFIG,
    MATCH_COLUMN,
    MATCH_KEY,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestCacheConfigConstruction:

    def test_default_values(self):
        c = CacheConfig()
        assert c.mode is Mode.APPEND
        assert c.anonymize == "remove"
        assert c.cleanup_ttl == dt.timedelta(days=1)
        assert c.tabular is None
        assert c.received_from is None
        assert c.received_to is None

    def test_custom_mode(self):
        c = CacheConfig(mode=Mode.OVERWRITE)
        assert c.mode is Mode.OVERWRITE

    def test_custom_anonymize(self):
        c = CacheConfig(anonymize="redact")
        assert c.anonymize == "redact"

    def test_custom_cleanup_ttl(self):
        ttl = dt.timedelta(hours=6)
        c = CacheConfig(cleanup_ttl=ttl)
        assert c.cleanup_ttl == ttl

    def test_cleanup_ttl_none_disables(self):
        c = CacheConfig(cleanup_ttl=None)
        assert c.cleanup_ttl is None

    def test_mode_coerced_from_string(self):
        # Mode.from_ accepts string inputs
        c = CacheConfig(mode="overwrite")
        assert c.mode is Mode.OVERWRITE

    def test_from_none_returns_default_singleton(self):
        result = CacheConfig.from_(None)
        assert result is CacheConfig.default()

    def test_from_cache_config_returns_same_instance(self):
        c = CacheConfig(mode=Mode.OVERWRITE)
        assert CacheConfig.from_(c) is c

    def test_from_dict_applies_values(self):
        c = CacheConfig.from_({"mode": "overwrite", "anonymize": "redact"})
        assert c.mode is Mode.OVERWRITE
        assert c.anonymize == "redact"

    def test_from_empty_dict_returns_default_singleton(self):
        result = CacheConfig.from_({})
        assert result is CacheConfig.default()

    def test_from_cache_config_with_overrides_creates_new(self):
        original = CacheConfig(mode=Mode.OVERWRITE)
        merged = CacheConfig.from_(original, anonymize="redact")
        assert merged is not original
        assert merged.mode is Mode.OVERWRITE
        assert merged.anonymize == "redact"

    def test_from_none_with_overrides(self):
        result = CacheConfig.from_(None, mode=Mode.OVERWRITE)
        assert result.mode is Mode.OVERWRITE
        assert result is not CacheConfig.default()

    def test_default_is_singleton(self):
        d1 = CacheConfig.default()
        d2 = CacheConfig.default()
        assert d1 is d2


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


class TestCacheConfigEquality:

    def test_equal_defaults(self):
        a = CacheConfig()
        b = CacheConfig()
        assert a == b

    def test_not_equal_different_mode(self):
        a = CacheConfig(mode=Mode.APPEND)
        b = CacheConfig(mode=Mode.OVERWRITE)
        assert a != b

    def test_not_equal_different_anonymize(self):
        a = CacheConfig(anonymize="remove")
        b = CacheConfig(anonymize="redact")
        assert a != b

    def test_not_equal_different_cleanup_ttl(self):
        a = CacheConfig(cleanup_ttl=dt.timedelta(days=1))
        b = CacheConfig(cleanup_ttl=dt.timedelta(hours=6))
        assert a != b

    def test_not_equal_cleanup_ttl_none_vs_set(self):
        a = CacheConfig(cleanup_ttl=None)
        b = CacheConfig(cleanup_ttl=dt.timedelta(days=1))
        assert a != b

    def test_eq_returns_not_implemented_for_non_cache_config(self):
        c = CacheConfig()
        assert c.__eq__(42) is NotImplemented
        assert c.__eq__("foo") is NotImplemented
        assert c.__eq__(None) is NotImplemented

    def test_hash_differs_for_different_configs(self):
        a = CacheConfig(mode=Mode.APPEND)
        b = CacheConfig(mode=Mode.OVERWRITE)
        # Different configs should (almost certainly) have different hashes
        assert hash(a) != hash(b)

    def test_hash_consistent_for_equal_configs(self):
        a = CacheConfig()
        b = CacheConfig()
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestCacheConfigRepr:

    def test_repr_default_is_minimal(self):
        c = CacheConfig()
        assert repr(c) == "CacheConfig()"

    def test_repr_includes_mode_when_not_append(self):
        c = CacheConfig(mode=Mode.OVERWRITE)
        r = repr(c)
        assert "CacheConfig(" in r
        assert "mode=" in r
        assert "OVERWRITE" in r

    def test_repr_omits_mode_when_append(self):
        c = CacheConfig(mode=Mode.APPEND)
        assert "mode=" not in repr(c)

    def test_repr_round_trips_format(self):
        c = CacheConfig()
        assert repr(c).startswith("CacheConfig(")
        assert repr(c).endswith(")")


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestReceivedWindowTruncation:
    """received_from / received_to snap to 15-minute blocks (not 1 hour)."""

    def test_received_from_floors_to_15min(self):
        c = CacheConfig(received_from=dt.datetime(2024, 1, 1, 10, 7, 30, tzinfo=dt.timezone.utc))
        assert c.received_from == dt.datetime(2024, 1, 1, 10, 0, tzinfo=dt.timezone.utc)

    def test_received_to_ceils_to_15min(self):
        c = CacheConfig(received_to=dt.datetime(2024, 1, 1, 10, 7, 30, tzinfo=dt.timezone.utc))
        assert c.received_to == dt.datetime(2024, 1, 1, 10, 15, tzinfo=dt.timezone.utc)

    def test_already_aligned_unchanged(self):
        c = CacheConfig(received_from=dt.datetime(2024, 1, 1, 10, 30, tzinfo=dt.timezone.utc))
        assert c.received_from == dt.datetime(2024, 1, 1, 10, 30, tzinfo=dt.timezone.utc)


class TestConstants:

    def test_match_key_value(self):
        assert MATCH_KEY == "public_hash"

    def test_match_column_value(self):
        assert MATCH_COLUMN == "request_public_hash"

    def test_match_key_is_str(self):
        assert isinstance(MATCH_KEY, str)

    def test_match_column_is_str(self):
        assert isinstance(MATCH_COLUMN, str)


class TestRefetch:
    """Refresh-mode caches bypass the read so the wire response replaces them."""

    def test_append_serves_hits(self):
        assert CacheConfig(mode=Mode.APPEND).refetch is False

    def test_refresh_modes_refetch(self):
        for mode in (Mode.UPSERT, Mode.MERGE, Mode.OVERWRITE, Mode.TRUNCATE):
            assert CacheConfig(mode=mode).refetch is True

    def test_received_window_overrides_refetch(self):
        c = CacheConfig(
            mode=Mode.UPSERT,
            received_from=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        )
        assert c.refetch is False


# ---------------------------------------------------------------------------
# DEFAULT_CACHE_CONFIG singleton
# ---------------------------------------------------------------------------


class TestDefaultCacheConfig:

    def test_exists_and_is_cache_config(self):
        assert isinstance(DEFAULT_CACHE_CONFIG, CacheConfig)

    def test_has_default_values(self):
        assert DEFAULT_CACHE_CONFIG.mode is Mode.APPEND
        assert DEFAULT_CACHE_CONFIG.anonymize == "remove"
        assert DEFAULT_CACHE_CONFIG.cleanup_ttl == dt.timedelta(days=1)
        assert DEFAULT_CACHE_CONFIG.tabular is None
        assert DEFAULT_CACHE_CONFIG.received_from is None
        assert DEFAULT_CACHE_CONFIG.received_to is None

    def test_equals_freshly_constructed_default(self):
        assert DEFAULT_CACHE_CONFIG == CacheConfig()
