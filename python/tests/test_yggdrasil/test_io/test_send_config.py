"""Tests for yggdrasil.io.send_config."""

from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.io.enums import Mode
from yggdrasil.io.send_config import CacheConfig, SendConfig, SendManyConfig

from ._helpers import make_request, make_response


# ---------------------------------------------------------------------------
# CacheConfig
# ---------------------------------------------------------------------------


class TestCacheConfigDefaults:
    def test_default_factory(self):
        cfg = CacheConfig.default()
        assert isinstance(cfg, CacheConfig)
        assert cfg.mode is Mode.APPEND

    def test_check_arg_none_returns_default(self):
        cfg = CacheConfig.check_arg(None)
        assert cfg == CacheConfig.default()


class TestCacheConfigParseMapping:
    def test_basic_mapping(self):
        cfg = CacheConfig.parse_mapping({"mode": "overwrite"})
        assert cfg.mode is Mode.OVERWRITE

    def test_invalid_request_by_raises(self):
        with pytest.raises(ValueError):
            CacheConfig(request_by=["not_a_real_field"])

    def test_received_ttl_back_calculates_from(self):
        cfg = CacheConfig(received_ttl=dt.timedelta(hours=1))
        assert cfg.received_to is not None
        assert cfg.received_from is not None
        delta = cfg.received_to - cfg.received_from
        assert abs(delta.total_seconds() - 3600) < 1


class TestCacheConfigCheckArgPolymorphism:
    def test_passthrough(self):
        cfg = CacheConfig()
        assert CacheConfig.check_arg(cfg) is cfg

    def test_dict_arg(self):
        cfg = CacheConfig.check_arg({"mode": "overwrite"})
        assert cfg.mode is Mode.OVERWRITE

    def test_received_from_datetime_arg(self):
        when = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
        cfg = CacheConfig.check_arg(when)
        assert cfg.received_from == when

    def test_path_arg(self, tmp_path):
        from yggdrasil.io.buffer.nested.folder_io import FolderIO

        cfg = CacheConfig.check_arg(tmp_path)
        assert isinstance(cfg.tabular, FolderIO)
        assert str(cfg.tabular.path) == str(tmp_path)


class TestCacheConfigEnabledFlags:
    def test_default_neither_local_nor_remote(self):
        cfg = CacheConfig.default()
        assert cfg.local_cache_enabled is False
        assert cfg.remote_cache_enabled is False

    def test_local_enabled_when_received_from_set(self):
        cfg = CacheConfig(received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc))
        assert cfg.local_cache_enabled is True


class TestCacheConfigSqlLiteral:
    def test_none(self):
        assert CacheConfig.sql_literal(None) == "null"

    def test_bool(self):
        assert CacheConfig.sql_literal(True) == "true"
        assert CacheConfig.sql_literal(False) == "false"

    def test_int(self):
        assert CacheConfig.sql_literal(42) == "42"

    def test_str_quoted_and_escaped(self):
        result = CacheConfig.sql_literal("O'Brien")
        assert result.startswith("'") and result.endswith("'")
        assert "''" in result  # escaped single quote

    def test_datetime(self):
        when = dt.datetime(2020, 1, 1, 12, 0, tzinfo=dt.timezone.utc)
        out = CacheConfig.sql_literal(when)
        assert out.startswith("timestamp '")


class TestCacheConfigMatchBy:
    def test_combines_request_by_and_response_by(self):
        cfg = CacheConfig(
            request_by=["method"],
            response_by=["status_code"],
        )
        assert cfg.match_by == ["method", "status_code"]


class TestCacheConfigSqlClauses:
    def test_request_clause_matches_values(self):
        cfg = CacheConfig(request_by=["method"])
        req = make_request()
        clause = cfg.sql_request_clause(req)
        assert "method" in clause
        assert "GET" in clause

    def test_request_clause_no_request_returns_truthy(self):
        cfg = CacheConfig(request_by=["method"])
        assert cfg.sql_request_clause(None) == "1=1"

    def test_response_clause_includes_received_window(self):
        cfg = CacheConfig(received_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc))
        clause = cfg.sql_response_clause(None)
        assert "received_at" in clause


class TestCacheConfigLookupSql:
    def test_make_lookup_sql_includes_select(self):
        cfg = CacheConfig(
            request_by=["method"],
        )
        sql = cfg.make_lookup_sql("cache_table", make_request())
        assert "SELECT * FROM cache_table" in sql

    def test_make_batch_lookup_sql_or_combines(self):
        cfg = CacheConfig(request_by=["method"])
        reqs = [make_request(method="GET"), make_request(method="POST")]
        sql = cfg.make_batch_lookup_sql("tbl", reqs)
        assert " OR " in sql or "method" in sql


# ---------------------------------------------------------------------------
# SendConfig
# ---------------------------------------------------------------------------


class TestSendConfig:
    def test_default(self):
        cfg = SendConfig.default()
        assert isinstance(cfg, SendConfig)
        assert cfg.raise_error is True
        assert cfg.stream is True

    def test_check_arg_passthrough(self):
        cfg = SendConfig()
        assert SendConfig.check_arg(cfg) is cfg

    def test_check_arg_mapping(self):
        cfg = SendConfig.check_arg({"raise_error": False})
        assert cfg.raise_error is False

    def test_check_arg_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            SendConfig.check_arg(42)

    def test_merge_overrides(self):
        cfg = SendConfig.default().merge(raise_error=False)
        assert cfg.raise_error is False

    def test_merge_unknown_field_raises(self):
        with pytest.raises(TypeError):
            SendConfig.default().merge(not_a_field=True)


# ---------------------------------------------------------------------------
# SendManyConfig
# ---------------------------------------------------------------------------


class TestSendManyConfig:
    def test_default(self):
        cfg = SendManyConfig.default()
        assert cfg.ordered is False

    def test_to_send_config_strips_caches_when_disabled(self):
        many = SendManyConfig()
        single = many.to_send_config(with_remote_cache=False, with_local_cache=False)
        assert isinstance(single, SendConfig)
        assert single.remote_cache == CacheConfig()
        assert single.local_cache == CacheConfig()

    def test_promote_from_send_config(self):
        single = SendConfig(raise_error=False)
        many = SendManyConfig.check_arg(single)
        assert many.raise_error is False

    def test_check_arg_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            SendManyConfig.check_arg(42)
