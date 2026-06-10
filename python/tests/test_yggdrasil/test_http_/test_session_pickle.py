"""Pickle round-trip tests for HTTPSession.

Validates that every constructor parameter (base_url, verify, proxy,
no_proxy, headers, auth) survives a pickle dump/load cycle, that
process-local resources (connection cache, retry policy, job pool)
are rebuilt fresh, and that the singleton cache correctly collapses
restored instances.
"""
from __future__ import annotations

import pickle
import warnings

import pytest

from yggdrasil.http_.session import HTTPSession


@pytest.fixture(autouse=True)
def _fresh_singletons():
    HTTPSession._INSTANCES.clear()
    yield
    HTTPSession._INSTANCES.clear()


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------


class TestBasicPickle:

    def test_default_session(self):
        s = HTTPSession(base_url="https://example.com")
        restored = pickle.loads(pickle.dumps(s))
        assert restored.base_url.to_string() == s.base_url.to_string()
        assert restored.verify is True
        assert restored.proxy is None
        assert restored.no_proxy is None

    def test_no_base_url(self):
        s = HTTPSession()
        restored = pickle.loads(pickle.dumps(s))
        assert restored.base_url is None


# ---------------------------------------------------------------------------
# verify variants
# ---------------------------------------------------------------------------


class TestVerifyPickle:

    def test_verify_true(self):
        s = HTTPSession(base_url="https://x.com", verify=True)
        restored = pickle.loads(pickle.dumps(s))
        assert restored.verify is True

    def test_verify_false(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = HTTPSession(base_url="https://x.com", verify=False)
        data = pickle.dumps(s)
        HTTPSession._INSTANCES.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            restored = pickle.loads(data)
        assert restored.verify is False

    def test_verify_ca_path(self, tmp_path):
        ca = tmp_path / "certs"
        ca.mkdir()
        s = HTTPSession(base_url="https://x.com", verify=str(ca))
        data = pickle.dumps(s)
        HTTPSession._INSTANCES.clear()
        restored = pickle.loads(data)
        assert restored.verify == str(ca)

    def test_verify_pathlib_normalised_before_pickle(self, tmp_path):
        ca = tmp_path / "certs"
        ca.mkdir()
        s = HTTPSession(base_url="https://x.com", verify=ca)
        assert isinstance(s.verify, str)
        data = pickle.dumps(s)
        HTTPSession._INSTANCES.clear()
        restored = pickle.loads(data)
        assert restored.verify == str(ca)


# ---------------------------------------------------------------------------
# proxy / no_proxy
# ---------------------------------------------------------------------------


class TestProxyPickle:

    def test_proxy_url(self):
        s = HTTPSession(base_url="https://x.com", proxy="http://proxy:3128")
        data = pickle.dumps(s)
        HTTPSession._INSTANCES.clear()
        restored = pickle.loads(data)
        assert restored.proxy is not None
        assert restored.proxy.host == "proxy"
        assert restored.proxy.port == 3128

    def test_proxy_with_auth(self):
        s = HTTPSession(base_url="https://x.com", proxy="http://user:pass@proxy:8080")
        data = pickle.dumps(s)
        HTTPSession._INSTANCES.clear()
        restored = pickle.loads(data)
        assert restored.proxy.user == "user"
        assert restored.proxy.password == "pass"

    def test_no_proxy(self):
        s = HTTPSession(base_url="https://x.com", no_proxy="localhost,.internal")
        data = pickle.dumps(s)
        HTTPSession._INSTANCES.clear()
        restored = pickle.loads(data)
        assert restored.no_proxy == "localhost,.internal"


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


class TestHeadersPickle:

    def test_custom_headers_preserved(self):
        s = HTTPSession(
            base_url="https://x.com",
            headers={"X-Custom": "value", "Accept": "application/json"},
        )
        data = pickle.dumps(s)
        HTTPSession._INSTANCES.clear()
        restored = pickle.loads(data)
        assert restored.headers.get("X-Custom") == "value"
        assert restored.headers.get("Accept") == "application/json"


# ---------------------------------------------------------------------------
# Transient state rebuilt
# ---------------------------------------------------------------------------


class TestTransientState:

    def test_connections_cache_empty_after_unpickle(self):
        s = HTTPSession(base_url="https://x.com")
        data = pickle.dumps(s)
        HTTPSession._INSTANCES.clear()
        restored = pickle.loads(data)
        assert restored._connections == {}

    def test_retry_policy_rebuilt(self):
        s = HTTPSession(base_url="https://x.com")
        data = pickle.dumps(s)
        HTTPSession._INSTANCES.clear()
        restored = pickle.loads(data)
        assert restored._retry is not None

    def test_lock_rebuilt(self):
        s = HTTPSession(base_url="https://x.com")
        data = pickle.dumps(s)
        HTTPSession._INSTANCES.clear()
        restored = pickle.loads(data)
        assert hasattr(restored, "_lock")


# ---------------------------------------------------------------------------
# Singleton collapse on unpickle
# ---------------------------------------------------------------------------


class TestSingletonCollapse:

    def test_unpickle_collapses_to_live_singleton(self):
        s = HTTPSession(base_url="https://singleton-test.example.com")
        data = pickle.dumps(s)
        restored = pickle.loads(data)
        assert restored is s

    def test_unpickle_fresh_cache_creates_new(self):
        s = HTTPSession(base_url="https://fresh-test.example.com")
        data = pickle.dumps(s)
        HTTPSession._INSTANCES.clear()
        restored = pickle.loads(data)
        assert restored is not s
        assert restored.base_url.to_string() == s.base_url.to_string()


# ---------------------------------------------------------------------------
# Combined config round-trip
# ---------------------------------------------------------------------------


class TestCombinedPickle:

    def test_full_config(self, tmp_path):
        ca = tmp_path / "certs"
        ca.mkdir()
        s = HTTPSession(
            base_url="https://api.example.com",
            verify=str(ca),
            headers={"Authorization": "Bearer tok"},
            proxy="http://user:pass@corp-proxy:8080",
            no_proxy="localhost,127.0.0.1,.internal",
        )
        data = pickle.dumps(s)
        HTTPSession._INSTANCES.clear()
        restored = pickle.loads(data)

        assert "api.example.com" in restored.base_url.to_string()
        assert restored.verify == str(ca)
        assert restored.headers.get("Authorization") == "Bearer tok"
        assert restored.proxy.host == "corp-proxy"
        assert restored.proxy.user == "user"
        assert restored.no_proxy == "localhost,127.0.0.1,.internal"
        assert restored._connections == {}
        assert restored._retry is not None
