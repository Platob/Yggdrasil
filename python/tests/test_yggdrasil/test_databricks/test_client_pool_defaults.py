"""Default HTTP pool sizing on :class:`DatabricksClient`.

The SDK's own defaults (``max_connection_pools=20`` /
``max_connections_per_pool=20`` with ``pool_block=True``) are too tight:
volume IO and statement-execution traffic share the same
:class:`requests.Session` inside the workspace client, so a handful of
in-flight volume transfers stall ``execute_statement``. The client
ships bumped defaults that get plumbed through to every SDK
:class:`Config` it builds.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks import client as client_module


@pytest.fixture
def captured_config(monkeypatch):
    """Replace SDK ``Config`` so we can observe the kwargs the client
    passes without triggering a live auth dance."""
    captured: dict = {}

    def _factory(**kwargs):
        captured.clear()
        captured.update(kwargs)
        cfg = MagicMock(name="Config")
        # ``make_config`` reads these back to update ``self`` post-build;
        # mirror the kwargs so the assignment loop sees consistent values.
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        cfg.client_type = client_module.ClientType.WORKSPACE
        return cfg

    monkeypatch.setattr(client_module, "Config", _factory)
    return captured


class TestPoolDefaults:
    def test_fields_default_to_bumped_values(self) -> None:
        """No explicit kwargs → bumped defaults applied at resolve time."""
        c = DatabricksClient(host="https://ws.example.com", token="t")

        assert c.max_connection_pools == client_module._DEFAULT_MAX_CONNECTION_POOLS
        assert (
            c.max_connections_per_pool
            == client_module._DEFAULT_MAX_CONNECTIONS_PER_POOL
        )

    def test_caller_overrides_are_honored(self) -> None:
        """Explicit kwargs win over the static defaults."""
        c = DatabricksClient(
            host="https://ws.example.com",
            token="t",
            max_connection_pools=8,
            max_connections_per_pool=16,
        )

        assert c.max_connection_pools == 8
        assert c.max_connections_per_pool == 16

    def test_defaults_flow_into_sdk_config(self, captured_config) -> None:
        """``make_config`` must hand the bumped sizes to the SDK ``Config``."""
        c = DatabricksClient(host="https://ws.example.com", token="t")

        c.make_config()

        assert (
            captured_config["max_connection_pools"]
            == client_module._DEFAULT_MAX_CONNECTION_POOLS
        )
        assert (
            captured_config["max_connections_per_pool"]
            == client_module._DEFAULT_MAX_CONNECTIONS_PER_POOL
        )

    def test_overrides_flow_into_sdk_config(self, captured_config) -> None:
        c = DatabricksClient(
            host="https://ws.example.com",
            token="t",
            max_connection_pools=12,
            max_connections_per_pool=24,
        )

        c.make_config()

        assert captured_config["max_connection_pools"] == 12
        assert captured_config["max_connections_per_pool"] == 24


class TestPoolFieldsRoundtripThroughUrl:
    """Init fields ride in the ``dbks://`` query string for URL clients.

    The values come back as strings on the receiving side — same shape
    as the other int-typed fields (``http_timeout_seconds``,
    ``rate_limit``); URL query coercion is a pre-existing limitation
    and not specific to the pool knobs.
    """

    def test_to_url_emits_pool_sizes(self) -> None:
        c = DatabricksClient(
            host="https://ws.example.com",
            token="t",
            max_connection_pools=10,
            max_connections_per_pool=20,
        )
        items = dict(c.to_url().query_items())
        assert items["max_connection_pools"] == "10"
        assert items["max_connections_per_pool"] == "20"
