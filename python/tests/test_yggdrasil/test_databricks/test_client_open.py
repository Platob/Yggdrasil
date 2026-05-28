"""Tests for :meth:`DatabricksClient.open`.

The thin wrapper that lets callers say ``client.open("/Volumes/...")``
without first constructing a :class:`DatabricksPath` by hand.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.databricks.path import DatabricksPath


@pytest.fixture(autouse=True)
def reset_singletons():
    DatabricksClient._INSTANCES.clear()
    VolumePath._INSTANCES.clear()
    DatabricksPath._INSTANCES.clear()
    yield
    DatabricksClient._INSTANCES.clear()
    VolumePath._INSTANCES.clear()
    DatabricksPath._INSTANCES.clear()


@pytest.fixture
def client():
    return DatabricksClient(host="https://ws.example.com", token="t")


class TestDatabricksClientOpen:

    def test_string_path_routes_through_databricks_path_from(self, client):
        # The string lands on :meth:`DatabricksPath.from_` (bound to
        # this client) and the returned path's ``.open`` does the work.
        fake_path = MagicMock()
        with patch(
            "yggdrasil.databricks.path.DatabricksPath.from_",
            return_value=fake_path,
        ) as from_:
            result = client.open("/Volumes/cat/sch/vol/x", "rb")
        from_.assert_called_once_with(
            obj="/Volumes/cat/sch/vol/x", client=client,
        )
        fake_path.open.assert_called_once_with(mode="rb")
        assert result is fake_path.open.return_value

    def test_existing_path_is_opened_verbatim(self, client):
        # Already a :class:`Path` — skip :meth:`DatabricksPath.from_`
        # and call its ``.open`` directly so callers can mix in
        # S3/HTTP/local paths without losing the workspace binding.
        path = MagicMock(spec=VolumePath)
        with patch(
            "yggdrasil.databricks.path.DatabricksPath.from_",
        ) as from_:
            client.open(path, "wb")
        from_.assert_not_called()
        path.open.assert_called_once_with(mode="wb")

    def test_default_mode_is_none(self, client):
        # ``mode=None`` flows through to ``Path.open`` which falls
        # through to ``Holder.open``'s default ("rb+").
        fake_path = MagicMock()
        with patch(
            "yggdrasil.databricks.path.DatabricksPath.from_",
            return_value=fake_path,
        ):
            client.open("/Volumes/cat/sch/vol/x")
        fake_path.open.assert_called_once_with(mode=None)

    def test_kwargs_ride_through_to_path_open(self, client):
        fake_path = MagicMock()
        with patch(
            "yggdrasil.databricks.path.DatabricksPath.from_",
            return_value=fake_path,
        ):
            client.open(
                "/Volumes/cat/sch/vol/x",
                mode="rb",
                media_type="text/csv",
                owns_holder=False,
            )
        fake_path.open.assert_called_once_with(
            mode="rb",
            media_type="text/csv",
            owns_holder=False,
        )


class TestDbfsPathDeprecation:

    def test_dbfs_path_emits_deprecation_warning(self, client):
        with pytest.warns(DeprecationWarning, match=r"deprecated"):
            with patch(
                "yggdrasil.databricks.path.DatabricksPath.from_",
                return_value=MagicMock(),
            ):
                client.dbfs_path("/Volumes/cat/sch/vol/x")

