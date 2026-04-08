from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks.sql.staging import StagingPath


@pytest.fixture(autouse=True)
def _reset_staging_sweep_state() -> None:
    StagingPath._EXPIRED_SWEEP_DONE = False


def test_for_table_builds_expected_volume_path_shape() -> None:
    client = MagicMock()

    staging = StagingPath.for_table(
        client=client,
        catalog_name="trading",
        schema_name="ba_3mv_polaris__p__volcano_output",
        table_name="positions",
        start_ts=1774938955,
        token="5ba89ace",
        max_lifetime=3600,
    )

    path_str = str(staging.path)

    assert path_str.startswith(
        "/Volumes/trading/ba_3mv_polaris__p__volcano_output/tmp/.sql/"
        "trading/ba_3mv_polaris__p__volcano_output/positions/"
    )
    assert path_str.endswith("tmp-1774938955-1774942555-5ba89ace.parquet")


def test_for_table_clamps_lifetime_to_max_one_hour() -> None:
    staging = StagingPath.for_table(
        client=MagicMock(),
        catalog_name="main",
        schema_name="sales",
        table_name="t",
        start_ts=100,
        token="abcd1234",
        max_lifetime=999999,
    )

    assert staging.start_ts == 100
    assert staging.end_ts == 3700
    assert staging.end_ts - staging.start_ts <= 3600


def test_for_table_normalizes_non_positive_lifetime_to_one_second() -> None:
    staging = StagingPath.for_table(
        client=MagicMock(),
        catalog_name="main",
        schema_name="sales",
        table_name="t",
        start_ts=100,
        token="abcd1234",
        max_lifetime=0,
    )

    assert staging.end_ts - staging.start_ts == 1


def test_register_and_unregister_shutdown_cleanup() -> None:
    path = MagicMock()
    staging = StagingPath(
        path=path,
        catalog_name="main",
        schema_name="sales",
        table_name="t",
        start_ts=1,
        end_ts=2,
        token="abcd",
    )

    fake_hook = object()

    with patch("yggdrasil.databricks.sql.staging.yg_shutdown.register", return_value=fake_hook) as reg:
        with patch("yggdrasil.databricks.sql.staging.yg_shutdown.unregister") as unreg:
            staging.register_shutdown_cleanup()
            reg.assert_called_once()

            staging.unregister_shutdown_cleanup()
            unreg.assert_called_once_with(fake_hook)


def test_cleanup_removes_path_and_unregisters_hook() -> None:
    path = MagicMock()
    staging = StagingPath(
        path=path,
        catalog_name="main",
        schema_name="sales",
        table_name="t",
        start_ts=1,
        end_ts=2,
        token="abcd",
    )

    fake_hook = object()

    with patch("yggdrasil.databricks.sql.staging.yg_shutdown.register", return_value=fake_hook):
        with patch("yggdrasil.databricks.sql.staging.yg_shutdown.unregister") as unreg:
            staging.register_shutdown_cleanup()
            staging.cleanup(allow_not_found=True, unregister=True)

            path.remove.assert_called_once_with(recursive=True, allow_not_found=True)
            unreg.assert_called_once_with(fake_hook)


def test_lazy_expired_sweep_runs_once_and_removes_only_expired_files() -> None:
    root = MagicMock()

    expired = MagicMock()
    expired.name = "tmp-100-101-deadbeef.parquet"

    active = MagicMock()
    active.name = "tmp-100-9999999999-cafebabe.parquet"

    other = MagicMock()
    other.name = "readme.txt"

    root.ls.return_value = [expired, active, other]

    final_1 = MagicMock()
    final_2 = MagicMock()

    with patch(
        "yggdrasil.databricks.sql.staging.DatabricksPath.parse",
        side_effect=[root, final_1, final_2],
    ) as parse:
        StagingPath.for_table(
            client=MagicMock(),
            catalog_name="main",
            schema_name="sales",
            table_name="t1",
            start_ts=10,
            token="aaaa1111",
            max_lifetime=3600,
        )
        # Second call should skip the sweep entirely.
        StagingPath.for_table(
            client=MagicMock(),
            catalog_name="main",
            schema_name="sales",
            table_name="t2",
            start_ts=20,
            token="bbbb2222",
            max_lifetime=3600,
        )

    # parse() called 3 times: root sweep + final path for call1 + final path for call2
    assert parse.call_count == 3
    root.ls.assert_called_once_with(recursive=True, allow_not_found=True)
    expired.remove.assert_called_once_with(recursive=True, allow_not_found=True)
    active.remove.assert_not_called()
    other.remove.assert_not_called()


