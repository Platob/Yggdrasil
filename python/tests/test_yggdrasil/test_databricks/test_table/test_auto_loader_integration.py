"""Live Auto Loader integration — staging via the storage path + a cloudFiles
ingestion job with a file-arrival trigger.

End-to-end against a real workspace:

1. provision an EXTERNAL Delta table at a writable S3 prefix;
2. stage Parquet through :meth:`Table.stage_insert` — which lands files in the
   direct cloud-storage Auto Loader staging path (S3, not the Files API);
3. deploy the Auto Loader job (``file_arrival`` trigger on the staging path,
   leveraging the ygg wheel + serverless environment); and
4. run it once and assert the staged rows were ingested into the table.

Requires the ``aws`` extra (the credential refresher rides botocore) and an
identity that can CREATE EXTERNAL VOLUME/TABLE, be granted
``EXTERNAL USE SCHEMA``, and create + run serverless jobs. Skips cleanly when
any prerequisite is missing. Set ``YGG_TEST_EXTERNAL_LOCATION`` to a writable
prefix (default the ``-a-apps`` dev location).

Run::

    DATABRICKS_HOST=... DATABRICKS_TOKEN=... \\
    uv run --extra dev --extra aws python -m pytest \\
      tests/test_yggdrasil/test_databricks/test_table/test_auto_loader_integration.py \\
      -v -s -m integration
"""
from __future__ import annotations

import os
import secrets
import time
import unittest

import pyarrow as pa
import pytest

from yggdrasil.data.schema import Schema, field
from yggdrasil.enums import Mode

from .. import DatabricksIntegrationCase


def _external_base() -> "str | None":
    base = os.environ.get("YGG_TEST_EXTERNAL_LOCATION")
    return base.rstrip("/") if base else None


@pytest.mark.integration
class TestAutoLoaderIngestion(DatabricksIntegrationCase):
    """Stage → file-arrival Auto Loader → ingested, on a real external table."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        from databricks.sdk.errors import DatabricksError
        from databricks.sdk.errors.platform import PermissionDenied

        try:
            import botocore  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"aws extra not installed: {exc}") from exc

        cls.integration_schema()
        cls._me = cls.workspace.current_user.me().user_name
        full_schema = f"{cls.INTEGRATION_CATALOG}.{cls.INTEGRATION_SCHEMA}"
        try:
            cls.client.sql.execute(
                f"GRANT EXTERNAL USE SCHEMA ON SCHEMA {full_schema} TO `{cls._me}`")
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"cannot grant EXTERNAL USE SCHEMA: {exc}") from exc

        base = _external_base()
        if not base:
            raise unittest.SkipTest("set YGG_TEST_EXTERNAL_LOCATION (writable s3:// base prefix)")
        runid = secrets.token_hex(4)
        cls._base = f"{base}/it/{runid}"
        cls.table = cls.client.tables.table(
            catalog_name=cls.INTEGRATION_CATALOG,
            schema_name=cls.INTEGRATION_SCHEMA,
            table_name=f"ygg_al_{runid}",
        )
        cls._schema = Schema([field("id", "int64"), field("v", "string")])
        try:
            cls.table.create(cls._schema, storage_location=f"{cls._base}/table")
        except (DatabricksError, PermissionDenied, NotImplementedError) as exc:
            raise unittest.SkipTest(f"cannot provision external table: {exc}") from exc

        # The staging area must be a real cloud-storage Path (direct S3) for
        # cloudFiles to watch it — pin the staging volume external at an s3://
        # base, then resolve its storage path. Skip if the credential vend /
        # grant doesn't actually permit direct access.
        try:
            cls.table.staging_location = f"{cls._base}/staging"
            cls.stage = (
                cls.table.ensure_staging_volume().storage_path(mode=Mode.AUTO)
                / cls.table.STAGE_SUBPATH
            )
            cls.source = cls.stage.full_path()
            assert cls.source.startswith("s3://")
        except Exception as exc:  # noqa: BLE001
            cls._purge()
            raise unittest.SkipTest(
                f"staging storage path unavailable (needs S3 + EXTERNAL USE "
                f"SCHEMA): {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls._purge()
        finally:
            super().tearDownClass()

    @classmethod
    def _purge(cls) -> None:
        # Delete the job, purge staged + table S3 data, drop the table.
        try:
            job = cls.client.jobs.get(f"[YGG][AUTOLOADER] {cls.table.full_name()}")
            if job is not None:
                job.delete()
        except Exception:
            pass
        for getter in (lambda: cls.stage, lambda: cls.table.storage_path()):
            try:
                p = getter()
                if p is not None:
                    p.remove(recursive=True, missing_ok=True)
            except Exception:
                pass
        try:
            cls.table.delete(missing_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def test_staging_volume_storage_writes_to_s3(self) -> None:
        leaf = self.stage / f"probe_{secrets.token_hex(3)}.parquet"
        leaf.write_table(pa.table({"id": [1], "v": ["a"]}), mode=Mode.OVERWRITE)
        # Round-trips off the bucket, and the path is a plain s3:// URL.
        assert leaf.full_path().startswith("s3://")
        assert leaf.read_arrow_table().num_rows == 1
        leaf.remove(missing_ok=True)

    def test_autoloader_deploys_with_file_trigger_on_staging(self) -> None:
        job = self.table.auto_loader(file_arrival=True)  # source defaults to staging
        assert getattr(job, "job_id", None) is not None
        settings = job.settings
        # File-arrival trigger pointed at the staging storage path.
        assert settings.trigger is not None
        assert settings.trigger.file_arrival.url == self.source.rstrip("/") + "/"
        # Single python-wheel task running the ygg auto-loader entry point.
        task = settings.tasks[0]
        assert task.python_wheel_task.package_name == "ygg"
        assert self.table.full_name() in task.python_wheel_task.parameters
        # The serverless env references the reusable, version-pinned ygg base
        # environment (written to /Workspace/Shared/environments as
        # ``ygg-<version>-py3XX.yml`` — the same file the seed writes) rather
        # than inlining the dependency list.
        from yggdrasil.databricks.environments.service import ygg_base_environment_name
        env = settings.environments[0]
        assert env.spec.base_environment is not None
        assert env.spec.base_environment.endswith(f"{ygg_base_environment_name()}.yml")
        # base_environment carries the version → no inline version, and an
        # ygg-only job layers nothing on top.
        assert env.spec.environment_version is None
        assert not env.spec.dependencies

    def test_autoload_external_s3_via_volume_smoke(self) -> None:
        """Smoke: ingest external-S3 data **through the UC external volume**.

        The same external location, but addressed as a governed ``/Volumes/...``
        path (Files-API) instead of a raw ``s3://`` URL — so Auto Loader watches
        the volume. Verifies the volume staging path resolves, a file written
        through it lands on the backing S3, and the deployed job targets the
        volume source with the version-pinned ygg base environment. The actual
        serverless run is gated behind ``YGG_TEST_AUTOLOADER_RUN=1`` (minutes)."""
        from yggdrasil.databricks.environments.service import ygg_base_environment_name

        # Volume-addressed staging: /Volumes/<cat>/<sch>/<vol>/.sql/tmp, backed by
        # the table's EXTERNAL S3 location (created in setUpClass).
        vol_dir = self.table.staging_folder()
        vol_source = vol_dir.full_path()
        assert vol_source.startswith("/Volumes/"), vol_source

        # A file written through the volume round-trips, and its data physically
        # lives on the external S3 prefix (same bytes, two addressing schemes).
        leaf = vol_dir / f"vol_probe_{secrets.token_hex(3)}.parquet"
        leaf.write_table(pa.table({"id": [7], "v": ["vol"]}), mode=Mode.OVERWRITE)
        assert leaf.read_arrow_table().num_rows == 1

        # Deploy the Auto Loader job watching the VOLUME path. Default trigger is
        # file-arrival on that volume; the source flows to the on-cluster entry
        # point and the env is the canonical wheel-built ygg base environment.
        job = self.table.auto_loader(source=vol_source)
        try:
            assert getattr(job, "job_id", None) is not None
            settings = job.settings
            task = settings.tasks[0]
            assert task.python_wheel_task.package_name == "ygg"
            assert vol_source in task.python_wheel_task.parameters
            assert settings.trigger.file_arrival.url == vol_source.rstrip("/") + "/"
            env = settings.environments[0]
            assert env.spec.base_environment.endswith(f"{ygg_base_environment_name()}.yml")

            if not os.environ.get("YGG_TEST_AUTOLOADER_RUN"):
                self.skipTest("set YGG_TEST_AUTOLOADER_RUN=1 for the live volume ingestion run")

            # One AvailableNow sweep over the volume, then confirm the row landed.
            run = self.table.auto_loader(
                source=vol_source, file_arrival=False, available_now=True,
            ).run(wait=1200, raise_error=True)
            assert run is not None
            deadline = time.time() + 180
            ids: list = []
            while time.time() < deadline:
                ids = self.table.read_arrow_table().column("id").to_pylist()
                if 7 in ids:
                    break
                time.sleep(5)
            assert 7 in ids, f"volume-staged row not ingested (saw {sorted(set(ids))})"
        finally:
            leaf.remove(missing_ok=True)

    def test_bulk_stage_ingest_fast_reuse_and_cleanup(self) -> None:
        # Heavy: builds + ships the ygg wheel and runs a serverless cloudFiles
        # job (minutes). Opt in with YGG_TEST_AUTOLOADER_RUN=1.
        if not os.environ.get("YGG_TEST_AUTOLOADER_RUN"):
            self.skipTest("set YGG_TEST_AUTOLOADER_RUN=1 to run the live ingestion sweep")

        # Stage 100 inserts via stage_insert — it lands each Parquet file
        # directly under the Auto Loader staging path, so the pipeline is
        # stage_insert → Auto Loader → table with no extra wiring.
        n = int(os.environ.get("YGG_TEST_AUTOLOADER_N", "100"))
        for i in range(n):
            staged = self.table.stage_insert(pa.table({"id": [i], "v": [f"r{i}"]}))
            assert staged.full_path().startswith(self.source.rstrip("/"))
        assert len(list(self.stage.iterdir())) >= n

        # Get-or-create twice — the warm second call reuses the deployed bundle
        # (only the project wheel is refreshed) and updates the *same* job, not a
        # duplicate. clean_source exercises the cloudFiles.cleanSource path with a
        # valid > 7-day retention (regression: retention 0 is rejected).
        job = self.table.auto_loader(available_now=True, clean_source=True)
        t = time.time()
        job2 = self.table.auto_loader(available_now=True, clean_source=True)
        warm = time.time() - t
        assert job2.job_id == job.job_id, "get-or-create should reuse the same job"
        # Warm get-or-create is cheap (no full ~100 MB bundle re-upload).
        assert warm < 90, f"warm get-or-create too slow: {warm:.1f}s"

        run = job2.run(wait=1200, raise_error=True)
        assert run is not None

        # All 100 rows land in the table.
        deadline = time.time() + 180
        ids: list = []
        while time.time() < deadline:
            ids = sorted(self.table.read_arrow_table().column("id").to_pylist())
            if len(set(ids)) >= n:
                break
            time.sleep(5)
        assert len(set(ids)) >= n, f"ingested {len(set(ids))}/{n}"

        # cloudFiles.cleanSource is a rolling janitor (> 7-day retention) so the
        # just-ingested files are NOT removed within this sweep; explicit staging
        # removal empties the source deterministically.
        self.stage.remove(recursive=True, missing_ok=True)
        assert len(list(self.stage.iterdir())) == 0
