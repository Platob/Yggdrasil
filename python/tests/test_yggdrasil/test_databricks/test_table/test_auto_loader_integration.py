"""Live Auto Loader integration — staging via the storage path + a cloudFiles
ingestion job with a file-arrival trigger.

End-to-end against a real workspace:

1. provision an EXTERNAL Delta table at a writable S3 prefix;
2. stage a Parquet file through :meth:`Table.stage_storage_path` (a direct
   cloud-storage Path — the file lands in S3, not via the Files API);
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


def _external_base() -> str:
    return os.environ.get(
        "YGG_TEST_EXTERNAL_LOCATION",
        "s3://odp-aws-dls3-eu-central-1-a-apps/3mv/ygg",
    ).rstrip("/")


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

        runid = secrets.token_hex(4)
        cls._base = f"{_external_base()}/it/{runid}"
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
        # cloudFiles to watch it — skip if the credential vend / grant doesn't
        # actually permit direct access.
        try:
            cls.stage = cls.table.stage_storage_path()
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
            job = cls.client.jobs.get(f"ygg_autoloader_{cls.table.full_name()}".replace(".", "_"))
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
    def test_stage_storage_path_writes_to_s3(self) -> None:
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
        assert settings.trigger.file_arrival.url == self.source
        # Single python-wheel task running the ygg auto-loader entry point.
        task = settings.tasks[0]
        assert task.python_wheel_task.package_name == "ygg"
        assert self.table.full_name() in task.python_wheel_task.parameters

    def test_ingestion_smoke(self) -> None:
        # Heavy: builds + ships the ygg wheel and runs a serverless cloudFiles
        # job (minutes). Opt in with YGG_TEST_AUTOLOADER_RUN=1.
        if not os.environ.get("YGG_TEST_AUTOLOADER_RUN"):
            self.skipTest("set YGG_TEST_AUTOLOADER_RUN=1 to run the live ingestion sweep")
        # Stage a file, then run the deployed job once (AvailableNow) and assert
        # the rows land in the table. (The file-arrival *trigger* is validated
        # above; here we drive one ingestion sweep deterministically.)
        leaf = self.stage / f"batch_{secrets.token_hex(3)}.parquet"
        leaf.write_table(pa.table({"id": [101, 102], "v": ["x", "y"]}), mode=Mode.OVERWRITE)

        job = self.table.auto_loader(available_now=True)  # source = staging
        run = job.run(wait=900, raise_error=True)
        assert run is not None

        # Give UC a beat, then read the table back.
        deadline = time.time() + 120
        ids: list = []
        while time.time() < deadline:
            ids = sorted(self.table.read_arrow_table().column("id").to_pylist())
            if 101 in ids and 102 in ids:
                break
            time.sleep(5)
        assert 101 in ids and 102 in ids, f"ingested ids={ids}"
        leaf.remove(missing_ok=True)
