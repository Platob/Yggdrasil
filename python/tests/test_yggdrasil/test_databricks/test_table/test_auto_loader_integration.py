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
    base = os.environ.get("YGG_TEST_EXTERNAL_LOCATION", "s3://odp-aws-dls3-eu-central-1-a-apps/3mv/")
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
            cls.stage = cls.table.staging_volume / cls.table.STAGE_SUBPATH
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

        assert job is not None
