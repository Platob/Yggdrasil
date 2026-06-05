"""``Table.auto_loader`` — get-or-create an Auto Loader ingestion job — and the
on-cluster :func:`auto_load` cloudFiles entry point (no live workspace/Spark)."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.table.table import Table
from yggdrasil.databricks.table.auto_loader import auto_load


def _table(catalog_name="cat", schema_name="sch", table_name="tbl") -> Table:
    service = MagicMock()
    ws = MagicMock()
    service.client.workspace_client.return_value = ws
    return Table(
        service=service,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_name=table_name,
    )


class TestTableAutoLoader:
    def test_deploys_get_or_create_job_with_ygg_entry_and_params(self):
        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            job = tbl.auto_loader("s3://bkt/landing", file_format="json")

        Flow.assert_called_once()
        args, kwargs = Flow.call_args
        # The on-cluster cloudFiles entry point is the job's target.
        assert args[0] is auto_load
        assert kwargs["name"] == "[YGG][AUTOLOADER] cat.sch.tbl"
        # File-arrival is the default trigger — the job fires when files land.
        assert kwargs["trigger"].file_arrival.url == "s3://bkt/landing/"
        # Positional job parameters: target table, source, format, checkpoint, mode.
        assert kwargs["parameters"] == [
            "cat.sch.tbl", "s3://bkt/landing", "json", "", True, False, "8 days",
        ]
        # Get-or-create happens through Flow.deploy(client) → create_or_update.
        Flow.return_value.deploy.assert_called_once_with(tbl.client)
        assert job is Flow.return_value.deploy.return_value

    def test_name_override_and_explicit_checkpoint(self):
        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader(
                "s3://bkt/landing", name="my_loader",
                checkpoint="s3://bkt/ckpt", available_now=False,
            )
        kwargs = Flow.call_args.kwargs
        assert kwargs["name"] == "my_loader"
        assert kwargs["parameters"] == [
            "cat.sch.tbl", "s3://bkt/landing", "parquet", "s3://bkt/ckpt", False, False, "8 days",
        ]

    def test_file_arrival_builds_trigger_on_source(self):
        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://bkt/landing", file_arrival=True)
        trigger = Flow.call_args.kwargs["trigger"]
        # A file-arrival trigger pointed at the source path.
        assert trigger is not None
        assert trigger.file_arrival.url == "s3://bkt/landing/"

    def test_file_arrival_disabled_deploys_without_trigger(self):
        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://bkt/landing", file_arrival=False)
        assert Flow.call_args.kwargs["trigger"] is None

    def test_explicit_trigger_overrides_default_file_arrival(self):
        tbl = _table()
        sentinel = object()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://bkt/landing", trigger=sentinel)
        # An explicit trigger wins over the default file-arrival one.
        assert Flow.call_args.kwargs["trigger"] is sentinel

    def test_deploy_false_returns_flow_without_creating(self):
        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            out = tbl.auto_loader("s3://bkt/landing", deploy=False)
        Flow.return_value.deploy.assert_not_called()
        assert out is Flow.return_value

    def test_job_name_uses_bracketed_prefix(self):
        tbl = _table(catalog_name="cat", schema_name="sch", table_name="tbl")
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://x", deploy=False)
        assert Flow.call_args.kwargs["name"] == "[YGG][AUTOLOADER] cat.sch.tbl"

    def test_defaults_bundle_and_canonical_ygg_environment(self):
        from yggdrasil.databricks.job.wheel import ygg_base_environment_name

        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://x", deploy=False)
        flow = Flow.return_value
        assert flow.bundle_dependencies is True
        # Default env is the version-pinned ygg image the seed writes, not "yellow".
        assert flow.base_environment_name == ygg_base_environment_name()
        assert flow.base_environment_name.startswith("ygg-")

    def test_environment_override_and_disable(self):
        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://x", environment="green", deploy=False)
        assert Flow.return_value.base_environment_name == "green"
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://x", environment=None, deploy=False)
        assert Flow.return_value.base_environment_name is None


class TestAutoLoadEntryPoint:
    def _spark(self, *, location="s3://bkt/tbl"):
        spark = MagicMock()
        spark.sql.return_value.collect.return_value = [{"location": location}]
        reader = spark.readStream.format.return_value
        reader.option.return_value = reader            # chainable
        frame = reader.load.return_value
        writer = frame.writeStream
        writer.option.return_value = writer
        writer.trigger.return_value = writer
        return spark, reader, frame, writer

    def _with_pyspark(self, spark):
        pyspark = types.ModuleType("pyspark")
        pyspark_sql = types.ModuleType("pyspark.sql")
        session = MagicMock()
        session.builder.getOrCreate.return_value = spark
        pyspark_sql.SparkSession = session
        return patch.dict(sys.modules, {"pyspark": pyspark, "pyspark.sql": pyspark_sql})

    def test_streams_cloudfiles_into_table_with_derived_checkpoint(self):
        spark, reader, frame, writer = self._spark()
        with self._with_pyspark(spark):
            out = auto_load("cat.sch.tbl", "s3://src/landing", file_format="json")

        # cloudFiles reader configured with the format + a schema location under
        # the checkpoint derived from the table's storage location.
        spark.readStream.format.assert_called_once_with("cloudFiles")
        reader.option.assert_any_call("cloudFiles.format", "json")
        reader.option.assert_any_call(
            "cloudFiles.schemaLocation", "s3://bkt/tbl/_ygg_autoloader/_schema",
        )
        reader.load.assert_called_once_with("s3://src/landing")
        # writeStream checkpoint + AvailableNow trigger, sinking into the table.
        writer.option.assert_any_call("checkpointLocation", "s3://bkt/tbl/_ygg_autoloader")
        writer.trigger.assert_called_once_with(availableNow=True)
        writer.toTable.assert_called_once_with("cat.sch.tbl")
        writer.toTable.return_value.awaitTermination.assert_called_once()
        assert out["checkpoint"] == "s3://bkt/tbl/_ygg_autoloader"

    def test_explicit_checkpoint_skips_describe_detail(self):
        spark, reader, _frame, writer = self._spark()
        with self._with_pyspark(spark):
            auto_load("cat.sch.tbl", "s3://src", checkpoint="s3://ckpt", available_now=False)
        spark.sql.assert_not_called()  # no DESCRIBE DETAIL when checkpoint given
        writer.option.assert_any_call("checkpointLocation", "s3://ckpt")
        writer.trigger.assert_called_once_with(processingTime="1 minute")

    def test_clean_source_sets_cleanSource_delete(self):
        spark, reader, _frame, _writer = self._spark()
        with self._with_pyspark(spark):
            auto_load("cat.sch.tbl", "s3://src", checkpoint="s3://ckpt", clean_source=True)
        reader.option.assert_any_call("cloudFiles.cleanSource", "DELETE")
        # Databricks requires a retention interval > 7 days.
        reader.option.assert_any_call("cloudFiles.cleanSource.retentionDuration", "8 days")

    def test_clean_source_retention_override(self):
        spark, reader, _frame, _writer = self._spark()
        with self._with_pyspark(spark):
            auto_load("cat.sch.tbl", "s3://src", checkpoint="s3://ckpt",
                      clean_source=True, clean_source_retention="30 days")
        reader.option.assert_any_call("cloudFiles.cleanSource.retentionDuration", "30 days")

    def test_clean_source_off_by_default(self):
        spark, reader, _frame, _writer = self._spark()
        with self._with_pyspark(spark):
            auto_load("cat.sch.tbl", "s3://src", checkpoint="s3://ckpt")
        calls = [c.args[0] for c in reader.option.call_args_list]
        assert "cloudFiles.cleanSource" not in calls


class TestStageStorageAndDefaultSource:
    def test_auto_loader_defaults_source_to_staging_volume_storage(self):
        # The default source is the staging volume's cloud storage path joined
        # with STAGE_SUBPATH, and (on this same zero-config path) the checkpoint
        # defaults to its CHECKPOINT_SUBPATH sibling — both resolved straight off
        # the staging volume's (writable, external) storage, never the MANAGED
        # table's governed location.
        from yggdrasil.enums import Mode
        tbl = _table()
        vol = MagicMock()
        storage_root = MagicMock()
        leaves = {
            Table.STAGE_SUBPATH: "s3://bkt/3mv/.ygg/stage",
            Table.CHECKPOINT_SUBPATH: "s3://bkt/3mv/.ygg/_autoloader",
        }
        def _join(sub):
            m = MagicMock()
            m.full_path.return_value = leaves[sub]
            return m
        storage_root.__truediv__.side_effect = _join
        vol.storage_path.return_value = storage_root
        with patch.object(Table, "ensure_staging_volume", return_value=vol), \
                patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader(file_arrival=True)  # no source, no checkpoint

        assert vol.storage_path.call_args.kwargs["mode"] is Mode.AUTO
        storage_root.__truediv__.assert_any_call(Table.STAGE_SUBPATH)
        storage_root.__truediv__.assert_any_call(Table.CHECKPOINT_SUBPATH)
        params = Flow.call_args.kwargs["parameters"]
        assert params[1] == "s3://bkt/3mv/.ygg/stage"             # source = staging storage path
        assert params[3] == "s3://bkt/3mv/.ygg/_autoloader"       # checkpoint = sibling, writable
        trig = Flow.call_args.kwargs["trigger"]
        assert trig.file_arrival.url == "s3://bkt/3mv/.ygg/stage/"  # file trigger on it

    def test_explicit_source_leaves_checkpoint_to_on_cluster_default(self):
        # With an explicit source and no checkpoint, the staging volume is not
        # touched and the checkpoint param stays empty so the on-cluster step
        # derives <table-location>/_ygg_autoloader (unchanged behavior).
        tbl = _table()
        with patch.object(Table, "ensure_staging_volume") as ensure, \
                patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://bkt/landing", file_arrival=False)
        ensure.assert_not_called()
        assert Flow.call_args.kwargs["parameters"][3] == ""
