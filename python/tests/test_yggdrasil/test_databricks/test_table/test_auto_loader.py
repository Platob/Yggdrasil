"""``Table.auto_loader`` — get-or-create an Auto Loader ingestion job — and the
on-cluster :func:`auto_load` cloudFiles entry point (no live workspace/Spark)."""
from __future__ import annotations

import sys
import types
from contextlib import ExitStack
from unittest.mock import MagicMock, PropertyMock, patch

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
    def test_deploys_get_or_create_job_with_ygg_entry_and_command(self):
        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            job = tbl.auto_loader("s3://bkt/landing", file_format="json")

        Flow.assert_called_once()
        args, kwargs = Flow.call_args
        # The on-cluster cloudFiles entry point is the job's target (so the
        # built wheel is the package that defines it).
        assert args[0] is auto_load
        assert kwargs["name"] == "[YGG][AUTOLOADER] cat.sch.tbl"
        # File-arrival is the default trigger — the job fires when files land.
        assert kwargs["trigger"].file_arrival.url == "s3://bkt/landing/"
        # The wheel-task command: ``ygg databricks table autoload`` with the
        # configurable --table / --source and the ingestion options. No
        # checkpoint flag here (none given → on-cluster derives it).
        assert kwargs["command"] == [
            "databricks", "table", "autoload",
            "--table", "cat.sch.tbl",
            "--source", "s3://bkt/landing",
            "--format", "json",
            "--available-now",
            "--clean-source-retention", "8 days",
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
        # Explicit checkpoint adds --checkpoint; available_now=False → --no-available-now.
        assert kwargs["command"] == [
            "databricks", "table", "autoload",
            "--table", "cat.sch.tbl",
            "--source", "s3://bkt/landing",
            "--format", "parquet",
            "--checkpoint", "s3://bkt/ckpt",
            "--no-available-now",
            "--clean-source-retention", "8 days",
        ]

    def test_clean_source_flag_in_command(self):
        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://bkt/landing", clean_source=True,
                            clean_source_retention="30 days")
        cmd = Flow.call_args.kwargs["command"]
        assert "--clean-source" in cmd
        assert cmd[-2:] == ["--clean-source-retention", "30 days"]

    def test_file_arrival_builds_trigger_on_source(self):
        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://bkt/landing", file_arrival=True)
        trigger = Flow.call_args.kwargs["trigger"]
        # A file-arrival trigger pointed at the source path, with the 60s
        # Databricks polling floor.
        assert trigger is not None
        assert trigger.file_arrival.url == "s3://bkt/landing/"
        assert trigger.file_arrival.min_time_between_triggers_seconds == 60

    def test_file_arrival_min_seconds_clamped_to_60(self):
        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://bkt/landing", file_arrival_min_seconds=5)
        trig = Flow.call_args.kwargs["trigger"]
        assert trig.file_arrival.min_time_between_triggers_seconds == 60

    def test_file_arrival_min_seconds_override(self):
        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://bkt/landing", file_arrival_min_seconds=300)
        trig = Flow.call_args.kwargs["trigger"]
        assert trig.file_arrival.min_time_between_triggers_seconds == 300

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
        from yggdrasil.databricks.environments.service import environment_stem

        tbl = _table()
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://x", deploy=False)
        flow = Flow.return_value
        assert flow.bundle_dependencies is True
        # Default env is the version-pinned ygg image the seed writes, not "yellow".
        assert flow.base_environment_name == environment_stem('ygg')
        assert flow.base_environment_name.startswith("ygg-")

    def test_job_is_tagged_with_table_source_format_and_trigger(self):
        tbl = _table()
        # Identity sanitizer so we can assert the literal tag values.
        tbl.service.client.safe_tag_value = lambda v, **k: v
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://bkt/landing", file_format="json", deploy=False)
        tags = Flow.return_value.job_tags
        assert tags["ygg"] == "autoloader"
        assert tags["ygg_kind"] == "autoloader"
        assert tags["ygg_catalog"] == "cat"
        assert tags["ygg_schema"] == "sch"
        assert tags["ygg_table"] == "cat.sch.tbl"
        assert tags["ygg_source"] == "s3://bkt/landing"
        assert tags["ygg_format"] == "json"
        assert tags["ygg_trigger"] == "file_arrival"
        assert tags["ygg_available_now"] == "true"
        assert tags["ygg_clean_source"] == "false"

    def test_manual_trigger_tag_when_file_arrival_disabled(self):
        tbl = _table()
        tbl.service.client.safe_tag_value = lambda v, **k: v
        with patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://x", file_arrival=False, deploy=False)
        assert Flow.return_value.job_tags["ygg_trigger"] == "manual"

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
        stack = ExitStack()
        stack.enter_context(
            patch.dict(sys.modules, {"pyspark": pyspark, "pyspark.sql": pyspark_sql})
        )
        # ``auto_load`` always casts the stream via ``yggdrasil.data.Schema``;
        # patch it with an identity cast so the writer chain is the frame's own
        # ``writeStream`` (tests that care assert on from_spark / cast_spark_tabular).
        self._Schema = stack.enter_context(patch("yggdrasil.data.Schema"))
        self._Schema.from_spark.return_value.cast_spark_tabular.side_effect = lambda f: f
        return stack

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
        # writeStream checkpoint + AvailableNow trigger, sinking into the table
        # via ``.toTable`` (Structured Streaming's built-in exactly-once).
        writer.option.assert_any_call("checkpointLocation", "s3://bkt/tbl/_ygg_autoloader")
        writer.trigger.assert_called_once_with(availableNow=True)
        writer.toTable.assert_called_once_with("cat.sch.tbl")
        writer.toTable.return_value.awaitTermination.assert_called_once()
        assert out["checkpoint"] == "s3://bkt/tbl/_ygg_autoloader"

    def test_casts_stream_to_target_schema_via_yggdrasil(self):
        spark, reader, frame, writer = self._spark()
        with self._with_pyspark(spark):
            auto_load("cat.sch.tbl", "s3://src", checkpoint="s3://ckpt")

            Schema = self._Schema
            # Target schema read off the live table; the stream cast to it via
            # yggdrasil's field casting before the ``.toTable`` sink.
            spark.table.assert_called_once_with("cat.sch.tbl")
            Schema.from_spark.assert_called_once_with(spark.table.return_value.schema)
            Schema.from_spark.return_value.cast_spark_tabular.assert_called_once_with(frame)
        # Identity cast here, so the casted frame is the frame that gets written.
        writer.toTable.assert_called_once_with("cat.sch.tbl")

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
    def test_auto_loader_defaults_source_to_governed_volume_path(self):
        # Default source/checkpoint = the governed ``/Volumes/…`` staging-volume
        # path via ``Volume.path`` (NOT the ``/`` operator, which would resolve
        # to a raw ``s3://`` URL for an external volume). On-cluster cloudFiles
        # reads the volume path through UC's optimized access — works for managed
        # *and* external volumes — while uploads keep the direct-S3 fast path.
        tbl = _table()
        vol = MagicMock()
        leaves = {
            Table.STAGE_SUBPATH: "/Volumes/cat/sch/tbl/.staging/data",
            Table.CHECKPOINT_SUBPATH: "/Volumes/cat/sch/tbl/.staging/_autoloader",
        }
        def _path(sub):
            m = MagicMock()
            m.full_path.return_value = leaves[sub]
            return m
        vol.path.side_effect = _path
        with patch.object(
            Table, "staging_volume", new_callable=PropertyMock, return_value=vol,
        ), patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader(file_arrival=True)  # no source, no checkpoint

        # The staging volume is ensured present before its paths are taken, and
        # the *volume* path (``.path``) is used — never the ``/`` operator.
        vol.get_or_create.assert_called_once()
        vol.path.assert_any_call(Table.STAGE_SUBPATH)
        vol.path.assert_any_call(Table.CHECKPOINT_SUBPATH)
        vol.__truediv__.assert_not_called()
        cmd = Flow.call_args.kwargs["command"]
        assert cmd[cmd.index("--source") + 1] == "/Volumes/cat/sch/tbl/.staging/data"
        assert cmd[cmd.index("--checkpoint") + 1] == "/Volumes/cat/sch/tbl/.staging/_autoloader"
        trig = Flow.call_args.kwargs["trigger"]
        assert trig.file_arrival.url == "/Volumes/cat/sch/tbl/.staging/data/"

    def test_explicit_source_leaves_checkpoint_to_on_cluster_default(self):
        # With an explicit source and no checkpoint, the staging volume is not
        # touched and the command carries no --checkpoint flag so the on-cluster
        # step derives <table-location>/_ygg_autoloader (unchanged behavior).
        tbl = _table()
        staging = PropertyMock()
        with patch.object(Table, "staging_volume", new=staging), \
                patch("yggdrasil.databricks.job.skeleton.Flow") as Flow:
            tbl.auto_loader("s3://bkt/landing", file_arrival=False)
        staging.assert_not_called()
        assert "--checkpoint" not in Flow.call_args.kwargs["command"]
