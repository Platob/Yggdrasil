"""Unit tests for :func:`deploy_scheduled_fxrate_job`.

These exercise the deploy plumbing — input coercion, schedule
shaping, compute-pin patching, staged-script content — against a
mocked Jobs service and workspace path. No real Databricks
involvement; the live integration path is covered separately by
the operator running it against their workspace.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# databricks-sdk is required for these tests — `Job.from_callable`
# imports it eagerly. Skip the module when the SDK isn't installed
# (matching the per-extra skip pattern the Databricks tests use).
pytest.importorskip("databricks.sdk")

from databricks.sdk.service.jobs import CronSchedule, PauseStatus

from yggdrasil.databricks.fs.workspace_path import WorkspacePath
from yggdrasil.databricks.tests import DatabricksTestCase
from yggdrasil.fxrate.job import (
    _coerce_cron_schedule,
    _pairs_to_json,
    deploy_scheduled_fxrate_job,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _StagedScriptCapture:
    """Capture every staged ``.py`` written via WorkspacePath.write_bytes.

    Lets us assert the rendered task script bakes the deploy-time
    args (target_table, pairs_json, lookback_days, sampling, geo) as
    literals.
    """

    def __init__(self) -> None:
        self.captured: dict[str, str] = {}

    def install(self, test_case) -> None:
        orig_write = WorkspacePath.write_bytes
        captured = self.captured

        def _write(this: WorkspacePath, data: bytes):
            captured[this.url.path or this.full_path()] = data.decode()

        WorkspacePath.write_bytes = _write
        test_case.addCleanup(setattr, WorkspacePath, "write_bytes", orig_write)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


class TestCoercion:
    """``_coerce_cron_schedule`` + ``_pairs_to_json`` standalone behaviour."""

    def test_cron_string_becomes_cron_schedule(self) -> None:
        cron = _coerce_cron_schedule(
            "0 0 6 * * ?", timezone="UTC", pause_status="PAUSED",
        )
        assert isinstance(cron, CronSchedule)
        assert cron.quartz_cron_expression == "0 0 6 * * ?"
        assert cron.timezone_id == "UTC"
        assert cron.pause_status == PauseStatus.PAUSED

    def test_pause_status_string_normalised(self) -> None:
        cron = _coerce_cron_schedule(
            "0 0 6 * * ?", timezone="Europe/Paris", pause_status="unpaused",
        )
        assert cron.pause_status == PauseStatus.UNPAUSED

    def test_cron_schedule_passthrough(self) -> None:
        original = CronSchedule(
            quartz_cron_expression="*/15 * * * * ?",
            timezone_id="UTC",
            pause_status=PauseStatus.PAUSED,
        )
        result = _coerce_cron_schedule(
            original, timezone="UTC", pause_status="PAUSED",
        )
        assert result is original

    def test_invalid_schedule_raises(self) -> None:
        with pytest.raises(TypeError, match="Quartz cron"):
            _coerce_cron_schedule(
                None, timezone="UTC", pause_status="PAUSED",
            )
        with pytest.raises(TypeError, match="Quartz cron"):
            _coerce_cron_schedule(
                42, timezone="UTC", pause_status="PAUSED",
            )

    def test_pairs_json_round_trip_uppercases_codes(self) -> None:
        payload = _pairs_to_json([("eur", "usd"), ("Gbp", "Jpy")])
        decoded = json.loads(payload)
        assert decoded == [["EUR", "USD"], ["GBP", "JPY"]]

    def test_pairs_json_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _pairs_to_json([])

    def test_pairs_json_accepts_aliases(self) -> None:
        payload = _pairs_to_json([("$", "€")])
        assert json.loads(payload) == [["USD", "EUR"]]


# ---------------------------------------------------------------------------
# End-to-end deploy plumbing
# ---------------------------------------------------------------------------


class TestDeploySchedule(DatabricksTestCase):
    """Trace + stage + upsert path against a mocked Jobs service."""

    def setUp(self) -> None:
        super().setUp()
        self._capture = _StagedScriptCapture()
        self._capture.install(self)
        # No pre-existing job → ``find`` returns nothing, create_or_update
        # falls through to ``create``.
        self.workspace_client.jobs.list.return_value = iter([])

    def _patch_create(self):
        # Patch ``Jobs.create_or_update`` to short-circuit the SDK round
        # trip and return a synthetic Job handle. Capture call kwargs so
        # the tests can assert what landed on the spec.
        return patch.object(
            self.client.jobs, "create_or_update",
        )

    def test_creates_job_with_schedule_and_tasks(self) -> None:
        with self._patch_create() as create_or_update:
            create_or_update.return_value = MagicMock(
                job_id=99, job_name="ygg-fxrate-ingestion",
            )
            job = deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD"), ("EUR", "GBP")],
                schedule="0 0 6 * * ?",
                client=self.client,
            )

        assert job.job_id == 99
        create_or_update.assert_called_once()
        call = create_or_update.call_args
        assert call.kwargs["name"] == "ygg-fxrate-ingestion"
        cron = call.kwargs["schedule"]
        assert isinstance(cron, CronSchedule)
        assert cron.quartz_cron_expression == "0 0 6 * * ?"
        assert cron.timezone_id == "UTC"
        assert cron.pause_status == PauseStatus.PAUSED

    def test_description_mentions_target_and_cron(self) -> None:
        with self._patch_create() as create_or_update:
            create_or_update.return_value = MagicMock(job_id=1, job_name="ygg-fxrate-ingestion")
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                client=self.client,
            )
        desc = create_or_update.call_args.kwargs["description"]
        assert "main.fx.raw_fxrate" in desc
        assert "0 0 6 * * ?" in desc

    def test_tags_pass_through(self) -> None:
        with self._patch_create() as create_or_update:
            create_or_update.return_value = MagicMock(job_id=1, job_name="ygg-fxrate-ingestion")
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                tags={"owner": "data-eng", "tier": "raw"},
                client=self.client,
            )
        tags = create_or_update.call_args.kwargs["tags"]
        assert tags == {"owner": "data-eng", "tier": "raw"}

    def test_pause_status_unpaused_passes_through(self) -> None:
        with self._patch_create() as create_or_update:
            create_or_update.return_value = MagicMock(job_id=1, job_name="ygg-fxrate-ingestion")
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                pause_status="UNPAUSED",
                client=self.client,
            )
        cron = create_or_update.call_args.kwargs["schedule"]
        assert cron.pause_status == PauseStatus.UNPAUSED

    def test_staged_script_bakes_deploy_time_args(self) -> None:
        with self._patch_create() as create_or_update:
            create_or_update.return_value = MagicMock(job_id=1, job_name="ygg-fxrate-ingestion")
            deploy_scheduled_fxrate_job(
                target_table="prod.fx.raw_fxrate_default",
                pairs=[("EUR", "USD"), ("USD", "BTC")],
                schedule="0 0 6 * * ?",
                lookback_days=14,
                sampling="1d",
                geo=True,
                client=self.client,
            )

        assert self._capture.captured, "no staged scripts captured"
        # Find the fxrate_ingestion script (only one task).
        script_bodies = [
            body for body in self._capture.captured.values()
            if "fxrate_ingestion_entrypoint" in body
        ]
        assert len(script_bodies) >= 1
        body = script_bodies[0]
        assert "'prod.fx.raw_fxrate_default'" in body
        # Pairs ride as a JSON literal — uppercased + JSON-shaped.
        assert '[["EUR","USD"],["USD","BTC"]]' in body
        assert "lookback_days=14" in body
        assert "sampling='1d'" in body
        assert "geo=True" in body

    def test_existing_cluster_pin_lands_on_task_spec(self) -> None:
        captured_tasks: list = []

        def _capture_task_create(self_task):
            # Intercept ``JobTask.create`` so we can inspect the
            # patched Task spec without going through the SDK update.
            captured_tasks.append(self_task)
            return self_task

        with self._patch_create() as create_or_update, \
                patch("yggdrasil.databricks.jobs.task.JobTask.create", _capture_task_create):
            create_or_update.return_value = MagicMock(job_id=1, job_name="ygg-fxrate-ingestion")
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                existing_cluster_id="0123-456789-abc",
                client=self.client,
            )

        assert captured_tasks, "JobTask.create was never called"
        details = captured_tasks[0]._details
        assert details.existing_cluster_id == "0123-456789-abc"
        # Cluster pin disables the serverless env default.
        assert details.environment_key is None

    def test_environment_key_pin_keeps_serverless(self) -> None:
        captured_tasks: list = []

        def _capture_task_create(self_task):
            captured_tasks.append(self_task)
            return self_task

        with self._patch_create() as create_or_update, \
                patch("yggdrasil.databricks.jobs.task.JobTask.create", _capture_task_create):
            create_or_update.return_value = MagicMock(job_id=1, job_name="ygg-fxrate-ingestion")
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                environment_key="ygg-internet",
                client=self.client,
            )

        details = captured_tasks[0]._details
        assert details.environment_key == "ygg-internet"
        assert details.existing_cluster_id is None
        assert details.new_cluster is None

    def test_mutually_exclusive_compute_pins_raise(self) -> None:
        with pytest.raises(ValueError, match="at most one compute pin"):
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                existing_cluster_id="0123-456789-abc",
                environment_key="ygg-internet",
                client=self.client,
            )

    def test_invalid_pair_raises_at_deploy_time(self) -> None:
        # A typo / unknown alias must fail at deploy time, not on the
        # first scheduled run — that's the whole point of routing
        # through ``_pairs_to_json`` here.
        with pytest.raises(ValueError):
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "EUR")],  # identical sides — rejected
                schedule="0 0 6 * * ?",
                client=self.client,
            )

    def test_auto_create_cluster_resolves_id_and_pins_task(self) -> None:
        # When auto_create_cluster=True the deploy path calls
        # ``client.compute.clusters.all_purpose_cluster(...)`` and
        # pins the returned cluster_id onto the staged task.
        captured_tasks: list = []

        def _capture_task_create(self_task):
            captured_tasks.append(self_task)
            return self_task

        fake_cluster = MagicMock(cluster_id="9999-cluster", cluster_name="fx-auto")

        with self._patch_create() as create_or_update, \
                patch("yggdrasil.databricks.jobs.task.JobTask.create", _capture_task_create), \
                patch.object(
                    self.client.compute.clusters, "all_purpose_cluster",
                    return_value=fake_cluster,
                ) as cluster_call:
            create_or_update.return_value = MagicMock(job_id=1, job_name="ygg-fxrate-ingestion")
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                auto_create_cluster=True,
                cluster_name="fx-auto",
                cluster_node_type_id="m5d.large",
                cluster_extra_libraries=["pyarrow-hotfix"],
                cluster_spec={"num_workers": 0, "autotermination_minutes": 20},
                client=self.client,
            )

        cluster_call.assert_called_once()
        kwargs = cluster_call.call_args.kwargs
        assert kwargs["name"] == "fx-auto"
        # extra_libraries forwards as the ``libraries`` arg.
        assert kwargs["libraries"] == ["pyarrow-hotfix"]
        # node_type_id + cluster_spec merge into the **cluster_spec
        # passthrough — caller wins on collision.
        assert kwargs["node_type_id"] == "m5d.large"
        assert kwargs["num_workers"] == 0
        assert kwargs["autotermination_minutes"] == 20

        # The staged task now pins the auto-resolved cluster_id.
        assert captured_tasks
        details = captured_tasks[0]._details
        assert details.existing_cluster_id == "9999-cluster"
        assert details.environment_key is None

    def test_auto_create_cluster_no_cluster_name(self) -> None:
        # ``cluster_name=None`` lets ``all_purpose_cluster`` pick the
        # default (per-user) name. We just verify the call goes
        # through and pins the result.
        captured_tasks: list = []
        fake_cluster = MagicMock(cluster_id="0001-default-cluster")

        with self._patch_create() as create_or_update, \
                patch("yggdrasil.databricks.jobs.task.JobTask.create",
                      lambda self_task: captured_tasks.append(self_task) or self_task), \
                patch.object(
                    self.client.compute.clusters, "all_purpose_cluster",
                    return_value=fake_cluster,
                ):
            create_or_update.return_value = MagicMock(job_id=1, job_name="ygg-fxrate-ingestion")
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                auto_create_cluster=True,
                client=self.client,
            )

        assert captured_tasks
        assert captured_tasks[0]._details.existing_cluster_id == "0001-default-cluster"

    def test_auto_create_cluster_conflicts_with_explicit_pin(self) -> None:
        with pytest.raises(ValueError, match="at most one compute pin"):
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                auto_create_cluster=True,
                existing_cluster_id="0123-456789-abc",
                client=self.client,
            )

    def test_auto_create_cluster_raises_when_id_missing(self) -> None:
        # Defensive: ``all_purpose_cluster`` should always return a
        # Cluster with a real cluster_id, but if it doesn't (mock
        # leak, workspace race condition) we must fail loud rather
        # than emit a Task spec with ``existing_cluster_id=None``.
        fake_cluster = MagicMock(cluster_id=None, cluster_name="fx-broken")
        with self._patch_create(), \
                patch.object(
                    self.client.compute.clusters, "all_purpose_cluster",
                    return_value=fake_cluster,
                ):
            with pytest.raises(RuntimeError, match="no cluster_id"):
                deploy_scheduled_fxrate_job(
                    target_table="main.fx.raw_fxrate",
                    pairs=[("EUR", "USD")],
                    schedule="0 0 6 * * ?",
                    auto_create_cluster=True,
                    cluster_name="fx-broken",
                    client=self.client,
                )

    def test_idempotent_redeploy_updates_in_place(self) -> None:
        # Simulate a pre-existing job on the workspace; the deploy
        # path should route through ``update`` (find returned a hit)
        # rather than ``create``. There are *two* update calls in
        # total — the first carries the schedule + description from
        # ``create_or_update``, the second carries the staged task
        # spec from ``JobTask.create``.
        existing = MagicMock(job_id=42, job_name="ygg-fxrate-ingestion")
        with patch.object(self.client.jobs, "find", return_value=existing) as find_mock, \
                patch.object(existing, "update", return_value=existing) as update_mock:
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                client=self.client,
            )

        find_mock.assert_called()
        assert update_mock.call_count >= 1
        # The first update carries the job-level shell (schedule +
        # description); the task update follows.
        first_call = update_mock.call_args_list[0]
        assert isinstance(first_call.kwargs["schedule"], CronSchedule)
        assert "main.fx.raw_fxrate" in first_call.kwargs["description"]


class TestDeployDashTask(DatabricksTestCase):
    """Coverage for the downstream ``dash_fxrate`` task wiring.

    The dash task only stages when *dash_table* is set on the deploy
    call. When it does, two ``JobTask`` instances flow through
    ``JobTask.create`` — the ingestion task plus the dash task — and
    the dash task's ``Task`` spec carries a ``depends_on`` link
    pointing at the ingestion task_key.
    """

    def setUp(self) -> None:
        super().setUp()
        self._capture = _StagedScriptCapture()
        self._capture.install(self)
        self.workspace_client.jobs.list.return_value = iter([])

    def _patch_create(self):
        return patch.object(self.client.jobs, "create_or_update")

    def test_no_dash_table_stages_one_task(self) -> None:
        captured_tasks: list = []
        with self._patch_create() as create_or_update, \
                patch(
                    "yggdrasil.databricks.jobs.task.JobTask.create",
                    lambda self_task: captured_tasks.append(self_task) or self_task,
                ):
            create_or_update.return_value = MagicMock(
                job_id=1, job_name="ygg-fxrate-ingestion",
            )
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                client=self.client,
            )
        # Single ingestion task, no dash task.
        assert len(captured_tasks) == 1
        assert captured_tasks[0].task_key == "fxrate_ingestion"

    def test_dash_table_stages_two_tasks_with_depends_on(self) -> None:
        captured_tasks: list = []
        with self._patch_create() as create_or_update, \
                patch(
                    "yggdrasil.databricks.jobs.task.JobTask.create",
                    lambda self_task: captured_tasks.append(self_task) or self_task,
                ):
            create_or_update.return_value = MagicMock(
                job_id=1, job_name="ygg-fxrate-ingestion",
            )
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                dash_table="main.fx.dash_fxrate",
                client=self.client,
            )

        # Ingestion task + dash task — exactly two staged.
        keys = [t.task_key for t in captured_tasks]
        assert keys == ["fxrate_ingestion", "dash_fxrate"]

        ingestion_details = captured_tasks[0]._details
        dash_details = captured_tasks[1]._details

        # The ingestion task carries no depends_on; the dash task
        # depends on the ingestion task_key.
        assert not ingestion_details.depends_on
        assert dash_details.depends_on is not None
        assert len(dash_details.depends_on) == 1
        assert dash_details.depends_on[0].task_key == "fxrate_ingestion"

    def test_dash_task_script_bakes_default_targets(self) -> None:
        with self._patch_create() as create_or_update:
            create_or_update.return_value = MagicMock(
                job_id=1, job_name="ygg-fxrate-ingestion",
            )
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                dash_table="main.fx.dash_fxrate",
                client=self.client,
            )

        scripts = self._capture.captured.values()
        dash_bodies = [s for s in scripts if "dash_fxrate_entrypoint" in s]
        assert len(dash_bodies) == 1
        body = dash_bodies[0]
        # Source / target tables baked as ``repr``-d literals.
        assert "'main.fx.raw_fxrate'" in body
        assert "'main.fx.dash_fxrate'" in body
        # Default EUR/USD/CHF targets ride as JSON literal — bake
        # carries the ISO codes inside the targets_json kwarg.
        for code in ("EUR", "USD", "CHF"):
            assert code in body
        # Default lookback is 30 days.
        assert "lookback_days=30" in body

    def test_dash_targets_override(self) -> None:
        with self._patch_create() as create_or_update:
            create_or_update.return_value = MagicMock(
                job_id=1, job_name="ygg-fxrate-ingestion",
            )
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD"), ("EUR", "JPY")],
                schedule="0 0 6 * * ?",
                dash_table="main.fx.dash_fxrate",
                dash_targets=("EUR", "JPY", "GBP"),
                dash_lookback_days=14,
                client=self.client,
            )

        scripts = self._capture.captured.values()
        dash_body = next(s for s in scripts if "dash_fxrate_entrypoint" in s)
        # Custom targets land in the staged script body.
        assert '"EUR"' in dash_body and '"JPY"' in dash_body and '"GBP"' in dash_body
        assert "lookback_days=14" in dash_body

    def test_dash_targets_invalid_raises_at_deploy_time(self) -> None:
        # ``Currency.from_`` enforces ISO 4217 — a 4-char code fails fast.
        with self._patch_create() as create_or_update:
            create_or_update.return_value = MagicMock(
                job_id=1, job_name="ygg-fxrate-ingestion",
            )
            with pytest.raises(ValueError, match="ISO-4217"):
                deploy_scheduled_fxrate_job(
                    target_table="main.fx.raw_fxrate",
                    pairs=[("EUR", "USD")],
                    schedule="0 0 6 * * ?",
                    dash_table="main.fx.dash_fxrate",
                    dash_targets=("EUROS",),  # 5-char — invalid ISO 4217
                    client=self.client,
                )

    def test_dash_task_inherits_cluster_pin(self) -> None:
        captured_tasks: list = []
        with self._patch_create() as create_or_update, \
                patch(
                    "yggdrasil.databricks.jobs.task.JobTask.create",
                    lambda self_task: captured_tasks.append(self_task) or self_task,
                ):
            create_or_update.return_value = MagicMock(
                job_id=1, job_name="ygg-fxrate-ingestion",
            )
            deploy_scheduled_fxrate_job(
                target_table="main.fx.raw_fxrate",
                pairs=[("EUR", "USD")],
                schedule="0 0 6 * * ?",
                dash_table="main.fx.dash_fxrate",
                existing_cluster_id="0123-456789-abc",
                client=self.client,
            )

        assert len(captured_tasks) == 2
        ingestion, dash_task = captured_tasks
        # Both tasks pinned to the same cluster — the dash task
        # inherits the ingestion task's compute pin so the deploy
        # story is uniform (CLAUDE.md "Pick compute by workload type").
        assert ingestion._details.existing_cluster_id == "0123-456789-abc"
        assert dash_task._details.existing_cluster_id == "0123-456789-abc"
