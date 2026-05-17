"""Tests for ``@flow(file_trigger=...)`` — file-arrival trigger config.

Covers the four input shapes :func:`_coerce_file_trigger` accepts (a
plain workspace path string, a dict of
:class:`FileArrivalTriggerConfiguration` kwargs, a pre-built
:class:`FileArrivalTriggerConfiguration`, and a pass-through
:class:`TriggerSettings`), the way :attr:`Flow.pause_status` flows
through into the coerced :class:`TriggerSettings`, the conflict check
that catches a caller passing both ``file_trigger=`` *and* an explicit
``trigger=`` through ``**job_settings``, and the end-to-end deploy
path which lands the trigger on the upserted job settings.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from databricks.sdk.service.jobs import (
    FileArrivalTriggerConfiguration,
    PauseStatus,
    TriggerSettings,
)

from yggdrasil.databricks.workflow import flow, task
from yggdrasil.databricks.workflow.flow import _coerce_file_trigger


@task
def _noop() -> None:
    return None


class TestCoerceFileTrigger(unittest.TestCase):
    """Direct tests for the input-shape coercion helper."""

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_coerce_file_trigger(None, pause_status=None))

    def test_string_lowers_to_trigger_settings_with_url(self) -> None:
        ts = _coerce_file_trigger("/Volumes/main/landing/inbox/", pause_status=None)

        self.assertIsInstance(ts, TriggerSettings)
        self.assertEqual(ts.file_arrival.url, "/Volumes/main/landing/inbox/")
        self.assertIsNone(ts.file_arrival.min_time_between_triggers_seconds)
        self.assertIsNone(ts.pause_status)

    def test_dict_form_carries_debounce_knobs(self) -> None:
        ts = _coerce_file_trigger(
            {
                "url": "/Volumes/main/landing/inbox/",
                "min_time_between_triggers_seconds": 60,
                "wait_after_last_change_seconds": 30,
            },
            pause_status=None,
        )

        self.assertEqual(ts.file_arrival.url, "/Volumes/main/landing/inbox/")
        self.assertEqual(ts.file_arrival.min_time_between_triggers_seconds, 60)
        self.assertEqual(ts.file_arrival.wait_after_last_change_seconds, 30)

    def test_dict_form_requires_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "must include 'url'"):
            _coerce_file_trigger(
                {"min_time_between_triggers_seconds": 60},
                pause_status=None,
            )

    def test_file_arrival_config_wrapped_with_pause_status(self) -> None:
        fac = FileArrivalTriggerConfiguration(
            url="/Volumes/x",
            wait_after_last_change_seconds=15,
        )

        ts = _coerce_file_trigger(fac, pause_status="PAUSED")

        self.assertIs(ts.file_arrival, fac)
        self.assertEqual(ts.pause_status, PauseStatus.PAUSED)

    def test_trigger_settings_passes_through_untouched(self) -> None:
        original = TriggerSettings(
            file_arrival=FileArrivalTriggerConfiguration(url="/Volumes/x"),
            pause_status=PauseStatus.PAUSED,
        )

        ts = _coerce_file_trigger(original, pause_status="UNPAUSED")

        # Pass-through — caller takes full ownership. We don't second-
        # guess their pause_status choice.
        self.assertIs(ts, original)
        self.assertEqual(ts.pause_status, PauseStatus.PAUSED)

    def test_pause_status_string_normalises_to_enum_member(self) -> None:
        ts = _coerce_file_trigger("/Volumes/x", pause_status="paused")

        self.assertEqual(ts.pause_status, PauseStatus.PAUSED)

    def test_unsupported_type_raises_with_hint(self) -> None:
        with self.assertRaisesRegex(TypeError, "expected a workspace path"):
            _coerce_file_trigger(42, pause_status=None)


class TestFlowFileTriggerKwarg(unittest.TestCase):
    """``@flow(file_trigger=...)`` end-to-end behaviour."""

    def test_flow_stores_file_trigger_on_instance(self) -> None:
        @flow(name="x", file_trigger="/Volumes/landing/inbox/")
        def f():
            _noop()

        self.assertEqual(f.file_trigger, "/Volumes/landing/inbox/")

    def test_conflict_with_explicit_trigger_kwarg_raises(self) -> None:
        ts = TriggerSettings(
            file_arrival=FileArrivalTriggerConfiguration(url="/x"),
        )

        with self.assertRaisesRegex(ValueError, "pass either file_trigger"):
            @flow(name="x", file_trigger="/y", trigger=ts)
            def bad():
                _noop()

    def test_deploy_lands_file_trigger_on_job_settings(self) -> None:
        @flow(name="on-file", prefix=False, file_trigger="/Volumes/landing/inbox/")
        def on_file():
            _noop()

        jobs = MagicMock()
        jobs.client = MagicMock()
        # Avoid the real staging pipeline — patch the helper that produces
        # the staged Task so we don't hit the workspace.
        from yggdrasil.databricks.jobs.task import stage_python_callable
        with unittest_mock_patch_stage(stage_python_callable):
            on_file.deploy(service=jobs)

        # Inspect the kwargs passed to ``Jobs.create_or_update`` — the
        # trigger must land on the JobSettings payload.
        _, kwargs = jobs.create_or_update.call_args
        self.assertIn("trigger", kwargs)
        trigger = kwargs["trigger"]
        self.assertIsInstance(trigger, TriggerSettings)
        self.assertEqual(trigger.file_arrival.url, "/Volumes/landing/inbox/")

    def test_deploy_threads_pause_status_into_trigger(self) -> None:
        @flow(
            name="on-file-paused",
            prefix=False,
            file_trigger="/Volumes/landing/inbox/",
            pause_status="PAUSED",
        )
        def on_file_paused():
            _noop()

        jobs = MagicMock()
        jobs.client = MagicMock()
        from yggdrasil.databricks.jobs.task import stage_python_callable
        with unittest_mock_patch_stage(stage_python_callable):
            on_file_paused.deploy(service=jobs)

        _, kwargs = jobs.create_or_update.call_args
        self.assertEqual(kwargs["trigger"].pause_status, PauseStatus.PAUSED)

    def test_schedule_and_file_trigger_coexist(self) -> None:
        @flow(
            name="dual-signal",
            prefix=False,
            schedule="0 0 * * * ?",
            file_trigger="/Volumes/landing/inbox/",
        )
        def dual_signal():
            _noop()

        jobs = MagicMock()
        jobs.client = MagicMock()
        from yggdrasil.databricks.jobs.task import stage_python_callable
        with unittest_mock_patch_stage(stage_python_callable):
            dual_signal.deploy(service=jobs)

        _, kwargs = jobs.create_or_update.call_args
        # Both signals land on the JobSettings — Databricks runs the
        # job on whichever fires first.
        self.assertIn("schedule", kwargs)
        self.assertIn("trigger", kwargs)


# ---------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------- #


def unittest_mock_patch_stage(stage_fn):
    """Patch :func:`stage_python_callable` to return a stub Task.

    The full staging pipeline writes to the workspace and sniffs imports
    — too heavy and stateful for a unit test that only cares about the
    final ``JobSettings`` payload. We return a minimal :class:`Task`
    that satisfies :meth:`Flow._stage_nodes`'s assembly path.
    """
    from contextlib import contextmanager
    from unittest.mock import patch

    from databricks.sdk.service.jobs import SparkPythonTask, Task

    def _fake_stage(client, func, *args, task_key=None, **kwargs):
        task_obj = Task(
            task_key=task_key or func.__name__,
            spark_python_task=SparkPythonTask(python_file="/Workspace/stub.py"),
            environment_key="ygg-default",
        )
        return task_obj, [], []

    @contextmanager
    def _ctx():
        # The import inside :meth:`WorkflowTask.stage` is function-scope
        # (``from yggdrasil.databricks.jobs.task import stage_python_callable``),
        # so we have to patch the source attribute on the jobs module —
        # patching the workflow module would miss the runtime lookup.
        with patch(
            "yggdrasil.databricks.jobs.task.stage_python_callable",
            new=_fake_stage,
        ):
            yield

    return _ctx()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
