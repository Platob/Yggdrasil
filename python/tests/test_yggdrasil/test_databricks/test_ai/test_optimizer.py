"""Unit tests for the propose-only repository optimizer agent.

The path layer (workspace walk + proposal writes) is faked so the test is
hermetic, while the serving call goes through the *real* ModelServing wiring
onto the mocked ``workspace_client.serving_endpoints`` SDK boundary. Asserts
filtering, per-file optimization, proposal writing, the propose-only
guarantee (source is never written), the ``max_files`` bound, and report
rendering.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from yggdrasil.databricks.ai import OptimizerConfig, RepoOptimizer, RepoOptimizerFlow
from yggdrasil.databricks.tests import DatabricksTestCase

ROOT = "/Workspace/Shared/monteleq"


class FakeFile:
    """A readable source file in the fake repo."""

    def __init__(self, rel: str, content: str, *, size: int | None = None):
        self.full = f"{ROOT}/{rel}"
        self.content = content
        self.suffix = "." + rel.rsplit(".", 1)[1] if "." in rel else ""
        self.size = size if size is not None else len(content.encode())

    def is_file(self) -> bool:
        return True

    def full_path(self) -> str:
        return self.full

    def read_text(self) -> str:
        return self.content


class FakeDir:
    def is_file(self) -> bool:
        return False

    def full_path(self) -> str:
        return f"{ROOT}/pkg"


class WritePath:
    """Records every ``write_text`` into a shared sink; navigable like a Path."""

    def __init__(self, path: str, sink: dict):
        self.path = path
        self.sink = sink

    def joinpath(self, *segs) -> "WritePath":
        return WritePath(self.path.rstrip("/") + "/" + "/".join(str(s) for s in segs), self.sink)

    @property
    def parent(self) -> "WritePath":
        return WritePath(self.path.rsplit("/", 1)[0], self.sink)

    def mkdir(self, parents=True, exist_ok=True):
        return self

    def write_text(self, content: str):
        self.sink[self.path] = content


class TestRepoOptimizer(DatabricksTestCase):

    def setUp(self):
        super().setUp()
        self.sink: dict[str, str] = {}
        self.files = [
            FakeFile("a.py", "print( 1 )\n"),
            FakeFile("pkg/b.sql", "select 1"),
            FakeDir(),
            FakeFile("notes.md", "# skip me, wrong suffix"),
            FakeFile("huge.py", "x" * 10, size=10_000_000),  # over byte limit
            FakeFile(".optimizer/proposals/old/files/a.py", "stale"),  # excluded dir
        ]
        root = SimpleNamespace(
            full_path=lambda: ROOT,
            ls=lambda recursive=False: iter(self.files),
        )
        # repo() always returns the fake root; client.path() returns write sinks
        self.optimizer = RepoOptimizer(
            client=self.client,
            config=OptimizerConfig(max_file_bytes=1000, max_tokens=256),
        )
        self.optimizer.repo = lambda: root
        self.client.path = lambda p, **kw: WritePath(p, self.sink)

        # Real ModelServing → mocked SDK boundary. Reply per requested file.
        def _query(name, messages, **kwargs):
            prompt = messages[-1].content
            if "a.py" in prompt:
                body = {"summary": "tidy spacing", "findings": ["extra spaces"],
                        "changed": True, "optimized_code": "print(1)\n"}
            else:
                body = {"summary": "already good", "findings": [], "changed": False,
                        "optimized_code": None}
            choice = SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(body)),
                finish_reason="stop", text=None,
            )
            return SimpleNamespace(choices=[choice], model="m", usage=None, predictions=None)

        self.workspace_client.serving_endpoints.query.side_effect = _query

    def test_iter_source_files_filters(self):
        rels = [f.full_path().split(ROOT + "/")[1] for f in self.optimizer.iter_source_files()]
        # .md (wrong suffix), the dir, the oversized file, and the .optimizer/ tree are all dropped
        self.assertEqual(rels, ["a.py", "pkg/b.sql"])

    def test_run_proposes_only_changed_files(self):
        report = self.optimizer.run()
        self.assertEqual(report.files_scanned, 2)
        self.assertEqual(len(report.changed), 1)
        self.assertEqual(report.changed[0].rel_path, "a.py")

        # The optimized a.py is written under the proposals/files tree...
        written = {k: v for k, v in self.sink.items() if k.endswith("/files/a.py")}
        self.assertEqual(len(written), 1)
        self.assertEqual(next(iter(written.values())), "print(1)\n")

        # ...and a REPORT.md is produced.
        reports = [k for k in self.sink if k.endswith("REPORT.md")]
        self.assertEqual(len(reports), 1)
        self.assertIn("tidy spacing", self.sink[reports[0]])
        self.assertIn("already good", self.sink[reports[0]])

    def test_propose_only_never_writes_source(self):
        self.optimizer.run()
        # Nothing is ever written outside the proposals subtree.
        for path in self.sink:
            self.assertIn("/.optimizer/proposals/", path)
        # The original source paths are untouched.
        self.assertNotIn(f"{ROOT}/a.py", self.sink)

    def test_max_files_bounds_the_pass(self):
        self.optimizer.config.max_files = 1
        report = self.optimizer.run()
        self.assertEqual(report.files_scanned, 1)

    def test_unparseable_reply_is_skipped_not_fatal(self):
        self.workspace_client.serving_endpoints.query.side_effect = None
        self.workspace_client.serving_endpoints.query.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="oops not json"),
                                     finish_reason="stop", text=None)],
            model="m", usage=None, predictions=None,
        )
        report = self.optimizer.run()
        self.assertTrue(all(p.skipped for p in report.proposals))
        self.assertEqual(len(report.changed), 0)


class TestRepoOptimizerFlow(DatabricksTestCase):

    def test_parameters_round_trip_through_cli_shape(self):
        flow = RepoOptimizerFlow(repo_path=ROOT, endpoint_name="my-llm", max_files=5, interval=12)
        params = flow.parameters()
        self.assertEqual(params[:3], ["databricks", "optimizer", "run"])
        self.assertIn("--repo", params)
        self.assertEqual(params[params.index("--repo") + 1], ROOT)
        self.assertEqual(params[params.index("--endpoint") + 1], "my-llm")
        self.assertEqual(params[params.index("--max-files") + 1], "5")

    def test_trigger_is_periodic(self):
        from databricks.sdk.service.jobs import PauseStatus

        flow = RepoOptimizerFlow(repo_path=ROOT, interval=3, unit="DAYS")
        trig = flow.trigger()
        self.assertEqual(trig.periodic.interval, 3)
        self.assertEqual(trig.periodic.unit.value, "DAYS")
        self.assertEqual(trig.pause_status, PauseStatus.UNPAUSED)

    def test_name_slug_from_repo(self):
        self.assertEqual(RepoOptimizerFlow(repo_path=ROOT).name, "optimize-Workspace_Shared_monteleq")
