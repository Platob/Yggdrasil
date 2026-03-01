import unittest
from yggdrasil.databricks.workspaces.workspace import Workspace


class DatabricksIntegrationBase(unittest.TestCase):
    """
    Real integration tests. No fakes, no mocks.

    Requirements:
      - Databricks auth configured for databricks-sdk (env vars / config file)
      - The cluster / workspace must allow DBFS + Workspace API.
      - Volume tests require an existing UC volume base path.

    Optional env vars:
      - DATABRICKS_TEST_DBFS_BASE:    default "/tmp/yggdrasil_databricks_path_it"
      - DATABRICKS_TEST_WORKSPACE_BASE: default "/Users/<me>/yggdrasil_databricks_path_it"
      - DATABRICKS_TEST_VOLUME_BASE:  e.g. "/Volumes/<catalog>/<schema>/<volume>/yggdrasil_databricks_path_it"
    """

    @classmethod
    def setUpClass(cls):
        from yggdrasil.databricks.workspaces.workspace import Workspace

        cls.workspace = Workspace()

        # hard gate: if auth/network is broken, skip all tests in this file
        try:
            cls.workspace.sdk().current_user.me()
        except Exception as e:
            raise unittest.SkipTest(f"Databricks auth not configured or API not reachable: {e}")

        cls.dbfs_root = os.getenv("DATABRICKS_TEST_DBFS_BASE", "/tmp/yggdrasil_databricks_path_it")
        cls.workspace_root = os.getenv(
            "DATABRICKS_TEST_WORKSPACE_BASE",
            f"/Users/{cls.workspace.current_user.user_name}/yggdrasil_databricks_path_it",
        )
        cls.schema_root = os.getenv(
            "DATABRICKS_TEST_VOLUME_BASE",
            "/Volumes/trading/unittest"
        )  # may be None

    def setUp(self):
        # Unique per test so parallel runs don’t punch each other
        self.test_id = "unittest"

        self.dbfs_base = DatabricksPath(f"{self.dbfs_root}/{self.test_id}", workspace=self.workspace)
        self.ws_base = DatabricksPath(f"{self.workspace_root}/{self.test_id}", workspace=self.workspace)
        self.vol_base = DatabricksPath(f"{self.schema_root}/{self.test_id}", workspace=self.workspace)

    def tearDown(self):
        # Best-effort cleanup; don’t fail teardown
        for p in (self.vol_base, self.ws_base, self.dbfs_base):
            if p is None:
                continue
            try:
                p.rmdir(recursive=True)
            except Exception:
                pass