import pytest

from yggdrasil.databricks.workspaces.service import Workspaces
from ..conftest import requires_databricks, DatabricksCase

pytestmark = [requires_databricks, pytest.mark.integration]


class TestWorkspaces(DatabricksCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.service = cls.workspace.workspaces

    def test_ok(self):
        assert self.service is not None

    def test_url(self):
        url = self.service.to_url()
        built = Workspaces.from_parsed_url(url)
        assert built.to_url() == url
