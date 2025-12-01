import base64
import io
import re
import zipfile
from pathlib import Path

from mongoengine import DynamicDocument, connect

from yggdrasil.databricks.compute.remote import (
    _build_remote_command,
    databricks_remote_compute,
)


def dump_os(*args, **kwargs):
    class Cities(DynamicDocument):
        meta = {'collection': 'cities'}

    # Connect to local MongoDB (standalone mode)
    connect(db="xxx",alias="default",host=prod)

    # Test the function
    docs = list(Cities.objects.limit(10))
    return [doc.to_mongo().to_dict() for doc in docs]

def test_remote_pyeval_executes_function_and_returns_value():
    result = databricks_remote_compute(
        "xxx",
        workspace="xxx.cloud.databricks.com",
    )(dump_os)

    assert result() == 5


def test_package_root_uploaded_for_remote_dill_imports():
    def sample_func() -> int:
        return 1

    command = _build_remote_command(
        sample_func,
        args=(),
        kwargs={},
        debug=False,
        debug_host=None,
        debug_port=5678,
        debug_suspend=True,
        upload_paths=None,
        remote_target=None,
    )

    match = re.search(r'_modules_zip = "(.*?)"', command)
    assert match is not None

    modules_zip = base64.b64decode(match.group(1))
    with zipfile.ZipFile(io.BytesIO(modules_zip)) as zf:
        names = set(zf.namelist())

    tests_root = Path(__file__).resolve().parents[3]
    expected_rel = Path(__file__).resolve().relative_to(tests_root)

    assert str(expected_rel) in names
