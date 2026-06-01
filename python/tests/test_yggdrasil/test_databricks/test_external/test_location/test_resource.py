"""Resource: yggdrasil.databricks.external.location.resource.ExternalLocation."""
from __future__ import annotations

import pytest

from databricks.sdk.service.catalog import ExternalLocationInfo

from yggdrasil.databricks.external.location.resource import ExternalLocation


def test_metadata(service):
    el = service.get("raw_zone")
    assert el.url == "s3://my-bucket/raw/"
    assert el.read_only is False and el.owner == "data-eng"
    assert service.get("ro_zone").read_only is True


def test_lazy_info_single_fetch(service):
    el = service.location("raw_zone")  # lazy handle, no fetch
    api = service.client.workspace_client.return_value.external_locations
    api.get.assert_not_called()
    _ = el.url  # triggers exactly one GET
    _ = el.comment  # cached
    api.get.assert_called_once()


def test_path_is_a_credential_backed_s3path(service):
    from yggdrasil.aws.fs.path import S3Path

    p = service.get("raw_zone").path
    assert isinstance(p, S3Path)
    assert p.bucket == "my-bucket" and p.key == "raw/"
    # built via the storage credential's AWS client.
    service.client.credentials.credential.assert_called_with("prod-cred")


def test_filesystem_ops_mirror_to_inner_path(service):
    el = service.get("raw_zone")
    # write + read through the wrapper delegate to the inner S3Path.
    (el / "f.txt").write_bytes(b"hello")
    assert bytes((el / "f.txt").read_bytes()) == b"hello"
    assert el.exists()
    assert [c.key for c in el.ls()] == ["raw/f.txt"]


def test_parent_and_children_use_inner_path(service):
    from yggdrasil.aws.fs.path import S3Path

    el = service.get("raw_zone")
    assert isinstance(el.parent, S3Path)        # navigation leaves the wrapper
    assert isinstance(el / "sub/x.parquet", S3Path)


def test_non_s3_scheme_raises(service, store):
    from databricks.sdk.service.catalog import ExternalLocationInfo

    store["az"] = ExternalLocationInfo(
        name="az", url="abfss://c@acct.dfs.core.windows.net/x", credential_name="az-cred",
    )
    with pytest.raises(NotImplementedError):
        service.get("az").path


def test_is_hashable(service):
    el = service.get("raw_zone")
    assert hash(el) is not None
    assert el in {el}


def test_explore_url_and_clickable_repr(service):
    el = service["raw_zone"]
    assert str(el.explore_url) == "https://dbc-x.cloud.databricks.com/explore/locations/raw_zone"
    assert repr(el) == f"ExternalLocation({el.explore_url!r})"
    assert el._repr_html_().startswith('<a href="https://dbc-x.cloud.databricks.com/explore/locations/raw_zone"')


def test_instance_update_refreshes_cache(service):
    el = service.get("raw_zone")
    el.update(comment="patched")
    assert el.comment == "patched"


def test_refresh_refetches(service, store):
    el = service.get("raw_zone")
    store["raw_zone"] = ExternalLocationInfo(
        name="raw_zone", url="s3://my-bucket/raw2/", credential_name="prod-cred",
    )
    assert el.refresh().url == "s3://my-bucket/raw2/"


def test_from_url_bare_name_returns_location(service):
    el = ExternalLocation.from_url("dbfs+extloc:///raw_zone", service=service)
    assert isinstance(el, ExternalLocation)
    assert el.name == "raw_zone"
    assert el.url == "s3://my-bucket/raw/"


def test_from_url_subpath_returns_inner_storage_path(service):
    from yggdrasil.aws.fs.path import S3Path

    child = ExternalLocation.from_url(
        "dbfs+extloc:///raw_zone/sub/f.parquet", service=service,
    )
    assert isinstance(child, S3Path)            # left the wrapper
    assert child.bucket == "my-bucket" and child.key == "raw/sub/f.parquet"


def test_databrickspath_posix_prefix_dispatches_to_location(service):
    # ``/ExternalLocations/<name>`` POSIX form coerces to the
    # ``dbfs+extloc://`` scheme and dispatches to ExternalLocation.
    from yggdrasil.databricks.path import DatabricksPath

    el = DatabricksPath.from_("/ExternalLocations/raw_zone", service=service)
    assert isinstance(el, ExternalLocation)
    assert el.name == "raw_zone"


def test_s3_url_at_location_root_resolves_to_location(service):
    from yggdrasil.databricks.path import DatabricksPath

    el = DatabricksPath.from_("s3://my-bucket/raw/", service=service)
    assert isinstance(el, ExternalLocation)
    assert el.name == "raw_zone"


def test_s3_url_child_resolves_to_inner_storage_path(service):
    from yggdrasil.aws.fs.path import S3Path
    from yggdrasil.databricks.path import DatabricksPath

    child = DatabricksPath.from_("s3://my-bucket/raw/sub/f.parquet", service=service)
    assert isinstance(child, S3Path)            # left the wrapper
    assert child.bucket == "my-bucket" and child.key == "raw/sub/f.parquet"


def test_s3_url_without_matching_location_falls_back_to_plain_s3(service):
    from yggdrasil.aws.fs.path import S3Path
    from yggdrasil.databricks.path import DatabricksPath

    p = DatabricksPath.from_("s3://unregistered-bucket/x.csv", service=service)
    assert type(p) is S3Path                    # not a Databricks surface


def test_find_url_longest_prefix_wins(service, store):
    from databricks.sdk.service.catalog import ExternalLocationInfo

    store["deep"] = ExternalLocationInfo(
        name="deep", url="s3://my-bucket/raw/deep/", credential_name="prod-cred",
    )
    # A child under the nested location resolves to that (more specific) one.
    assert service.find_url("s3://my-bucket/raw/deep/x.csv").name == "deep"
    # A child only under the outer one resolves to it.
    assert service.find_url("s3://my-bucket/raw/other.csv").name == "raw_zone"
    # Nothing covers a different bucket.
    assert service.find_url("s3://elsewhere/x") is None
