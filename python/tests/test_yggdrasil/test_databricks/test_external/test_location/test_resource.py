"""Resource: yggdrasil.databricks.external.location.resource.ExternalLocation."""
from __future__ import annotations

from databricks.sdk.service.catalog import ExternalLocationInfo


def test_metadata(service):
    el = service.get("raw_zone")
    assert el.url == "s3://my-bucket/raw/"
    assert el.read_only is False and el.owner == "data-eng"
    assert service.get("ro_zone").read_only is True


def test_lazy_info_single_fetch(service):
    el = service["raw_zone"]  # lazy, no fetch
    api = service.client.workspace_client.return_value.external_locations
    api.get.assert_not_called()
    _ = el.url  # triggers exactly one GET
    _ = el.comment  # cached
    api.get.assert_called_once()


def test_path_is_an_s3path(service):
    from yggdrasil.aws.fs.path import S3Path

    p = service.get("raw_zone").path
    assert isinstance(p, S3Path)
    assert p.bucket == "my-bucket" and p.key == "raw/"


def test_explore_url_and_clickable_repr(service):
    el = service["raw_zone"]
    assert str(el.explore_url) == "https://dbc-x.cloud.databricks.com/explore/location/raw_zone"
    assert repr(el) == f"ExternalLocation({el.explore_url!r})"
    assert el._repr_html_().startswith('<a href="https://dbc-x.cloud.databricks.com/explore/location/raw_zone"')


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
