"""Mock-driven tests for :class:`VolumePath`."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.fs import VolumePath
from yggdrasil.io.io_stats import IOKind


class NotFound(Exception):
    pass


class InternalError(Exception):
    pass


class PermissionDenied(Exception):
    pass


@pytest.fixture(autouse=True)
def reset_remote_stat_cache():
    from yggdrasil.io.path.remote_path import RemotePath
    RemotePath._STAT_CACHE.clear()
    yield
    RemotePath._STAT_CACHE.clear()


@pytest.fixture
def workspace():
    return MagicMock()


def _file_meta(size: int, mtime_ms: int = 0):
    return SimpleNamespace(
        content_length=size,
        modification_time=mtime_ms,
    )


class TestConstruction:

    def test_legacy_posix_string(self, workspace) -> None:
        p = VolumePath(
            "/Volumes/cat/sch/vol/data.parquet", workspace=workspace,
        )
        assert p.full_path() == "/Volumes/cat/sch/vol/data.parquet"
        assert p.api_path == "/Volumes/cat/sch/vol/data.parquet"

    def test_url_form(self, workspace) -> None:
        p = VolumePath("dbfs+volume:///cat/sch/vol/x", workspace=workspace)
        assert p.full_path() == "/Volumes/cat/sch/vol/x"


class TestStat:

    def test_existing_file(self, workspace) -> None:
        workspace.files.get_metadata.return_value = _file_meta(42)
        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        s = p._stat_uncached()
        assert s.kind is IOKind.FILE
        assert s.size == 42

    def test_directory_fallback(self, workspace) -> None:
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.return_value = SimpleNamespace()
        p = VolumePath("/Volumes/c/s/v/dir", workspace=workspace)
        assert p._stat_uncached().kind is IOKind.DIRECTORY

    def test_missing(self, workspace) -> None:
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()
        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        assert p._stat_uncached().kind is IOKind.MISSING


class TestRead:

    def test_full_object_read(self, workspace) -> None:
        workspace.files.get_metadata.return_value = _file_meta(5)
        body = SimpleNamespace(read=lambda: b"hello")
        workspace.files.download.return_value = SimpleNamespace(contents=body)

        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        assert p.read_bytes() == b"hello"
        workspace.files.download.assert_called_once_with("/Volumes/c/s/v/x")

    def test_missing_raises(self, workspace) -> None:
        workspace.files.get_metadata.return_value = _file_meta(5)
        workspace.files.download.side_effect = NotFound()
        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        with pytest.raises(FileNotFoundError):
            p.read_bytes()


class TestWrite:

    def test_overwrite(self, workspace) -> None:
        # Initial probe: missing.
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()
        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        p.write_bytes(b"abcdef")
        kwargs = workspace.files.upload.call_args.kwargs
        assert kwargs["file_path"] == "/Volumes/c/s/v/x"
        assert kwargs["overwrite"] is True
        # ``contents`` is a stdlib ``BytesIO`` — read it for the
        # payload we sent.
        assert kwargs["contents"].getvalue() == b"abcdef"

    def test_pwrite_does_rmw(self, workspace) -> None:
        workspace.files.get_metadata.return_value = _file_meta(5)
        body = SimpleNamespace(read=lambda: b"abcde")
        workspace.files.download.return_value = SimpleNamespace(contents=body)
        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        p.pwrite(b"XX", 1)
        sent = workspace.files.upload.call_args.kwargs["contents"].getvalue()
        assert sent == b"aXXde"


class TestMutators:

    def test_unlink(self, workspace) -> None:
        workspace.files.get_metadata.return_value = _file_meta(0)
        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        p.unlink()
        workspace.files.delete.assert_called_once_with("/Volumes/c/s/v/x")

    def test_mkdir(self, workspace) -> None:
        p = VolumePath("/Volumes/c/s/v/folder", workspace=workspace)
        p.mkdir()
        workspace.files.create_directory.assert_called_once_with(
            "/Volumes/c/s/v/folder",
        )


class TestListing:

    def test_iterdir_preserves_catalog(self, workspace) -> None:
        # ``list_directory_contents`` returns canonical
        # ``/Volumes/<cat>/<sch>/<vol>/...`` paths; the catalog
        # segment must round-trip through child construction.
        workspace.files.list_directory_contents.return_value = [
            SimpleNamespace(path="/Volumes/trading/sch/vol/folder/a.bin",
                            is_directory=False),
            SimpleNamespace(path="/Volumes/trading/sch/vol/folder/sub",
                            is_directory=True),
        ]
        p = VolumePath("/Volumes/trading/sch/vol/folder", workspace=workspace)
        children = list(p.iterdir())
        assert [c.full_path() for c in children] == [
            "/Volumes/trading/sch/vol/folder/a.bin",
            "/Volumes/trading/sch/vol/folder/sub",
        ]
        assert all(isinstance(c, VolumePath) for c in children)


class TestStagingPath:

    def test_default_layout(self, workspace) -> None:
        p = VolumePath.staging_path(
            catalog_name="cat",
            schema_name="sch",
            resource_name="tbl",
            workspace=workspace,
        )
        full = p.full_path()
        assert full.startswith("/Volumes/cat/sch/tmp/.sql/cat/sch/tbl/part-")
        assert full.endswith(".parquet")
        assert p.temporary is True
        assert p.workspace is workspace

    def test_temporary_false(self, workspace) -> None:
        p = VolumePath.staging_path(
            catalog_name="cat",
            schema_name="sch",
            temporary=False,
            workspace=workspace,
        )
        assert p.temporary is False
        assert "/cat/sch/tmp/.sql/cat/sch/default/" in p.full_path()

    def test_unique_per_call(self, workspace) -> None:
        a = VolumePath.staging_path(
            catalog_name="c", schema_name="s", workspace=workspace,
        )
        b = VolumePath.staging_path(
            catalog_name="c", schema_name="s", workspace=workspace,
        )
        assert a.full_path() != b.full_path()

    def test_client_aggregator(self, workspace) -> None:
        client = MagicMock()
        client.workspace_client.return_value = workspace
        p = VolumePath.staging_path(
            catalog_name="c", schema_name="s", client=client,
        )
        assert p.workspace is workspace

    def test_segments_are_sanitized(self, workspace) -> None:
        p = VolumePath.staging_path(
            catalog_name="`cat`",
            schema_name="  sch  ",
            resource_name="a/b",
            workspace=workspace,
        )
        full = p.full_path()
        assert "/cat/sch/tmp/.sql/cat/sch/a_b/" in full


class TestVolumeAutoCreate:
    """``_call_ensuring_parents`` should walk the cheap path first
    (``create_directory`` on the parent) and only blind-create the
    catalog / schema / managed volume when that fails NotFound."""

    def test_only_subdir_missing_skips_volume_create(self, workspace) -> None:
        # Upload fails because parent dir missing; one parent
        # ``create_directory`` is enough — no catalog/schema/volume
        # creates should happen.
        uploads = [NotFound("does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.upload.side_effect = upload
        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/file.bin", workspace=workspace,
        )
        p.write_bytes(b"payload")

        workspace.files.create_directory.assert_called_once_with(
            "/Volumes/cat/sch/vol/sub",
        )
        workspace.catalogs.create.assert_not_called()
        workspace.schemas.create.assert_not_called()
        workspace.volumes.create.assert_not_called()
        assert workspace.files.upload.call_count == 2

    def test_volume_missing_only_creates_volume(self, workspace) -> None:
        # Common case: catalog + schema already exist, only the volume
        # is missing. After a NotFound on upload + create_directory,
        # we should try ``volumes.create`` first and stop there.
        uploads = [NotFound("Volume does not exist"), None]
        create_dirs = [NotFound("does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def create_directory(_path):
            r = create_dirs.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.upload.side_effect = upload
        workspace.files.create_directory.side_effect = create_directory

        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/file.bin", workspace=workspace,
        )
        p.write_bytes(b"payload")

        workspace.catalogs.get.assert_not_called()
        workspace.schemas.get.assert_not_called()
        workspace.volumes.read.assert_not_called()

        # Volume created first; catalog/schema untouched because volume
        # create succeeded.
        workspace.catalogs.create.assert_not_called()
        workspace.schemas.create.assert_not_called()
        vol_kwargs = workspace.volumes.create.call_args.kwargs
        assert vol_kwargs["catalog_name"] == "cat"
        assert vol_kwargs["schema_name"] == "sch"
        assert vol_kwargs["name"] == "vol"
        vt = vol_kwargs["volume_type"]
        assert getattr(vt, "name", str(vt)).upper() == "MANAGED"

    def test_schema_missing_creates_schema_then_volume(
        self, workspace,
    ) -> None:
        # Schema is also missing — first ``volumes.create`` fails
        # NotFound, then ``schemas.create`` succeeds, then volume create
        # is retried. Catalog should not be touched.
        uploads = [NotFound("Volume does not exist"), None]
        create_dirs = [NotFound("does not exist"), None]
        volume_creates = [NotFound("Schema does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def create_directory(_path):
            r = create_dirs.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def volumes_create(**_kwargs):
            r = volume_creates.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.upload.side_effect = upload
        workspace.files.create_directory.side_effect = create_directory
        workspace.volumes.create.side_effect = volumes_create

        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/file.bin", workspace=workspace,
        )
        p.write_bytes(b"payload")

        workspace.catalogs.create.assert_not_called()
        workspace.schemas.create.assert_called_once_with(
            name="sch", catalog_name="cat",
        )
        assert workspace.volumes.create.call_count == 2

    def test_catalog_missing_creates_full_chain(self, workspace) -> None:
        # Both schema and catalog are missing — volume.create then
        # schema.create both fail NotFound, falling through to catalog
        # → schema → volume creation.
        uploads = [NotFound("Volume does not exist"), None]
        create_dirs = [NotFound("does not exist"), None]
        volume_creates = [NotFound("Schema does not exist"), None]
        schema_creates = [NotFound("Catalog does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def create_directory(_path):
            r = create_dirs.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def volumes_create(**_kwargs):
            r = volume_creates.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def schemas_create(**_kwargs):
            r = schema_creates.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.upload.side_effect = upload
        workspace.files.create_directory.side_effect = create_directory
        workspace.volumes.create.side_effect = volumes_create
        workspace.schemas.create.side_effect = schemas_create

        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/file.bin", workspace=workspace,
        )
        p.write_bytes(b"payload")

        workspace.catalogs.create.assert_called_once_with(name="cat")
        assert workspace.schemas.create.call_count == 2
        assert workspace.volumes.create.call_count == 2

    def test_already_exists_swallowed(self, workspace) -> None:
        # Volume create races with another caller — ``AlreadyExists``
        # is treated as success, no retry storm.
        uploads = [NotFound("Volume does not exist"), None]
        create_dirs = [NotFound("does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def create_directory(_path):
            r = create_dirs.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        class AlreadyExists(Exception):
            pass

        workspace.files.upload.side_effect = upload
        workspace.files.create_directory.side_effect = create_directory
        workspace.volumes.create.side_effect = AlreadyExists()

        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/file.bin", workspace=workspace,
        )
        p.write_bytes(b"payload")  # should not raise

        # Volume.create raised AlreadyExists → no schema/catalog touch.
        workspace.catalogs.create.assert_not_called()
        workspace.schemas.create.assert_not_called()

    def test_propagates_when_not_a_volume_path(self, workspace) -> None:
        # Path too shallow to address a volume — auto-create can't help,
        # so the original error must surface.
        workspace.files.upload.side_effect = NotFound("does not exist")
        p = VolumePath("/Volumes/onlycat", workspace=workspace)
        with pytest.raises(NotFound):
            p.write_bytes(b"x")
        workspace.volumes.create.assert_not_called()


class TestRetryPolicy:

    @pytest.fixture
    def sleeps(self):
        recorded: list[float] = []
        return recorded, recorded.append

    def test_internal_error_retries(self, workspace, sleeps) -> None:
        recorded, spy = sleeps
        attempts = [InternalError(), InternalError(), _file_meta(3)]

        def get_metadata(path):
            r = attempts.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.get_metadata.side_effect = get_metadata
        p = VolumePath(
            "/Volumes/c/s/v/x", workspace=workspace, retry_sleep=spy,
        )
        assert p.size == 3
        assert recorded == [1.0, 2.0]

    def test_permission_retries_once(self, workspace, sleeps) -> None:
        recorded, spy = sleeps
        attempts = [PermissionDenied(), _file_meta(2)]

        def get_metadata(path):
            r = attempts.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.get_metadata.side_effect = get_metadata
        p = VolumePath(
            "/Volumes/c/s/v/x", workspace=workspace, retry_sleep=spy,
        )
        assert p.size == 2
        assert recorded == [1.0]


# ---------------------------------------------------------------------------
# Native S3 storage fast path — storage_location + temporary_credentials + s3_path
# ---------------------------------------------------------------------------


def _volume_info(
    *,
    catalog: str = "cat",
    schema: str = "sch",
    name: str = "vol",
    volume_id: str = "volume-uuid-0001",
    storage_location: str = "s3://my-bucket/__unitystorage/cat/sch/vol",
):
    return SimpleNamespace(
        catalog_name=catalog,
        schema_name=schema,
        name=name,
        volume_id=volume_id,
        volume_type="MANAGED",
        storage_location=storage_location,
        full_name=f"{catalog}.{schema}.{name}",
        access_point=None,
    )


def _aws_creds_response(
    *,
    access_key_id: str = "AKIA-test",
    secret_access_key: str = "secret-test",
    session_token: str = "session-test",
):
    import datetime as _dt
    return SimpleNamespace(
        aws_temp_credentials=SimpleNamespace(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            access_point=None,
        ),
        expiration_time=_dt.datetime(2030, 1, 1, tzinfo=_dt.timezone.utc),
        url=None,
        azure_aad=None,
        azure_user_delegation_sas=None,
        gcp_oauth_token=None,
        r2_temp_credentials=None,
    )


class TestVolumeInfoCaching:

    def test_volume_info_caches_after_first_read(self, workspace) -> None:
        workspace.volumes.read.return_value = _volume_info()
        p = VolumePath("/Volumes/cat/sch/vol/x", workspace=workspace)
        first = p.volume_info()
        second = p.volume_info()
        assert first is second
        workspace.volumes.read.assert_called_once_with("cat.sch.vol")

    def test_volume_info_refresh_forces_reread(self, workspace) -> None:
        workspace.volumes.read.side_effect = [_volume_info(), _volume_info()]
        p = VolumePath("/Volumes/cat/sch/vol/x", workspace=workspace)
        p.volume_info()
        p.volume_info(refresh=True)
        assert workspace.volumes.read.call_count == 2

    def test_storage_location_resolves_from_volume_info(self, workspace) -> None:
        workspace.volumes.read.return_value = _volume_info(
            storage_location="s3://bkt/__unitystorage/c/s/v",
        )
        p = VolumePath("/Volumes/c/s/v/sub/y.parquet", workspace=workspace)
        assert p.storage_location() == "s3://bkt/__unitystorage/c/s/v"

    def test_storage_location_caches_independently_of_volume_info(
        self, workspace,
    ) -> None:
        workspace.volumes.read.return_value = _volume_info()
        p = VolumePath("/Volumes/cat/sch/vol/x", workspace=workspace)
        p.storage_location()
        p.storage_location()
        # One read call drives both — the value is snapshotted onto
        # ``_storage_location`` and the second call returns the cached
        # string without re-touching ``VolumeInfo``.
        workspace.volumes.read.assert_called_once()

    def test_storage_location_missing_raises(self, workspace) -> None:
        workspace.volumes.read.return_value = _volume_info(storage_location=None)
        p = VolumePath("/Volumes/cat/sch/vol/x", workspace=workspace)
        with pytest.raises(ValueError, match="storage_location"):
            p.storage_location()


class TestTemporaryCredentials:

    def test_vends_via_volume_id(self, workspace) -> None:
        workspace.volumes.read.return_value = _volume_info(volume_id="vid-42")
        gen = workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/cat/sch/vol/x", workspace=workspace)
        resp = p.temporary_credentials(operation="read")
        assert resp.aws_temp_credentials.access_key_id == "AKIA-test"

        # Call kwargs must include the volume_id from VolumeInfo plus
        # the SDK enum (READ_VOLUME for read-only modes).
        from databricks.sdk.service.catalog import VolumeOperation
        gen.assert_called_once()
        kwargs = gen.call_args.kwargs
        assert kwargs["volume_id"] == "vid-42"
        assert kwargs["operation"] is VolumeOperation.READ_VOLUME

    def test_write_operation_maps_to_write_volume(self, workspace) -> None:
        workspace.volumes.read.return_value = _volume_info()
        gen = workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        gen.return_value = _aws_creds_response()
        p = VolumePath("/Volumes/cat/sch/vol/x", workspace=workspace)
        p.temporary_credentials(operation="overwrite")

        from databricks.sdk.service.catalog import VolumeOperation
        assert gen.call_args.kwargs["operation"] is VolumeOperation.WRITE_VOLUME

    def test_missing_volume_id_raises(self, workspace) -> None:
        workspace.volumes.read.return_value = _volume_info(volume_id=None)
        p = VolumePath("/Volumes/cat/sch/vol/x", workspace=workspace)
        with pytest.raises(ValueError, match="volume_id"):
            p.temporary_credentials()


class TestAWSAndS3Path:

    def test_aws_returns_refreshable_client(self, workspace) -> None:
        workspace.volumes.read.return_value = _volume_info()
        gen = workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/cat/sch/vol/x", workspace=workspace)
        aws = p.aws(operation="read", region="us-east-1")

        # Snapshot creds bound to the AWSConfig — these came from the
        # initial refresher call. The refresher itself is wired into
        # the config so botocore can re-invoke it on token expiry.
        assert aws.config.access_key_id == "AKIA-test"
        assert aws.config.region == "us-east-1"
        assert aws.config.has_refresher()

    def test_aws_refresher_rotates_credentials(self, workspace) -> None:
        # The refresher is the same callable on every cycle — botocore
        # would re-invoke it via the config. We invoke it manually here
        # to prove fresh creds flow through on every call.
        workspace.volumes.read.return_value = _volume_info()
        gen = workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        gen.side_effect = [
            _aws_creds_response(access_key_id="AKIA-1"),
            _aws_creds_response(access_key_id="AKIA-2"),
        ]
        p = VolumePath("/Volumes/cat/sch/vol/x", workspace=workspace)
        aws = p.aws(operation="read")
        # First call seeded the config; invoke the refresher once more.
        refreshed = aws.config.refresher()
        assert refreshed.access_key_id == "AKIA-2"

    def test_aws_raises_for_non_s3_volume(self, workspace) -> None:
        workspace.volumes.read.return_value = _volume_info()
        gen = workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        # Azure-style response — no ``aws_temp_credentials``.
        gen.return_value = SimpleNamespace(
            aws_temp_credentials=None,
            azure_aad=SimpleNamespace(aad_token="token"),
            azure_user_delegation_sas=None,
            gcp_oauth_token=None,
            r2_temp_credentials=None,
            expiration_time=None,
            url=None,
        )
        p = VolumePath("/Volumes/cat/sch/vol/x", workspace=workspace)
        with pytest.raises(RuntimeError, match="aws_temp_credentials"):
            p.aws(operation="read")

    def test_s3_path_joins_storage_with_subvolume_tail(self, workspace) -> None:
        workspace.volumes.read.return_value = _volume_info(
            storage_location="s3://my-bucket/__unitystorage/cat/sch/vol",
        )
        gen = workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        gen.return_value = _aws_creds_response()

        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/dir/file.parquet", workspace=workspace,
        )
        s3 = p.s3_path(operation="read", region="us-east-1")

        from yggdrasil.aws.fs.path import S3Path
        assert isinstance(s3, S3Path)
        assert s3.full_path() == (
            "s3://my-bucket/__unitystorage/cat/sch/vol/sub/dir/file.parquet"
        )

    def test_s3_path_at_volume_root(self, workspace) -> None:
        workspace.volumes.read.return_value = _volume_info(
            storage_location="s3://b/__unitystorage/c/s/v/",
        )
        gen = workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/c/s/v", workspace=workspace)
        s3 = p.s3_path()
        # No sub-volume tail to append; storage root drives the URL.
        # ``S3Path`` strips the trailing slash on canonical rendering
        # (the canonical S3 key has no trailing slash).
        assert s3.full_path() == "s3://b/__unitystorage/c/s/v"
        assert s3.bucket == "b"
        assert s3.key == "__unitystorage/c/s/v"
