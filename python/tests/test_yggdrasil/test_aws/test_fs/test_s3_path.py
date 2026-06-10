"""Behavior tests for the pure-HTTP :class:`yggdrasil.aws.fs.path.S3Path`.

S3Path no longer drives boto3 — it redirects every backend op to its
:class:`S3Bucket`, which signs requests (SigV4, proven in ``test_sigv4``) and
sends them over an :class:`S3HttpClient`. Here we prime the bucket with a
fake-backed client (``_fake_s3``) and exercise the genuine RemotePath contract
end-to-end: stat, full + ranged reads, single-PUT vs multipart writes, delete,
recursive prune, and listing (+ cache).
"""
from __future__ import annotations

import pytest

from yggdrasil.io.io_stats import IOKind
from yggdrasil.aws.fs.path import S3Bucket

from tests.test_yggdrasil.test_aws._fake_s3 import FakeS3, wire_s3_path, reset_s3_singletons


@pytest.fixture(autouse=True)
def _isolate_singletons():
    reset_s3_singletons()
    yield
    reset_s3_singletons()


@pytest.fixture
def fake():
    return FakeS3()


def _path(fake, key="dir/obj.bin"):
    return wire_s3_path(fake, f"s3://bkt/{key}")


# --- bucket plumbing -------------------------------------------------------
class TestS3Bucket:
    def test_path_resolves_shared_long_lived_bucket(self, fake):
        p = _path(fake, "a/x")
        assert p.s3_bucket is S3Bucket(bucket="bkt")
        assert p.s3_bucket.name == "bkt"

    def test_bucket_ttl_is_longer_than_path_ttl(self):
        from yggdrasil.path.remote_path import _STAT_CACHE_TTL

        assert S3Bucket._SINGLETON_TTL > _STAT_CACHE_TTL

    def test_sibling_paths_reuse_bucket(self, fake):
        p = _path(fake, "a/x")
        sib = p.parent / "y"
        assert sib.s3_bucket is p.s3_bucket


# --- stat ------------------------------------------------------------------
class TestStat:
    def test_file_stat(self, fake):
        p = _path(fake)
        p.write_bytes(b"hello")
        st = p.stat()
        assert st.kind == IOKind.FILE and st.size == 5

    def test_missing_is_missing(self, fake):
        assert _path(fake, "nope").stat().kind == IOKind.MISSING

    def test_prefix_is_directory(self, fake):
        _path(fake, "d/child").write_bytes(b"z")
        assert wire_s3_path(fake, "s3://bkt/d").stat().kind == IOKind.DIRECTORY

    def test_bucket_root_is_directory(self, fake):
        assert wire_s3_path(fake, "s3://bkt/").stat().kind == IOKind.DIRECTORY


# --- read ------------------------------------------------------------------
class TestRead:
    def test_whole_object(self, fake):
        p = _path(fake)
        p.write_bytes(b"0123456789")
        assert p.read_bytes() == b"0123456789"

    def test_ranged_read(self, fake):
        p = _path(fake)
        p.write_bytes(b"0123456789")
        assert bytes(p.read_mv(4, 2)) == b"2345"

    def test_missing_read_raises(self, fake):
        with pytest.raises(FileNotFoundError):
            _path(fake, "ghost").read_bytes()


# --- write -----------------------------------------------------------------
class TestWrite:
    def test_small_upload_single_put(self, fake):
        p = _path(fake)
        p._upload(b"small payload")
        assert fake.objects["dir/obj.bin"] == b"small payload"
        assert fake.calls.get("put") == 1
        assert fake.calls.get("create_multipart", 0) == 0

    def test_large_upload_uses_multipart(self, fake, monkeypatch):
        p = _path(fake)
        monkeypatch.setattr(type(p), "MULTIPART_THRESHOLD", 1024)
        monkeypatch.setattr(type(p), "MULTIPART_CHUNKSIZE", 256)
        payload = b"x" * 4096
        p._upload(payload)
        assert fake.objects["dir/obj.bin"] == payload
        assert fake.calls.get("create_multipart") == 1
        assert fake.calls.get("upload_part") == 16  # 4096 / 256
        assert fake.calls.get("complete_multipart") == 1
        assert fake.calls.get("put", 0) == 0

    def test_round_trip(self, fake):
        p = _path(fake)
        p.write_bytes(b"round trip")
        assert p.read_bytes() == b"round trip"


# --- delete ----------------------------------------------------------------
class TestDelete:
    def test_delete_file(self, fake):
        p = _path(fake)
        p.write_bytes(b"z")
        p.remove(missing_ok=False)
        assert "dir/obj.bin" not in fake.objects

    def test_recursive_dir_delete(self, fake):
        for k in ["d/1", "d/2", "d/sub/3"]:
            _path(fake, k).write_bytes(b"z")
        wire_s3_path(fake, "s3://bkt/d").remove(recursive=True, missing_ok=True)
        assert not any(k.startswith("d/") for k in fake.objects)


# --- not folder-oriented ---------------------------------------------------
class TestNotFolderOriented:
    """S3 is an object store, not a directory tree — there are no folders to
    create, so mkdir is inert and writing a deep key never materializes
    intermediate prefixes or probes for them."""

    def test_mkdir_issues_no_requests(self, fake):
        d = wire_s3_path(fake, "s3://bkt/a/b/c/")
        d.mkdir(parents=True, exist_ok=True)
        assert fake.calls == {}  # nothing created, nothing probed

    def test_deep_write_creates_no_intermediate_prefixes(self, fake):
        p = _path(fake, "a/b/c/d.txt")
        p.write_bytes(b"hello", overwrite=True)
        # Exactly one PUT for the object — no list/head/mkdir for the "folders".
        assert fake.calls == {"put": 1}
        assert set(fake.objects) == {"a/b/c/d.txt"}


# --- listing ---------------------------------------------------------------
class TestListing:
    def test_shallow_lists_children_and_prefixes(self, fake):
        for k in ["p/a", "p/b", "p/sub/c"]:
            _path(fake, k).write_bytes(b"z")
        names = sorted(child.key for child in wire_s3_path(fake, "s3://bkt/p")._ls())
        assert names == ["p/a", "p/b", "p/sub/"]

    def test_recursive_lists_all(self, fake):
        for k in ["p/a", "p/sub/c", "p/sub/deep/d"]:
            _path(fake, k).write_bytes(b"z")
        names = sorted(child.key for child in wire_s3_path(fake, "s3://bkt/p")._ls(recursive=True))
        assert names == ["p/a", "p/sub/c", "p/sub/deep/d"]

    def test_listing_is_never_cached(self, fake):
        # No listing cache — every ``_ls`` is a fresh ListObjectsV2 so
        # concurrent / external mutations show up immediately.
        for k in ["p/a", "p/b"]:
            _path(fake, k).write_bytes(b"z")
        d = wire_s3_path(fake, "s3://bkt/p")
        list(d._ls())
        before = fake.calls.get("list", 0)
        list(d._ls())  # second walk re-lists
        assert fake.calls.get("list", 0) == before + 1
