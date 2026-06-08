"""S3Path tests using a stub backend backed by a plain dict."""
from __future__ import annotations

import time
from typing import Any, ClassVar, Iterator

import pyarrow as pa
import pytest

from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.enums import Mode
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.path.remote_path import RemotePath
from yggdrasil.url import URL

# Try to import the real S3Path so we can verify construction/repr
# against the actual class when boto3 is available.
try:
    from yggdrasil.aws.fs.path import S3Path as _RealS3Path
except Exception:
    _RealS3Path = None


# ---------------------------------------------------------------------------
# Stub backend -- dict-backed S3-like RemotePath for hermetic testing
# ---------------------------------------------------------------------------


class _StubS3Path(RemotePath):
    _STORAGE: ClassVar[dict[str, bytes]] = {}
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=300.0,
        max_size=10_000,
    )

    _ACCEPTED_SCHEMES: ClassVar[frozenset[str]] = frozenset({"s3", "s3a", "s3n"})

    def __init__(
        self,
        data: Any = None,
        *,
        url: URL | None = None,
        singleton_ttl: Any = ...,
        **kwargs: Any,
    ) -> None:
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return
        if url is None and isinstance(data, str):
            url = URL.from_(data)
        if url is None and isinstance(data, URL):
            url = data
            data = None
        # Normalize s3a/s3n to s3
        if url is not None and url.scheme in self._ACCEPTED_SCHEMES and url.scheme != "s3":
            url = URL(scheme="s3", host=url.host, path=url.path,
                      port=url.port, query=url.query, fragment=url.fragment)
        super().__init__(data=data, url=url, singleton_ttl=False, **kwargs)
        self._initialized = True

    @property
    def bucket(self) -> str:
        host = self.url.host
        if not host:
            raise ValueError(f"S3 path has no bucket: {self.url!r}")
        return host

    @property
    def key(self) -> str:
        path = self.url.path or ""
        return path.lstrip("/")

    def full_path(self) -> str:
        key = self.key
        return f"s3://{self.bucket}/{key}" if key else f"s3://{self.bucket}/"

    # -- backend primitives ------------------------------------------------

    def _stat_uncached(self) -> IOStats:
        storage_key = f"{self.bucket}/{self.key}"
        if storage_key in self._STORAGE:
            return IOStats(
                size=len(self._STORAGE[storage_key]),
                kind=IOKind.FILE,
                mtime=time.time(),
            )
        # Check if any keys start with this prefix for directory semantics.
        prefix = storage_key.rstrip("/") + "/"
        if any(k.startswith(prefix) for k in self._STORAGE):
            return IOStats(size=0, kind=IOKind.DIRECTORY, mtime=0.0)
        return IOStats(size=0, kind=IOKind.MISSING, mtime=0.0)

    def _read_mv(self, n: int, pos: int) -> memoryview:
        storage_key = f"{self.bucket}/{self.key}"
        data = self._STORAGE.get(storage_key)
        if data is None:
            raise FileNotFoundError(self.full_path())
        if n < 0:
            return memoryview(data[pos:])
        return memoryview(data[pos : pos + n])

    def _upload(self, content: bytes) -> int:
        storage_key = f"{self.bucket}/{self.key}"
        self._STORAGE[storage_key] = content
        self._persist_stat_cache(
            IOStats(
                size=len(content),
                kind=IOKind.FILE,
                mtime=time.time(),
            )
        )
        return len(content)

    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator[_StubS3Path]:
        prefix = f"{self.bucket}/{self.key}"
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        seen: set[str] = set()
        for k in sorted(self._STORAGE):
            if not k.startswith(prefix):
                continue
            remainder = k[len(prefix):]
            if not recursive:
                top = remainder.split("/")[0]
                if top in seen:
                    continue
                seen.add(top)
                child_key = prefix + top
            else:
                child_key = k
            # Reconstruct as s3:// URL
            bucket_end = child_key.index("/")
            child_bucket = child_key[:bucket_end]
            child_obj_key = child_key[bucket_end + 1:]
            child_url = URL(scheme="s3", host=child_bucket, path="/" + child_obj_key)
            yield _StubS3Path(url=child_url, singleton_ttl=False)

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        # S3 has no directory concept -- no-op.
        pass

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        storage_key = f"{self.bucket}/{self.key}"
        if storage_key in self._STORAGE:
            del self._STORAGE[storage_key]
            self.invalidate_singleton()
        elif not missing_ok:
            raise FileNotFoundError(self.full_path())

    def _remove_dir(
        self,
        recursive: bool,
        missing_ok: bool,
        wait: WaitingConfig,
    ) -> None:
        prefix = f"{self.bucket}/{self.key}"
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        to_delete = [k for k in self._STORAGE if k.startswith(prefix)]
        if not to_delete and not missing_ok:
            raise FileNotFoundError(self.full_path())
        for k in to_delete:
            del self._STORAGE[k]
        self.invalidate_singleton()

    def _from_url(self, url: URL) -> _StubS3Path:
        return _StubS3Path(url=url, singleton_ttl=False)

    def __repr__(self) -> str:
        return f"S3Path({self.full_path()!r})"


# Post-class scheme assignment to match the _StubRemotePath pattern.
_StubS3Path.scheme = "s3"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_stub_state():
    _StubS3Path._STORAGE.clear()
    _StubS3Path._INSTANCES = ExpiringDict(
        default_ttl=300.0,
        max_size=10_000,
    )
    yield
    _StubS3Path._STORAGE.clear()
    _StubS3Path._INSTANCES = ExpiringDict(
        default_ttl=300.0,
        max_size=10_000,
    )


def _make(url: str = "s3://test-bucket/data.bin", **kwargs: Any) -> _StubS3Path:
    return _StubS3Path(url, singleton_ttl=False, **kwargs)


# ---------------------------------------------------------------------------
# TestS3PathConstruction
# ---------------------------------------------------------------------------


class TestS3PathConstruction:

    def test_from_s3_url_string(self) -> None:
        p = _make("s3://my-bucket/some/key.parquet")
        assert p.url.scheme == "s3"
        assert p.url.host == "my-bucket"
        assert p.url.path == "/some/key.parquet"

    def test_bucket_and_key_parsing_from_url(self) -> None:
        p = _make("s3://analytics-prod/warehouse/events/2024/data.parquet")
        assert p.bucket == "analytics-prod"
        assert p.key == "warehouse/events/2024/data.parquet"

    def test_repr_includes_s3_scheme(self) -> None:
        p = _make("s3://my-bucket/path/to/file.csv")
        r = repr(p)
        assert r.startswith("S3Path(")
        assert "s3://" in r
        assert "my-bucket" in r
        assert "path/to/file.csv" in r


# ---------------------------------------------------------------------------
# TestS3PathReadWrite
# ---------------------------------------------------------------------------


class TestS3PathReadWrite:

    def test_write_bytes_then_read_back(self) -> None:
        p = _make("s3://bucket/rw/file.bin")
        p.write_bytes(b"hello s3 world")
        assert p.read_bytes() == b"hello s3 world"

    def test_write_arrow_table_via_as_media_read_back(self) -> None:
        p = _make("s3://bucket/rw/table.arrow")
        leaf = p.as_media("arrow")
        table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        assert result.column("x").to_pylist() == [1, 2, 3]
        assert result.column("y").to_pylist() == ["a", "b", "c"]

    def test_size_matches_written_data(self) -> None:
        p = _make("s3://bucket/rw/sized.bin")
        payload = b"x" * 256
        p.write_bytes(payload)
        assert p.size == 256

    def test_overwrite_replaces_content(self) -> None:
        p = _make("s3://bucket/rw/overwrite.bin")
        p.write_bytes(b"first version")
        p.write_bytes(b"replaced", overwrite=True)
        assert p.read_bytes() == b"replaced"


# ---------------------------------------------------------------------------
# TestS3PathDirectory
# ---------------------------------------------------------------------------


class TestS3PathDirectory:

    def test_mkdir_is_noop(self) -> None:
        p = _make("s3://bucket/prefix/subdir")
        # Should not raise -- S3 mkdir is a no-op.
        p.mkdir()
        # The path should not exist as a file in storage.
        assert "bucket/prefix/subdir" not in _StubS3Path._STORAGE

    def test_iterdir_lists_keys_with_common_prefix(self) -> None:
        _StubS3Path._STORAGE["bucket/data/file1.csv"] = b"row1"
        _StubS3Path._STORAGE["bucket/data/file2.csv"] = b"row2"
        _StubS3Path._STORAGE["bucket/data/nested/deep.csv"] = b"deep"
        _StubS3Path._STORAGE["bucket/other/unrelated.txt"] = b"nope"
        d = _make("s3://bucket/data")
        children = sorted(c.key for c in d.iterdir())
        # Non-recursive: should see file1.csv, file2.csv, and nested (as prefix).
        assert children == ["data/file1.csv", "data/file2.csv", "data/nested"]

    def test_remove_deletes_object(self) -> None:
        _StubS3Path._STORAGE["bucket/doomed/target.bin"] = b"bye"
        p = _make("s3://bucket/doomed/target.bin")
        assert p.exists()
        p.remove()
        assert "bucket/doomed/target.bin" not in _StubS3Path._STORAGE


# ---------------------------------------------------------------------------
# TestS3PathSingleton
# ---------------------------------------------------------------------------


class TestS3PathSingleton:

    def test_same_url_returns_same_instance(self) -> None:
        a = _StubS3Path(
            url=URL(scheme="s3", host="bucket", path="/key.bin"),
            singleton_ttl=300,
        )
        b = _StubS3Path(
            url=URL(scheme="s3", host="bucket", path="/key.bin"),
            singleton_ttl=300,
        )
        assert a is b

    def test_different_url_returns_different_instance(self) -> None:
        a = _StubS3Path(
            url=URL(scheme="s3", host="bucket", path="/key_a.bin"),
            singleton_ttl=300,
        )
        b = _StubS3Path(
            url=URL(scheme="s3", host="bucket", path="/key_b.bin"),
            singleton_ttl=300,
        )
        assert a is not b


# ---------------------------------------------------------------------------
# TestS3MultipartUpload — large objects route through boto3 managed transfer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_RealS3Path is None, reason="S3Path not available")
class TestS3MultipartUpload:
    """A small upload is a single signed PUT; a large one streams as a
    pure-HTTP multipart upload (create / parts / complete)."""

    @staticmethod
    def _path(key="obj.bin"):
        from tests.test_yggdrasil.test_aws._fake_s3 import FakeS3, wire_s3_path, reset_s3_singletons

        reset_s3_singletons()
        fake = FakeS3()
        return wire_s3_path(fake, f"s3://bucket/{key}", bucket="bucket"), fake

    def test_small_upload_uses_put_object(self):
        p, fake = self._path()
        p._upload(b"small payload")
        assert fake.calls.get("put") == 1
        assert fake.calls.get("create_multipart", 0) == 0
        assert fake.objects["obj.bin"] == b"small payload"

    def test_large_upload_uses_managed_multipart(self, monkeypatch):
        p, fake = self._path()
        monkeypatch.setattr(type(p), "MULTIPART_THRESHOLD", 1024)
        monkeypatch.setattr(type(p), "MULTIPART_CHUNKSIZE", 256)
        p._upload(b"x" * 4096)
        assert fake.calls.get("create_multipart") == 1
        assert fake.calls.get("upload_part") == 16
        assert fake.calls.get("complete_multipart") == 1
        assert fake.calls.get("put", 0) == 0
        assert fake.objects["obj.bin"] == b"x" * 4096


# ---------------------------------------------------------------------------
# TestRangedProjection — shared RemotePath.arrow_random_access_file
# ---------------------------------------------------------------------------


class TestRangedProjection:
    """A ranged-capable backend (SUPPORTS_RANGED_RANDOM_ACCESS) projects
    Parquet columns through bounded reads, not a whole-object snapshot —
    the same mechanism VolumePath uses, exercised here over the stub."""

    def test_column_projection_uses_bounded_ranged_reads(self):
        import io as _io

        import pyarrow as pa
        import pyarrow.parquet as pq

        from yggdrasil.io.parquet_file import ParquetFile

        class _RangedStub(_StubS3Path):
            SUPPORTS_RANGED_RANDOM_ACCESS: ClassVar[bool] = True
            reads: ClassVar[list] = []

            def _read_mv(self, n: int, pos: int) -> memoryview:
                type(self).reads.append((n, pos))
                return super()._read_mv(n, pos)

        table = pa.table(
            {f"c{i}": pa.array(range(5000), type=pa.int64()) for i in range(8)}
        )
        sink = _io.BytesIO()
        pq.write_table(table, sink, row_group_size=500)
        _RangedStub._STORAGE["b/wide.parquet"] = sink.getvalue()
        _RangedStub.reads.clear()

        p = _RangedStub(url=URL.from_("s3://b/wide.parquet"), singleton_ttl=False)
        got = ParquetFile(holder=p, owns_holder=False).read_arrow_table(
            target=pa.schema([("c3", pa.int64())]),
        )
        assert got.column_names == ["c3"]
        assert got.num_rows == 5000
        assert got.column("c3").to_pylist()[:3] == [0, 1, 2]
        # Ranged: the reader seeked to footer + column chunks (several
        # bounded reads), not one whole-object snapshot.
        assert len(_RangedStub.reads) >= 2
        assert all(n != -1 for (n, _pos) in _RangedStub.reads)
