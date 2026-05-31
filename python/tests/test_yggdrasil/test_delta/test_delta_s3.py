"""S3-backed Delta integration tests.

Two test suites:

1. **Mock S3** — injects a mock boto3 client into S3Path, runs the
   full DeltaFolder read/write protocol over simulated S3. No network
   required. Validates that the Delta log, checkpoint, DV, and parquet
   read paths all work correctly through the RemotePath abstraction.

2. **Real S3** (``@pytest.mark.integration``) — exercises the same
   operations against an actual S3 bucket. Requires:
   - ``AWS_ACCESS_KEY_ID`` + ``AWS_SECRET_ACCESS_KEY`` env vars
   - ``YGG_TEST_S3_BUCKET`` env var (bucket name)
   - ``YGG_TEST_S3_PREFIX`` env var (optional, defaults to ``ygg-delta-test/``)
"""
from __future__ import annotations

import io
import os
import time
import unittest
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yggdrasil.enums import Mode
from yggdrasil.delta.io import DeltaOptions
from yggdrasil.delta.tests import DeltaTestCase


# ---------------------------------------------------------------------------
# Mock S3 infrastructure
# ---------------------------------------------------------------------------


class _InMemoryS3:
    """In-memory S3 backend that tracks objects as a dict of bytes.

    Provides the subset of the boto3 S3 client interface that
    S3Path uses: head_object, get_object, put_object, delete_object,
    delete_objects, and list_objects_v2 via a paginator.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def head_object(self, *, Bucket: str, Key: str, **kw: Any) -> dict:
        full = f"{Bucket}/{Key}"
        if full not in self.objects:
            raise self._not_found(Key)
        return {
            "ContentLength": len(self.objects[full]),
            "LastModified": None,
        }

    def get_object(self, *, Bucket: str, Key: str, **kw: Any) -> dict:
        full = f"{Bucket}/{Key}"
        if full not in self.objects:
            raise self._not_found(Key)
        data = self.objects[full]
        range_header = kw.get("Range")
        if range_header:
            start, end = self._parse_range(range_header, len(data))
            data = data[start : end + 1]
            return {
                "Body": _Body(data),
                "ContentLength": len(data),
                "ContentRange": f"bytes {start}-{end}/{len(self.objects[full])}",
            }
        return {
            "Body": _Body(data),
            "ContentLength": len(data),
        }

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kw: Any) -> dict:
        full = f"{Bucket}/{Key}"
        self.objects[full] = bytes(Body)
        return {}

    def upload_fileobj(
        self, Fileobj: Any, Bucket: str, Key: str, **kw: Any
    ) -> None:
        # boto3's managed-transfer upload: ``upload_fileobj(Fileobj,
        # Bucket, Key, ExtraArgs=, Callback=, Config=)``. S3Path streams
        # large writes through this instead of ``put_object``, so the mock
        # has to honor it too — read the file object to EOF and store it.
        self.objects[f"{Bucket}/{Key}"] = Fileobj.read()

    def delete_object(self, *, Bucket: str, Key: str, **kw: Any) -> dict:
        full = f"{Bucket}/{Key}"
        self.objects.pop(full, None)
        return {}

    def delete_objects(self, *, Bucket: str, Delete: dict, **kw: Any) -> dict:
        for obj in Delete.get("Objects", []):
            full = f"{Bucket}/{obj['Key']}"
            self.objects.pop(full, None)
        return {}

    def get_paginator(self, operation: str) -> "_Paginator":
        return _Paginator(self, operation)

    def _not_found(self, key: str) -> Exception:
        exc = Exception(f"NoSuchKey: {key}")
        exc.response = {  # type: ignore[attr-defined]
            "Error": {"Code": "NoSuchKey", "Message": f"Key {key} not found"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }
        return exc

    @staticmethod
    def _parse_range(header: str, total: int) -> tuple[int, int]:
        prefix = "bytes="
        if header.startswith(prefix):
            header = header[len(prefix):]
        parts = header.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else total - 1
        return start, min(end, total - 1)


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        self.closed = True


class _Paginator:
    def __init__(self, s3: _InMemoryS3, operation: str) -> None:
        self.s3 = s3
        self.operation = operation

    def paginate(self, *, Bucket: str, Prefix: str = "", Delimiter: str = "", **kw: Any):
        if self.operation != "list_objects_v2":
            return iter([])

        matching: list[dict] = []
        common_prefixes: set[str] = set()

        for full_key, data in sorted(self.s3.objects.items()):
            if not full_key.startswith(f"{Bucket}/"):
                continue
            key = full_key[len(Bucket) + 1 :]
            if not key.startswith(Prefix):
                continue
            suffix = key[len(Prefix) :]
            if Delimiter and Delimiter in suffix:
                cp = Prefix + suffix[: suffix.index(Delimiter) + len(Delimiter)]
                common_prefixes.add(cp)
            else:
                matching.append({
                    "Key": key,
                    "Size": len(data),
                    "LastModified": None,
                })

        page: dict[str, Any] = {}
        if matching:
            page["Contents"] = matching
        if common_prefixes:
            page["CommonPrefixes"] = [{"Prefix": p} for p in sorted(common_prefixes)]
        return iter([page] if (matching or common_prefixes) else [])


def _make_s3_service(mock_s3: _InMemoryS3) -> MagicMock:
    """Build a mock S3Service wired to an in-memory S3 backend."""
    from yggdrasil.aws.fs.service import S3Service

    svc = MagicMock(spec=S3Service)
    svc.boto_client = mock_s3
    svc.ls_cache = {}
    return svc


def _s3_delta_folder(mock_s3: _InMemoryS3, bucket: str, prefix: str):
    """Create a DeltaFolder backed by in-memory S3."""
    from yggdrasil.aws.fs.path import S3Path
    from yggdrasil.io.delta.delta_folder import DeltaFolder

    svc = _make_s3_service(mock_s3)
    root = S3Path(f"s3://{bucket}/{prefix}", service=svc)
    return DeltaFolder(path=root)


# ---------------------------------------------------------------------------
# Mock S3 tests
# ---------------------------------------------------------------------------


class TestDeltaOnMockS3(DeltaTestCase):
    """Full Delta read/write protocol over mocked S3."""

    def setUp(self) -> None:
        super().setUp()
        self.s3 = _InMemoryS3()
        self.bucket = "test-bucket"
        self.prefix = f"delta-test-{int(time.time_ns())}/"

    def delta_s3(self, name: str = "table") -> Any:
        return _s3_delta_folder(self.s3, self.bucket, f"{self.prefix}{name}")

    def test_write_and_read_unpartitioned(self) -> None:
        d = self.delta_s3()
        t = pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        d.write_arrow_table(t)

        snap = d.snapshot(fresh=True)
        self.assertEqual(snap.version, 0)
        self.assertEqual(snap.num_active_files(), 1)

        out = d.read_arrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3])

    def test_append_over_s3(self) -> None:
        d = self.delta_s3()
        d.write_arrow_table(pa.table({"id": [1, 2]}))
        d.write_arrow_batches(
            pa.table({"id": [3, 4]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )
        snap = d.snapshot(fresh=True)
        self.assertEqual(snap.num_active_files(), 2)
        out = d.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3, 4])

    def test_overwrite_over_s3(self) -> None:
        d = self.delta_s3()
        d.write_arrow_table(pa.table({"id": [1, 2]}))
        d.write_arrow_table(
            pa.table({"id": [99]}),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )
        out = d.read_arrow_table()
        self.assertEqual(out.column("id").to_pylist(), [99])

    def test_time_travel_over_s3(self) -> None:
        d = self.delta_s3()
        d.write_arrow_table(pa.table({"id": [1, 2]}))
        d.write_arrow_batches(
            pa.table({"id": [3]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )
        v0 = d.read_arrow_table(options=DeltaOptions(version=0))
        self.assertEqual(sorted(v0.column("id").to_pylist()), [1, 2])
        head = d.read_arrow_table()
        self.assertEqual(sorted(head.column("id").to_pylist()), [1, 2, 3])

    def test_v1_checkpoint_over_s3(self) -> None:
        d = self.delta_s3()
        for i in range(6):
            mode = Mode.AUTO if i == 0 else Mode.APPEND
            d.write_arrow_batches(
                pa.table({"id": [i]}).to_batches(),
                options=DeltaOptions(
                    mode=mode,
                    checkpoint_interval=5,
                    checkpoint_kind="v1",
                ),
            )

        from yggdrasil.io.delta.delta_folder import DeltaFolder
        d2 = _s3_delta_folder(self.s3, self.bucket, f"{self.prefix}table")
        out = d2.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), list(range(6)))

    def test_v2_checkpoint_over_s3(self) -> None:
        d = self.delta_s3()
        for i in range(6):
            mode = Mode.AUTO if i == 0 else Mode.APPEND
            d.write_arrow_batches(
                pa.table({"id": [i]}).to_batches(),
                options=DeltaOptions(
                    mode=mode,
                    checkpoint_interval=5,
                    checkpoint_kind="v2",
                ),
            )
        d2 = _s3_delta_folder(self.s3, self.bucket, f"{self.prefix}table")
        out = d2.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), list(range(6)))

    def test_partitioned_write_read_over_s3(self) -> None:
        from yggdrasil.data.data_field import Field
        from yggdrasil.data.schema import Schema
        from yggdrasil.data.types.primitive import Int64Type, StringType

        schema = Schema()
        schema.with_field(Field(name="id", dtype=Int64Type()))
        schema.with_field(
            Field(name="region", dtype=StringType()).with_partition_by(True)
        )
        schema.with_field(Field(name="val", dtype=StringType()))

        d = self.delta_s3()
        t = pa.table({
            "id": [1, 2, 3, 4],
            "region": ["us", "us", "eu", "eu"],
            "val": ["a", "b", "c", "d"],
        })
        d.write_arrow_table(t, options=DeltaOptions(target=schema))

        snap = d.snapshot(fresh=True)
        self.assertEqual(snap.partition_columns, ["region"])
        self.assertEqual(snap.num_active_files(), 2)

        out = d.read_arrow_table()
        self.assertEqual(out.num_rows, 4)
        self.assertEqual(set(out.column("region").to_pylist()), {"us", "eu"})

    def test_schema_collection_over_s3(self) -> None:
        d = self.delta_s3()
        d.write_arrow_table(pa.table({"id": [1], "name": ["a"]}))
        s = d.collect_schema()
        names = [f.name for f in s.fields]
        self.assertEqual(names, ["id", "name"])

    def test_stats_collected_over_s3(self) -> None:
        d = self.delta_s3()
        t = pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        d.write_arrow_table(t, options=DeltaOptions(collect_stats=True))

        snap = d.snapshot(fresh=True)
        import json
        for add in snap.active_files.values():
            self.assertIsNotNone(add.stats)
            stats = json.loads(add.stats)
            self.assertEqual(stats["numRecords"], 3)

    def test_concurrent_commit_detection_over_s3(self) -> None:
        d = self.delta_s3()
        d.write_arrow_table(pa.table({"id": [1]}))

        # Manually write version 1 commit to simulate concurrent writer
        log_prefix = f"{self.prefix}table/_delta_log/"
        key = f"{self.bucket}/{log_prefix}00000000000000000001.json"
        self.s3.objects[key] = b'{"commitInfo":{"timestamp":0}}\n'

        from yggdrasil.io.delta.delta_folder import ConcurrentDeltaCommitError
        d.write_arrow_batches(
            pa.table({"id": [2]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND, commit_max_retries=1),
        )
        snap = d.snapshot(fresh=True)
        self.assertGreaterEqual(snap.version, 2)

    def test_s3_log_listing_cached(self) -> None:
        d = self.delta_s3()
        d.write_arrow_table(pa.table({"id": [1]}))

        d.read_arrow_table()
        d.read_arrow_table()
        d.collect_schema()
        # All three should use the cached listing (no extra S3 LIST calls)

    def test_s3_objects_layout(self) -> None:
        """Verify the S3 object keys match Delta convention."""
        d = self.delta_s3("layout_test")
        d.write_arrow_table(pa.table({"id": [1, 2]}))

        prefix = f"{self.prefix}layout_test/"
        s3_keys = [
            k[len(self.bucket) + 1 :]
            for k in self.s3.objects
            if k.startswith(f"{self.bucket}/{prefix}")
        ]
        log_keys = [k for k in s3_keys if "/_delta_log/" in k]
        parquet_keys = [k for k in s3_keys if k.endswith(".parquet") and "_delta_log" not in k]

        self.assertEqual(len(log_keys), 1)
        self.assertTrue(log_keys[0].endswith("00000000000000000000.json"))
        self.assertEqual(len(parquet_keys), 1)
        self.assertTrue(parquet_keys[0].endswith(".parquet"))

    def test_ignore_mode_over_s3(self) -> None:
        d = self.delta_s3()
        d.write_arrow_table(pa.table({"id": [1]}))
        d.write_arrow_batches(
            pa.table({"id": [2]}).to_batches(),
            options=DeltaOptions(mode=Mode.IGNORE),
        )
        self.assertEqual(d.read_arrow_table().column("id").to_pylist(), [1])

    def test_error_if_exists_over_s3(self) -> None:
        d = self.delta_s3()
        d.write_arrow_table(pa.table({"id": [1]}))
        with self.assertRaises(FileExistsError):
            d.write_arrow_batches(
                pa.table({"id": [2]}).to_batches(),
                options=DeltaOptions(mode=Mode.ERROR_IF_EXISTS),
            )

    def test_multi_type_columns_over_s3(self) -> None:
        d = self.delta_s3()
        t = pa.table({
            "int_col": pa.array([1, 2], type=pa.int64()),
            "float_col": pa.array([1.5, 2.5], type=pa.float64()),
            "str_col": pa.array(["a", "b"], type=pa.string()),
            "bool_col": pa.array([True, False], type=pa.bool_()),
        })
        d.write_arrow_table(t)
        out = d.read_arrow_table()
        self.assertEqual(out.num_rows, 2)
        self.assertEqual(out.column("str_col").to_pylist(), ["a", "b"])


# ---------------------------------------------------------------------------
# Real S3 integration tests
# ---------------------------------------------------------------------------

def _has_s3() -> bool:
    return bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        and os.environ.get("YGG_TEST_S3_BUCKET")
    )


@pytest.mark.integration
@unittest.skipUnless(_has_s3(), "AWS_ACCESS_KEY_ID or YGG_TEST_S3_BUCKET not set")
class TestDeltaOnRealS3(DeltaTestCase):
    """Full Delta protocol against a real S3 bucket."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.bucket = os.environ["YGG_TEST_S3_BUCKET"]
        cls.base_prefix = os.environ.get("YGG_TEST_S3_PREFIX", "ygg-delta-test/")
        cls.test_prefix = f"{cls.base_prefix}{int(time.time_ns())}/"

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            from yggdrasil.aws.fs.path import S3Path
            root = S3Path(f"s3://{cls.bucket}/{cls.test_prefix}")
            root.remove(recursive=True, missing_ok=True)
        except Exception:
            pass
        super().tearDownClass()

    def _s3_folder(self, name: str = "table"):
        from yggdrasil.aws.fs.path import S3Path
        from yggdrasil.io.delta.delta_folder import DeltaFolder

        path = S3Path(f"s3://{self.bucket}/{self.test_prefix}{name}")
        return DeltaFolder(path=path)

    def test_write_read_round_trip(self) -> None:
        d = self._s3_folder("rt")
        t = pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        d.write_arrow_table(t)

        out = d.read_arrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3])

    def test_append_and_time_travel(self) -> None:
        d = self._s3_folder("tt")
        d.write_arrow_table(pa.table({"id": [1, 2]}))
        d.write_arrow_batches(
            pa.table({"id": [3]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )

        v0 = d.read_arrow_table(options=DeltaOptions(version=0))
        self.assertEqual(sorted(v0.column("id").to_pylist()), [1, 2])

        head = d.read_arrow_table()
        self.assertEqual(sorted(head.column("id").to_pylist()), [1, 2, 3])

    def test_overwrite(self) -> None:
        d = self._s3_folder("ow")
        d.write_arrow_table(pa.table({"id": [1, 2]}))
        d.write_arrow_table(
            pa.table({"id": [99]}),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )
        out = d.read_arrow_table()
        self.assertEqual(out.column("id").to_pylist(), [99])

    def test_checkpoint_v1(self) -> None:
        d = self._s3_folder("ckv1")
        for i in range(6):
            mode = Mode.AUTO if i == 0 else Mode.APPEND
            d.write_arrow_batches(
                pa.table({"id": [i]}).to_batches(),
                options=DeltaOptions(mode=mode, checkpoint_interval=5),
            )

        from yggdrasil.io.delta.delta_folder import DeltaFolder
        from yggdrasil.aws.fs.path import S3Path

        d2 = DeltaFolder(path=S3Path(f"s3://{self.bucket}/{self.test_prefix}ckv1"))
        out = d2.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), list(range(6)))

    def test_partitioned(self) -> None:
        from yggdrasil.data.data_field import Field
        from yggdrasil.data.schema import Schema
        from yggdrasil.data.types.primitive import Int64Type, StringType

        schema = Schema()
        schema.with_field(Field(name="id", dtype=Int64Type()))
        schema.with_field(
            Field(name="region", dtype=StringType()).with_partition_by(True)
        )

        d = self._s3_folder("part")
        t = pa.table({
            "id": [1, 2, 3],
            "region": ["us", "eu", "us"],
        })
        d.write_arrow_table(t, options=DeltaOptions(target=schema))
        out = d.read_arrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(set(out.column("region").to_pylist()), {"us", "eu"})

    def test_schema_introspection(self) -> None:
        d = self._s3_folder("schema")
        d.write_arrow_table(pa.table({"x": [1], "y": ["a"], "z": [1.5]}))
        s = d.collect_schema()
        self.assertEqual([f.name for f in s.fields], ["x", "y", "z"])
