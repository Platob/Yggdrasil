"""DeltaFolder over an S3 path — the object-store commit path.

The local-FS tests cover the write/read/concurrency *algebra*; these pin
that the same DeltaFolder works against an ``s3://`` path, driving the
genuine pure-HTTP S3 code path (data-file PUTs, ``_delta_log`` listing via
ListObjectsV2, ranged GETs) through the in-memory :class:`FakeS3`
transport — no network. The S3-specific guarantee is the **atomic commit**:
the commit JSON lands via a conditional ``PUT`` (``If-None-Match: *`` → HTTP
412 → :class:`FileExistsError` → rebase), the object-store analogue of the
local ``O_EXCL`` create.
"""
from __future__ import annotations

import secrets

import pyarrow as pa

from yggdrasil.enums import Mode
from yggdrasil.io.delta import DeltaFolder, DeltaOptions
from yggdrasil.io.delta.tests import DeltaTestCase

from tests.test_yggdrasil.test_aws._fake_s3 import (
    FakeS3, reset_s3_singletons, wire_s3_path,
)


class _S3DeltaBase(DeltaTestCase):
    """DeltaTestCase wired to a fresh in-memory S3 per test.

    DeltaFolder / S3Bucket / S3Path are all process-lifetime singletons
    keyed on URL, so the registries are cleared around every test to keep
    one test's table (and its cached snapshot) from leaking into the next.
    """

    def setUp(self) -> None:
        super().setUp()
        reset_s3_singletons()
        DeltaFolder._INSTANCES.clear()
        self.fake = FakeS3()
        # Unique per test run so no two tables ever collide on a singleton
        # key (DeltaFolder / S3Path are URL-keyed singletons).
        self._run = secrets.token_hex(4)

    def tearDown(self) -> None:
        reset_s3_singletons()
        DeltaFolder._INSTANCES.clear()
        super().tearDown()

    def _delta(self, name: str = "t") -> DeltaFolder:
        path = wire_s3_path(self.fake, f"s3://bkt/tmp/{self._run}/{name}")
        return DeltaFolder(path=path)

    def _keys(self, suffix: str, *, under: str | None = None) -> list[str]:
        return sorted(
            k for k in self.fake.objects
            if k.endswith(suffix) and (under is None or under in k)
        )


class TestS3WriteRead(_S3DeltaBase):
    def test_create_append_read_round_trip(self) -> None:
        d = self._delta()
        d.write_arrow_table(pa.table({"id": [1, 2, 3]}))
        self.assertEqual(d.snapshot(fresh=True).version, 0)

        d.write_arrow_batches(pa.table({"id": [4, 5]}).to_batches(),
                              options=DeltaOptions(mode=Mode.APPEND))
        self.assertEqual(d.snapshot(fresh=True).version, 1)

        # A cold reader (cleared singleton) reads the converged table off S3.
        DeltaFolder._INSTANCES.clear()
        reader = DeltaFolder(path=wire_s3_path(self.fake, d.path.full_path()))
        self.assertEqual(sorted(reader.read_arrow_table().column("id").to_pylist()),
                         [1, 2, 3, 4, 5])

    def test_commit_uses_conditional_put(self) -> None:
        # Every commit JSON must be written with If-None-Match so a version
        # race is atomic. Proven indirectly: re-committing an existing version
        # is rejected by the conditional PUT (412 → FileExistsError).
        d = self._delta()
        d.write_arrow_table(pa.table({"id": [1]}))  # version 0
        with self.assertRaises(FileExistsError):
            d._commit_atomic(
                0, [d._build_commit_info(options=DeltaOptions(), mode=Mode.APPEND)],
            )
        self.assertGreaterEqual(self.fake.calls.get("put_precondition_failed", 0), 1)


class TestS3WriteNewVersion(_S3DeltaBase):
    def test_write_new_version_false_skips_commit(self) -> None:
        d = self._delta()
        d.write_arrow_table(pa.table({"id": [1, 2, 3]}))
        version_before = d.snapshot(fresh=True).version
        log_before = self._keys(".json", under="_delta_log")
        data_before = self._keys(".parquet")

        d.write_arrow_batches(pa.table({"id": [4, 5, 6]}).to_batches(),
                              options=DeltaOptions(write_new_version=False))

        # Version + _delta_log untouched ...
        self.assertEqual(d.snapshot(fresh=True).version, version_before)
        self.assertEqual(self._keys(".json", under="_delta_log"), log_before)
        # ... but the parquet object was physically PUT to the bucket ...
        self.assertGreater(len(self._keys(".parquet")), len(data_before))
        # ... and the uncommitted object is invisible to a reader.
        self.assertEqual(sorted(d.read_arrow_table().column("id").to_pylist()),
                         [1, 2, 3])


class TestS3ConcurrentRebase(_S3DeltaBase):
    def _smuggle_rival_commit(self, d: DeltaFolder, rows: list[int], at_version: int) -> None:
        """Land a rival append at *at_version* directly in *d*'s S3 keyspace.

        Builds the rival in a sibling table, then copies its newest commit
        JSON + the parquet that commit adds into *d*'s key prefix — a real,
        conditional-PUT-visible commit the next writer must collide with and
        rebase past (no second process needed; the conflict is genuine
        because the key now exists in the shared FakeS3 store).
        """
        import json

        rival = self._delta("rival")
        rival.write_arrow_table(pa.table({"id": [0]}))            # rival v0
        rival.write_arrow_batches(pa.table({"id": rows}).to_batches(),
                                  options=DeltaOptions(mode=Mode.APPEND))  # rival v1
        rival_prefix = rival.path.key.rstrip("/")
        target_prefix = d.path.key.rstrip("/")
        commit_name = f"{1:020d}.json"
        commit_key = f"{rival_prefix}/_delta_log/{commit_name}"
        body = self.fake.objects[commit_key]
        # Copy the parquet the rival's commit adds (its AddFile path is
        # table-relative, so it must physically exist under our prefix).
        for line in body.decode().splitlines():
            obj = json.loads(line)
            if "add" in obj:
                rel = obj["add"]["path"]
                self.fake.objects[f"{target_prefix}/{rel}"] = \
                    self.fake.objects[f"{rival_prefix}/{rel}"]
        # Drop the rival commit JSON in at our target version.
        self.fake.objects[f"{target_prefix}/_delta_log/{at_version:020d}.json"] = body

    def test_append_rebases_past_concurrent_s3_commit(self) -> None:
        d = self._delta("ours")
        d.write_arrow_table(pa.table({"id": [1]}))  # version 0

        # Land the rival at the exact version our writer is about to take —
        # *inside* the commit, just before the real conditional PUT runs — so
        # the PUT itself collides (real 412), not a pre-seeded log the
        # snapshot would have already absorbed. Our append must then refresh,
        # rebase, and land at v2 with no lost write.
        orig_commit = d._commit_atomic
        calls = {"n": 0}

        def _smuggling_commit(version, actions):
            calls["n"] += 1
            if calls["n"] == 1:
                self._smuggle_rival_commit(d, [10, 11], at_version=version)
            return orig_commit(version, actions)  # real If-None-Match PUT

        d._commit_atomic = _smuggling_commit  # type: ignore[assignment]
        d.write_arrow_batches(pa.table({"id": [2]}).to_batches(),
                              options=DeltaOptions(mode=Mode.APPEND,
                                                   commit_retry_backoff=0))

        self.assertEqual(d.snapshot(fresh=True).version, 2)
        # The conditional PUT actually rejected our first (v1) attempt — i.e.
        # the atomicity came from S3's 412, not from our own listing.
        self.assertGreaterEqual(self.fake.calls.get("put_precondition_failed", 0), 1)
        # Ours + the rival's rows all survive.
        DeltaFolder._INSTANCES.clear()
        reader = DeltaFolder(path=wire_s3_path(self.fake, d.path.full_path()))
        self.assertEqual(sorted(reader.read_arrow_table().column("id").to_pylist()),
                         [1, 2, 10, 11])
