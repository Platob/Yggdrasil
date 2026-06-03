"""DeltaFolder snapshot cache — 30s TTL + incremental log advance.

The latest snapshot is cached for ``_SNAPSHOT_TTL``; past it, ``snapshot()``
re-lists the ``_delta_log`` and replays *only* the commits written since the
cached version (no checkpoint / prior-commit re-read), rather than rebuilding
from scratch. ``fresh=True`` still forces a full re-read.
"""
from __future__ import annotations

from unittest.mock import patch

from yggdrasil.enums import Mode
from yggdrasil.io.delta.delta_folder import _SNAPSHOT_TTL
from yggdrasil.io.delta.snapshot import Snapshot
from yggdrasil.io.delta.tests import DeltaTestCase


class TestSnapshotCacheTTL(DeltaTestCase):
    def test_ttl_is_30s(self) -> None:
        self.assertEqual(_SNAPSHOT_TTL, 30.0)

    def test_within_ttl_returns_cached_instance(self) -> None:
        d = self.new_table(self.pa.table({"id": [1, 2, 3]}))
        s1 = d.snapshot()
        s2 = d.snapshot()
        self.assertIs(s2, s1)  # cached within the TTL — no re-read

    def test_expired_ttl_rechecks_same_state(self) -> None:
        d = self.new_table(self.pa.table({"id": [1]}))
        v = d.snapshot().version
        d._snapshot_at = 0.0  # expire the TTL
        again = d.snapshot()
        self.assertEqual(again.version, v)  # re-checked; no new commits → same state

    def test_fresh_forces_full_reread(self) -> None:
        d = self.new_table(self.pa.table({"id": [1]}))
        with patch.object(Snapshot, "from_log", wraps=Snapshot.from_log) as spy:
            d.snapshot(fresh=True)
        self.assertTrue(spy.called)  # full read rebuilds via Snapshot.from_log


class TestIncrementalAdvance(DeltaTestCase):
    def _three_versions(self):
        d = self.new_table(self.pa.table({"id": [1]}))                       # v0
        v0 = d.snapshot(fresh=True)
        d.write_arrow_table(self.pa.table({"id": [2]}), mode=Mode.APPEND)    # v1
        d.write_arrow_table(self.pa.table({"id": [3]}), mode=Mode.APPEND)    # v2
        return d, v0

    def test_commits_after_lists_only_newer(self) -> None:
        d, _ = self._three_versions()
        d._log.invalidate()
        after0 = d._log.commits_after(0)
        self.assertEqual(len(after0), 2)  # versions 1 and 2
        names = [p.name for p in after0]
        self.assertEqual(names, sorted(names))  # ascending

    def test_advanced_equals_full_read(self) -> None:
        d, v0 = self._three_versions()
        d._log.invalidate()
        full = Snapshot.from_log(d._log, None)
        advanced = v0.advanced(d._log, d._log.commits_after(v0.version), full.version)
        self.assertEqual(advanced.version, full.version)
        self.assertEqual(set(advanced.active_files), set(full.active_files))

    def test_snapshot_advances_incrementally_without_full_segment(self) -> None:
        d, v0 = self._three_versions()
        full = d.snapshot(fresh=True)
        # Restore the stale v0 cache and expire the TTL so the next snapshot()
        # takes the incremental path.
        d._snapshot = v0
        d._snapshot_at = 0.0

        with patch.object(Snapshot, "from_log", wraps=Snapshot.from_log) as spy:
            result = d.snapshot()

        self.assertEqual(result.version, full.version)
        self.assertEqual(set(result.active_files), set(full.active_files))
        # Incremental: advanced via commits_after + replay, never a full
        # Snapshot.from_log (checkpoint + all commits) rebuild.
        self.assertFalse(spy.called)
