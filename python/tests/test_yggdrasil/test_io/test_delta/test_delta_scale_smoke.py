"""DeltaFolder scale smoke — stays correct and cheap as the table grows.

Many small commits is the worst case for a Delta reader (the snapshot must
know every active file). This asserts the structural properties that keep
DeltaFolder scaling — rather than wall-clock timings (flaky in CI):

- every appended row survives; the version advances by exactly the commit
  count, with one active file per commit;
- a checkpoint is written (so log replay is bounded — a snapshot reads the
  checkpoint + the few commits after it, not all N commit JSONs);
- data-skipping prunes a 1/N-selective predicate down to a single file
  (so a selective read stays O(1) in the file count); and
- advancing a cached snapshot replays only the new commits and lands on the
  same state as a full read.

See ``benchmarks/io/delta/bench_delta_scale.py`` for the timing profile.
"""
from __future__ import annotations

from yggdrasil.enums import Mode
from yggdrasil.execution.expr import Expression
from yggdrasil.io.delta import DeltaFolder, DeltaOptions
from yggdrasil.io.delta.delta_folder import _data_skip_adds
from yggdrasil.io.delta.snapshot import Snapshot
from yggdrasil.io.delta.tests import DeltaTestCase

_COMMITS = 40   # > checkpoint_interval (10) so checkpoints kick in
_ROWS = 1000


class TestDeltaScaleSmoke(DeltaTestCase):
    def _grow(self) -> "DeltaFolder":
        d = self.delta_io("scale")
        for i in range(_COMMITS):
            d.write_arrow_table(
                self.pa.table({
                    "id": self.pa.array(range(i * _ROWS, (i + 1) * _ROWS), self.pa.int64()),
                    "g": self.pa.array([i] * _ROWS, self.pa.int32()),
                }),
                mode=Mode.APPEND,
            )
        return d

    def test_all_rows_present_and_versioned(self) -> None:
        d = self._grow()
        out = d.read_arrow_table()
        self.assertEqual(out.num_rows, _COMMITS * _ROWS)
        self.assertEqual(sorted(out.column("id").to_pylist())[:3], [0, 1, 2])
        snap = d.snapshot(fresh=True)
        self.assertEqual(snap.version, _COMMITS - 1)        # 0-indexed commits
        self.assertEqual(snap.num_active_files(), _COMMITS)  # one file per append

    def test_checkpoint_bounds_log_replay(self) -> None:
        d = self._grow()
        names = [c.name for c in (d.path / "_delta_log").iterdir()]
        self.assertTrue(
            any(".checkpoint." in n for n in names),
            f"no checkpoint written after {_COMMITS} commits — log replay is "
            f"unbounded: {names}",
        )

    def test_data_skipping_prunes_to_one_file(self) -> None:
        d = self._grow()
        snap = d.snapshot(fresh=True)
        pred = Expression.from_sql(f"g = {_COMMITS // 2}")
        kept = list(_data_skip_adds(snap, list(snap.active_files.values()), pred))
        # One distinct ``g`` per file → a 1/N predicate keeps exactly one.
        self.assertEqual(len(kept), 1)
        self.assertLess(len(kept), snap.num_active_files())
        # And the pruned read returns exactly the matching rows.
        out = d.read_arrow_table(options=DeltaOptions(predicate=pred))
        self.assertEqual(out.num_rows, _ROWS)
        self.assertEqual(set(out.column("g").to_pylist()), {_COMMITS // 2})

    def test_incremental_advance_matches_full_read(self) -> None:
        d = self._grow()
        d.log.invalidate()
        full = Snapshot.from_log(d.log, None)
        base = Snapshot.from_log(d.log, _COMMITS // 2)  # midway version
        advanced = base.advanced(d.log, d.log.commits_after(base.version), full.version)
        self.assertEqual(advanced.version, full.version)
        self.assertEqual(set(advanced.active_files), set(full.active_files))
