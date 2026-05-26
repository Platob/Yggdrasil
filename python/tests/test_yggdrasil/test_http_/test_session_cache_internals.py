"""Unit tests for the private helpers behind the session cache.

These pin the contracts of the small, side-effect-free helpers the
:class:`Session` pipeline relies on:

* ``CacheConfig.make_lookup_predicate`` / ``.make_batch_lookup_predicate``
  — the per-request and batch :class:`Predicate` shape the partitioned
  local cache pushes through :meth:`FolderPath._read_arrow_batches`.
  Mirrors the SQL the remote cache uses, so the partition prune fires
  on both backends with the same logical clause.
* The unified :meth:`CacheConfig.cache_tabular` surface — both
  local (FolderPath) and remote (Databricks Table) plug into the
  Session through :meth:`Tabular.read_arrow_batches` +
  :meth:`Tabular.insert`, so the cache pipeline is backend-
  agnostic.
* ``HTTPSession._remote_write_group_key`` — the bucket key used to fan
  one batch of remote-cache inserts into per-(table, mode, match_by,
  wait, anonymize) groups.

Full end-to-end coverage through ``send`` / ``send_many`` lives in
``test_session_cache_integration.py`` — this file deliberately stays
unit-scoped so a regression in any of the helpers above lights up
without dragging the whole pipeline in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from yggdrasil.enums import Mode
from yggdrasil.http_ import HTTPSession
from yggdrasil.io.send_config import CacheConfig
from yggdrasil.io.session import Session
from yggdrasil.io.tabular import Tabular

from ._helpers import make_request, make_response


# ---------------------------------------------------------------------------
# Singleton-cache hygiene — keeps StubSessions from leaking between tests.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_session_singleton_cache():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


# ---------------------------------------------------------------------------
# _remote_write_group_key
# ---------------------------------------------------------------------------


class _StubTabular(Tabular):
    """Minimal Tabular — only the attributes the group key reads."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        from yggdrasil.url import URL
        self.url = URL.from_(name)

    def _read_arrow_batches(self, options=None): return iter(())
    def _write_arrow_batches(self, batches, options=None): pass


class TestRemoteWriteGroupKey:

    def _cfg(self, **overrides: Any) -> CacheConfig:
        # ``tabular`` bypasses ``__post_init__`` validation and is what
        # ``_remote_write_group_key`` actually reads — a stub is enough.
        return CacheConfig(
            tabular=_StubTabular(overrides.pop("name", "ws.cache.responses")),
            mode=overrides.pop("mode", Mode.APPEND),
            request_by=overrides.pop("request_by", ["public_url_hash"]),
            response_by=overrides.pop("response_by", None),
            anonymize=overrides.pop("anonymize", "remove"),
            wait=overrides.pop("wait", False),
        )

    def test_identical_configs_share_group(self) -> None:
        a = self._cfg()
        b = self._cfg()
        assert HTTPSession._remote_write_group_key(a) == HTTPSession._remote_write_group_key(b)

    def test_distinct_table_splits(self) -> None:
        a = self._cfg(name="ws.a.responses")
        b = self._cfg(name="ws.b.responses")
        assert HTTPSession._remote_write_group_key(a) != HTTPSession._remote_write_group_key(b)

    def test_distinct_mode_splits(self) -> None:
        a = self._cfg(mode=Mode.APPEND)
        b = self._cfg(mode=Mode.UPSERT)
        assert HTTPSession._remote_write_group_key(a) != HTTPSession._remote_write_group_key(b)

    def test_distinct_match_by_splits(self) -> None:
        a = self._cfg(request_by=["public_url_hash"])
        b = self._cfg(request_by=["public_url_hash", "method"])
        assert HTTPSession._remote_write_group_key(a) != HTTPSession._remote_write_group_key(b)

    def test_distinct_wait_splits(self) -> None:
        a = self._cfg(wait=False)
        b = self._cfg(wait=True)
        assert HTTPSession._remote_write_group_key(a) != HTTPSession._remote_write_group_key(b)

    def test_distinct_anonymize_splits(self) -> None:
        a = self._cfg(anonymize="remove")
        b = self._cfg(anonymize="redact")
        assert HTTPSession._remote_write_group_key(a) != HTTPSession._remote_write_group_key(b)


# ---------------------------------------------------------------------------
# CacheConfig predicate builders — the only lookup surface
# ---------------------------------------------------------------------------


class TestCacheLookupPredicates:
    """:meth:`CacheConfig.make_lookup_predicate` /
    :meth:`CacheConfig.make_batch_lookup_predicate` build the
    :class:`Predicate` the Session pushes through
    :meth:`Tabular.read_arrow_batches` — same call shape for both
    the local :class:`FolderPath` cache and remote :class:`Tabular`
    backends (Databricks Table, …). Asserted shape: partition key
    prune + per-request match + response time window.
    """

    def _cfg(self, **overrides: Any) -> CacheConfig:
        return CacheConfig(
            request_by=overrides.pop("request_by", ["public_url_hash"]),
            **overrides,
        )

    def _free_columns(self, pred):
        from yggdrasil.execution.expr import free_columns
        return free_columns(pred)

    def test_single_request_predicate_partitions_and_matches(self) -> None:
        cfg = self._cfg()
        req = make_request("https://example.com/x")
        pred = cfg.make_lookup_predicate(request=req)
        # Both the partition column and the per-request match key
        # show up as free columns in the AST.
        free = self._free_columns(pred)
        assert "partition_key" in free
        assert "request_public_url_hash" in free

    def test_batch_predicate_emits_partition_in_clause(self) -> None:
        from yggdrasil.execution.expr import extract_partition_filters

        cfg = self._cfg()
        reqs = [
            make_request("https://example.com/a"),
            make_request("https://example.com/b"),
        ]
        pred = cfg.make_batch_lookup_predicate(requests=reqs)
        # Distinct partition_keys collapse into an IN-list the
        # extractor surfaces as a finite accepted-value set.
        extracted = extract_partition_filters(pred, ("partition_key",))
        assert extracted["partition_key"] == frozenset(
            r.partition_key for r in reqs
        )

    def test_predicate_extracts_partition_filters(self) -> None:
        from yggdrasil.execution.expr import (
            extract_partition_filters,
        )

        cfg = self._cfg()
        reqs = [
            make_request("https://example.com/a"),
            make_request("https://example.com/b"),
        ]
        pred = cfg.make_batch_lookup_predicate(requests=reqs)
        # The partition pruner walks the predicate and returns the
        # finite accepted-value set — that's what
        # FolderPath.iter_children probes against.
        extracted = extract_partition_filters(pred, ("partition_key",))
        assert "partition_key" in extracted
        assert extracted["partition_key"] == frozenset(
            r.partition_key for r in reqs
        )

    def test_received_window_not_in_predicate(self) -> None:
        import datetime as dt

        from_ts = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        to_ts = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)
        cfg = self._cfg(received_from=from_ts, received_to=to_ts)
        req = make_request("https://example.com/x")
        pred = cfg.make_lookup_predicate(request=req)
        assert "received_at" not in self._free_columns(pred)

    def test_empty_batch_predicate_is_none(self) -> None:
        cfg = self._cfg()
        assert cfg.make_batch_lookup_predicate(requests=[]) is None

    def test_batch_predicate_top_level_is_and_of_partition_and_match(self) -> None:
        """``make_batch_lookup_predicate`` returns an AND of the
        partition-key IN clause and the OR-of-match-clauses. Leaf
        shapes stay as the builder produced them — there's no
        post-hoc simplify pass.
        """
        from yggdrasil.execution.expr import InList, Logical, LogicalOp

        cfg = self._cfg()
        reqs = [make_request(f"https://example.com/r{i}") for i in range(32)]
        pred = cfg.make_batch_lookup_predicate(requests=reqs)
        # Top-level: AND of (partition_key IN ...) and the per-request
        # OR / single-leaf match.
        assert isinstance(pred, Logical)
        assert pred.op is LogicalOp.AND
        # The partition-key constraint is an explicit ``is_in`` so it's
        # an :class:`InList`; the rest of the operands are whatever
        # ``request_predicate`` / ``response_predicate`` built.
        partition_clauses = [
            c for c in pred.operands
            if isinstance(c, InList) and getattr(c.target, "name", None) == "partition_key"
        ]
        assert partition_clauses, "expected a partition_key InList"


# ---------------------------------------------------------------------------
# Partitioned local-cache layout (FolderPath under CacheConfig)
# ---------------------------------------------------------------------------


class TestPartitionedLocalCache:
    """Round-trip a real ``Response`` through the unified
    :meth:`CacheConfig.cache_tabular` surface.

    The asserted contract is the on-disk shape — Hive-style
    ``partition_key=<int>/part-*.<ext>`` under the cache root —
    plus the listing-time partition prune on
    :meth:`FolderPath.iter_children` shrinking the read to only the
    matching directories. Both write and read go through the same
    :meth:`Tabular.write_arrow_batches` /
    :meth:`Tabular.read_arrow_batches` calls the Session uses, so
    the test stays representative of the production path.
    """

    def _seed(self, tmp_path: Path, *requests) -> tuple[CacheConfig, "Any"]:
        from yggdrasil.io.nested.folder_path import FolderOptions

        cfg = CacheConfig(tabular=str(tmp_path))
        tabular = cfg.cache_tabular()
        opts = FolderOptions(mode=cfg.mode)
        # Partition layout is auto-detected from the response batch's
        # per-field metadata — no explicit ``partition_columns`` on
        # the FolderOptions: Response.to_arrow_batch stamps
        # ``t:partition_by=true`` on ``partition_key`` straight from
        # RESPONSE_SCHEMA, and the FolderPath reads that to drive the
        # Hive layout.
        for req in requests:
            resp = make_response(request=req, body=b'{"ok":true}')
            tabular.write_arrow_batches(
                (resp.to_arrow_batch(parse=False),), options=opts,
            )
        return cfg, tabular

    def _read(self, tabular, predicate):
        from yggdrasil.io.nested.folder_path import FolderOptions
        # No explicit partition hint: the folder's ``collect_schema``
        # reads ``.ygg/schema.arrow`` (persisted on prior writes),
        # ``Field.partition_by`` flags the partition columns, and
        # ``_resolve_partition_columns`` picks them up automatically.
        return list(tabular.read_arrow_batches(
            options=FolderOptions(predicate=predicate),
        ))

    def test_writes_land_under_partition_key_directory(self, tmp_path) -> None:
        req = make_request("https://example.com/x")
        cfg, tabular = self._seed(tmp_path, req)
        # Hive-encoded directory name: partition_key=<int>
        expected_dir = tmp_path / f"partition_key={req.partition_key}"
        assert expected_dir.is_dir()
        # One part file per write (mode=APPEND mints a fresh leaf).
        part_files = list(expected_dir.glob("part-*"))
        assert len(part_files) == 1

    def test_lookup_predicate_round_trip(self, tmp_path) -> None:
        req = make_request("https://example.com/x")
        cfg, tabular = self._seed(tmp_path, req)

        predicate = cfg.make_lookup_predicate(request=req)
        batches = self._read(tabular, predicate)
        assert batches, "expected one batch back from the partitioned read"
        rows = sum(b.num_rows for b in batches)
        assert rows == 1

    def test_batch_predicate_only_probes_matching_partitions(
        self, tmp_path, monkeypatch,
    ) -> None:
        """The candidate-probe path stat()s only the partitions the
        predicate accepts — never calls ``iterdir`` on the cache root.
        """
        wanted = make_request("https://example.com/wanted")
        other = make_request("https://example.com/other")
        cfg, tabular = self._seed(tmp_path, wanted, other)
        assert wanted.partition_key != other.partition_key

        # Spy on the root path's iterdir — the candidate-probe path
        # should NOT call it because the predicate pins partition_key.
        root_path = tabular.path
        calls = {"iterdir": 0}
        original = root_path.iterdir

        def _count(*args, **kwargs):
            calls["iterdir"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(root_path, "iterdir", _count)

        predicate = cfg.make_batch_lookup_predicate(requests=[wanted])
        batches = self._read(tabular, predicate)
        rows = sum(b.num_rows for b in batches)
        assert rows == 1
        assert calls["iterdir"] == 0, (
            "partition pushdown should skip iterdir() on the cache root"
        )

    def test_lookup_misses_when_partition_directory_empty(self, tmp_path) -> None:
        # Empty cache → predicate yields no rows, no exceptions.
        cfg = CacheConfig(tabular=str(tmp_path))
        tabular = cfg.cache_tabular()
        req = make_request("https://example.com/x")
        predicate = cfg.make_lookup_predicate(request=req)
        # Folder doesn't exist on disk yet → empty stream.
        if tabular.path.exists():
            assert self._read(tabular, predicate) == []
        else:
            # Match the Session's defensive guard.
            assert list(tabular.read_arrow_batches()) == []

