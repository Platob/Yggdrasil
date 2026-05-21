"""Unit tests for the private helpers behind the session cache.

These pin the contracts of the small, side-effect-free helpers the
:class:`Session` pipeline relies on:

* ``CacheConfig.make_lookup_predicate`` / ``.make_batch_lookup_predicate``
  — the per-request and batch :class:`Predicate` shape the partitioned
  local cache pushes through :meth:`FolderIO._read_arrow_batches`.
  Mirrors the SQL the remote cache uses, so the partition prune fires
  on both backends with the same logical clause.
* The unified :meth:`CacheConfig.cache_tabular` surface — both
  local (FolderIO) and remote (Databricks Table) plug into the
  Session through :meth:`Tabular.read_arrow_batches` +
  :meth:`Tabular.insert`, so the cache pipeline is backend-
  agnostic.
* ``_maybe_autocompress_body_for_cache`` — pre-persistence gzip
  heuristic (threshold, MIME gate, ratio bailout, header sync).
* ``Session._remote_write_group_key`` — the bucket key used to fan
  one batch of remote-cache inserts into per-(table, mode, match_by,
  wait, anonymize) groups.

Full end-to-end coverage through ``send`` / ``send_many`` lives in
``test_session_cache_integration.py`` — this file deliberately stays
unit-scoped so a regression in any of the helpers above lights up
without dragging the whole pipeline in.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from yggdrasil.data.enums import Mode
from yggdrasil.io.send_config import CacheConfig
from yggdrasil.io.session import (
    Session,
    _BODY_AUTOCOMPRESS_MIN_SIZE,
    _maybe_autocompress_body_for_cache,
)

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


class _StubTabular:
    """Minimal Tabular-like object — only the attributes the group key reads."""

    def __init__(self, name: str) -> None:
        self._name = name

    def full_name(self, safe: bool = False) -> str:
        return self._name


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
        assert Session._remote_write_group_key(a) == Session._remote_write_group_key(b)

    def test_distinct_table_splits(self) -> None:
        a = self._cfg(name="ws.a.responses")
        b = self._cfg(name="ws.b.responses")
        assert Session._remote_write_group_key(a) != Session._remote_write_group_key(b)

    def test_distinct_mode_splits(self) -> None:
        a = self._cfg(mode=Mode.APPEND)
        b = self._cfg(mode=Mode.UPSERT)
        assert Session._remote_write_group_key(a) != Session._remote_write_group_key(b)

    def test_distinct_match_by_splits(self) -> None:
        a = self._cfg(request_by=["public_url_hash"])
        b = self._cfg(request_by=["public_url_hash", "method"])
        assert Session._remote_write_group_key(a) != Session._remote_write_group_key(b)

    def test_distinct_wait_splits(self) -> None:
        a = self._cfg(wait=False)
        b = self._cfg(wait=True)
        assert Session._remote_write_group_key(a) != Session._remote_write_group_key(b)

    def test_distinct_anonymize_splits(self) -> None:
        a = self._cfg(anonymize="remove")
        b = self._cfg(anonymize="redact")
        assert Session._remote_write_group_key(a) != Session._remote_write_group_key(b)


# ---------------------------------------------------------------------------
# CacheConfig predicate builders — same shape as make_*_lookup_sql
# ---------------------------------------------------------------------------


class TestCacheLookupPredicates:
    """The :class:`Predicate` mirrors of :meth:`CacheConfig.make_lookup_sql` /
    :meth:`CacheConfig.make_batch_lookup_sql` carry the same logical
    shape — partition key prune + per-request match + response time
    window — as their SQL counterparts, so the local
    :class:`FolderIO` cache and the remote :class:`Tabular` cache
    prune through the same primitive.
    """

    def _cfg(self, **overrides: Any) -> CacheConfig:
        return CacheConfig(
            request_by=overrides.pop("request_by", ["public_url_hash"]),
            **overrides,
        )

    def test_single_request_predicate_partitions_and_matches(self) -> None:
        cfg = self._cfg()
        req = make_request("https://example.com/x")
        pred = cfg.make_lookup_predicate(request=req)
        sql = pred.to_sql()
        # Both the partition prune and the per-request match key
        # land in the WHERE clause via AND. The SQL emitter wraps
        # identifiers in backticks (`partition_key`), so match on the
        # column name embedded in the rendered string.
        assert "partition_key" in sql
        assert str(req.partition_key) in sql
        assert "request_public_url_hash" in sql

    def test_batch_predicate_emits_partition_in_clause(self) -> None:
        cfg = self._cfg()
        reqs = [
            make_request("https://example.com/a"),
            make_request("https://example.com/b"),
        ]
        pred = cfg.make_batch_lookup_predicate(requests=reqs)
        sql = pred.to_sql()
        # Distinct partition_keys collapse into an IN (...) — same
        # shape the SQL-side make_batch_lookup_sql emits. Strip
        # backticks the emitter wraps identifiers in.
        normalized = sql.replace("`", "")
        assert "partition_key IN" in normalized
        for r in reqs:
            assert str(r.partition_key) in sql

    def test_predicate_extracts_partition_filters(self) -> None:
        from yggdrasil.io.tabular.execution.expr import (
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
        # FolderIO.iter_children probes against.
        extracted = extract_partition_filters(pred, ("partition_key",))
        assert "partition_key" in extracted
        assert extracted["partition_key"] == frozenset(
            r.partition_key for r in reqs
        )

    def test_received_window_lands_in_predicate(self) -> None:
        import datetime as dt

        from_ts = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        to_ts = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)
        cfg = self._cfg(received_from=from_ts, received_to=to_ts)
        req = make_request("https://example.com/x")
        pred = cfg.make_lookup_predicate(request=req)
        sql = pred.to_sql().replace("`", "")
        # The time-window clause from sql_response_clause comes
        # through unchanged. The SQL emitter wraps identifiers in
        # backticks; strip them before substring matching.
        assert "received_at >=" in sql
        assert "received_at <" in sql

    def test_empty_batch_predicate_is_none(self) -> None:
        cfg = self._cfg()
        assert cfg.make_batch_lookup_predicate(requests=[]) is None


# ---------------------------------------------------------------------------
# Partitioned local-cache layout (FolderIO under CacheConfig)
# ---------------------------------------------------------------------------


class TestPartitionedLocalCache:
    """Round-trip a real ``Response`` through the unified
    :meth:`CacheConfig.cache_tabular` surface.

    The asserted contract is the on-disk shape — Hive-style
    ``partition_key=<int>/part-*.<ext>`` under the cache root —
    plus the listing-time partition prune on
    :meth:`FolderIO.iter_children` shrinking the read to only the
    matching directories. Both write and read go through the same
    :meth:`Tabular.write_arrow_batches` /
    :meth:`Tabular.read_arrow_batches` calls the Session uses, so
    the test stays representative of the production path.
    """

    def _seed(self, tmp_path: Path, *requests) -> tuple[CacheConfig, "Any"]:
        from yggdrasil.io.nested.folder_io import FolderOptions

        cfg = CacheConfig(path=str(tmp_path))
        tabular = cfg.cache_tabular()
        opts = FolderOptions(mode=cfg.mode)
        # Partition layout is auto-detected from the response batch's
        # per-field metadata — no explicit ``partition_columns`` on
        # the FolderOptions: Response.to_arrow_batch stamps
        # ``t:partition_by=true`` on ``partition_key`` straight from
        # RESPONSE_SCHEMA, and the FolderIO reads that to drive the
        # Hive layout.
        for req in requests:
            resp = make_response(request=req, body=b'{"ok":true}')
            tabular.write_arrow_batches(
                (resp.to_arrow_batch(parse=False),), options=opts,
            )
        return cfg, tabular

    def _read(self, tabular, predicate):
        from yggdrasil.io.nested.folder_io import FolderOptions
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
        cfg = CacheConfig(path=str(tmp_path))
        tabular = cfg.cache_tabular()
        req = make_request("https://example.com/x")
        predicate = cfg.make_lookup_predicate(request=req)
        # Folder doesn't exist on disk yet → empty stream.
        if tabular.path.exists():
            assert self._read(tabular, predicate) == []
        else:
            # Match the Session's defensive guard.
            assert list(tabular.read_arrow_batches()) == []


# ---------------------------------------------------------------------------
# _maybe_autocompress_body_for_cache
# ---------------------------------------------------------------------------


class TestMaybeAutocompressBodyForCache:
    """Smart body gzipping run before cache persistence.

    Pinned behavior:

    * threshold is :data:`_BODY_AUTOCOMPRESS_MIN_SIZE` (skip below);
    * skip when ``Content-Encoding`` is already set (no recompress);
    * skip when the resolved MIME is not in the compressible set —
      binary entropy-dense formats (image/png, parquet, …) don't
      benefit and we don't want to burn CPU on them;
    * skip when the gzip ratio is < 10% — random / already-compact
      input bails out so the read side doesn't pay decompress cost
      for ~no savings;
    * on a hit, the swap is consistent: the buffer carries the gzipped
      bytes, ``Content-Encoding`` is ``gzip``, ``Content-Length``
      matches the new size, ``media_type.codec`` reflects the encoding.
    """

    def _big_json(self, size: int) -> bytes:
        # JSON-shaped repetitive bytes — highly compressible and just
        # over the autocompress threshold by default.
        chunk = b'{"key":"value"}'
        repeats = size // len(chunk) + 1
        return chunk * repeats

    def test_small_body_below_threshold_skips(self) -> None:
        resp = make_response(body=b'{"k":1}' * 100)  # well under 1 MiB
        before = resp.buffer.size
        _maybe_autocompress_body_for_cache(resp)
        assert resp.buffer.size == before
        assert resp.headers.get("Content-Encoding") is None

    def test_already_encoded_body_skips(self) -> None:
        big = self._big_json(_BODY_AUTOCOMPRESS_MIN_SIZE + 1024)
        resp = make_response(
            body=big,
            headers={"Content-Encoding": "br"},
        )
        before = resp.buffer.size
        _maybe_autocompress_body_for_cache(resp)
        assert resp.buffer.size == before
        assert resp.headers.get("Content-Encoding") == "br"

    def test_binary_mime_skips(self) -> None:
        big = self._big_json(_BODY_AUTOCOMPRESS_MIN_SIZE + 1024)
        resp = make_response(body=big, content_type="image/png")
        before = resp.buffer.size
        _maybe_autocompress_body_for_cache(resp)
        assert resp.buffer.size == before
        assert resp.headers.get("Content-Encoding") is None

    def test_text_outside_enum_skips(self) -> None:
        # ``text/css`` is not in the compressible :class:`MimeTypes` set
        # — strict enum membership keeps the rule predictable; add a
        # MIME to the set in one place rather than maintaining a string
        # prefix list at every caller.
        big = self._big_json(_BODY_AUTOCOMPRESS_MIN_SIZE + 1024)
        resp = make_response(body=big, content_type="text/css")
        _maybe_autocompress_body_for_cache(resp)
        assert resp.headers.get("Content-Encoding") is None

    def test_random_bytes_ratio_bailout(self) -> None:
        # ``os.urandom`` is incompressible — the gzip output is within
        # 1% of the input, well above the 10% bail threshold. The
        # helper has to skip so we don't pay decompress cost on read
        # for ~no storage win.
        random = os.urandom(_BODY_AUTOCOMPRESS_MIN_SIZE + 1024)
        resp = make_response(body=random, content_type="text/plain")
        _maybe_autocompress_body_for_cache(resp)
        assert resp.headers.get("Content-Encoding") is None

    def test_large_json_gets_compressed(self) -> None:
        big = self._big_json(_BODY_AUTOCOMPRESS_MIN_SIZE + 1024)
        resp = make_response(body=big, content_type="application/json")
        before = resp.buffer.size
        _maybe_autocompress_body_for_cache(resp)
        assert resp.headers.get("Content-Encoding") == "gzip"
        assert resp.buffer.size < before
        # Content-Length must be resynced to the compressed bytes —
        # otherwise the cache row's headers would lie about the payload
        # and the read-side codec dispatch would break.
        assert resp.headers.get("Content-Length") == str(resp.buffer.size)
        assert resp.media_type.codec is not None
        assert resp.media_type.codec.name == "gzip"

    def test_large_csv_gets_compressed(self) -> None:
        csv = (b"col1,col2\n1,2\n" * (
            _BODY_AUTOCOMPRESS_MIN_SIZE // len(b"col1,col2\n1,2\n") + 1
        ))
        resp = make_response(body=csv, content_type="text/csv")
        before = resp.buffer.size
        _maybe_autocompress_body_for_cache(resp)
        assert resp.headers.get("Content-Encoding") == "gzip"
        assert resp.buffer.size < before
