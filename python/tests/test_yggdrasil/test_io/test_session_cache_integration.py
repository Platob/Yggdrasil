"""Integration tests for the :class:`Session` cache pipeline.

These tests fill gaps left by :mod:`test_http_integration` (single-send
local cache only) and :mod:`test_session_concurrent_cache` (helpers
only). They drive the staged ``send_many`` pipeline end-to-end with a
:class:`StubSession` transport and a partitioned :class:`YGGFolderIO`
local cache, plus a hand-rolled fake remote :class:`Tabular` for the
remote-cache flow.

Coverage:

* ``Session._remote_write_group_key`` actually splits responses by
  every dimension that affects the insert call (table, mode,
  match_by, wait, anonymize) — collapsing on any one of those would
  silently drop per-request config divergence on the floor.
* ``Session._split_local_cache`` groups by the effective config's
  ``tabular.path``, so a per-request override pointing at a different
  cache folder takes a separate folder read instead of bleeding into
  the session-level bucket.
* ``Session.send_many`` end-to-end with a mix of cache hits and
  misses: hits skip the network, the writeback persists the misses,
  and a re-run of the same batch reads everything from disk.
* Per-request ``local_cache_config`` override survives the batch
  pipeline (a request pointing at a fresh empty folder must miss the
  pre-seeded session cache).
* ``filter_response`` rejection: a row outside the
  ``received_from``/``received_to`` window is treated as a miss even
  when the partition + tuple match.
* ``_lookup_local_responses`` picks the latest row when the same
  ``request_by`` tuple appears multiple times — the local cache is
  append-only so duplicate keys are real and the tie-break has to
  pick by ``received_at``.
* The body-hash predicate keeps two POSTs with distinct bodies from
  aliasing each other through the cache.
* Remote-cache integration (no Databricks required): fake
  :class:`Tabular` with ``sql.execute`` returning seeded Arrow rows
  exercises ``_load_remote_cached_response``,
  ``_store_remote_cached_response``, the
  ``TABLE_OR_VIEW_NOT_FOUND`` recovery, and the
  ``mirror_local_to_remote`` writeback.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pytest

from yggdrasil.data.enums import Mode
from yggdrasil.io.nested.ygg_folder_io import YGGFolderIO
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.response import RESPONSE_SCHEMA, Response
from yggdrasil.io.send_config import CacheConfig
from yggdrasil.io.session import (
    Session,
    _combine_predicates,
    _lookup_local_responses,
    _request_body_hash_predicate,
    _request_match_by_predicate,
)

from ._helpers import StubSession, make_request, make_response


# ---------------------------------------------------------------------------
# Singleton-cache hygiene — keeps StubSessions from leaking between tests.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_session_singleton_cache():
    Session._singleton_cache.clear()
    yield
    Session._singleton_cache.clear()


def _local_cache(tmp_path: Path, **overrides: Any) -> CacheConfig:
    folder = YGGFolderIO(
        path=LocalPath(str(tmp_path)),
        schema=RESPONSE_SCHEMA,
    )
    return CacheConfig(tabular=folder, mode=Mode.APPEND, **overrides)


def _wait_for_readable(cache: CacheConfig, *, timeout: float = 3.0) -> bool:
    """Poll until the cache reads back a non-empty Arrow table.

    Picks up both layouts the session writes through: the
    partitioned ``col=val/...`` tree (legacy bulk path) and the
    flat per-``public_hash`` fast-path files at the cache root.
    """
    from pathlib import Path as _P

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            cache.tabular.invalidate_listing()
            with cache.tabular:
                table = cache.tabular.read_arrow_table()
            if table.num_rows > 0:
                return True
        except Exception:
            pass
        try:
            root = _P(str(cache.tabular.path))
            if root.exists() and any(
                p.is_file()
                and p.suffix == ".arrow"
                and not p.name.startswith(".")
                for p in root.iterdir()
            ):
                return True
        except OSError:
            pass
        time.sleep(0.05)
    return False


def _seed(cache: CacheConfig, response: Response) -> None:
    """Synchronously write a response into the cache, no fire-and-forget race."""
    batch = response.to_arrow_batch(parse=False)
    cache.tabular.write_arrow_batches([batch], options=None)
    cache.tabular.invalidate_listing()


# ---------------------------------------------------------------------------
# _remote_write_group_key
# ---------------------------------------------------------------------------


class _StubTabular:
    """Minimal Tabular-like object — only attributes the group key reads."""

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
# Predicate helpers
# ---------------------------------------------------------------------------


class TestPredicateHelpers:

    def test_combine_predicates_drops_none(self) -> None:
        assert _combine_predicates(None, None) is None
        # A single non-None predicate flows through unchanged.
        from yggdrasil.io.tabular.execution.expr import col
        p = col("partition_key").is_in([1, 2])
        assert _combine_predicates(None, p, None) is p

    def test_match_by_predicate_skips_dotted_keys(self) -> None:
        # Dotted keys (struct paths) can't be expressed as a flat
        # column term — the helper must skip them rather than mangle
        # the predicate.
        reqs = [make_request("https://example.com/a")]
        expr = _request_match_by_predicate(reqs, ("headers.X-API-Key",))
        assert expr is None

    def test_match_by_predicate_translates_request_prefix(self) -> None:
        # ``public_url_hash`` is a request-side key stored on the
        # response cache as ``request_public_url_hash``. The helper
        # must rewrite the column name so the predicate references a
        # real column on the response schema.
        reqs = [make_request("https://example.com/a")]
        expr = _request_match_by_predicate(reqs, ("public_url_hash",))
        assert expr is not None
        # The exact predicate AST is implementation-defined; just
        # check the column it reads from is the prefixed name.
        assert "request_public_url_hash" in repr(expr)

    def test_body_hash_predicate_distinguishes_post_bodies(self) -> None:
        a = make_request("https://example.com/x", method="POST", body=b"payload-A")
        b = make_request("https://example.com/x", method="POST", body=b"payload-B")
        assert a.body_hash != b.body_hash
        expr = _request_body_hash_predicate([a, b])
        assert expr is not None
        # The predicate should mention both hashes — anything that
        # collapsed them would mean POST aliasing.
        rendered = repr(expr)
        assert str(a.body_hash) in rendered
        assert str(b.body_hash) in rendered

    def test_body_hash_predicate_keeps_null_for_legacy_zero(self) -> None:
        # Empty-body GETs hash to 0 on writes (schema is non-null).
        # Older rows can carry a literal NULL — the helper has to
        # OR is_null() in when 0 is among the incoming hashes so
        # legacy data isn't silently dropped.
        empty = make_request("https://example.com/x")
        assert empty.body_hash == 0
        expr = _request_body_hash_predicate([empty])
        assert expr is not None
        assert "isnull" in repr(expr).lower()


# ---------------------------------------------------------------------------
# _lookup_local_responses tie-break behavior
# ---------------------------------------------------------------------------


class TestLookupLocalResponsesLatestWins:

    def test_latest_received_at_wins_on_duplicate_key(self, tmp_path) -> None:
        # The local cache is append-only, so two writes for the same
        # request show up as two rows. The lookup has to return the
        # newest by ``received_at`` — otherwise a stale UPSERT-mode
        # row could outlive a fresher one.
        folder = YGGFolderIO(
            path=LocalPath(str(tmp_path)),
            schema=RESPONSE_SCHEMA,
        )
        req = make_request("https://example.com/x")
        old = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        new = dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc)
        old_resp = make_response(request=req, body=b'{"v":"old"}', received_at=old)
        new_resp = make_response(request=req, body=b'{"v":"new"}', received_at=new)
        folder.write_arrow_batches(
            [old_resp.to_arrow_batch(parse=False)], options=None,
        )
        folder.write_arrow_batches(
            [new_resp.to_arrow_batch(parse=False)], options=None,
        )
        folder.invalidate_listing()

        out = _lookup_local_responses(
            folder, [req], match_by=("public_url_hash",),
        )
        # Exactly one returned row, and it's the newer one.
        assert len(out) == 1
        result = next(iter(out.values()))
        assert result.json() == {"v": "new"}

    def test_empty_cache_returns_empty(self, tmp_path) -> None:
        folder = YGGFolderIO(
            path=LocalPath(str(tmp_path)),
            schema=RESPONSE_SCHEMA,
        )
        # Writing an empty marker sets up the partition layout but
        # leaves no rows — the lookup must just return ``{}``.
        out = _lookup_local_responses(
            folder, [make_request()], match_by=("public_url_hash",),
        )
        assert out == {}

    def test_missing_path_returns_empty(self, tmp_path) -> None:
        # A folder whose root never existed (cold cache) shouldn't
        # raise — the caller treats it as an all-miss batch.
        folder = YGGFolderIO(
            path=LocalPath(str(tmp_path / "no-such-dir")),
            schema=RESPONSE_SCHEMA,
        )
        out = _lookup_local_responses(
            folder, [make_request()], match_by=("public_url_hash",),
        )
        assert out == {}


# ---------------------------------------------------------------------------
# send_many end-to-end through the local cache
# ---------------------------------------------------------------------------


class TestSendManyLocalCacheIntegration:

    def test_mixed_hits_and_misses(self, tmp_path) -> None:
        # Pre-seed two of three URLs; the third must reach the
        # network. Streamed output must include all three responses.
        cache = _local_cache(tmp_path)

        seeded_a = make_request("https://example.com/a")
        seeded_b = make_request("https://example.com/b")
        miss = make_request("https://example.com/c")

        _seed(cache, make_response(request=seeded_a, body=b'{"k":"a"}'))
        _seed(cache, make_response(request=seeded_b, body=b'{"k":"b"}'))

        s = StubSession()
        s.queue(make_response(request=miss, body=b'{"k":"c"}'))

        out = list(s.send_many(
            iter([seeded_a, seeded_b, miss]),
            local_cache=cache,
        ))
        assert {r.json()["k"] for r in out} == {"a", "b", "c"}
        # Only the miss touched the wire.
        assert len(s.calls) == 1
        assert s.calls[0].url.path == "/c"

    def test_writeback_round_trip_via_send_many(self, tmp_path) -> None:
        # First batch goes to the network; second batch reads from
        # disk. The fire-and-forget writeback must finish before the
        # second batch runs — poll the cache instead of sleeping.
        cache = _local_cache(tmp_path)
        s = StubSession()
        req = make_request("https://example.com/x")
        s.queue(make_response(request=req, body=b'{"v":"first"}'))

        first = list(s.send_many(iter([req]), local_cache=cache))
        assert first[0].json() == {"v": "first"}
        assert _wait_for_readable(cache), "writeback never landed"

        second = list(s.send_many(iter([req]), local_cache=cache))
        assert len(s.calls) == 1, "second batch must hit disk, not network"
        assert second[0].json() == {"v": "first"}

    def test_per_request_local_cache_override_misses_pre_seeded_session(
        self, tmp_path,
    ) -> None:
        # Session-level cache holds a row that *would* satisfy the
        # request; per-request override points at an empty alt
        # folder, so the batch must miss and refetch from the
        # network.
        session_cache = _local_cache(tmp_path / "session")
        seed_req = make_request("https://example.com/x")
        _seed(session_cache, make_response(request=seed_req, body=b'{"v":"cached"}'))

        alt_dir = tmp_path / "alt"
        alt_dir.mkdir()
        per_req_cache = _local_cache(alt_dir)
        req = seed_req.copy(local_cache_config=per_req_cache)

        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":"network"}'))

        out = list(s.send_many(iter([req]), local_cache=session_cache))
        assert len(s.calls) == 1
        assert out[0].json() == {"v": "network"}

    def test_filter_response_outside_window_misses(self, tmp_path) -> None:
        # The cached row is too old for the configured received_from
        # window — must be treated as a miss even though the
        # match-by tuple matches.
        old = dt.datetime(2010, 1, 1, tzinfo=dt.timezone.utc)
        cache = _local_cache(
            tmp_path,
            received_from=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
            received_to=dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc),
        )
        req = make_request("https://example.com/x")
        _seed(cache, make_response(request=req, body=b'{"v":"old"}', received_at=old))

        s = StubSession()
        s.queue(make_response(request=req, body=b'{"v":"fresh"}'))

        out = list(s.send_many(iter([req]), local_cache=cache))
        assert len(s.calls) == 1
        assert out[0].json() == {"v": "fresh"}

    def test_post_body_distinct_bodies_default_request_by_aliases(
        self, tmp_path,
    ) -> None:
        # KNOWN LIMITATION: with the default ``request_by=("public_url_hash",)``
        # two POSTs to the same URL with different bodies share the same
        # cache key. ``public_url_hash`` is method+URL only — body bytes
        # don't enter the digest. The :func:`_request_body_hash_predicate`
        # helper prunes file reads to rows whose digest is in the batch's
        # set, but the row-tuple match in :func:`_lookup_local_responses`
        # uses ``match_by`` only, so a seeded row passes the prune and
        # gets handed to *both* requests. POST callers who need body
        # discrimination must opt into ``request_by=["public_url_hash",
        # "body_hash"]`` — see the next test.
        cache = _local_cache(tmp_path)
        url = "https://example.com/echo"
        req_a = make_request(url, method="POST", body=b"payload-A")
        req_b = make_request(url, method="POST", body=b"payload-B")
        _seed(cache, make_response(request=req_a, body=b'{"got":"A"}'))

        s = StubSession()
        out = list(s.send_many(iter([req_a, req_b]), local_cache=cache))
        # Both requests are reported as hits — req_b gets req_a's body.
        assert len(s.calls) == 0
        assert all(r.json()["got"] == "A" for r in out), (
            "default request_by aliases POSTs by URL — captured for visibility"
        )

    def test_post_body_distinct_with_request_body_hash_in_request_by(
        self, tmp_path,
    ) -> None:
        # The supported path: opt the cache key into ``request_body_hash``
        # — the explicit prefixed column name. ``"body_hash"`` alone
        # would be wrong on the response side because
        # ``Response.match_value("body_hash")`` returns the *response*
        # body's hash, not the request's. ``request_body_hash``
        # disambiguates: requests resolve it via the prefix-strip
        # branch of ``PreparedRequest.match_value``, responses read
        # the flattened arrow column directly. With both sides
        # agreeing, the cache key gates on the request body digest
        # and POSTs to the same URL with different bodies don't
        # alias.
        cache = _local_cache(
            tmp_path,
            request_by=["public_url_hash", "request_body_hash"],
        )
        url = "https://example.com/echo"
        req_a = make_request(url, method="POST", body=b"payload-A")
        req_b = make_request(url, method="POST", body=b"payload-B")
        _seed(cache, make_response(request=req_a, body=b'{"got":"A"}'))

        s = StubSession()
        s.queue(make_response(request=req_b, body=b'{"got":"B"}'))

        out = list(s.send_many(iter([req_a, req_b]), local_cache=cache))
        assert len(s.calls) == 1
        assert s.calls[0].buffer.to_bytes() == b"payload-B"
        assert {r.json()["got"] for r in out} == {"A", "B"}


# ---------------------------------------------------------------------------
# Concurrent local-cache writeback
# ---------------------------------------------------------------------------


class TestConcurrentWriteback:

    def test_send_many_writeback_eventually_consistent(self, tmp_path) -> None:
        # Many simultaneous writes against the same partition-key
        # must not race — :class:`YGGFolderIO` writes UUID-named
        # leaves so concurrent fire-and-forget workers can't trample
        # each other. Polling the cache for the expected row count
        # is the reliable way to wait for the daemon writers.
        cache = _local_cache(tmp_path)
        s = StubSession()
        n = 8
        reqs = [make_request(f"https://example.com/p{i}") for i in range(n)]
        s.queue(*[
            make_response(request=r, body=f'{{"i":{i}}}'.encode())
            for i, r in enumerate(reqs)
        ])
        list(s.send_many(iter(reqs), local_cache=cache))

        # Poll for the expected row count; one row per request.
        deadline = time.monotonic() + 5.0
        last_count = 0
        while time.monotonic() < deadline:
            try:
                cache.tabular.invalidate_listing()
                with cache.tabular:
                    last_count = cache.tabular.read_arrow_table().num_rows
            except Exception:
                last_count = 0
            if last_count >= n:
                break
            time.sleep(0.05)
        assert last_count >= n


# ---------------------------------------------------------------------------
# Remote-cache integration via a fake Tabular
# ---------------------------------------------------------------------------


class _FakeStatementResult:
    """Minimal :class:`StatementResult` stand-in.

    Only the surface ``Session._load_remote_cached_response`` /
    ``_lookup_remote_table`` actually use is implemented:
    ``read_arrow_batches`` returning an iterator of record batches.
    """

    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = batches

    def read_arrow_batches(self) -> Iterator[pa.RecordBatch]:
        return iter(self._batches)


class _FakeSql:
    def __init__(self, parent: "_FakeRemoteTabular") -> None:
        self._parent = parent

    def execute(self, query: str, *, spark_session: Any = None) -> _FakeStatementResult:
        self._parent.queries.append(query)
        if self._parent.raise_table_not_found and not self._parent.created:
            # Simulate Databricks' first-touch failure when the cache
            # table doesn't exist yet — the session catches this
            # exact substring and recovers via ``create``.
            raise RuntimeError("[TABLE_OR_VIEW_NOT_FOUND] table missing")
        # Return any rows that have been stored so far via
        # :meth:`_FakeRemoteTabular.insert`.
        return _FakeStatementResult(list(self._parent.rows))


class _FakeRemoteTabular:
    """Hand-rolled remote Tabular for cache-flow tests.

    Tracks every ``sql.execute`` query, every ``insert`` call, and
    holds the seeded rows in memory so a subsequent lookup can
    return them. ``is_local_tabular`` resolves to ``False`` (it is
    not a :class:`FolderIO`), so :meth:`CacheConfig.remote_cache_enabled`
    fires.
    """

    def __init__(self, name: str = "ws.cache.responses") -> None:
        self._name = name
        self.rows: list[pa.RecordBatch] = []
        self.queries: list[str] = []
        self.inserts: list[dict[str, Any]] = []
        self.created = False
        self.raise_table_not_found = False
        self.sql = _FakeSql(self)
        self.path = name  # str works for dict-key purposes

    def full_name(self, safe: bool = False) -> str:
        return self._name

    def create(self, schema: pa.Schema, if_not_exists: bool = False) -> None:
        self.created = True
        # Once "created" the next sql.execute returns rows normally.
        self.raise_table_not_found = False

    def insert(
        self,
        batch: Any,
        *,
        mode: Mode = Mode.APPEND,
        match_by: Any = None,
        wait: bool = False,
        prune_values: Any = None,
        prune_by: Any = None,
        spark_session: Any = None,
    ) -> None:
        # Normalise both ``RecordBatch`` and ``Table`` inputs into a
        # list of batches we can store (the session passes both
        # shapes depending on the code path).
        if isinstance(batch, pa.Table):
            new_batches = batch.to_batches()
        elif isinstance(batch, pa.RecordBatch):
            new_batches = [batch]
        else:
            new_batches = []
        self.inserts.append({
            "mode": mode,
            "match_by": match_by,
            "wait": wait,
            "rows": sum(b.num_rows for b in new_batches),
        })
        self.rows.extend(new_batches)


def _remote_cfg(tab: _FakeRemoteTabular, **overrides: Any) -> CacheConfig:
    return CacheConfig(
        tabular=tab,
        mode=overrides.pop("mode", Mode.APPEND),
        request_by=overrides.pop("request_by", ["public_url_hash"]),
        wait=overrides.pop("wait", False),
        **overrides,
    )


class TestRemoteCacheIntegration:

    def test_remote_miss_then_writeback(self) -> None:
        # Empty remote → first send goes to network; the response is
        # written back via ``insert``.
        tab = _FakeRemoteTabular()
        cfg = _remote_cfg(tab)
        s = StubSession()
        req = make_request("https://example.com/x")
        s.queue(make_response(request=req, body=b'{"v":1}'))

        s.send(req, remote_cache=cfg)

        assert len(s.calls) == 1, "remote miss must touch the network"
        assert tab.queries, "lookup query must run before the network fetch"
        assert any(call["rows"] == 1 for call in tab.inserts), (
            "successful response must be written back to the remote cache"
        )

    def test_remote_hit_skips_network(self) -> None:
        # Pre-seed the fake remote with a row matching the request.
        tab = _FakeRemoteTabular()
        cfg = _remote_cfg(tab)
        req = make_request("https://example.com/x")
        seeded = make_response(request=req, body=b'{"v":"cached"}')
        tab.rows.append(seeded.to_arrow_batch(parse=False))

        s = StubSession()
        out = s.send(req, remote_cache=cfg)
        assert len(s.calls) == 0, "remote hit must skip the network"
        assert out.json() == {"v": "cached"}

    def test_table_or_view_not_found_recovers(self) -> None:
        # First lookup raises TABLE_OR_VIEW_NOT_FOUND; the session
        # must call ``create`` and retry the lookup transparently.
        tab = _FakeRemoteTabular()
        tab.raise_table_not_found = True
        cfg = _remote_cfg(tab)
        s = StubSession()
        req = make_request("https://example.com/x")
        s.queue(make_response(request=req, body=b'{"v":1}'))

        s.send(req, remote_cache=cfg)
        assert tab.created, "missing table must be created on first miss"

    def test_remote_hit_backfills_local_cache(self, tmp_path) -> None:
        # Remote has the row; local cache is empty. After the send,
        # the local cache must have been written back so a subsequent
        # offline send hits disk.
        tab = _FakeRemoteTabular()
        remote_cfg = _remote_cfg(tab)
        local = _local_cache(tmp_path)
        req = make_request("https://example.com/x")
        seeded = make_response(request=req, body=b'{"v":"from-remote"}')
        tab.rows.append(seeded.to_arrow_batch(parse=False))

        s = StubSession()
        s.send(req, remote_cache=remote_cfg, local_cache=local)
        assert len(s.calls) == 0

        # Backfill is fire-and-forget — poll the cache before the
        # offline check.
        assert _wait_for_readable(local), "remote hit must backfill local cache"
        out = s.send(req, local_cache=local)
        assert out.json() == {"v": "from-remote"}

    def test_mirror_local_to_remote_writes_pre_network(self, tmp_path) -> None:
        # ``mirror_local_to_remote=True`` — a local cache hit during
        # ``send_many`` must produce a remote insert *without* going
        # to the network.
        tab = _FakeRemoteTabular()
        remote_cfg = _remote_cfg(tab, mirror_local_to_remote=True)
        local = _local_cache(tmp_path)
        req = make_request("https://example.com/x")
        _seed(local, make_response(request=req, body=b'{"v":"local"}'))

        s = StubSession()
        list(s.send_many(
            iter([req]),
            local_cache=local,
            remote_cache=remote_cfg,
        ))
        assert len(s.calls) == 0, "local hit must not touch the network"
        # The mirror path goes through ``_persist_remote`` →
        # ``insert`` with ``mode=APPEND``; assert at least one
        # writeback fired with our row.
        assert any(call["rows"] >= 1 for call in tab.inserts), (
            "mirror_local_to_remote must push the local hit upstream"
        )

    def test_mirror_disabled_keeps_remote_silent(self, tmp_path) -> None:
        # Default config — no mirror flag — keeps the remote
        # untouched on a local-only batch.
        tab = _FakeRemoteTabular()
        remote_cfg = _remote_cfg(tab)  # mirror_local_to_remote defaults to False
        local = _local_cache(tmp_path)
        req = make_request("https://example.com/x")
        _seed(local, make_response(request=req, body=b'{"v":"local"}'))

        s = StubSession()
        list(s.send_many(
            iter([req]),
            local_cache=local,
            remote_cache=remote_cfg,
        ))
        assert tab.inserts == [], (
            "default config must not push local-only hits to remote"
        )

    def test_upsert_mode_disables_remote_cache_path(self) -> None:
        # ``CacheConfig.cache_enabled`` is gated on ``mode in (APPEND, AUTO)``,
        # so :attr:`remote_cache_enabled` is False for UPSERT and the entire
        # remote cache flow short-circuits — no lookup query, no writeback
        # insert. This pins that contract: a caller who wants UPSERT today
        # gets *no* cache activity (not "always refetch + write back").
        tab = _FakeRemoteTabular()
        cfg = _remote_cfg(tab, mode=Mode.UPSERT)
        seed_req = make_request("https://example.com/x")
        tab.rows.append(
            make_response(request=seed_req, body=b'{"v":"old"}').to_arrow_batch(parse=False)
        )

        s = StubSession()
        s.queue(make_response(request=seed_req, body=b'{"v":"fresh"}'))
        out = s.send(seed_req, remote_cache=cfg)

        assert len(s.calls) == 1, "UPSERT must always go to the network"
        assert out.json() == {"v": "fresh"}
        assert tab.queries == [], "UPSERT must not issue a lookup query"
        assert tab.inserts == [], "UPSERT short-circuits the writeback too"
