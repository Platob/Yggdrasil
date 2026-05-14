"""Integration tests for the :class:`Session` cache pipeline.

Drives the staged ``send_many`` pipeline end-to-end with a
:class:`StubSession` transport and the URL-mirrored fast-path
local cache (one ``.arrow`` per ``public_hash`` under the cache
root), plus a hand-rolled fake remote :class:`Tabular` for the
remote-cache flow.

Coverage:

* ``Session._remote_write_group_key`` actually splits responses by
  every dimension that affects the insert call (table, mode,
  match_by, wait, anonymize) — collapsing on any one of those would
  silently drop per-request config divergence on the floor.
* ``Session._split_local_cache`` groups by the effective config's
  resolved ``path``, so a per-request override pointing at a
  different cache folder lands in its own bucket instead of bleeding
  into the session-level one.
* ``Session.send_many`` end-to-end with a mix of cache hits and
  misses: hits skip the network, the writeback persists the misses,
  and a re-run of the same batch reads everything from disk.
* Per-request ``local_cache_config`` override survives the batch
  pipeline (a request pointing at a fresh empty folder must miss
  the pre-seeded session cache).
* ``received_from`` / ``received_to`` window rejection: a fast-path
  row outside the window is treated as a miss.
* The fast-path ``public_hash`` keys two POSTs with distinct bodies
  to distinct files so they can't alias each other through the
  cache.
* ``_cleanup_local_fast_path`` unlinks ``.arrow`` files older than
  the configured TTL and is throttled by an in-tree sentinel.
* Remote-cache integration (no Databricks required): fake
  :class:`Tabular` with ``sql.execute`` returning seeded Arrow rows
  exercises ``_load_remote_cached_response``,
  ``_store_remote_cached_response``, the
  ``TABLE_OR_VIEW_NOT_FOUND`` recovery, and the
  ``mirror_local_to_remote`` writeback.
"""
from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pytest

from yggdrasil.data.enums import Mode
from yggdrasil.io.response import Response
from yggdrasil.io.send_config import CacheConfig
from yggdrasil.io.session import (
    Session,
    _FAST_PATH_SEGMENT_MAX_BYTES,
    _cleanup_local_fast_path,
    _local_fast_path_relative,
    _safe_fast_path_segment,
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
    overrides.setdefault("mode", Mode.APPEND)
    return CacheConfig(path=str(tmp_path), **overrides)


def _wait_for_readable(cache: CacheConfig, *, timeout: float = 3.0) -> bool:
    """Poll until the URL-mirrored fast-path tree under the cache root holds an entry.

    The session writes responses via a fire-and-forget Job, so the
    file may not be on disk immediately after :meth:`Session.send`
    returns. We watch for any non-hidden ``.arrow`` file anywhere
    under the cache root.
    """
    from pathlib import Path as _P

    deadline = time.monotonic() + timeout
    root = _P(cache.path)
    while time.monotonic() < deadline:
        try:
            if root.exists() and any(
                p.is_file() and not p.name.startswith(".")
                for p in root.rglob("*.arrow")
            ):
                return True
        except OSError:
            pass
        time.sleep(0.05)
    return False


def _seed(cache: CacheConfig, response: Response) -> None:
    """Synchronously write a response into the cache, no fire-and-forget race."""
    from yggdrasil.io.session import (
        _local_fast_path_relative,
        _store_fast_path_arrow_batch,
    )

    req = response.request
    rel = _local_fast_path_relative(req.method, req.url, req.public_hash)
    _store_fast_path_arrow_batch(
        cache.path, rel, response.to_arrow_batch(parse=False),
    )


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
# Fast-path URL-mirrored layout (_local_fast_path_relative)
# ---------------------------------------------------------------------------


class TestFastPathLocalLayout:

    def test_layout_mirrors_method_host_and_path(self) -> None:
        req = make_request("https://api.example.com/v1/users/42", method="GET")
        rel = _local_fast_path_relative(req.method, req.url, req.public_hash)
        parts = rel.split(os.sep)
        # Expected: GET / api.example.com / v1 / users / 42 / <16hex>.arrow
        assert parts[:5] == ["GET", "api.example.com", "v1", "users", "42"]
        assert parts[-1].endswith(".arrow")
        assert len(parts[-1]) == len("0123456789abcdef.arrow")

    def test_leaf_filename_is_public_hash_hex(self) -> None:
        req = make_request("https://example.com/x")
        rel = _local_fast_path_relative(req.method, req.url, req.public_hash)
        leaf = rel.rsplit(os.sep, 1)[-1]
        expected = f"{req.public_hash & 0xFFFFFFFFFFFFFFFF:016x}.arrow"
        assert leaf == expected

    def test_root_path_yields_method_and_host_only(self) -> None:
        req = make_request("https://example.com/", method="POST")
        rel = _local_fast_path_relative(req.method, req.url, req.public_hash)
        parts = rel.split(os.sep)
        # No real path segments between host and the .arrow leaf.
        assert parts[0] == "POST"
        assert parts[1] == "example.com"
        assert parts[-1].endswith(".arrow")
        assert len(parts) == 3

    def test_distinct_paths_land_in_distinct_dirs(self) -> None:
        a = make_request("https://example.com/api/users")
        b = make_request("https://example.com/api/orders")
        rel_a = _local_fast_path_relative(a.method, a.url, a.public_hash)
        rel_b = _local_fast_path_relative(b.method, b.url, b.public_hash)
        assert rel_a.rsplit(os.sep, 1)[0] != rel_b.rsplit(os.sep, 1)[0]

    def test_same_path_different_query_share_dir_not_file(self) -> None:
        a = make_request("https://example.com/api/items?id=1")
        b = make_request("https://example.com/api/items?id=2")
        rel_a = _local_fast_path_relative(a.method, a.url, a.public_hash)
        rel_b = _local_fast_path_relative(b.method, b.url, b.public_hash)
        # Same directory tree (URL path mirrors), distinct leaf files
        # because public_hash mixes query string in.
        assert rel_a.rsplit(os.sep, 1)[0] == rel_b.rsplit(os.sep, 1)[0]
        assert rel_a != rel_b

    def test_long_segment_is_hashed(self) -> None:
        long_seg = "x" * 256
        req = make_request(f"https://example.com/api/{long_seg}/end")
        rel = _local_fast_path_relative(req.method, req.url, req.public_hash)
        parts = rel.split(os.sep)
        # The "api", "end", and method/host parts stay short; the rogue
        # segment should be folded under the per-segment byte cap.
        for p in parts:
            assert len(p.encode("utf-8")) <= max(
                _FAST_PATH_SEGMENT_MAX_BYTES, len("0123456789abcdef.arrow"),
            )
        # And the folded segment must still distinguish two long but
        # different tokens at the same position.
        other = make_request(f"https://example.com/api/{'y' * 256}/end")
        rel_other = _local_fast_path_relative(
            other.method, other.url, other.public_hash,
        )
        assert rel != rel_other

    def test_unsafe_chars_are_replaced(self) -> None:
        # Backslashes and reserved chars should never appear as path
        # separators in the result — they must be sanitized.
        out = _safe_fast_path_segment('a\\b:c*d?e"f<g>h|i')
        assert "\\" not in out
        assert ":" not in out
        assert "*" not in out
        assert "?" not in out
        assert '"' not in out
        assert "<" not in out
        assert ">" not in out
        assert "|" not in out

    def test_empty_segment_normalizes_to_placeholder(self) -> None:
        # ``""`` and ``"   "`` would otherwise produce an empty
        # directory name; the helper has to fall back to a sentinel.
        assert _safe_fast_path_segment("") == "_"
        assert _safe_fast_path_segment("   ") in {"_", " "}  # rstrip→empty→"_"

    def test_method_defaults_when_missing(self) -> None:
        # Defensive: a request without an explicit method should still
        # produce a valid relative path (the leaf hex tells uniqueness).
        url = make_request("https://example.com/x").url
        rel = _local_fast_path_relative(None, url, 0xDEADBEEF)
        assert rel.split(os.sep)[0] == "GET"

    def test_nul_and_control_chars_are_replaced(self) -> None:
        # NUL bytes and ASCII control characters (0x00-0x1f) are illegal
        # in directory names on every supported OS; they must be folded.
        for bad in ("\x00", "\x01", "\t", "\n", "\r", "\x1f"):
            out = _safe_fast_path_segment(f"seg{bad}value")
            assert "\x00" not in out
            assert bad not in out

    def test_unicode_segment_within_limit_passes_through(self) -> None:
        # Short unicode segment (fits within max_bytes in UTF-8) → kept as-is.
        seg = "café"
        out = _safe_fast_path_segment(seg)
        assert out == seg

    def test_long_unicode_segment_is_folded_on_byte_boundary(self) -> None:
        # A 3-byte-per-char CJK string that overflows max_bytes must be
        # truncated on a valid UTF-8 character boundary (no half-char).
        seg = "数" * 100  # 100 × 3 = 300 bytes — well over 80
        out = _safe_fast_path_segment(seg)
        # Result must be valid UTF-8 (no truncated multi-byte sequence).
        out.encode("utf-8")
        assert len(out.encode("utf-8")) <= _FAST_PATH_SEGMENT_MAX_BYTES + 17  # +17 for -<16-hex>

    def test_same_input_same_output_lru_cache(self) -> None:
        # The LRU cache must return identical results for repeated inputs.
        seg = "api-v2-endpoint"
        r1 = _safe_fast_path_segment(seg)
        r2 = _safe_fast_path_segment(seg)
        assert r1 == r2
        assert r1 is r2  # same object — cache hit, not a new str

    def test_different_long_inputs_produce_different_outputs(self) -> None:
        # Two distinct long tokens that collide on prefix must still produce
        # distinct outputs (the xxh3 digest restores uniqueness).
        prefix = "x" * 90  # longer than max_bytes
        a = _safe_fast_path_segment(prefix + "A")
        b = _safe_fast_path_segment(prefix + "B")
        assert a != b


# ---------------------------------------------------------------------------
# Fast-path TTL cleanup walker (_cleanup_local_fast_path)
# ---------------------------------------------------------------------------


class TestCleanupLocalFastPath:

    def _write_arrow(self, path: Path, mtime_offset: float = 0.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Minimal Arrow IPC payload — just an empty batch, contents
        # don't matter for the cleanup walker.
        sink = pa.BufferOutputStream()
        schema = pa.schema([("x", pa.int64())])
        with pa.ipc.new_stream(sink, schema):
            pass
        path.write_bytes(sink.getvalue().to_pybytes())
        if mtime_offset:
            now = time.time()
            os.utime(path, (now + mtime_offset, now + mtime_offset))

    def test_unlinks_files_older_than_ttl(self, tmp_path) -> None:
        old = tmp_path / "GET" / "host" / "stale.arrow"
        fresh = tmp_path / "GET" / "host" / "fresh.arrow"
        # Backdate ``old`` so its mtime sits an hour outside the TTL.
        self._write_arrow(old, mtime_offset=-3600)
        self._write_arrow(fresh)

        removed = _cleanup_local_fast_path(
            str(tmp_path), ttl_seconds=60.0, throttle_seconds=0.0,
        )
        assert removed == 1
        assert not old.exists()
        assert fresh.exists()

    def test_throttled_by_sentinel(self, tmp_path) -> None:
        # Two back-to-back calls inside the throttle window: the
        # second must short-circuit (return 0) even though there's a
        # fresh stale file to unlink.
        first_stale = tmp_path / "GET" / "host" / "a.arrow"
        self._write_arrow(first_stale, mtime_offset=-3600)
        first_removed = _cleanup_local_fast_path(
            str(tmp_path), ttl_seconds=60.0, throttle_seconds=60.0,
        )
        assert first_removed == 1

        second_stale = tmp_path / "GET" / "host" / "b.arrow"
        self._write_arrow(second_stale, mtime_offset=-3600)
        second_removed = _cleanup_local_fast_path(
            str(tmp_path), ttl_seconds=60.0, throttle_seconds=60.0,
        )
        # Throttled: walker doesn't even look, so b.arrow stays.
        assert second_removed == 0
        assert second_stale.exists()

    def test_missing_root_is_no_op(self, tmp_path) -> None:
        ghost = tmp_path / "no-such-dir"
        assert _cleanup_local_fast_path(str(ghost), ttl_seconds=60.0) == 0

    def test_skips_hidden_tmp_files(self, tmp_path) -> None:
        # Tmp files written by _store_fast_path_arrow_batch start with
        # a dot — the cleanup walker must not touch them or it'd race
        # with concurrent writes.
        live = tmp_path / "GET" / "host" / ".x.tmp.arrow"
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_bytes(b"in-flight")
        os.utime(live, (time.time() - 3600, time.time() - 3600))

        removed = _cleanup_local_fast_path(
            str(tmp_path), ttl_seconds=60.0, throttle_seconds=0.0,
        )
        assert removed == 0
        assert live.exists()


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

    def test_post_body_distinct_bodies_dont_alias(self, tmp_path) -> None:
        # Fast-path files are keyed by ``public_hash`` (xxh3_64 over
        # method + anonymized URL + headers + body), so two POSTs to
        # the same URL with different bodies land at distinct
        # ``.arrow`` files. The seeded body must not be served to
        # the request that carries different bytes.
        cache = _local_cache(tmp_path)
        url = "https://example.com/echo"
        req_a = make_request(url, method="POST", body=b"payload-A")
        req_b = make_request(url, method="POST", body=b"payload-B")
        _seed(cache, make_response(request=req_a, body=b'{"got":"A"}'))

        s = StubSession()
        s.queue(make_response(request=req_b, body=b'{"got":"B"}'))

        out = list(s.send_many(iter([req_a, req_b]), local_cache=cache))
        # req_a hits the seeded file, req_b misses and goes to network.
        assert len(s.calls) == 1
        assert s.calls[0].buffer.to_bytes() == b"payload-B"
        assert {r.json()["got"] for r in out} == {"A", "B"}


# ---------------------------------------------------------------------------
# Concurrent local-cache writeback
# ---------------------------------------------------------------------------


class TestConcurrentWriteback:

    def test_send_many_writeback_eventually_consistent(self, tmp_path) -> None:
        # Many simultaneous writes against distinct request identities
        # must not race — each public_hash maps to a unique fast-path
        # filename so concurrent fire-and-forget workers can't trample
        # each other. Polling the cache for the expected file count is
        # the reliable way to wait for the daemon writers.
        cache = _local_cache(tmp_path)
        s = StubSession()
        n = 8
        reqs = [make_request(f"https://example.com/p{i}") for i in range(n)]
        s.queue(*[
            make_response(request=r, body=f'{{"i":{i}}}'.encode())
            for i, r in enumerate(reqs)
        ])
        list(s.send_many(iter(reqs), local_cache=cache))

        deadline = time.monotonic() + 5.0
        root = Path(cache.path)
        last_count = 0
        while time.monotonic() < deadline:
            try:
                last_count = sum(
                    1 for p in root.rglob("*.arrow")
                    if p.is_file() and not p.name.startswith(".")
                )
            except OSError:
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
    return them. The CacheConfig holding it satisfies
    :meth:`CacheConfig.is_remote` so :meth:`remote_cache_enabled`
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
