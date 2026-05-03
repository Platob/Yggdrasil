"""Unit tests for :mod:`yggdrasil.io.local_response_cache`.

These cover the partitioned-folder layout, lookup tie-break on
:attr:`response_received_at`, the received-window filter, and bulk
store / lookup so the session-side wiring can stay focused on its
own integration tests.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.local_response_cache import (
    DEFAULT_LOCAL_CACHE_FOLDER,
    DEFAULT_LOCAL_CACHE_MATCH_BY,
    DEFAULT_LOCAL_CACHE_PARTITIONS,
    LocalResponseCache,
)
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response


def _make_response(
    url: str,
    *,
    method: str = "GET",
    received_at: dt.datetime | None = None,
    status: int = 200,
) -> Response:
    request = PreparedRequest.prepare(method=method, url=url)
    return Response(
        request=request,
        status_code=status,
        headers={},
        tags={},
        buffer=BytesIO(b""),
        received_at=received_at or dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
    )


class TestLocalResponseCacheLayout:
    def test_defaults_match_documented_constants(self, tmp_path):
        # The exported defaults are part of the public contract —
        # callers tune off them. Pin both the constants and the
        # constructor's behaviour against drift.
        cache = LocalResponseCache(path=tmp_path)
        assert cache.partition_columns == DEFAULT_LOCAL_CACHE_PARTITIONS
        assert cache.match_by == DEFAULT_LOCAL_CACHE_MATCH_BY
        assert cache.folder_name == DEFAULT_LOCAL_CACHE_FOLDER
        # Folder path is rooted at the cache root + folder_name so
        # the configured root can host sibling artefacts.
        assert cache.folder_path == Path(str(tmp_path)) / DEFAULT_LOCAL_CACHE_FOLDER

    def test_store_lays_down_hive_partitioned_directories(self, tmp_path):
        # Stored responses land under
        # ``cache/request_method=.../request_url_host=.../<leaf>``
        # — Hive convention, partition columns stripped from the
        # leaf payload.
        cache = LocalResponseCache(path=tmp_path)
        cache.store(_make_response("https://api.example.com/foo"))

        leaves = [p for p in cache.folder_path.rglob("*") if p.is_file()]
        assert len(leaves) == 1
        rel = leaves[0].relative_to(cache.folder_path)
        assert rel.parts[0] == "request_method=GET"
        assert rel.parts[1] == "request_url_host=api.example.com"
        # Leaf is a UUID-named ``part-<hex>.<ext>`` file (avoids
        # the FolderIO sequential ``part-NNNNN`` race the legacy
        # path was vulnerable to). Extension comes from the Arrow
        # IPC mime-type's primary extension.
        leaf_name = rel.parts[-1]
        assert leaf_name.startswith("part-")
        assert leaf_name.endswith((".ipc", ".arrow", ".feather"))

    def test_distinct_partitions_split_onto_distinct_subtrees(self, tmp_path):
        cache = LocalResponseCache(path=tmp_path)
        cache.store_many([
            _make_response("https://a.example.com/1", method="GET"),
            _make_response("https://b.example.com/1", method="GET"),
            _make_response("https://a.example.com/2", method="POST"),
        ])

        rel_dirs = sorted({
            "/".join(p.relative_to(cache.folder_path).parts[:-1])
            for p in cache.folder_path.rglob("*")
            if p.is_file()
        })
        assert rel_dirs == [
            "request_method=GET/request_url_host=a.example.com",
            "request_method=GET/request_url_host=b.example.com",
            "request_method=POST/request_url_host=a.example.com",
        ]


class TestLocalResponseCacheLookup:
    def test_lookup_round_trips_a_stored_response(self, tmp_path):
        cache = LocalResponseCache(path=tmp_path)
        request = PreparedRequest.prepare(
            method="GET", url="https://api.example.com/round-trip",
        )
        cache.store(
            Response(
                request=request,
                status_code=200,
                headers={},
                tags={},
                buffer=BytesIO(b""),
                received_at=dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
            )
        )

        loaded = cache.lookup(request)
        assert loaded is not None
        assert str(loaded.request.url) == "https://api.example.com/round-trip"
        assert loaded.status_code == 200

    def test_lookup_returns_none_when_no_match(self, tmp_path):
        cache = LocalResponseCache(path=tmp_path)
        cache.store(_make_response("https://api.example.com/stored"))

        miss = PreparedRequest.prepare(
            method="GET", url="https://api.example.com/never-cached",
        )
        assert cache.lookup(miss) is None

    def test_lookup_many_picks_latest_on_duplicate_keys(self, tmp_path):
        # When several stored rows share a request key (UPSERT-style
        # repeated fetches), lookup_many returns the one with the
        # max ``response_received_at`` — so callers don't have to
        # special-case eviction; fresh writes win on read.
        cache = LocalResponseCache(path=tmp_path)
        url = "https://api.example.com/duplicate"
        cache.store(
            _make_response(
                url,
                received_at=dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
                status=500,
            )
        )
        cache.store(
            _make_response(
                url,
                received_at=dt.datetime(2025, 6, 1, tzinfo=dt.timezone.utc),
                status=200,
            )
        )

        # The 500 response was filtered out at store-time (only ok
        # responses persist) — confirm the surviving 200 wins. Also
        # store an ok-but-older row so the tie-break is exercised
        # against equally-valid candidates.
        cache.store(
            _make_response(
                url,
                received_at=dt.datetime(2024, 12, 1, tzinfo=dt.timezone.utc),
                status=200,
            )
        )
        latest = cache.lookup(PreparedRequest.prepare(method="GET", url=url))
        assert latest is not None
        # The June 2025 row is the most recent ok store.
        assert latest.received_at == dt.datetime(
            2025, 6, 1, tzinfo=dt.timezone.utc,
        )

    def test_received_window_filters_out_stale_rows(self, tmp_path):
        # ``received_from`` / ``received_to`` apply on every read.
        # A row outside the window is treated as if it were never
        # stored, even though the leaf still holds it.
        early = _make_response(
            "https://api.example.com/old",
            received_at=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        )
        cache = LocalResponseCache(path=tmp_path)
        cache.store(early)

        windowed = LocalResponseCache(
            path=tmp_path,
            received_from=dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc),
        )
        assert windowed.lookup(early.request) is None
        assert windowed.count() == 0
        # Without a window, the row is visible — confirms the
        # filter is the only thing hiding it.
        assert cache.count() == 1


class TestLocalResponseCacheBulk:
    def test_store_many_skips_non_ok_responses(self, tmp_path):
        cache = LocalResponseCache(path=tmp_path)
        ok = _make_response("https://api.example.com/ok", status=200)
        bad = _make_response("https://api.example.com/bad", status=500)
        cache.store_many([ok, bad])
        assert cache.count() == 1
        assert cache.lookup(ok.request) is not None
        assert cache.lookup(bad.request) is None

    def test_store_many_groups_by_partition_in_one_call(self, tmp_path):
        cache = LocalResponseCache(path=tmp_path)
        cache.store_many([
            _make_response("https://x.example.com/1"),
            _make_response("https://x.example.com/2"),
            _make_response("https://y.example.com/3"),
        ])
        # Two distinct (method, host) partitions → exactly two leaf
        # files (one per partition tuple). The legacy per-file
        # layout would have produced three leaves; the new layout
        # batches per partition.
        leaves = [p for p in cache.folder_path.rglob("*") if p.is_file()]
        assert len(leaves) == 2
        assert cache.count() == 3

    def test_lookup_many_returns_keyed_dict(self, tmp_path):
        cache = LocalResponseCache(path=tmp_path)
        urls = [
            "https://api.example.com/a",
            "https://api.example.com/b",
            "https://api.example.com/c",
        ]
        cache.store_many([_make_response(u) for u in urls])
        miss_url = "https://api.example.com/missing"

        requests = [PreparedRequest.prepare(method="GET", url=u) for u in urls]
        miss_request = PreparedRequest.prepare(method="GET", url=miss_url)
        results = cache.lookup_many(requests + [miss_request])

        # Stored URLs hit; the missing one is absent (not present
        # with a None value — keeps the API simple to consume).
        for u, r in zip(urls, requests):
            key = cache._request_key(r)
            assert key in results
            assert str(results[key].request.url) == u
        assert cache._request_key(miss_request) not in results

    def test_clear_wipes_the_subtree(self, tmp_path):
        cache = LocalResponseCache(path=tmp_path)
        cache.store(_make_response("https://api.example.com/clearable"))
        assert cache.count() == 1
        cache.clear()
        assert cache.count() == 0
        assert not cache.folder_path.exists()
