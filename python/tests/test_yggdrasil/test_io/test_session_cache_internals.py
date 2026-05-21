"""Unit tests for the private helpers behind the session cache.

These pin the contracts of the small, side-effect-free helpers the
:class:`Session` pipeline relies on:

* ``_safe_fast_path_segment`` / ``_local_fast_path_relative`` — the
  URL-mirrored on-disk layout used by the local fast-path cache.
* ``_cleanup_local_fast_path`` — TTL-based cleanup walker, throttled
  by an in-tree sentinel.
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
    _FAST_PATH_SEGMENT_MAX_BYTES,
    _cleanup_local_fast_path,
    _local_fast_path_relative,
    _maybe_autocompress_body_for_cache,
    _read_fast_path_arrow_batch,
    _safe_fast_path_segment,
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
# Cache helpers accept any path-like root (str / pathlib / yggdrasil Path)
# ---------------------------------------------------------------------------


class TestCacheRootShapes:
    """The store / read / cleanup helpers all route their ``cache_root``
    argument through :func:`Path.from_` — so any shape :meth:`Path.from_`
    accepts works: ``str`` (back-compat with the old string-only call
    sites), :class:`pathlib.Path` (handy in tests), or a live abstract
    :class:`yggdrasil.io.path.Path` (any backend — LocalPath today,
    VolumePath / S3Path / … without touching the helpers when those
    integrations land their fast-path glue).
    """

    def _batch(self) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays(
            [pa.array([1, 2, 3])],
            schema=pa.schema([("x", pa.int64())]),
        )

    def _rel(self) -> str:
        return os.sep.join(["GET", "example.com", "v1", "0123456789abcdef.arrow"])

    def test_str_root_roundtrip(self, tmp_path) -> None:
        from yggdrasil.io.session import _store_fast_path_arrow_batch

        rel = self._rel()
        _store_fast_path_arrow_batch(str(tmp_path), rel, self._batch())
        out = _read_fast_path_arrow_batch(str(tmp_path), rel)
        assert out is not None
        assert out.num_rows == 3

    def test_pathlib_root_roundtrip(self, tmp_path) -> None:
        from yggdrasil.io.session import _store_fast_path_arrow_batch

        rel = self._rel()
        _store_fast_path_arrow_batch(tmp_path, rel, self._batch())
        out = _read_fast_path_arrow_batch(tmp_path, rel)
        assert out is not None
        assert out.num_rows == 3

    def test_abstract_path_root_roundtrip(self, tmp_path) -> None:
        from yggdrasil.io.path import LocalPath
        from yggdrasil.io.session import _store_fast_path_arrow_batch

        root = LocalPath(str(tmp_path))
        rel = self._rel()
        _store_fast_path_arrow_batch(root, rel, self._batch())
        out = _read_fast_path_arrow_batch(root, rel)
        assert out is not None
        assert out.num_rows == 3

    def test_cleanup_accepts_abstract_path(self, tmp_path) -> None:
        from yggdrasil.io.path import LocalPath
        from yggdrasil.io.session import _store_fast_path_arrow_batch

        root = LocalPath(str(tmp_path))
        rel = self._rel()
        _store_fast_path_arrow_batch(root, rel, self._batch())
        # Backdate the entry so the TTL sweep sees it as stale.
        stored = tmp_path / rel
        os.utime(stored, (time.time() - 3600, time.time() - 3600))
        removed = _cleanup_local_fast_path(
            root, ttl_seconds=60.0, throttle_seconds=0.0,
        )
        assert removed == 1
        assert not stored.exists()


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
