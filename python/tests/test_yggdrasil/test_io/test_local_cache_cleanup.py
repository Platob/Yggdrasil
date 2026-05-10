"""Tests for the local response-cache stale-file sweep.

Exercises :func:`yggdrasil.io.send_config._clean_local_cache_once`
and its hookup through :meth:`CacheConfig.local_cache`:

* old ``part-{epoch_ms}-{seed}.{ext}`` files get unlinked,
* recent files and unrelated names (including the ``.ygg/``
  sidecar) are left alone,
* the ``.ygg/.last_cleanup`` sentinel throttles repeat sweeps
  across processes (mtime within 1 day → skip),
* the in-process done-set throttles repeats within a Python run.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from yggdrasil.io import send_config
from yggdrasil.io.send_config import (
    _LOCAL_CACHE_SENTINEL_NAME,
    _LOCAL_CACHE_TTL_SECONDS,
    CacheConfig,
    _clean_local_cache_once,
)


@pytest.fixture(autouse=True)
def _reset_cleanup_state():
    send_config._LOCAL_CACHE_CLEANUP_DONE.clear()
    yield
    send_config._LOCAL_CACHE_CLEANUP_DONE.clear()


def _make_part(root: Path, epoch_ms: int, seed: str = "abcd", ext: str = "parquet") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"part-{epoch_ms}-{seed}.{ext}"
    p.write_bytes(b"x")
    return p


def test_unlinks_old_part_files_keeps_fresh_ones(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    old_ms = now_ms - int((_LOCAL_CACHE_TTL_SECONDS + 60) * 1000)
    fresh_ms = now_ms - 60_000  # 1 minute ago

    leaf = tmp_path / "partition_key=1"
    old = _make_part(leaf, old_ms)
    fresh = _make_part(leaf, fresh_ms)

    _clean_local_cache_once(tmp_path)

    assert not old.exists()
    assert fresh.exists()


def test_writes_sentinel_after_sweep(tmp_path: Path) -> None:
    _make_part(tmp_path / "p", int(time.time() * 1000) - 60_000)

    _clean_local_cache_once(tmp_path)

    sentinel = tmp_path / ".ygg" / _LOCAL_CACHE_SENTINEL_NAME
    assert sentinel.is_file()


def test_sentinel_skips_repeat_sweep_across_processes(tmp_path: Path) -> None:
    # Pre-create a fresh sentinel — emulates a sibling process that
    # already swept within the TTL window.
    (tmp_path / ".ygg").mkdir(parents=True)
    sentinel = tmp_path / ".ygg" / _LOCAL_CACHE_SENTINEL_NAME
    sentinel.write_text("0")
    # mtime = now → well within TTL.
    now = time.time()
    import os
    os.utime(sentinel, (now, now))

    old_ms = int((time.time() - _LOCAL_CACHE_TTL_SECONDS - 600) * 1000)
    old = _make_part(tmp_path / "p", old_ms)

    _clean_local_cache_once(tmp_path)

    # Sentinel was fresh → sweep skipped → old file remains.
    assert old.exists()


def test_in_process_done_set_throttles_repeat_calls(tmp_path: Path) -> None:
    old_ms = int((time.time() - _LOCAL_CACHE_TTL_SECONDS - 600) * 1000)
    leaf = tmp_path / "p"
    _make_part(leaf, old_ms, seed="aaaa")

    _clean_local_cache_once(tmp_path)
    # First call swept and updated the done-set; second call must
    # be a no-op even if a freshly-stale file appears mid-process.
    new_old = _make_part(leaf, old_ms, seed="bbbb")
    _clean_local_cache_once(tmp_path)

    assert new_old.exists()


def test_ignores_non_part_files_and_ygg_sidecar(tmp_path: Path) -> None:
    (tmp_path / ".ygg").mkdir(parents=True)
    schema = tmp_path / ".ygg" / ".schema"
    schema.write_bytes(b"schema-bytes")
    plain = tmp_path / "README.md"
    plain.write_text("hi")

    old_ms = int((time.time() - _LOCAL_CACHE_TTL_SECONDS - 60) * 1000)
    _make_part(tmp_path / "p", old_ms)

    _clean_local_cache_once(tmp_path)

    assert schema.exists()
    assert plain.exists()


def test_local_cache_triggers_sweep(tmp_path: Path) -> None:
    old_ms = int((time.time() - _LOCAL_CACHE_TTL_SECONDS - 60) * 1000)
    old = _make_part(tmp_path / "partition_key=1", old_ms)

    cfg = CacheConfig.check_arg(tmp_path)
    cfg.local_cache(session=None)

    assert not old.exists()
    assert (tmp_path / ".ygg" / _LOCAL_CACHE_SENTINEL_NAME).is_file()


def test_missing_root_is_a_no_op(tmp_path: Path) -> None:
    # Should not raise even when the directory has never been
    # created (e.g. cache hasn't received a single write yet).
    _clean_local_cache_once(tmp_path / "does-not-exist")
