"""Tests for :meth:`YGGFolderIO.cleanup_stale` / ``cleanup_stale_once``.

The cleanup feature lives on the :class:`YGGFolderIO` protocol so
every consumer (response cache, future Hive-partitioned caches)
inherits it. Coverage:

* old ``part-{epoch_ms}-{seed}.{ext}`` files get unlinked,
* recent files and the ``.ygg/`` sidecar (schema, sentinel) are
  left alone,
* the ``.ygg/.last_cleanup`` sentinel throttles repeat sweeps
  across processes (mtime within TTL → skip),
* the in-process done-set throttles repeats within a Python run,
* :meth:`CacheConfig.local_cache` triggers the sweep end-to-end.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.primitive import Int64Type
from yggdrasil.io.nested import ygg_folder_io
from yggdrasil.io.nested.ygg_folder_io import (
    _CLEANUP_SENTINEL_FILENAME,
    _DEFAULT_CLEANUP_TTL_SECONDS,
    YGGFolderIO,
)
from yggdrasil.io.path import LocalPath
from yggdrasil.io.send_config import CacheConfig


@pytest.fixture(autouse=True)
def _reset_cleanup_state():
    ygg_folder_io._CLEANUP_DONE.clear()
    yield
    ygg_folder_io._CLEANUP_DONE.clear()


def _make_part(root: Path, epoch_ms: int, seed: str = "abcd", ext: str = "parquet") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"part-{epoch_ms}-{seed}.{ext}"
    p.write_bytes(b"x")
    return p


def _folderio(root: Path, *, partitioned: bool = False) -> YGGFolderIO:
    schema = None
    if partitioned:
        schema = Schema()
        schema.with_field(
            Field(name="partition_key", dtype=Int64Type()).with_partition_by(True)
        )
    return YGGFolderIO(path=LocalPath(root), schema=schema)


def test_cleanup_stale_unlinks_old_parts_keeps_fresh(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    old_ms = now_ms - int((_DEFAULT_CLEANUP_TTL_SECONDS + 60) * 1000)
    fresh_ms = now_ms - 60_000

    leaf = tmp_path / "partition_key=1"
    old = _make_part(leaf, old_ms)
    fresh = _make_part(leaf, fresh_ms)

    folder = _folderio(tmp_path, partitioned=True)
    deleted = folder.cleanup_stale()

    assert deleted == 1
    assert not old.exists()
    assert fresh.exists()


def test_cleanup_stale_walks_unpartitioned_root(tmp_path: Path) -> None:
    old_ms = int((time.time() - _DEFAULT_CLEANUP_TTL_SECONDS - 60) * 1000)
    old = _make_part(tmp_path, old_ms)

    folder = _folderio(tmp_path, partitioned=False)
    folder.cleanup_stale()

    assert not old.exists()


def test_cleanup_stale_ignores_ygg_sidecar_and_unrelated_names(tmp_path: Path) -> None:
    (tmp_path / ".ygg").mkdir()
    schema = tmp_path / ".ygg" / ".schema"
    schema.write_bytes(b"schema-bytes")
    plain = tmp_path / "README.md"
    plain.write_text("hi")

    old_ms = int((time.time() - _DEFAULT_CLEANUP_TTL_SECONDS - 60) * 1000)
    _make_part(tmp_path / "partition_key=1", old_ms)

    folder = _folderio(tmp_path, partitioned=True)
    folder.cleanup_stale()

    assert schema.exists()
    assert plain.exists()


def test_cleanup_stale_once_writes_sentinel(tmp_path: Path) -> None:
    _make_part(tmp_path, int(time.time() * 1000) - 60_000)

    folder = _folderio(tmp_path, partitioned=False)
    folder.cleanup_stale_once()

    sentinel = tmp_path / ".ygg" / _CLEANUP_SENTINEL_FILENAME
    assert sentinel.is_file()


def test_cleanup_stale_once_skipped_when_sentinel_fresh(tmp_path: Path) -> None:
    (tmp_path / ".ygg").mkdir()
    sentinel = tmp_path / ".ygg" / _CLEANUP_SENTINEL_FILENAME
    sentinel.write_text("0")
    now = time.time()
    os.utime(sentinel, (now, now))

    old_ms = int((time.time() - _DEFAULT_CLEANUP_TTL_SECONDS - 600) * 1000)
    old = _make_part(tmp_path, old_ms)

    folder = _folderio(tmp_path, partitioned=False)
    deleted = folder.cleanup_stale_once()

    assert deleted == 0
    assert old.exists()


def test_cleanup_stale_once_in_process_dedup(tmp_path: Path) -> None:
    old_ms = int((time.time() - _DEFAULT_CLEANUP_TTL_SECONDS - 600) * 1000)
    _make_part(tmp_path, old_ms, seed="aaaa")

    folder = _folderio(tmp_path, partitioned=False)
    folder.cleanup_stale_once()

    # Second call must be a no-op even if a freshly-stale file
    # appears mid-process.
    new_old = _make_part(tmp_path, old_ms, seed="bbbb")
    deleted = folder.cleanup_stale_once()

    assert deleted == 0
    assert new_old.exists()


def test_cache_config_local_cache_triggers_sweep(tmp_path: Path) -> None:
    old_ms = int((time.time() - _DEFAULT_CLEANUP_TTL_SECONDS - 60) * 1000)
    old = _make_part(tmp_path / "partition_key=1", old_ms)

    cfg = CacheConfig.check_arg(tmp_path)
    cfg.local_cache(session=None)

    assert not old.exists()
    assert (tmp_path / ".ygg" / _CLEANUP_SENTINEL_FILENAME).is_file()


def test_cleanup_stale_on_missing_path_is_a_no_op(tmp_path: Path) -> None:
    folder = _folderio(tmp_path / "does-not-exist", partitioned=False)
    assert folder.cleanup_stale() == 0
    assert folder.cleanup_stale_once() == 0
