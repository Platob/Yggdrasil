# tests/test_path.py
from __future__ import annotations

import io
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

from yggdrasil.enums import SaveMode
from yggdrasil.enums.io.file_format import FileFormat
from yggdrasil.io.path import LocalDataPath, _ensure_bytes, _rand_str


# -------------------------
# helpers
# -------------------------
def _mk_tree(root: LocalDataPath, mapping: dict[str, bytes]) -> None:
    for rel, data in mapping.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def _tree_bytes(root: LocalDataPath) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for f in root.ls(recursive=True, allow_not_found=False):
        if f.is_file():
            rel = os.path.relpath(str(f), start=str(root))
            out[rel] = f.read_bytes()
    return out

def test_rand_str_len_and_charset():
    s = _rand_str(32)
    assert len(s) == 32
    assert re.fullmatch(r"[A-Za-z0-9]{32}", s)


@pytest.mark.parametrize(
    "buf",
    [
        b"abc",
        bytearray(b"abc"),
        memoryview(b"abc"),
        io.BytesIO(b"abc"),
    ],
)
def test_ensure_bytes(buf):
    assert _ensure_bytes(buf) == b"abc"


# -------------------------
# LocalDataPath basics
# -------------------------

def test_localdatapath_open_returns_python_file_handle(tmp_path):
    p = LocalDataPath(tmp_path / "x.txt")
    with p.open("w", encoding="utf-8") as f:
        # Should be a normal Python file object, not a custom wrapper
        f.write("yo")

    with p.open("r", encoding="utf-8") as f:
        assert f.read() == "yo"


def test_extension_and_file_format_inference(tmp_path):
    p = LocalDataPath(tmp_path / "data.parquet")
    assert p.extension == "parquet"
    assert p.file_format == FileFormat.PARQUET


def test_check_file_format_arg_defaults_to_inferred(tmp_path):
    p = LocalDataPath(tmp_path / "data.csv")
    assert p.check_file_format_arg(None) == FileFormat.CSV
    assert p.check_file_format_arg("parquet") == FileFormat.PARQUET
    assert p.check_file_format_arg(FileFormat.JSON) == FileFormat.JSON


@pytest.mark.parametrize(
    "path_str, expected",
    [
        ("some_dir", True),          # no dot => dir sink heuristic
        ("some_dir/", True),         # trailing slash => dir sink
        ("file.csv", False),         # file exists => not dir sink (if created)
    ],
)
def test_is_dir_sink_heuristics(tmp_path, path_str, expected):
    p = LocalDataPath(tmp_path / path_str)
    # For file.csv case, create it so is_file() becomes True
    if p.name.endswith(".csv"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("a,b\n1,2\n", encoding="utf-8")
    else:
        # do not create; keep it "not found" to hit heuristic code
        pass

    assert p.is_dir_sink() is expected


# -------------------------
# ls() behavior
# -------------------------

def test_ls_non_recursive_lists_direct_children(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "d" / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "d" / "sub").mkdir()
    (tmp_path / "d" / "sub" / "c.txt").write_text("c", encoding="utf-8")

    p = LocalDataPath(tmp_path / "d")
    names = sorted([x.name for x in p.ls(recursive=False)])
    # includes directory entries when non-recursive
    assert names == ["a.txt", "b.txt", "sub"]


def test_ls_recursive_yields_files_only(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "d" / "sub").mkdir()
    (tmp_path / "d" / "sub" / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "d" / "sub" / "sub2").mkdir()
    (tmp_path / "d" / "sub" / "sub2" / "c.txt").write_text("c", encoding="utf-8")

    p = LocalDataPath(tmp_path / "d")
    out = sorted([str(x) for x in p.ls(recursive=True)])
    assert out == sorted(
        [
            str(LocalDataPath(tmp_path / "d" / "a.txt")),
            str(LocalDataPath(tmp_path / "d" / "sub" / "b.txt")),
            str(LocalDataPath(tmp_path / "d" / "sub" / "sub2" / "c.txt")),
        ]
    )


def test_ls_allow_not_found_true_returns_empty_iterator(tmp_path):
    p = LocalDataPath(tmp_path / "missing")
    assert list(p.ls(allow_not_found=True)) == []


def test_ls_allow_not_found_false_raises(tmp_path):
    p = LocalDataPath(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        list(p.ls(allow_not_found=False))


# -------------------------
# remove/rmdir/rmfile
# -------------------------

def test_rmfile_removes_file(tmp_path):
    p = LocalDataPath(tmp_path / "x.bin")
    p.write_bytes(b"abc")  # NOTE: this is currently buggy in implementation; see xfail test below
    # If write_bytes is fixed, this test should pass. For now, skip if it raises.
    assert p.exists()
    p.rmfile()
    assert not p.exists()


def test_rmdir_removes_dir_tree(tmp_path):
    d = LocalDataPath(tmp_path / "d")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "d" / "sub").mkdir()
    (tmp_path / "d" / "sub" / "b.txt").write_text("b", encoding="utf-8")

    assert d.exists()
    d.rmdir(recursive=True, allow_not_found=False)
    assert not d.exists()


# -------------------------
# Known bugs (xfail)
# These are real issues in the snippet:
# - LocalDataPath.write_bytes opens with "rb" (should be "wb") and returns f.write(...)
# - read_arrow_table_file / write_arrow_table_file reference "path" that doesn't exist
# - write_arrow_table_file has broken control flow and raises for PARQUET/CSV/etc
# -------------------------

@pytest.mark.xfail(reason="BUG: LocalDataPath.write_bytes opens file with 'rb' instead of 'wb'")
def test_write_bytes_and_read_bytes_roundtrip(tmp_path):
    p = LocalDataPath(tmp_path / "x.bin")
    p.write_bytes(b"abc")
    assert p.read_bytes() == b"abc"


@pytest.mark.xfail(reason="BUG: read_arrow_table_file uses undefined variable `path` instead of file handle/path str")
def test_read_arrow_table_file_parquet(tmp_path):
    t = pa.table({"a": [1, 2, 3]})
    p = LocalDataPath(tmp_path / "t.parquet")
    # write using pyarrow directly so read path code is isolated
    import pyarrow.parquet as pq
    pq.write_table(t, str(p))

    out = p.read_arrow_table_file(file_format=FileFormat.PARQUET)
    assert out.equals(t)


@pytest.mark.xfail(reason="BUG: write_arrow_table_file control flow + undefined `path` breaks all formats")
@pytest.mark.parametrize("fmt, ext", [(FileFormat.PARQUET, "parquet"), (FileFormat.CSV, "csv"), (FileFormat.ARROW_IPC, "arrow")])
def test_write_arrow_table_file_smoke(tmp_path, fmt, ext):
    t = pa.table({"a": [1, 2, 3]})
    p = LocalDataPath(tmp_path / f"t.{ext}")
    p.write_arrow_table_file(t, file_format=fmt, mode=SaveMode.OVERWRITE)
    assert p.exists()
    assert p.stat().st_size > 0


# -------------------------
# Directory sharding behavior (should pass if write_arrow_table_file is fixed)
# Here we mainly verify the *naming / extension* logic in write_arrow_table().
# If the underlying file writer is still broken, this will xfail too.
# -------------------------

@pytest.mark.xfail(reason="Depends on write_arrow_table_file which is currently broken in snippet")
def test_write_arrow_table_dir_sink_uses_fmt_extension_for_part_files(tmp_path):
    # dir sink: no '.' in last part
    out_dir = LocalDataPath(tmp_path / "dataset_out")
    t = pa.table({"a": list(range(10))})

    out_dir.write_arrow_table(
        t,
        file_format=FileFormat.PARQUET,
        mode=SaveMode.OVERWRITE,
        batch_size=3,
    )

    files = list(out_dir.ls(recursive=True))
    assert files, "should have written part files"
    assert all(f.name.endswith(".parquet") for f in files)
    assert all(f.name.startswith("part-") for f in files)


@pytest.mark.xfail(reason="Depends on write_polars_file / yggdrasil.polars which may not be installed in test env")
def test_write_polars_dir_sink_uses_fmt_extension_for_part_files(tmp_path):
    import polars as pl

    out_dir = LocalDataPath(tmp_path / "pl_out")
    df = pl.DataFrame({"a": list(range(10))})

    out_dir.write_polars(
        df,
        file_format=FileFormat.CSV,
        mode=SaveMode.OVERWRITE,
        batch_size=4,
    )

    files = list(out_dir.ls(recursive=True))
    assert files
    assert all(f.name.endswith(".csv") for f in files)


def test_write_databricks():
    from yggdrasil.databricks.workspaces import DatabricksPath

    out_dir = LocalDataPath("dbfs://dbc-e646c5f9-8a44.cloud.databricks.com/Volumes/trading/unittest/tmp/part-00000-1770826374288-jtCS.parquet")

    assert isinstance(out_dir, DatabricksPath)

    out_parquet_file = out_dir / "file.parquet"

    try:
        out_parquet_file.write_table({
            "c0": [1, 2, 3, 4],
            "str": ["s", None, None, "other"],
        })

        df = out_parquet_file.read_polars(limit=3)

        assert not df.is_empty()
        assert df.shape[0] == 3
    finally:
        out_parquet_file.remove()



def test_sync_file_overwrite_streaming(tmp_path):
    src = LocalDataPath(tmp_path / "src.bin")
    dst = LocalDataPath(tmp_path / "dst.bin")

    src.write_bytes(b"hello")
    dst.write_bytes(b"old")

    src.sync_file(dst, mode=SaveMode.OVERWRITE, parallel=None)
    assert dst.read_bytes() == b"hello"


def test_sync_file_ignore_existing(tmp_path):
    src = LocalDataPath(tmp_path / "src.bin")
    dst = LocalDataPath(tmp_path / "dst.bin")

    src.write_bytes(b"new")
    dst.write_bytes(b"existing")

    src.sync_file(dst, mode=SaveMode.IGNORE, parallel=None)
    assert dst.read_bytes() == b"existing"


def test_sync_file_error_if_exists(tmp_path):
    src = LocalDataPath(tmp_path / "src.bin")
    dst = LocalDataPath(tmp_path / "dst.bin")

    src.write_bytes(b"new")
    dst.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        src.sync_file(dst, mode=SaveMode.ERROR_IF_EXISTS, parallel=None)


def test_sync_file_append_local_to_local(tmp_path):
    src = LocalDataPath(tmp_path / "src.bin")
    dst = LocalDataPath(tmp_path / "dst.bin")

    dst.write_bytes(b"AAA")
    src.write_bytes(b"BBB")

    src.sync_file(dst, mode=SaveMode.APPEND, parallel=None)
    assert dst.read_bytes() == b"AAABBB"


def test_sync_file_parallel_executor_returns_future(tmp_path):
    src = LocalDataPath(tmp_path / "src.bin")
    dst = LocalDataPath(tmp_path / "dst.bin")
    src.write_bytes(b"yo")

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut = src.sync_file(dst, mode=SaveMode.OVERWRITE, parallel=ex)
        fut.result()

    assert dst.read_bytes() == b"yo"


def test_sync_file_allow_not_found_true_noop(tmp_path):
    src = LocalDataPath(tmp_path / "missing.bin")
    dst = LocalDataPath(tmp_path / "dst.bin")

    # should not raise, should not create dst
    src.sync_file(dst, allow_not_found=True, parallel=None)
    assert not dst.exists()


def test_sync_file_allow_not_found_false_raises(tmp_path):
    src = LocalDataPath(tmp_path / "missing.bin")
    dst = LocalDataPath(tmp_path / "dst.bin")

    with pytest.raises(FileNotFoundError):
        src.sync_file(dst, allow_not_found=False, parallel=None)


def test_sync_dir_copies_tree_sequential(tmp_path):
    src_dir = LocalDataPath(tmp_path / "src")
    dst_dir = LocalDataPath(tmp_path / "dst")

    _mk_tree(
        src_dir,
        {
            "a.bin": b"a",
            "sub/b.bin": b"b",
            "sub/deeper/c.bin": b"c",
        },
    )

    src_dir.sync_dir(dst_dir, mode=SaveMode.OVERWRITE, parallel=None)
    assert _tree_bytes(dst_dir) == {
        "a.bin": b"a",
        os.path.join("sub", "b.bin"): b"b",
        os.path.join("sub", "deeper", "c.bin"): b"c",
    }


def test_sync_dir_parallel_int(tmp_path):
    src_dir = LocalDataPath(tmp_path / "src")
    dst_dir = LocalDataPath(tmp_path / "dst")

    _mk_tree(src_dir, {f"files/{i}.bin": f"v{i}".encode() for i in range(50)})

    src_dir.sync_dir(dst_dir, mode=SaveMode.OVERWRITE, parallel=8)
    assert _tree_bytes(dst_dir) == {
        os.path.join("files", f"{i}.bin"): f"v{i}".encode() for i in range(50)
    }


def test_sync_dir_parallel_executor(tmp_path):
    src_dir = LocalDataPath(tmp_path / "src")
    dst_dir = LocalDataPath(tmp_path / "dst")

    _mk_tree(src_dir, {"x.bin": b"x", "y.bin": b"y"})

    with ThreadPoolExecutor(max_workers=4) as ex:
        src_dir.sync_dir(dst_dir, mode=SaveMode.OVERWRITE, parallel=ex)

    assert _tree_bytes(dst_dir) == {"x.bin": b"x", "y.bin": b"y"}


def test_sync_dir_ignore_existing_file(tmp_path):
    src_dir = LocalDataPath(tmp_path / "src")
    dst_dir = LocalDataPath(tmp_path / "dst")

    _mk_tree(src_dir, {"a.bin": b"new", "b.bin": b"b"})
    _mk_tree(dst_dir, {"a.bin": b"existing"})  # should be kept

    src_dir.sync_dir(dst_dir, mode=SaveMode.IGNORE, parallel=None)
    assert _tree_bytes(dst_dir) == {"a.bin": b"existing", "b.bin": b"b"}


def test_sync_dir_error_if_exists(tmp_path):
    src_dir = LocalDataPath(tmp_path / "src")
    dst_dir = LocalDataPath(tmp_path / "dst")

    _mk_tree(src_dir, {"a.bin": b"new"})
    _mk_tree(dst_dir, {"a.bin": b"existing"})

    with pytest.raises(FileExistsError):
        src_dir.sync_dir(dst_dir, mode=SaveMode.ERROR_IF_EXISTS, parallel=None)


def test_sync_dir_does_not_delete_extras(tmp_path):
    src_dir = LocalDataPath(tmp_path / "src")
    dst_dir = LocalDataPath(tmp_path / "dst")

    _mk_tree(src_dir, {"a.bin": b"a"})
    _mk_tree(dst_dir, {"extra.bin": b"zzz"})

    src_dir.sync_dir(dst_dir, mode=SaveMode.OVERWRITE, parallel=None)
    assert _tree_bytes(dst_dir) == {"a.bin": b"a", "extra.bin": b"zzz"}


def test_sync_dir_allow_not_found_true_noop(tmp_path):
    src_dir = LocalDataPath(tmp_path / "missing_dir")
    dst_dir = LocalDataPath(tmp_path / "dst")

    src_dir.sync_dir(dst_dir, allow_not_found=True, parallel=None)
    assert not dst_dir.exists()


def test_sync_dir_allow_not_found_false_raises(tmp_path):
    src_dir = LocalDataPath(tmp_path / "missing_dir")
    dst_dir = LocalDataPath(tmp_path / "dst")

    with pytest.raises(FileNotFoundError):
        src_dir.sync_dir(dst_dir, allow_not_found=False, parallel=None)


def test_sync_dispatch_file(tmp_path):
    src = LocalDataPath(tmp_path / "src.bin")
    dst = LocalDataPath(tmp_path / "dst.bin")
    src.write_bytes(b"data")

    src.sync(dst, parallel=None)
    assert dst.read_bytes() == b"data"


def test_sync_dispatch_dir(tmp_path):
    src_dir = LocalDataPath(tmp_path / "src")
    dst_dir = LocalDataPath(tmp_path / "dst")
    _mk_tree(src_dir, {"a.bin": b"a"})

    src_dir.sync(dst_dir, parallel=None)
    assert _tree_bytes(dst_dir) == {"a.bin": b"a"}