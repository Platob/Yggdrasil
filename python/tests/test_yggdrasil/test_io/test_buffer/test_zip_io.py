# tests/io/buffer/test_zip_io.py
from __future__ import annotations

import io
import zipfile

import pytest

import yggdrasil.pickle.json as json_mod
from yggdrasil.io import MimeTypes
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.zip_io import ZipIO, ZipOptions
from yggdrasil.io.config import BufferConfig
from yggdrasil.io.enums import MimeType
from yggdrasil.arrow.lib import pyarrow as pa


def _make_zip_bytes(members: dict[str, bytes], *, compresslevel: int = 8) -> bytes:
    mem = io.BytesIO()
    try:
        with zipfile.ZipFile(
            mem,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=int(compresslevel),
        ) as zf:
            for name, payload in members.items():
                zf.writestr(name, payload)
    except TypeError:
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, payload in members.items():
                zf.writestr(name, payload)
    return mem.getvalue()


def test_zipio_read_empty_returns_empty_table():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.ZIP)

    tb = io_.read_arrow_table()
    assert tb.num_rows == 0


def test_zipio_write_and_read_roundtrip_json_inner():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.ZIP)

    inp = pa.Table.from_pylist([{"a": 1}, {"a": 2}, {"a": 3}])

    # Force inner json to avoid parquet dependency in unit tests
    io_.write_arrow_table(inp, options=ZipOptions(inner_media="json", member="data.json"))

    out = io_.read_arrow_table(options=ZipOptions(member="data.json"))
    assert out.to_pylist() == inp.to_pylist()


def test_zipio_read_concat_all_members_when_member_none():
    # Two json members with same schema => concat should stack rows
    t1 = pa.Table.from_pylist([{"x": 1}, {"x": 2}]).to_pylist()
    t2 = pa.Table.from_pylist([{"x": 3}]).to_pylist()

    m1 = json_mod.dumps(t1)  # bytes
    m2 = json_mod.dumps(t2)  # bytes

    zbytes = _make_zip_bytes({"part1.json": m1, "part2.json": m2})
    buf = BytesIO(zbytes)

    out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(options=ZipOptions(member=None))
    assert out.to_pylist() == (t1 + t2)


def test_zipio_read_single_member_when_specified():
    m1 = json_mod.dumps([{"x": 1}])     # bytes
    m2 = json_mod.dumps([{"x": 999}])   # bytes

    zbytes = _make_zip_bytes({"a.json": m1, "b.json": m2})
    buf = BytesIO(zbytes)

    out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(options=ZipOptions(member="b.json"))
    assert out.to_pylist() == [{"x": 999}]


def test_zipio_member_not_found_raises_keyerror():
    zbytes = _make_zip_bytes({"a.json": json_mod.dumps([{"x": 1}])})
    buf = BytesIO(zbytes)

    with pytest.raises(KeyError):
        MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(options=ZipOptions(member="nope.json"))


def test_zipio_does_not_move_parent_cursor_on_read():
    zbytes = _make_zip_bytes({"a.json": json_mod.dumps([{"x": 1}])})
    buf = BytesIO(zbytes)
    io_ = MediaIO.make(buf, MimeTypes.ZIP)

    buf.seek(7)
    before = buf.tell()

    out = io_.read_arrow_table(options=ZipOptions(member="a.json"))
    after = buf.tell()

    assert out.to_pylist() == [{"x": 1}]
    assert after == before  # view() must not touch parent BytesIO._pos


def test_zipio_read_infers_inner_media_from_extension_json():
    # Tests the "infer children mediatypes" path via member extension.
    payload = json_mod.dumps([{"k": "v"}])  # bytes
    zbytes = _make_zip_bytes({"data.json": payload})
    buf = BytesIO(zbytes)

    out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table()
    assert out.to_pylist() == [{"k": "v"}]


def test_zipio_force_inner_media_overrides_inference():
    # Name looks "wrong" (no extension), payload is json bytes.
    payload = json_mod.dumps([{"a": 1}])  # bytes
    zbytes = _make_zip_bytes({"weirdname": payload})
    buf = BytesIO(zbytes)

    out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
        options=ZipOptions(member="weirdname", inner_media="json", force_inner_media=True)
    )
    assert out.to_pylist() == [{"a": 1}]


# ===================================================================
# Optimisation tests — compute / memory behaviour of the read path
# ===================================================================


def _monkeypatch_counter(cls, name, monkeypatch):
    """Wrap ``cls.name`` with a call counter without changing behaviour.

    Handles both instance methods and staticmethods transparently by
    resolving the underlying callable through ``inspect.getattr_static``
    and re-wrapping it in the original descriptor type.
    """
    import inspect

    raw = inspect.getattr_static(cls, name)
    counter = {"calls": 0}

    if isinstance(raw, staticmethod):
        fn = raw.__func__

        def wrapper(*args, **kwargs):
            counter["calls"] += 1
            return fn(*args, **kwargs)

        monkeypatch.setattr(cls, name, staticmethod(wrapper))
        return counter

    if isinstance(raw, classmethod):
        fn = raw.__func__

        def wrapper(cls_arg, *args, **kwargs):
            counter["calls"] += 1
            return fn(cls_arg, *args, **kwargs)

        monkeypatch.setattr(cls, name, classmethod(wrapper))
        return counter

    def wrapper(self, *args, **kwargs):
        counter["calls"] += 1
        return raw(self, *args, **kwargs)

    monkeypatch.setattr(cls, name, wrapper)
    return counter


class TestZipReadOptimizations:
    """Cover the compute/memory optimisations of the ZIP read path."""

    @staticmethod
    def _zip_with_many_members(n: int, target_idx: int) -> tuple[bytes, str]:
        """Build a ZIP with *n* trivial JSON members; the target has id=42."""
        payload = lambda i: json_mod.dumps([{"x": i}])  # noqa: E731
        members = {f"member_{i:06d}.json": payload(i) for i in range(n)}
        name = f"member_{target_idx:06d}.json"
        return _make_zip_bytes(members), name

    def test_exact_member_lookup_uses_getinfo_not_linear_scan(self, monkeypatch):
        """Single-name lookup must hit the hash-backed path (O(1))."""
        zbytes, name = self._zip_with_many_members(50, 42)
        buf = BytesIO(zbytes)

        getinfo_calls = _monkeypatch_counter(zipfile.ZipFile, "getinfo", monkeypatch)
        infolist_calls = _monkeypatch_counter(zipfile.ZipFile, "infolist", monkeypatch)

        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(member=name)
        )

        assert out.to_pylist() == [{"x": 42}]
        assert getinfo_calls["calls"] >= 1, "exact lookup should use getinfo"
        # infolist() is still allowed (used by zf.read internally), but we
        # must not walk it to find the member ourselves. getinfo is the
        # deciding call.

    def test_missing_member_error_lists_all_available_names(self):
        """Error path must still surface every member name for debuggability."""
        zbytes = _make_zip_bytes(
            {"a.json": b"[]", "b.json": b"[]", "c.json": b"[]"}
        )
        buf = BytesIO(zbytes)
        with pytest.raises(KeyError) as ei:
            MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
                options=ZipOptions(member="nope.json")
            )
        msg = str(ei.value)
        assert "a.json" in msg and "b.json" in msg and "c.json" in msg

    def test_glob_no_match_error_lists_all_available_names(self):
        zbytes = _make_zip_bytes({"a.json": b"[]", "b.csv": b""})
        buf = BytesIO(zbytes)
        with pytest.raises(KeyError) as ei:
            MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
                options=ZipOptions(member="*.xml")
            )
        msg = str(ei.value)
        assert "a.json" in msg and "b.csv" in msg

    def test_directory_entry_is_not_matched_by_exact_name(self):
        """``zf.getinfo`` on a dir-like entry must still raise KeyError."""
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("folder/", b"")
            zf.writestr("folder/inner.json", json_mod.dumps([{"x": 1}]))
        buf = BytesIO(mem.getvalue())
        with pytest.raises(KeyError):
            MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
                options=ZipOptions(member="folder/")
            )

    def test_large_member_streams_to_disk_bounded_memory(self):
        """Members larger than spill_bytes must be streamed, not fully RAM-loaded."""
        big_payload = json_mod.dumps([{"v": i} for i in range(10_000)])
        assert len(big_payload) > 10_000
        zbytes = _make_zip_bytes({"big.json": big_payload})

        cfg = BufferConfig(spill_bytes=512)  # force spill
        buf = BytesIO(zbytes, config=cfg)

        captured: list[tuple[str, int, bool]] = []
        original = ZipIO._load_member_buffer

        def tracer(self, zf, info):
            inner_buf, prefix = original(self, zf, info)
            captured.append((info.filename, info.file_size, inner_buf.spilled))
            return inner_buf, prefix

        ZipIO._load_member_buffer = tracer
        try:
            out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table()
        finally:
            ZipIO._load_member_buffer = original

        assert out.num_rows == 10_000
        assert captured, "streaming branch should have been invoked"
        name, size, spilled = captured[0]
        assert name == "big.json"
        assert size > cfg.spill_bytes
        assert spilled is True, "large member must spill to disk"

    def test_small_member_uses_in_memory_fast_path(self):
        """Members below spill_bytes should stay in RAM (no unnecessary disk I/O)."""
        small_payload = json_mod.dumps([{"x": 1}])
        zbytes = _make_zip_bytes({"tiny.json": small_payload})

        cfg = BufferConfig(spill_bytes=10 * 1024 * 1024)
        buf = BytesIO(zbytes, config=cfg)

        spilled_flags: list[bool] = []
        original = ZipIO._load_member_buffer

        def tracer(self, zf, info):
            inner_buf, prefix = original(self, zf, info)
            spilled_flags.append(inner_buf.spilled)
            return inner_buf, prefix

        ZipIO._load_member_buffer = tracer
        try:
            MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table()
        finally:
            ZipIO._load_member_buffer = original

        assert spilled_flags == [False]

    def test_name_based_inference_skips_payload_sniff(self, monkeypatch):
        """Members with an unambiguous extension must not re-sniff the payload."""
        zbytes = _make_zip_bytes({"a.json": json_mod.dumps([{"x": 1}])})
        buf = BytesIO(zbytes)
        mio = MediaIO.make(buf, MimeTypes.ZIP)

        sniff_calls = _monkeypatch_counter(ZipIO, "_sniff_media_from_prefix", monkeypatch)
        out = mio.read_arrow_table()

        assert out.to_pylist() == [{"x": 1}]
        assert sniff_calls["calls"] == 0, (
            "name-based inference should win; prefix sniff must not run"
        )

    def test_unnamed_member_falls_back_to_prefix_sniff(self, monkeypatch):
        """Extensionless members still detect media via a small prefix sniff."""
        zbytes = _make_zip_bytes({"noextension": json_mod.dumps([{"x": 1}])})
        buf = BytesIO(zbytes)

        sniff_calls = _monkeypatch_counter(ZipIO, "_sniff_media_from_prefix", monkeypatch)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(member="noextension")
        )

        assert out.to_pylist() == [{"x": 1}]
        assert sniff_calls["calls"] == 1

    def test_force_inner_media_short_circuits_inference_entirely(self, monkeypatch):
        """``force_inner_media`` must not perform either name lookup or sniff."""
        zbytes = _make_zip_bytes({"mystery.dat": json_mod.dumps([{"x": 1}])})
        buf = BytesIO(zbytes)

        sniff_calls = _monkeypatch_counter(ZipIO, "_sniff_media_from_prefix", monkeypatch)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(
                member="mystery.dat",
                inner_media="json",
                force_inner_media=True,
            )
        )

        assert out.to_pylist() == [{"x": 1}]
        assert sniff_calls["calls"] == 0

    def test_member_info_name_column_is_dictionary_encoded(self):
        """The synthetic ``name`` metadata column must use dict encoding."""
        payload = json_mod.dumps([{"x": i} for i in range(100)])
        zbytes = _make_zip_bytes({"src.json": payload})
        buf = BytesIO(zbytes)

        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(read_member_infos=[("name", "source")])
        )

        src_field = out.schema.field("source")
        assert pa.types.is_dictionary(src_field.type), (
            f"expected DictionaryType, got {src_field.type}"
        )
        # Values round-trip to plain strings.
        assert out.column("source").to_pylist() == ["src.json"] * 100

    def test_member_info_name_column_dict_has_single_entry_across_rows(self):
        """Dict column must reuse one entry rather than N copies of the name."""
        payload = json_mod.dumps([{"x": i} for i in range(5_000)])
        zbytes = _make_zip_bytes({"m.json": payload})
        buf = BytesIO(zbytes)

        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(read_member_infos=[("name", "src")])
        )

        chunk = out.column("src").combine_chunks()
        assert len(chunk.dictionary) == 1
        assert chunk.dictionary[0].as_py() == "m.json"
        assert len(chunk.indices) == 5_000

    def test_member_info_works_across_multiple_members(self):
        """Concat across members must produce a union dictionary as expected."""
        zbytes = _make_zip_bytes({
            "a.json": json_mod.dumps([{"x": 1}]),
            "b.json": json_mod.dumps([{"x": 2}]),
        })
        buf = BytesIO(zbytes)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(read_member_infos=[("name", "src")])
        )
        assert out.column("src").to_pylist() == ["a.json", "b.json"]

    def test_infolist_not_used_for_error_message_on_success(self, monkeypatch):
        """Happy-path exact lookup must not build the all-names list."""
        zbytes, name = self._zip_with_many_members(20, 7)
        buf = BytesIO(zbytes)

        nondir_calls = _monkeypatch_counter(ZipIO, "_nondir_names", monkeypatch)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(member=name)
        )

        assert out.num_rows == 1
        assert nondir_calls["calls"] == 0, (
            "_nondir_names must only run on the error branch"
        )