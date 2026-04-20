"""Unit tests for :class:`yggdrasil.io.buffer.zip_io.ZipIO`.

Covers:

* options validation
* roundtrip with inner JSON
* concat across members, single-member selection
* glob patterns
* inner media inference (name, force, sniff)
* memory optimizations (exact lookup via getinfo, streaming spill, dict-encoded info)
* error messages list all members
* parent-cursor invariant
* save modes (OVERWRITE, IGNORE, ERROR_IF_EXISTS, APPEND, UPSERT)
* group iteration API (read_arrow_groups, read_arrow_tables_by_group)
* read_member_infos dictionary encoding
* write path: one-member, multi-member APPEND, member replacement
* edge cases (directory entries, empty buffer, spilled parent buffer)
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

import yggdrasil.pickle.json as json_mod
from yggdrasil.arrow.lib import pyarrow as pa
from yggdrasil.io import MimeTypes
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.zip_io import ZipIO, ZipOptions
from yggdrasil.io.config import BufferConfig
from yggdrasil.io.enums import SaveMode


# =====================================================================
# Helpers
# =====================================================================

def _make_zip_bytes(
    members: dict[str, bytes],
    *,
    compresslevel: int = 8,
) -> bytes:
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


def _monkeypatch_counter(cls, name, monkeypatch):
    """Wrap cls.name with a call counter. Handles static/class/instance methods."""
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


@pytest.fixture()
def cfg(tmp_path: Path) -> BufferConfig:
    return BufferConfig(
        spill_bytes=1024 * 1024,
        tmp_dir=tmp_path,
        prefix="test_zipio_",
        suffix=".zip",
        keep_spilled_file=False,
    )


# =====================================================================
# Options validation
# =====================================================================

class TestZipOptions:
    def test_defaults(self):
        opt = ZipOptions()
        assert opt.member is None
        assert opt.inner_media is None
        assert opt.force_inner_media is False
        assert opt.read_member_infos is None
        assert opt.group_by is None
        assert opt.zip_compression == zipfile.ZIP_DEFLATED
        assert opt.zip_compresslevel is None

    def test_member_must_be_str_or_none(self):
        with pytest.raises(TypeError, match="member"):
            ZipOptions(member=42)  # type: ignore[arg-type]

    def test_member_cannot_be_empty(self):
        with pytest.raises(ValueError, match="member"):
            ZipOptions(member="")

    def test_force_inner_media_requires_inner_media(self):
        with pytest.raises(ValueError, match="force_inner_media"):
            ZipOptions(force_inner_media=True)

    def test_force_inner_media_with_inner_media_ok(self):
        opt = ZipOptions(force_inner_media=True, inner_media="json")
        assert opt.force_inner_media is True

    def test_read_member_infos_validates_shape(self):
        with pytest.raises(TypeError, match="read_member_infos"):
            ZipOptions(read_member_infos=[("name",)])  # type: ignore[list-item]

    def test_read_member_infos_rejects_empty_keys(self):
        with pytest.raises(ValueError, match="info_key"):
            ZipOptions(read_member_infos=[("", "col")])

    def test_read_member_infos_rejects_empty_column_name(self):
        with pytest.raises(ValueError, match="column_name"):
            ZipOptions(read_member_infos=[("name", "")])

    def test_group_by_accepts_member_all_callable_none(self):
        ZipOptions(group_by=None)
        ZipOptions(group_by="all")
        ZipOptions(group_by="member")
        ZipOptions(group_by=lambda name: name)

    def test_group_by_rejects_unknown_string(self):
        with pytest.raises(ValueError, match="group_by"):
            ZipOptions(group_by="nope")

    def test_zip_compression_must_be_valid_constant(self):
        with pytest.raises(ValueError, match="zip_compression"):
            ZipOptions(zip_compression=999)


# =====================================================================
# Factory
# =====================================================================

class TestFactory:
    def test_media_io_make_returns_zip_io(self):
        io_ = MediaIO.make(BytesIO(), MimeTypes.ZIP)
        assert isinstance(io_, ZipIO)


# =====================================================================
# Baseline read contract (existing tests)
# =====================================================================

class TestBasicRead:
    def test_read_empty_returns_empty_table(self):
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        tb = io_.read_arrow_table()
        assert tb.num_rows == 0

    def test_write_and_read_roundtrip_json_inner(self):
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ZIP)

        inp = pa.Table.from_pylist([{"a": 1}, {"a": 2}, {"a": 3}])
        io_.write_arrow_table(
            inp, options=ZipOptions(inner_media="json", member="data.json")
        )

        out = io_.read_arrow_table(options=ZipOptions(member="data.json"))
        assert out.to_pylist() == inp.to_pylist()

    def test_concat_all_members_when_member_none(self):
        t1 = [{"x": 1}, {"x": 2}]
        t2 = [{"x": 3}]

        zbytes = _make_zip_bytes({
            "part1.json": json_mod.dumps(t1),
            "part2.json": json_mod.dumps(t2),
        })
        buf = BytesIO(zbytes)

        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(member=None)
        )
        assert out.to_pylist() == t1 + t2

    def test_single_member_when_specified(self):
        zbytes = _make_zip_bytes({
            "a.json": json_mod.dumps([{"x": 1}]),
            "b.json": json_mod.dumps([{"x": 999}]),
        })
        buf = BytesIO(zbytes)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(member="b.json")
        )
        assert out.to_pylist() == [{"x": 999}]

    def test_member_not_found_raises_keyerror(self):
        zbytes = _make_zip_bytes({"a.json": json_mod.dumps([{"x": 1}])})
        buf = BytesIO(zbytes)
        with pytest.raises(KeyError):
            MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
                options=ZipOptions(member="nope.json")
            )

    def test_does_not_move_parent_cursor_on_read(self):
        zbytes = _make_zip_bytes({"a.json": json_mod.dumps([{"x": 1}])})
        buf = BytesIO(zbytes)
        io_ = MediaIO.make(buf, MimeTypes.ZIP)

        buf.seek(7)
        before = buf.tell()

        out = io_.read_arrow_table(options=ZipOptions(member="a.json"))
        after = buf.tell()

        assert out.to_pylist() == [{"x": 1}]
        assert after == before


# =====================================================================
# Inner-media resolution
# =====================================================================

class TestInnerMediaResolution:
    def test_read_infers_from_extension_json(self):
        payload = json_mod.dumps([{"k": "v"}])
        zbytes = _make_zip_bytes({"data.json": payload})
        buf = BytesIO(zbytes)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table()
        assert out.to_pylist() == [{"k": "v"}]

    def test_force_inner_media_overrides_inference(self):
        payload = json_mod.dumps([{"a": 1}])
        zbytes = _make_zip_bytes({"weirdname": payload})
        buf = BytesIO(zbytes)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(
                member="weirdname",
                inner_media="json",
                force_inner_media=True,
            )
        )
        assert out.to_pylist() == [{"a": 1}]


# =====================================================================
# Memory + compute optimizations (from the original suite)
# =====================================================================

class TestReadOptimizations:
    @staticmethod
    def _zip_with_many_members(n: int, target_idx: int) -> tuple[bytes, str]:
        payload = lambda i: json_mod.dumps([{"x": i}])  # noqa: E731
        members = {f"member_{i:06d}.json": payload(i) for i in range(n)}
        name = f"member_{target_idx:06d}.json"
        return _make_zip_bytes(members), name

    def test_exact_lookup_uses_getinfo(self, monkeypatch):
        zbytes, name = self._zip_with_many_members(50, 42)
        buf = BytesIO(zbytes)

        getinfo_calls = _monkeypatch_counter(zipfile.ZipFile, "getinfo", monkeypatch)

        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(member=name)
        )

        assert out.to_pylist() == [{"x": 42}]
        assert getinfo_calls["calls"] >= 1

    def test_missing_member_error_lists_all_names(self):
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

    def test_glob_no_match_error_lists_all_names(self):
        zbytes = _make_zip_bytes({"a.json": b"[]", "b.csv": b""})
        buf = BytesIO(zbytes)
        with pytest.raises(KeyError) as ei:
            MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
                options=ZipOptions(member="*.xml")
            )
        msg = str(ei.value)
        assert "a.json" in msg and "b.csv" in msg

    def test_directory_entry_is_not_matched_by_exact_name(self):
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("folder/", b"")
            zf.writestr("folder/inner.json", json_mod.dumps([{"x": 1}]))
        buf = BytesIO(mem.getvalue())
        with pytest.raises(KeyError):
            MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
                options=ZipOptions(member="folder/")
            )

    def test_large_member_spills_to_disk(self, tmp_path: Path):
        big_payload = json_mod.dumps([{"v": i} for i in range(10_000)])
        assert len(big_payload) > 10_000
        zbytes = _make_zip_bytes({"big.json": big_payload})

        cfg = BufferConfig(spill_bytes=512, tmp_dir=tmp_path)
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
        assert spilled is True

    def test_small_member_stays_in_memory(self, tmp_path: Path):
        small_payload = json_mod.dumps([{"x": 1}])
        zbytes = _make_zip_bytes({"tiny.json": small_payload})

        cfg = BufferConfig(spill_bytes=10 * 1024 * 1024, tmp_dir=tmp_path)
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

    def test_name_based_inference_skips_sniff(self, monkeypatch):
        zbytes = _make_zip_bytes({"a.json": json_mod.dumps([{"x": 1}])})
        buf = BytesIO(zbytes)
        mio = MediaIO.make(buf, MimeTypes.ZIP)

        sniff_calls = _monkeypatch_counter(ZipIO, "_sniff_media_from_prefix", monkeypatch)
        out = mio.read_arrow_table()

        assert out.to_pylist() == [{"x": 1}]
        assert sniff_calls["calls"] == 0

    def test_unnamed_member_falls_back_to_sniff(self, monkeypatch):
        zbytes = _make_zip_bytes({"noextension": json_mod.dumps([{"x": 1}])})
        buf = BytesIO(zbytes)

        sniff_calls = _monkeypatch_counter(ZipIO, "_sniff_media_from_prefix", monkeypatch)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(member="noextension")
        )

        assert out.to_pylist() == [{"x": 1}]
        assert sniff_calls["calls"] == 1

    def test_force_inner_media_skips_sniff(self, monkeypatch):
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

    def test_happy_path_does_not_build_error_name_list(self, monkeypatch):
        zbytes, name = self._zip_with_many_members(20, 7)
        buf = BytesIO(zbytes)

        nondir_calls = _monkeypatch_counter(ZipIO, "_nondir_names", monkeypatch)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(member=name)
        )

        assert out.num_rows == 1
        assert nondir_calls["calls"] == 0


# =====================================================================
# Dictionary-encoded member-info columns
# =====================================================================

class TestMemberInfoColumns:
    def test_name_column_is_dictionary_encoded(self):
        payload = json_mod.dumps([{"x": i} for i in range(100)])
        zbytes = _make_zip_bytes({"src.json": payload})
        buf = BytesIO(zbytes)

        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(read_member_infos=[("name", "source")])
        )
        src_field = out.schema.field("source")
        assert pa.types.is_dictionary(src_field.type)
        assert out.column("source").to_pylist() == ["src.json"] * 100

    def test_dict_has_single_entry_across_large_member(self):
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

    def test_works_across_multiple_members(self):
        zbytes = _make_zip_bytes({
            "a.json": json_mod.dumps([{"x": 1}]),
            "b.json": json_mod.dumps([{"x": 2}]),
        })
        buf = BytesIO(zbytes)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(read_member_infos=[("name", "src")])
        )
        assert out.column("src").to_pylist() == ["a.json", "b.json"]

    def test_file_size_info_column(self):
        """A non-name info key (file_size) produces an integer dict column."""
        payload = json_mod.dumps([{"x": 1}])
        zbytes = _make_zip_bytes({"m.json": payload})
        buf = BytesIO(zbytes)

        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(read_member_infos=[("file_size", "bytes")])
        )
        assert "bytes" in out.schema.names
        # file_size is the uncompressed size of the member payload.
        assert out.column("bytes").to_pylist()[0] == len(payload)

    def test_unknown_info_key_raises(self):
        zbytes = _make_zip_bytes({"a.json": b"[]"})
        buf = BytesIO(zbytes)
        with pytest.raises(ValueError, match="read_member_infos"):
            MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
                options=ZipOptions(read_member_infos=[("bogus_key", "x")])
            )


# =====================================================================
# Glob patterns
# =====================================================================

class TestGlobPatterns:
    def test_star_json_selects_only_json_members(self):
        zbytes = _make_zip_bytes({
            "a.json": json_mod.dumps([{"k": 1}]),
            "b.json": json_mod.dumps([{"k": 2}]),
            "c.csv": b"",  # not a member we can parse
        })
        buf = BytesIO(zbytes)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(member="*.json")
        )
        assert out.to_pylist() == [{"k": 1}, {"k": 2}]

    def test_question_mark_glob(self):
        zbytes = _make_zip_bytes({
            "a.json": json_mod.dumps([{"v": 1}]),
            "ab.json": json_mod.dumps([{"v": 2}]),
        })
        buf = BytesIO(zbytes)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(member="?.json")
        )
        # "?" matches exactly one char → only "a.json".
        assert out.to_pylist() == [{"v": 1}]


# =====================================================================
# Group iteration API (new)
# =====================================================================

class TestGroupIteration:
    def test_group_by_member_yields_one_group_per_member(self):
        zbytes = _make_zip_bytes({
            "a.json": json_mod.dumps([{"v": 1}, {"v": 2}]),
            "b.json": json_mod.dumps([{"v": 3}]),
            "c.json": json_mod.dumps([{"v": 4}, {"v": 5}, {"v": 6}]),
        })
        buf = BytesIO(zbytes)
        io_ = MediaIO.make(buf, MimeTypes.ZIP)

        groups: list[tuple[str, list[dict]]] = []
        for key, tbl in io_.read_arrow_tables_by_group(
            options=ZipOptions(group_by="member")
        ):
            groups.append((key, tbl.to_pylist()))

        assert groups == [
            ("a.json", [{"v": 1}, {"v": 2}]),
            ("b.json", [{"v": 3}]),
            ("c.json", [{"v": 4}, {"v": 5}, {"v": 6}]),
        ]

    def test_group_by_none_yields_single_group(self):
        zbytes = _make_zip_bytes({
            "a.json": json_mod.dumps([{"v": 1}]),
            "b.json": json_mod.dumps([{"v": 2}]),
        })
        buf = BytesIO(zbytes)
        io_ = MediaIO.make(buf, MimeTypes.ZIP)

        groups = list(io_.read_arrow_tables_by_group(
            options=ZipOptions(group_by=None)
        ))
        assert len(groups) == 1
        assert groups[0][1].to_pylist() == [{"v": 1}, {"v": 2}]

    def test_group_by_all_behaves_like_none(self):
        zbytes = _make_zip_bytes({
            "a.json": json_mod.dumps([{"v": 1}]),
            "b.json": json_mod.dumps([{"v": 2}]),
        })
        buf = BytesIO(zbytes)
        io_ = MediaIO.make(buf, MimeTypes.ZIP)

        groups = list(io_.read_arrow_tables_by_group(
            options=ZipOptions(group_by="all")
        ))
        assert len(groups) == 1

    def test_group_by_callable_uses_custom_key(self):
        zbytes = _make_zip_bytes({
            "2024_jan.json": json_mod.dumps([{"v": 1}]),
            "2024_feb.json": json_mod.dumps([{"v": 2}]),
            "2025_jan.json": json_mod.dumps([{"v": 3}]),
        })
        buf = BytesIO(zbytes)
        io_ = MediaIO.make(buf, MimeTypes.ZIP)

        groups = list(io_.read_arrow_tables_by_group(
            options=ZipOptions(group_by=lambda name: name.split("_")[0])
        ))
        # Members 2024_jan and 2024_feb share the "2024" group; 2025_jan
        # is its own group.
        keys = [g[0] for g in groups]
        assert keys == ["2024", "2025"]

    def test_read_arrow_groups_yields_batch_iterators(self):
        """Low-level API returns (key, Iterator[RecordBatch]) pairs."""
        zbytes = _make_zip_bytes({
            "a.json": json_mod.dumps([{"v": 1}]),
            "b.json": json_mod.dumps([{"v": 2}]),
        })
        buf = BytesIO(zbytes)
        io_ = MediaIO.make(buf, MimeTypes.ZIP)

        collected: list[tuple[str, int]] = []
        for key, batches in io_.read_arrow_groups(
            options=ZipOptions(group_by="member")
        ):
            rows = 0
            for b in batches:
                rows += b.num_rows
            collected.append((key, rows))

        assert collected == [("a.json", 1), ("b.json", 1)]

    def test_empty_buffer_groups_yields_nothing(self):
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ZIP)

        groups = list(io_.read_arrow_groups(
            options=ZipOptions(group_by="member")
        ))
        assert groups == []


# =====================================================================
# Save modes
# =====================================================================

class TestSaveModes:
    def _write_initial(self, buf: BytesIO) -> None:
        MediaIO.make(buf, MimeTypes.ZIP).write_arrow_table(
            pa.Table.from_pylist([{"v": 1}]),
            options=ZipOptions(
                inner_media="json",
                member="first.json",
                mode=SaveMode.OVERWRITE,
            ),
        )

    def test_ignore_mode(self):
        buf = BytesIO()
        self._write_initial(buf)
        size1 = buf.size
        bytes1 = buf.to_bytes()

        MediaIO.make(buf, MimeTypes.ZIP).write_arrow_table(
            pa.Table.from_pylist([{"v": 99}]),
            options=ZipOptions(
                inner_media="json",
                member="second.json",
                mode=SaveMode.IGNORE,
            ),
        )
        assert buf.size == size1
        assert buf.to_bytes() == bytes1

    def test_error_if_exists_raises(self):
        buf = BytesIO()
        self._write_initial(buf)
        with pytest.raises(IOError):
            MediaIO.make(buf, MimeTypes.ZIP).write_arrow_table(
                pa.Table.from_pylist([{"v": 2}]),
                options=ZipOptions(
                    inner_media="json",
                    member="x.json",
                    mode=SaveMode.ERROR_IF_EXISTS,
                ),
            )

    def test_overwrite_replaces_whole_archive(self):
        buf = BytesIO()
        self._write_initial(buf)

        MediaIO.make(buf, MimeTypes.ZIP).write_arrow_table(
            pa.Table.from_pylist([{"v": 99}]),
            options=ZipOptions(
                inner_media="json",
                member="only.json",
                mode=SaveMode.OVERWRITE,
            ),
        )

        # "first.json" should be gone, "only.json" present.
        with zipfile.ZipFile(io.BytesIO(buf.to_bytes())) as zf:
            names = zf.namelist()
        assert "first.json" not in names
        assert "only.json" in names

    def test_append_adds_new_member_preserves_others(self):
        buf = BytesIO()
        self._write_initial(buf)

        MediaIO.make(buf, MimeTypes.ZIP).write_arrow_table(
            pa.Table.from_pylist([{"v": 2}]),
            options=ZipOptions(
                inner_media="json",
                member="second.json",
                mode=SaveMode.APPEND,
            ),
        )

        with zipfile.ZipFile(io.BytesIO(buf.to_bytes())) as zf:
            names = sorted(zf.namelist())
        assert names == ["first.json", "second.json"]

        # Both round-trip correctly.
        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        t1 = io_.read_arrow_table(options=ZipOptions(member="first.json"))
        t2 = io_.read_arrow_table(options=ZipOptions(member="second.json"))
        assert t1.to_pylist() == [{"v": 1}]
        assert t2.to_pylist() == [{"v": 2}]

    def test_append_with_existing_name_replaces(self):
        """APPEND with a member name that exists replaces it."""
        buf = BytesIO()
        self._write_initial(buf)

        MediaIO.make(buf, MimeTypes.ZIP).write_arrow_table(
            pa.Table.from_pylist([{"v": 999}]),
            options=ZipOptions(
                inner_media="json",
                member="first.json",  # same name
                mode=SaveMode.APPEND,
            ),
        )

        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        out = io_.read_arrow_table(options=ZipOptions(member="first.json"))
        assert out.to_pylist() == [{"v": 999}]

    def test_upsert_replaces_named_member(self):
        buf = BytesIO()
        self._write_initial(buf)
        # Add a second member via APPEND.
        MediaIO.make(buf, MimeTypes.ZIP).write_arrow_table(
            pa.Table.from_pylist([{"v": 2}]),
            options=ZipOptions(
                inner_media="json",
                member="second.json",
                mode=SaveMode.APPEND,
            ),
        )

        # UPSERT on "first.json" should replace it, leave "second.json".
        MediaIO.make(buf, MimeTypes.ZIP).write_arrow_table(
            pa.Table.from_pylist([{"v": 1000}]),
            options=ZipOptions(
                inner_media="json",
                member="first.json",
                mode=SaveMode.UPSERT,
            ),
        )

        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        first = io_.read_arrow_table(options=ZipOptions(member="first.json"))
        second = io_.read_arrow_table(options=ZipOptions(member="second.json"))
        assert first.to_pylist() == [{"v": 1000}]
        assert second.to_pylist() == [{"v": 2}]


# =====================================================================
# Write-path validation
# =====================================================================

class TestWriteValidation:
    def test_write_requires_member(self):
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        with pytest.raises(ValueError, match="member"):
            io_.write_arrow_table(
                pa.Table.from_pylist([{"v": 1}]),
                options=ZipOptions(inner_media="json"),
            )

    def test_write_requires_inner_media(self):
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        with pytest.raises(ValueError, match="inner_media"):
            io_.write_arrow_table(
                pa.Table.from_pylist([{"v": 1}]),
                options=ZipOptions(member="data.json"),
            )

    def test_write_rejects_glob_as_member(self):
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        with pytest.raises(ValueError, match="member"):
            io_.write_arrow_table(
                pa.Table.from_pylist([{"v": 1}]),
                options=ZipOptions(
                    inner_media="json",
                    member="*.json",
                    mode=SaveMode.OVERWRITE,
                ),
            )


# =====================================================================
# Schema inspection
# =====================================================================

class TestSchema:
    def test_empty_buffer_schema_is_empty(self):
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        schema = io_._collect_arrow_schema()
        assert schema == pa.schema([])

    def test_schema_from_first_member(self):
        zbytes = _make_zip_bytes({
            "a.json": json_mod.dumps([{"id": 1, "name": "x"}]),
            "b.json": json_mod.dumps([{"id": 2, "extra": 42}]),
        })
        buf = BytesIO(zbytes)
        io_ = MediaIO.make(buf, MimeTypes.ZIP)

        schema = io_._collect_arrow_schema(full=False)
        # Only first member inspected.
        assert "id" in schema.names
        assert "name" in schema.names
        assert "extra" not in schema.names

    def test_full_schema_unifies_all_members(self):
        zbytes = _make_zip_bytes({
            "a.json": json_mod.dumps([{"id": 1, "name": "x"}]),
            "b.json": json_mod.dumps([{"id": 2, "extra": 42}]),
        })
        buf = BytesIO(zbytes)
        io_ = MediaIO.make(buf, MimeTypes.ZIP)

        schema = io_._collect_arrow_schema(full=True)
        names = set(schema.names)
        assert "id" in names
        assert "name" in names
        assert "extra" in names


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_column_projection_on_concat_result(self):
        zbytes = _make_zip_bytes({
            "a.json": json_mod.dumps([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]),
        })
        buf = BytesIO(zbytes)
        io_ = MediaIO.make(buf, MimeTypes.ZIP)

        out = io_.read_arrow_table(options=ZipOptions(columns=["x"]))
        assert out.column_names == ["x"]
        assert out.num_rows == 2

    def test_columns_can_include_injected_metadata(self):
        """Projection can reference a column injected by read_member_infos."""
        zbytes = _make_zip_bytes({"a.json": json_mod.dumps([{"v": 1}])})
        buf = BytesIO(zbytes)
        out = MediaIO.make(buf, MimeTypes.ZIP).read_arrow_table(
            options=ZipOptions(
                read_member_infos=[("name", "source")],
                columns=["source"],
            )
        )
        assert out.column_names == ["source"]
        assert out.column("source").to_pylist() == ["a.json"]

    def test_read_arrow_batches_empty_buffer_yields_nothing(self):
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        assert list(io_.read_arrow_batches()) == []

    def test_roundtrip_with_many_rows(self):
        """Multi-batch member (many rows) round-trips correctly."""
        rows = [{"i": i, "s": f"row_{i}"} for i in range(1000)]
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        io_.write_arrow_table(
            pa.Table.from_pylist(rows),
            options=ZipOptions(inner_media="json", member="bulk.json"),
        )

        out = io_.read_arrow_table(options=ZipOptions(member="bulk.json"))
        assert out.num_rows == 1000
        assert out.to_pylist() == rows


# =====================================================================
# Cast integration
# =====================================================================

class TestCastIntegration:
    def test_default_cast_is_identity(self):
        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        inp = pa.Table.from_pylist([{"a": 1}, {"a": 2}])

        io_.write_arrow_table(
            inp, options=ZipOptions(inner_media="json", member="d.json")
        )
        out = io_.read_arrow_table(options=ZipOptions(member="d.json"))
        assert out.to_pylist() == inp.to_pylist()

    def test_cast_reaches_write_path(self):
        try:
            from yggdrasil.data.cast.options import CastOptions
        except ImportError:
            pytest.skip("CastOptions not importable")

        target = pa.schema([pa.field("id", pa.int64())])

        try:
            cast = CastOptions(target_field=target)
        except TypeError:
            pytest.skip("CastOptions(target_field=...) signature mismatch")

        buf = BytesIO()
        io_ = MediaIO.make(buf, MimeTypes.ZIP)
        src = pa.table({"id": pa.array([1, 2, 3], type=pa.int32())})

        io_.write_arrow_table(
            src,
            options=ZipOptions(
                inner_media="json",
                member="d.json",
                cast=cast,
            ),
        )

        # Just verify write didn't crash and data is readable.
        out = io_.read_arrow_table(options=ZipOptions(member="d.json"))
        assert out.num_rows == 3