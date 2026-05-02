"""Tests for ``yggdrasil.io.buffer.nested.delta``.

The Delta package is large and reaches into pyroaring for deletion-
vector support. The tests here focus on the units that DON'T need
pyroaring or a full DeltaIO write round-trip:

- The action dataclasses' parse/serialize round-trip and the
  envelope dispatch in :func:`parse_action`.
- The :class:`DeletionVectorDescriptor` validation and JSON
  round-trip (without touching the bitmap layer).
- The Z85 codec used for inline DV descriptors.
- :func:`build_commit_body` and :func:`write_commit`'s atomicity
  refusal on duplicate version.
- :func:`replay_log` over a hand-written commit, plus its empty
  / gap / missing-folder behaviour.
- The :class:`DeltaOptions` defaults and :class:`DeltaIO` registry
  hook.
- :class:`Protocol` validation refuses unsupported reader features.

DV bitmap encode/decode tests are gated by ``pytest.importorskip``
so the suite passes on a base install.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yggdrasil.data.schema import Field, Schema
from yggdrasil.io.buffer.nested.delta import (
    AddFile,
    CommitInfo,
    DeltaIO,
    DeltaOptions,
    DeletionVectorDescriptor,
    DomainMetadata,
    MAX_INLINE_DV_BYTES,
    Metadata,
    Protocol,
    RemoveFile,
    ReplayResult,
    Txn,
    replay_log,
)
from yggdrasil.io.buffer.nested.delta.actions import (
    parse_action,
    parse_add,
    parse_commit_info,
    parse_metadata,
    parse_protocol,
    parse_remove,
    parse_txn,
    serialize_action,
    serialize_add,
    serialize_commit_info,
    serialize_metadata,
    serialize_protocol,
    serialize_remove,
)
from yggdrasil.io.buffer.nested.delta.commit import (
    build_commit_body,
    commit_path_for_version,
    write_commit,
)
from yggdrasil.io.buffer.nested.delta.constants import (
    COMMIT_FILE_RE,
    DEFAULT_ENGINE_INFO,
    DV_DIR_NAME,
    READER_VERSION_FEATURES,
    WRITER_VERSION_FEATURES,
)
from yggdrasil.io.buffer.nested.delta.deletion_vector import (
    _z85_decode,
    _z85_encode,
)
from yggdrasil.io.buffer.nested.delta.replay import (
    latest_commit_version,
    read_last_checkpoint,
)
from yggdrasil.io.buffer.nested.delta.schema_codec import (
    delta_schema_string_to_schema,
)
from yggdrasil.io.enums import MimeTypes
from yggdrasil.io.fs import LocalPath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _local(p: Path) -> LocalPath:
    return LocalPath.from_(str(p))


def _empty_struct_schema_string() -> str:
    return json.dumps({"type": "struct", "fields": []})


def _seed_commit_zero(
    log_dir: Path,
    *,
    schema_string: str | None = None,
    partition_columns: list[str] | None = None,
) -> None:
    """Write a minimal version-0 commit with Protocol + Metadata."""
    log_dir.mkdir(parents=True, exist_ok=True)
    schema_string = schema_string or _empty_struct_schema_string()

    actions = [
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        {
            "metaData": {
                "id": "tbl",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": schema_string,
                "partitionColumns": partition_columns or [],
                "configuration": {},
            }
        },
    ]
    body = "\n".join(json.dumps(a) for a in actions) + "\n"
    (log_dir / "00000000000000000000.json").write_bytes(body.encode())


# ---------------------------------------------------------------------------
# DeltaOptions / DeltaIO registry hook
# ---------------------------------------------------------------------------


class TestDeltaIOSurface:
    def test_default_mime_type(self):
        assert DeltaIO.default_mime_type() == MimeTypes.DELTA_FOLDER

    def test_options_defaults(self):
        opts = DeltaOptions()
        assert opts.parquet_compression == "snappy"
        assert opts.commit_info_engine == DEFAULT_ENGINE_INFO
        assert opts.require_existing_table is False
        assert opts.dv_inline_threshold == MAX_INLINE_DV_BYTES
        assert opts.checkpoint_v2 is False

    def test_constructor_requires_path(self):
        with pytest.raises(ValueError):
            DeltaIO()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_dv_dir_name(self):
        assert DV_DIR_NAME == "deletion_vectors"

    def test_reader_writer_feature_versions(self):
        # The "table-features" model gates kicked in at reader v3 /
        # writer v7.
        assert READER_VERSION_FEATURES == 3
        assert WRITER_VERSION_FEATURES == 7

    def test_commit_file_re_matches_zero_padded(self):
        assert COMMIT_FILE_RE.match("00000000000000000000.json")
        assert COMMIT_FILE_RE.match("00000000000000000123.json")

    def test_commit_file_re_rejects_other(self):
        assert COMMIT_FILE_RE.match("foo.json") is None
        assert COMMIT_FILE_RE.match("00000000000000000000.parquet") is None


# ---------------------------------------------------------------------------
# Action dataclasses — parse/serialize round-trip
# ---------------------------------------------------------------------------


class TestProtocolRoundTrip:
    def test_basic(self):
        p = Protocol(min_reader_version=1, min_writer_version=2)
        ser = serialize_protocol(p)
        assert ser == {"minReaderVersion": 1, "minWriterVersion": 2}
        assert parse_protocol(ser) == p

    def test_with_features(self):
        p = Protocol(
            min_reader_version=3,
            min_writer_version=7,
            reader_features=("deletionVectors",),
            writer_features=("deletionVectors", "appendOnly"),
        )
        ser = serialize_protocol(p)
        assert "readerFeatures" in ser
        assert ser["writerFeatures"] == ["deletionVectors", "appendOnly"]
        assert parse_protocol(ser) == p

    def test_no_features_omits_keys(self):
        p = Protocol(min_reader_version=1, min_writer_version=2)
        ser = serialize_protocol(p)
        assert "readerFeatures" not in ser
        assert "writerFeatures" not in ser


class TestAddFileRoundTrip:
    def test_round_trip(self):
        add = AddFile(
            path="part-001.parquet",
            partition_values={"year": "2024"},
            size=100,
            modification_time=12345,
            data_change=True,
            stats="{}",
        )
        ser = serialize_add(add)
        assert ser["path"] == "part-001.parquet"
        assert ser["partitionValues"] == {"year": "2024"}
        assert ser["size"] == 100
        assert parse_add(ser) == add

    def test_path_url_encoded_on_serialize(self):
        # Slash and = characters are preserved (Hive partitions);
        # other unsafe characters get percent-encoded.
        add = AddFile(
            path="year=2024/data file.parquet",
            partition_values={"year": "2024"},
            size=10,
            modification_time=0,
            data_change=True,
        )
        ser = serialize_add(add)
        assert "data%20file.parquet" in ser["path"]
        assert "year=2024/" in ser["path"]
        # parse_add reverses the encoding.
        assert parse_add(ser).path == "year=2024/data file.parquet"

    def test_optional_fields_omitted_when_none(self):
        add = AddFile(
            path="x.parquet",
            partition_values={},
            size=10,
            modification_time=0,
            data_change=True,
        )
        ser = serialize_add(add)
        assert "stats" not in ser
        assert "tags" not in ser
        assert "deletionVector" not in ser


class TestRemoveFileRoundTrip:
    def test_basic(self):
        rm = RemoveFile(
            path="part-001.parquet",
            deletion_timestamp=12345,
            data_change=True,
        )
        ser = serialize_remove(rm)
        assert ser["path"] == "part-001.parquet"
        assert parse_remove(ser) == rm


class TestCommitInfoRoundTrip:
    def test_basic(self):
        ci = CommitInfo(timestamp=1, operation="WRITE", engine_info="ygg")
        ser = serialize_commit_info(ci)
        assert ser["timestamp"] == 1
        assert ser["operation"] == "WRITE"
        assert ser["engineInfo"] == "ygg"
        # parse_commit_info round-trip
        got = parse_commit_info(ser)
        assert got.timestamp == ci.timestamp
        assert got.operation == ci.operation
        assert got.engine_info == ci.engine_info


class TestActionEnvelope:
    def test_serialize_action_dispatch(self):
        for value, expected_key in (
            (
                AddFile(
                    path="x", partition_values={}, size=1,
                    modification_time=1, data_change=True,
                ),
                "add",
            ),
            (
                RemoveFile(path="x", deletion_timestamp=1, data_change=True),
                "remove",
            ),
            (Protocol(min_reader_version=1, min_writer_version=2), "protocol"),
            (CommitInfo(timestamp=1, operation="x"), "commitInfo"),
            (Txn(app_id="a", version=1), "txn"),
            (
                DomainMetadata(domain="foo", configuration="{}"),
                "domainMetadata",
            ),
        ):
            envelope = serialize_action(value)
            assert list(envelope) == [expected_key]

    def test_serialize_action_unknown_raises(self):
        with pytest.raises(TypeError):
            serialize_action(object())

    def test_parse_action_dispatch(self):
        envelope = {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}}
        result = parse_action(envelope)
        assert isinstance(result, Protocol)

    def test_parse_action_skips_cdc(self):
        # CDC actions are deliberately skipped during replay.
        assert parse_action({"cdc": {"path": "anything"}}) is None

    def test_parse_action_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            parse_action({"someUnknownKind": {}})


# ---------------------------------------------------------------------------
# DeletionVectorDescriptor — validation + JSON round-trip
# ---------------------------------------------------------------------------


class TestDeletionVectorDescriptor:
    def test_inline_descriptor_basic(self):
        d = DeletionVectorDescriptor(
            storage_type="i",
            path_or_inline="abcde",
            size_in_bytes=10,
            cardinality=2,
        )
        assert d.is_inline
        assert not d.is_empty
        assert d.relative_path() is None

    def test_path_descriptor_basic(self):
        d = DeletionVectorDescriptor(
            storage_type="p",
            path_or_inline="rel/path.bin",
            size_in_bytes=10,
            cardinality=0,
            offset=0,
        )
        assert not d.is_inline
        assert d.is_empty
        assert d.relative_path() == "rel/path.bin"

    def test_uuid_descriptor_relative_path(self):
        d = DeletionVectorDescriptor(
            storage_type="u",
            path_or_inline="abcdefghij1234567890/sub.bin",
            size_in_bytes=10,
            cardinality=1,
            offset=0,
        )
        assert d.relative_path() == "abcdefghij1234567890/sub.bin"

    def test_unknown_storage_type_rejected(self):
        with pytest.raises(ValueError, match="storageType"):
            DeletionVectorDescriptor(
                storage_type="X",
                path_or_inline="",
                size_in_bytes=10,
                cardinality=0,
            )

    def test_negative_cardinality_rejected(self):
        with pytest.raises(ValueError, match="cardinality"):
            DeletionVectorDescriptor(
                storage_type="i",
                path_or_inline="abc",
                size_in_bytes=10,
                cardinality=-1,
            )

    def test_zero_size_rejected(self):
        with pytest.raises(ValueError, match="sizeInBytes"):
            DeletionVectorDescriptor(
                storage_type="i",
                path_or_inline="abc",
                size_in_bytes=0,
                cardinality=0,
            )

    def test_inline_with_offset_rejected(self):
        with pytest.raises(ValueError, match="offset"):
            DeletionVectorDescriptor(
                storage_type="i",
                path_or_inline="abc",
                size_in_bytes=10,
                cardinality=0,
                offset=4,
            )

    def test_json_round_trip_inline(self):
        d = DeletionVectorDescriptor(
            storage_type="i",
            path_or_inline="abcde",
            size_in_bytes=10,
            cardinality=2,
        )
        raw = d.to_json()
        assert raw["storageType"] == "i"
        assert "offset" not in raw
        got = DeletionVectorDescriptor.from_json(raw)
        assert got == d

    def test_json_round_trip_path_with_offset(self):
        d = DeletionVectorDescriptor(
            storage_type="p",
            path_or_inline="x.bin",
            size_in_bytes=20,
            cardinality=3,
            offset=128,
        )
        raw = d.to_json()
        assert raw["offset"] == 128
        got = DeletionVectorDescriptor.from_json(raw)
        assert got == d


# ---------------------------------------------------------------------------
# Z85 codec
# ---------------------------------------------------------------------------


class TestZ85:
    def test_round_trip_known_pattern(self):
        # 4-byte input → 5-char output.
        data = b"\x00\x01\x02\x03"
        encoded = _z85_encode(data)
        assert len(encoded) == 5
        assert _z85_decode(encoded) == data

    def test_round_trip_random(self):
        data = b"".join(bytes([i % 256]) for i in range(64))
        encoded = _z85_encode(data)
        assert _z85_decode(encoded) == data

    def test_input_must_be_multiple_of_4(self):
        with pytest.raises(ValueError):
            _z85_encode(b"abc")  # 3 bytes

    def test_decode_input_must_be_multiple_of_5(self):
        with pytest.raises(ValueError):
            _z85_decode("abc")

    def test_decode_invalid_char_rejected(self):
        # `"` is outside the Z85 alphabet.
        with pytest.raises(ValueError):
            _z85_decode('"' * 5)


# ---------------------------------------------------------------------------
# DV blob round-trip — needs pyroaring
# ---------------------------------------------------------------------------


class TestDvBlob:
    def test_roaring_round_trip(self):
        pyroaring = pytest.importorskip("pyroaring")
        from yggdrasil.io.buffer.nested.delta.deletion_vector import (
            bitmap_from_iter,
            decode_dv_blob,
            decode_inline_descriptor,
            encode_dv_blob,
            make_inline_descriptor,
        )
        bm = bitmap_from_iter([1, 5, 9, 100])
        blob = encode_dv_blob(bm, include_crc=True)
        got = decode_dv_blob(blob, expected_cardinality=4, has_crc=True)
        assert sorted(got) == [1, 5, 9, 100]

        desc = make_inline_descriptor(bm)
        assert desc.is_inline
        round_tripped = decode_inline_descriptor(desc)
        assert sorted(round_tripped) == [1, 5, 9, 100]

    def test_decode_too_short(self):
        pytest.importorskip("pyroaring")
        from yggdrasil.io.buffer.nested.delta.deletion_vector import (
            decode_dv_blob,
        )
        with pytest.raises(ValueError):
            decode_dv_blob(b"\x01\x00", expected_cardinality=0, has_crc=False)

    def test_decode_unsupported_format_byte(self):
        pytest.importorskip("pyroaring")
        from yggdrasil.io.buffer.nested.delta.deletion_vector import (
            decode_dv_blob,
        )
        # Format byte = 99, payload size = 0.
        blob = b"\x63\x00\x00\x00\x00"
        with pytest.raises(ValueError, match="format byte"):
            decode_dv_blob(blob, expected_cardinality=0, has_crc=False)


# ---------------------------------------------------------------------------
# Commit body builder + writer
# ---------------------------------------------------------------------------


class TestCommitBody:
    def test_build_commit_body_emits_one_line_per_action(self):
        body = build_commit_body(
            [
                Protocol(min_reader_version=1, min_writer_version=2),
                CommitInfo(timestamp=10, operation="WRITE"),
            ]
        )
        text = body.decode()
        # Trailing newline + two non-empty lines.
        lines = [ln for ln in text.split("\n") if ln]
        assert len(lines) == 2
        assert text.endswith("\n")
        # First line is the protocol envelope.
        first = json.loads(lines[0])
        assert "protocol" in first

    def test_build_commit_body_accepts_dict_envelopes(self):
        # build_commit_body accepts pre-wrapped action dicts directly.
        body = build_commit_body([{"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}}])
        text = body.decode()
        assert "protocol" in text

    def test_commit_path_for_version(self, tmp_path: Path):
        log = _local(tmp_path / "_delta_log")
        out = commit_path_for_version(log, 5)
        assert out.name == "00000000000000000005.json"


class TestWriteCommit:
    def test_write_commit_creates_file(self, tmp_path: Path):
        log = _local(tmp_path / "_delta_log")
        target = write_commit(
            log, 0, [Protocol(min_reader_version=1, min_writer_version=2)]
        )
        assert target.exists()
        assert target.name == "00000000000000000000.json"

    def test_duplicate_version_raises(self, tmp_path: Path):
        log = _local(tmp_path / "_delta_log")
        write_commit(
            log, 0, [Protocol(min_reader_version=1, min_writer_version=2)]
        )
        with pytest.raises(FileExistsError):
            write_commit(
                log, 0, [Protocol(min_reader_version=1, min_writer_version=2)]
            )


# ---------------------------------------------------------------------------
# replay_log + helpers
# ---------------------------------------------------------------------------


class TestReplayHelpers:
    def test_latest_commit_version_missing_log(self, tmp_path: Path):
        log = _local(tmp_path / "missing")
        assert latest_commit_version(log) == -1

    def test_latest_commit_version_picks_highest(self, tmp_path: Path):
        log = tmp_path / "_delta_log"
        log.mkdir()
        for v in (0, 1, 2, 7):
            (log / f"{v:020d}.json").write_text("{}\n")
        assert latest_commit_version(_local(log)) == 7

    def test_latest_commit_version_ignores_non_commit_files(self, tmp_path: Path):
        log = tmp_path / "_delta_log"
        log.mkdir()
        (log / "_last_checkpoint").write_text("{}")
        (log / "00000000000000000003.json").write_text("{}\n")
        assert latest_commit_version(_local(log)) == 3

    def test_read_last_checkpoint_missing(self, tmp_path: Path):
        log = tmp_path / "_delta_log"
        log.mkdir()
        assert read_last_checkpoint(_local(log)) is None

    def test_read_last_checkpoint_present(self, tmp_path: Path):
        log = tmp_path / "_delta_log"
        log.mkdir()
        (log / "_last_checkpoint").write_text(json.dumps({"version": 12}))
        out = read_last_checkpoint(_local(log))
        assert out == {"version": 12}


class TestReplayLog:
    def test_empty_returns_empty_result(self, tmp_path: Path):
        log = _local(tmp_path / "missing_dir")
        result = replay_log(log)
        assert isinstance(result, ReplayResult)
        assert result.is_empty
        assert result.version == -1

    def test_single_commit(self, tmp_path: Path):
        log = tmp_path / "_delta_log"
        _seed_commit_zero(log)
        result = replay_log(_local(log))
        assert result.version == 0
        assert isinstance(result.protocol, Protocol)
        assert isinstance(result.metadata, Metadata)
        assert result.metadata.id == "tbl"

    def test_add_then_remove_drops_from_live_set(self, tmp_path: Path):
        log = tmp_path / "_delta_log"
        _seed_commit_zero(log)
        # version 1: add file
        actions_1 = [
            {
                "add": {
                    "path": "a.parquet",
                    "partitionValues": {},
                    "size": 1,
                    "modificationTime": 0,
                    "dataChange": True,
                }
            }
        ]
        (log / "00000000000000000001.json").write_bytes(
            ("\n".join(json.dumps(a) for a in actions_1) + "\n").encode()
        )
        # version 2: remove same file
        actions_2 = [
            {
                "remove": {
                    "path": "a.parquet",
                    "deletionTimestamp": 1,
                    "dataChange": True,
                }
            }
        ]
        (log / "00000000000000000002.json").write_bytes(
            ("\n".join(json.dumps(a) for a in actions_2) + "\n").encode()
        )

        result = replay_log(_local(log))
        assert result.version == 2
        # Removed file should be absent from the live set.
        assert all(f.path != "a.parquet" for f in result.live_files)

    def test_gap_in_commit_sequence_raises(self, tmp_path: Path):
        log = tmp_path / "_delta_log"
        _seed_commit_zero(log)
        # Skip version 1; jump to 2.
        (log / "00000000000000000002.json").write_bytes(
            b'{"commitInfo":{"timestamp":1,"operation":"x"}}\n'
        )
        with pytest.raises(FileNotFoundError, match="gap"):
            replay_log(_local(log))

    def test_no_protocol_or_metadata_raises(self, tmp_path: Path):
        log = tmp_path / "_delta_log"
        log.mkdir(parents=True, exist_ok=True)
        # Commit-zero without Protocol/Metadata.
        (log / "00000000000000000000.json").write_bytes(
            b'{"commitInfo":{"timestamp":1,"operation":"x"}}\n'
        )
        with pytest.raises(ValueError, match="Protocol/Metadata"):
            replay_log(_local(log))

    def test_unsupported_reader_features_refused(self, tmp_path: Path):
        log = tmp_path / "_delta_log"
        log.mkdir(parents=True, exist_ok=True)
        # min_reader_version=3 turns reader_features into authoritative
        # gating; an unknown feature must be refused.
        actions = [
            {
                "protocol": {
                    "minReaderVersion": 3,
                    "minWriterVersion": 7,
                    "readerFeatures": ["someUnknownFutureFeature"],
                    "writerFeatures": [],
                }
            },
            {
                "metaData": {
                    "id": "tbl",
                    "format": {"provider": "parquet", "options": {}},
                    "schemaString": _empty_struct_schema_string(),
                    "partitionColumns": [],
                    "configuration": {},
                }
            },
        ]
        body = "\n".join(json.dumps(a) for a in actions) + "\n"
        (log / "00000000000000000000.json").write_bytes(body.encode())

        with pytest.raises(ValueError):
            replay_log(_local(log))


# ---------------------------------------------------------------------------
# Schema codec — parse direction
# ---------------------------------------------------------------------------


class TestSchemaCodec:
    def test_parse_empty_struct(self):
        s = delta_schema_string_to_schema(_empty_struct_schema_string())
        assert isinstance(s, Schema)


# ---------------------------------------------------------------------------
# ReplayResult helpers
# ---------------------------------------------------------------------------


class TestReplayResult:
    def test_empty_factory(self):
        empty = ReplayResult.empty()
        assert empty.is_empty
        assert empty.version == -1
        assert empty.live_files == ()
        assert empty.protocol is None
        assert empty.metadata is None

    def test_non_empty_is_empty_false(self):
        result = ReplayResult(
            version=0,
            protocol=Protocol(min_reader_version=1, min_writer_version=2),
            metadata=None,
            live_files=(),
            domain_metadata={},
        )
        assert not result.is_empty
