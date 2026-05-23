"""Buffered-page cache on :class:`RemotePath`.

Reads against the same byte range collapse to a single backend GET;
buffered writes batch into one PUT on :meth:`flush` / release. Tests
drive the surface through :class:`S3Path` with a counting mock so the
backend hit count is observable.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from yggdrasil.aws.fs.path import S3Path
from yggdrasil.aws.fs.service import S3Service


def _service_for(client: MagicMock) -> MagicMock:
    """Wrap a boto-shaped mock client in a mock :class:`S3Service`.

    :class:`S3Path` reaches the boto surface through
    ``self.service.boto_client``; tests build the boto mock then
    hand back a service-shaped wrapper that exposes it.
    """
    svc = MagicMock(spec=S3Service)
    svc.boto_client = client
    return svc


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        pass


def _counting_client(
    payload: bytes = b"",
) -> tuple[MagicMock, dict[str, int], dict[str, Any]]:
    """Mock S3 client that honours ``Range`` and counts every call."""
    state: dict[str, Any] = {"buf": payload}
    counts = {"get": 0, "head": 0, "put": 0}

    def head_object(*, Bucket: str, Key: str) -> dict:
        counts["head"] += 1
        buf = state["buf"]
        if buf is None:
            err = Exception("NoSuchKey")
            err.response = {  # type: ignore[attr-defined]
                "Error": {"Code": "NoSuchKey"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            }
            raise err
        return {"ContentLength": len(buf), "LastModified": None}

    def get_object(*, Bucket: str, Key: str, Range: str | None = None) -> dict:
        counts["get"] += 1
        buf = state["buf"] or b""
        if Range:
            spec = Range.split("=", 1)[1]
            start_str, end_str = spec.split("-", 1)
            start = int(start_str)
            end = int(end_str) if end_str else len(buf) - 1
            data = buf[start : end + 1]
        else:
            data = buf
        return {"Body": _Body(data), "ContentLength": len(data)}

    def put_object(*, Bucket: str, Key: str, Body: Any) -> dict:
        counts["put"] += 1
        state["buf"] = Body if isinstance(Body, (bytes, bytearray)) else bytes(Body)
        return {}

    client = MagicMock()
    client.head_object.side_effect = head_object
    client.get_object.side_effect = get_object
    client.put_object.side_effect = put_object
    client.list_objects_v2.side_effect = lambda **_: {"KeyCount": 0}
    return client, counts, state


class TestPagedReads:

    def test_repeated_reads_collapse_to_one_get(self) -> None:
        payload = b"x" * 10_000
        client, counts, _ = _counting_client(payload)
        # Page comfortably fits the whole file.
        p = S3Path("s3://b/k", service=_service_for(client), page_size=64 * 1024)
        assert p.read_bytes(100, 0) == payload[:100]
        assert p.read_bytes(50, 50) == payload[50:100]
        assert p.read_bytes(200, 0) == payload[:200]
        assert counts["get"] == 1

    def test_multi_page_read(self) -> None:
        payload = b"abcdefghij" * 1_024  # 10 KB
        client, counts, _ = _counting_client(payload)
        # 2 KB page → 5 pages.
        p = S3Path("s3://b/k", service=_service_for(client), page_size=2048)
        assert p.read_bytes(5000, 1000) == payload[1000:6000]
        # Window touches pages 0, 1, 2.
        assert counts["get"] == 3
        # Re-read inside the populated pages — zero new GETs.
        counts["get"] = 0
        assert p.read_bytes(100, 2000) == payload[2000:2100]
        assert counts["get"] == 0
        # Reach into a fresh page — one more GET.
        assert p.read_bytes(100, 8000) == payload[8000:8100]
        assert counts["get"] == 1

    def test_page_size_none_disables_paging(self) -> None:
        payload = b"y" * 100
        client, counts, _ = _counting_client(payload)
        p = S3Path("s3://b/k", service=_service_for(client), page_size=None)
        assert p.page_size is None
        p.read_bytes(10, 0)
        p.read_bytes(10, 0)
        assert counts["get"] == 2  # No paging → one GET per call.

    def test_page_size_string_parses_via_byteunit(self) -> None:
        p = S3Path("s3://b/k", service=_service_for(MagicMock()), page_size="8 KB")
        assert p.page_size == 8 * 1024
        q = S3Path("s3://b/k2", service=_service_for(MagicMock()), page_size="4 MB")
        assert q.page_size == 4 * 1024 * 1024

    def test_page_size_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="page_size"):
            S3Path("s3://b/bad", service=_service_for(MagicMock()), page_size="oops")

    def test_default_page_size_is_4mib(self) -> None:
        # Documents the default the CLAUDE.md guidance refers to.
        p = S3Path("s3://b/k", service=_service_for(MagicMock()))
        assert p.page_size == 4 * 1024 * 1024


class TestBufferedWrites:

    def test_writes_batch_into_one_put_inside_with_block(self) -> None:
        client, counts, store = _counting_client(b"")
        p = S3Path("s3://b/k", service=_service_for(client), page_size=64 * 1024)
        with p:
            p.write_mv(memoryview(b"AAAA"), offset=0)
            p.write_mv(memoryview(b"BBBB"), offset=10)
            assert counts["put"] == 0  # Deferred while acquired.
        assert counts["put"] == 1
        assert store["buf"][:4] == b"AAAA"
        assert store["buf"][10:14] == b"BBBB"

    def test_partial_write_preserves_existing_tail(self) -> None:
        client, counts, store = _counting_client(b"ABCDEFGHIJ")
        p = S3Path("s3://b/k", service=_service_for(client), page_size=64 * 1024)
        # Outside a ``with`` block — buffered write flushes immediately
        # so the closed-state direct-call contract still holds.
        p.write_mv(memoryview(b"XY"), offset=2)
        assert store["buf"] == b"ABXYEFGHIJ"

    def test_closed_state_write_flushes_immediately(self) -> None:
        client, counts, store = _counting_client(b"")
        p = S3Path("s3://b/k", service=_service_for(client), page_size=64 * 1024)
        p.write_bytes(b"hello")
        assert counts["put"] == 1
        assert store["buf"] == b"hello"

    def test_explicit_flush(self) -> None:
        client, counts, store = _counting_client(b"")
        p = S3Path("s3://b/k", service=_service_for(client), page_size=64 * 1024)
        p.acquire()
        try:
            p.write_mv(memoryview(b"x" * 500), offset=0)
            assert counts["put"] == 0
            p.flush()
            assert counts["put"] == 1
            assert store["buf"] == b"x" * 500
            # Second flush with no dirty pages → no extra PUT.
            p.flush()
            assert counts["put"] == 1
        finally:
            p.close()

    def test_open_wb_then_open_rb_round_trips_payload(self) -> None:
        """``with path.open("wb") as fh: fh.write(...)`` must flush on
        cursor close — the borrowed-parent cursor doesn't close the
        path, so without an explicit flush hop the buffered pages
        would never land. The follow-up ``open("rb")`` then reads
        the just-uploaded bytes from the backend.

        The OVERWRITE-mode truncate is a separate, pre-existing
        round trip (``put_object`` with empty body); we don't pin
        the exact PUT count here, only that the final state
        carries the payload and a follow-up read sees it."""
        client, counts, store = _counting_client(b"")
        p = S3Path("s3://b/k", service=_service_for(client), page_size=64 * 1024)
        with p.open("wb") as fh:
            fh.write(b"hello context")
        # Cursor close → parent.flush() → buffered page lands.
        assert store["buf"] == b"hello context"
        with p.open("rb") as fh:
            assert fh.read() == b"hello context"

    def test_invalidate_singleton_drops_pages(self) -> None:
        client, counts, _ = _counting_client(b"z" * 100)
        p = S3Path("s3://b/k", service=_service_for(client), page_size=64 * 1024)
        # Warm the cache.
        p.read_bytes(10, 0)
        assert counts["get"] == 1
        p.invalidate_singleton()
        # Pages dropped — next read re-fetches.
        p.read_bytes(10, 0)
        assert counts["get"] == 2
