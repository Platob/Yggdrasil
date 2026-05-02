"""Arrow-side behavior for raw :class:`BytesIO`.

A raw byte buffer with no tabular ``media_type`` cannot read or
write Arrow batches on its own — the redirect through
``_tabular_view`` returns ``None`` and the call must raise rather
than silently no-op. With a tabular media type set, the buffer
delegates to the registered concrete leaf via ``as_media``.
"""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.enums import MimeTypes
from .._helpers import sample_table


class TestBytesIOOpaqueRaises:
    def test_read_arrow_batches_raises_without_media(self):
        bio = BytesIO(b"opaque")
        with pytest.raises(NotImplementedError):
            list(bio.read_arrow_batches())

    def test_write_arrow_batches_raises_without_media(self):
        bio = BytesIO()
        with pytest.raises(NotImplementedError):
            bio.write_arrow_batches(iter(sample_table().to_batches()))

    def test_persist_raises_without_media(self):
        bio = BytesIO(b"opaque")
        with pytest.raises(NotImplementedError):
            bio.persist()


class TestBytesIOTabularViaMedia:
    def test_round_trip_via_as_media_arrow_ipc(self, tmp_path):
        # When the buffer carries a tabular mime, `as_media` builds
        # the right leaf wrapping the same byte storage.
        path = tmp_path / "x.arrow"
        bio = BytesIO(path=str(path), media_type=MimeTypes.ARROW_IPC)
        # Construction with a tabular mime redirects through PrimitiveIO,
        # so the resulting `bio` is actually an ArrowIPCIO.
        bio.write_arrow_table(sample_table())
        bio.close()

        # Re-open via the same path and read back through
        # the format leaf.
        reader = BytesIO(path=str(path), media_type=MimeTypes.ARROW_IPC)
        out = reader.read_arrow_table()
        reader.close()
        assert out.num_rows == 3
        assert out.column_names == ["a", "b"]
