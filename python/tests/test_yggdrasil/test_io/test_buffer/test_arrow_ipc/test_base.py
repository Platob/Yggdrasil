"""ArrowIPCIO core: registration, mime, save modes, byte round-trip."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.primitive import ArrowIPCIO
from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.buffer.primitive.arrow_ipc_io import ArrowIPCOptions
from yggdrasil.io.enums import Mode, MimeTypes
from .._helpers import sample_table


class TestArrowIPCIOBase:
    def test_default_media_type(self):
        assert ArrowIPCIO.default_media_type() == MimeTypes.ARROW_IPC

    def test_options_class(self):
        assert ArrowIPCIO.options_class() is ArrowIPCOptions

    def test_dispatch_via_path(self, tmp_path):
        p = tmp_path / "x.arrow"
        p.touch()
        io = BytesIO(path=str(p))
        assert isinstance(io, ArrowIPCIO)


class TestArrowIPCSaveMode:
    def test_auto_resolves_overwrite_on_empty(self):
        io = ArrowIPCIO()
        assert io._resolve_save_mode(Mode.AUTO) is Mode.OVERWRITE

    def test_truncate_resolves_overwrite(self):
        io = ArrowIPCIO()
        assert io._resolve_save_mode(Mode.TRUNCATE) is Mode.OVERWRITE

    def test_error_if_exists_raises_on_non_empty(self, tmp_path):
        p = tmp_path / "x.arrow"
        ArrowIPCIO(path=str(p)).write_arrow_table(sample_table())
        io = ArrowIPCIO(path=str(p))
        with pytest.raises(FileExistsError):
            io._resolve_save_mode(Mode.ERROR_IF_EXISTS)
