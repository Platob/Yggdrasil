"""NDJsonIO core."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.primitive import NDJsonIO
from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.buffer.primitive.ndjson_io import NDJsonOptions
from yggdrasil.io.enums import MimeTypes
from .._helpers import sample_table


class TestNDJsonBase:
    def test_default_mime_type(self):
        assert NDJsonIO.default_mime_type() == MimeTypes.NDJSON

    def test_options_class(self):
        assert NDJsonIO.options_class() is NDJsonOptions

    def test_dispatch_via_path(self, tmp_path):
        p = tmp_path / "x.ndjson"
        p.touch()
        io = BytesIO(path=str(p))
        assert isinstance(io, NDJsonIO)


class TestNDJsonWrite:
    def test_one_record_per_line(self, tmp_path):
        p = tmp_path / "x.ndjson"
        NDJsonIO(path=str(p)).write_arrow_table(sample_table())
        lines = p.read_text().strip().splitlines()
        # NDJSON: one JSON record per line.
        assert len(lines) == 3
        for line in lines:
            assert line.startswith("{")
