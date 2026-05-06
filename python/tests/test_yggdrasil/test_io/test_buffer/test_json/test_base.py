"""JsonIO core: registration, mime, byte-level write."""

from __future__ import annotations

from yggdrasil.io.primitive import JsonIO
from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.primitive import JsonOptions
from yggdrasil.data.enums import MimeTypes
from .._helpers import sample_table


class TestJsonBase:
    def test_default_media_type(self):
        assert JsonIO.default_media_type() == MimeTypes.JSON

    def test_options_class(self):
        assert JsonIO.options_class() is JsonOptions

    def test_dispatch_via_path(self, tmp_path):
        p = tmp_path / "x.json"
        p.touch()
        io = BytesIO(path=str(p))
        assert isinstance(io, JsonIO)


class TestJsonWrite:
    def test_write_emits_json(self, tmp_path):
        p = tmp_path / "x.json"
        JsonIO(path=str(p)).write_arrow_table(sample_table())
        text = p.read_text()
        # JsonIO writes a JSON array of records.
        assert text.strip().startswith("[")
        assert "henry" in text
