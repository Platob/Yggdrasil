"""Image :class:`Tabular` leaf — a one-row metadata projection.

An image isn't rows, but yggdrasil's io layer still wants a tabular view of
*any* recognised body. This leaf claims the image mime types (PNG / JPEG /
GIF / WEBP / BMP / TIFF) and yields a single Arrow row of metadata —
``format``, ``width``, ``height``, ``mode``, ``bytes`` — so
``IO.from_("logo.png").to_polars()`` (and a fetched image response) returns a
usable frame. Dimensions come from Pillow when present; otherwise they're
null. The raw bytes stay reachable via the holder (``read_bytes`` /
``content``); the leaf is read-only (a metadata row can't rebuild an image).
"""
from __future__ import annotations

import dataclasses
from typing import ClassVar, Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.enums import MimeTypes
from yggdrasil.io.base import IO
from yggdrasil.io.holder import _HOLDER_FORMAT_REGISTRY

__all__ = ["ImageFile", "ImageOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class ImageOptions(CastOptions):
    """:class:`CastOptions` for image metadata reads (no extra knobs yet)."""


class ImageFile(IO[bytes, ImageOptions]):
    """:class:`Tabular` leaf for images — one row of metadata."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.PNG

    @classmethod
    def options_class(cls):
        return ImageOptions

    def _read_arrow_batches(self, options: ImageOptions) -> Iterator[pa.RecordBatch]:
        data = self.read_bytes()
        fmt = width = height = mode = None
        try:
            import io as _io

            from PIL import Image

            with Image.open(_io.BytesIO(data)) as im:
                fmt, width, height, mode = im.format, im.width, im.height, im.mode
        except Exception:
            pass
        table = pa.table({
            "format": pa.array([fmt], pa.string()),
            "width": pa.array([width], pa.int64()),
            "height": pa.array([height], pa.int64()),
            "mode": pa.array([mode], pa.string()),
            "bytes": pa.array([len(data)], pa.int64()),
        })
        for batch in table.to_batches():
            yield options.cast_arrow_batch(batch)

    def _write_arrow_batches(
        self, batches: Iterable[pa.RecordBatch], options: ImageOptions
    ) -> None:
        raise NotImplementedError(
            "ImageFile is read-only — a metadata projection cannot rebuild an "
            "image. Write the raw bytes through a blob sink instead."
        )


# One leaf serves every image mime; PNG self-registers via __init_subclass__,
# the rest are aliased onto it here (setdefault never clobbers a real leaf).
for _mt in (MimeTypes.JPEG, MimeTypes.GIF, MimeTypes.WEBP, MimeTypes.BMP, MimeTypes.TIFF):
    _HOLDER_FORMAT_REGISTRY.setdefault(_mt.name, ImageFile)
