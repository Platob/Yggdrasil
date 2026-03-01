# yggdrasil/io/buffer/zip_io.py
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Optional, Self

from yggdrasil.io.enums import MediaType, MimeType, SaveMode
from .bytes_io import BytesIO
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["ZipOptions", "ZipIO"]


@dataclass(slots=True)
class ZipOptions(MediaOptions):
    """
    Options for ZipIO.

    member:
      - None: read ALL members (files) and concat
      - exact name: read only that member
      - glob pattern (contains '*', '?', or '['): read all matching members and concat

    inner_media:
      - when writing: media type used for the inner payload (default parquet)
      - when reading: optional override if force_inner_media=True

    force_inner_media:
      - if True and inner_media provided, use it for ALL members (skip inference)

    zip_compression:
      - deflate compresslevel (0..9-ish); method is ZIP_DEFLATED
    """

    member: str | None = None

    inner_media: MediaType | str | None = None
    force_inner_media: bool = False

    zip_compression: int = 8

    @classmethod
    def resolve(cls, *, options: Self | None = None, **overrides) -> Self:
        base = options or cls()
        valid = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        unknown = set(overrides) - set(valid)
        if unknown:
            raise TypeError(f"{cls.__name__}.resolve(): unknown option(s): {sorted(unknown)}")
        for k, v in overrides.items():
            setattr(base, k, v)
        return base


@dataclass(slots=True)
class ZipIO(MediaIO[ZipOptions]):
    """
    ZIP container IO on top of BytesIO.

    Read:
      - opens zip from a cursor-owned view (does not touch parent cursor)
      - selects members via ZipOptions.member (exact or glob or None)
      - infers inner media per member:
          1) by member name/extension (MediaType.parse_str)
          2) fallback: magic bytes via BytesIO(payload).media_type
      - reads each member via BytesIO(payload).media_io(inner_media).read_arrow_table()
      - concatenates tables (schema-relaxed) when multiple members

    Write:
      - serializes provided Arrow table into a single inner payload (inner_media)
      - writes a single zip member (options.member or default name)
      - replaces outer BytesIO bytes
    """

    buffer: BytesIO

    @classmethod
    def check_options(cls, options: Optional[ZipOptions], *args, **kwargs) -> ZipOptions:
        return ZipOptions.check_parameters(options=options, **kwargs)

    # ------------------------------------------------------------------
    # Member selection (exact or glob)
    # ------------------------------------------------------------------

    def _member_names(self, zf: zipfile.ZipFile) -> list[str]:
        return [n for n in zf.namelist() if n and not n.endswith("/")]

    def _is_glob(self, s: str) -> bool:
        return any(ch in s for ch in ("*", "?", "["))

    def _select_members(
        self,
        zf: zipfile.ZipFile,
        *,
        options: ZipOptions
    ) -> list[str]:
        names = self._member_names(zf)

        m = options.member
        if not m:
            return names

        # exact
        if not self._is_glob(m):
            if m not in names:
                raise KeyError(f"ZipIO: member not found: {m!r}. Available: {names}")
            return [m]

        # glob
        matched = sorted({n for n in names if fnmatchcase(n, m)})
        if not matched:
            raise KeyError(f"ZipIO: no members matched pattern={m!r}. Available: {names}")
        return matched

    # ------------------------------------------------------------------
    # Inner media inference
    # ------------------------------------------------------------------

    def _infer_inner_media_from_name(self, name: str) -> MediaType:
        try:
            return MediaType.parse_str(name)
        except Exception:
            return MediaType(MimeType.OCTET_STREAM)

    def _infer_inner_media(self, name: str, payload: bytes) -> MediaType:
        # 1) extension / name-based
        mt = self._infer_inner_media_from_name(name)
        if not mt.is_octet:
            return mt

        # 2) magic bytes fallback (works for parquet/ipc/zip/etc; JSON has no magic)
        try:
            return BytesIO(payload, config=self.buffer.config).media_type
        except Exception:
            return MediaType(MimeType.OCTET_STREAM)

    def _pick_inner_media_for_write(self, options: ZipOptions) -> MediaType:
        if options.inner_media is not None:
            mt = MediaType.parse(options.inner_media)
            if not mt.is_octet:
                return mt
        return MediaType(MimeType.PARQUET)

    def _default_member_for_write(self, inner_media: MediaType) -> str:
        ext = inner_media.full_extension
        return f"data.{ext}" if ext else "data"

    # ------------------------------------------------------------------
    # Zipfile kwargs (compression)
    # ------------------------------------------------------------------

    def _zipfile_kwargs(self, options: ZipOptions) -> dict:
        kw = {"compression": zipfile.ZIP_DEFLATED}
        # compresslevel support varies by Python build
        try:
            zipfile.ZipFile(
                io.BytesIO(),
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=int(options.zip_compression),
            ).close()
            kw["compresslevel"] = int(options.zip_compression)
        except TypeError:
            pass
        return kw

    # ------------------------------------------------------------------
    # MediaIO required impl
    # ------------------------------------------------------------------

    def _read_arrow_table(self, *, options: ZipOptions) -> "pyarrow.Table":
        from yggdrasil.arrow.lib import pyarrow as _pa

        if self.buffer.size <= 0:
            return _pa.table({})

        tables: list[_pa.Table] = []

        # cursor-safe: view owns its own cursor
        with self.buffer.view(text=False) as f:
            with zipfile.ZipFile(f, mode="r") as zf:
                members = self._select_members(zf, options=options)

                for name in members:
                    payload = zf.read(name)

                    if options.force_inner_media and options.inner_media is not None:
                        inner_media = MediaType.parse(options.inner_media)
                        if inner_media.is_octet:
                            inner_media = self._infer_inner_media(name, payload)
                    else:
                        inner_media = self._infer_inner_media(name, payload)

                    inner_buf = BytesIO(payload, config=self.buffer.config)
                    t = inner_buf.media_io(inner_media).read_arrow_table()
                    tables.append(t)

        if not tables:
            return _pa.table({})
        if len(tables) == 1:
            return tables[0]

        return _pa.concat_tables(tables, promote_options="default")

    def _write_arrow_table(self, *, table: "pyarrow.Table", options: ZipOptions) -> None:
        inner_media = self._pick_inner_media_for_write(options)

        inner_buf = BytesIO(config=self.buffer.config)
        inner_buf.media_io(inner_media).write_arrow_table(
            table,
            mode=SaveMode.OVERWRITE,
            match_by=options.match_by,
        )
        inner_payload = inner_buf.to_bytes()

        member = options.member or self._default_member_for_write(inner_media)

        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", **self._zipfile_kwargs(options)) as zf:
            zf.writestr(member, inner_payload)

        # overwrite outer bytes
        self.buffer._replace_with_payload(mem.getvalue())