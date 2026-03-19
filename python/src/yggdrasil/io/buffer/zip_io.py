"""ZIP container I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Reads ZIP archives by selecting members (exact name, glob, or all),
inferring each member's inner media type, and concatenating the resulting
Arrow tables.  Writes serialise an Arrow table into a single inner
payload (defaulting to Parquet) inside a new ZIP archive.

Transport-level compression is handled transparently by the base class.
"""
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
    """Options for ZIP I/O.

    Parameters
    ----------
    member:
        Which member(s) to read:

        * ``None`` — read **all** members and concatenate.
        * An exact name — read only that member.
        * A glob pattern (``*``, ``?``, ``[``) — read matching members
          and concatenate.
    inner_media:
        Media type for inner payloads.  Used as the write format (default
        Parquet) and optionally forced on read when *force_inner_media* is
        ``True``.
    force_inner_media:
        When ``True`` and *inner_media* is set, skip per-member media
        inference and use *inner_media* for all members.
    zip_compression:
        Deflate compression level (0–9).  Method is always
        ``ZIP_DEFLATED``.
    """

    member: str | None = None
    inner_media: MediaType | str | None = None
    force_inner_media: bool = False
    zip_compression: int = 8

    @classmethod
    def resolve(cls, *, options: Self | None = None, **overrides) -> Self:
        """Merge *overrides* into *options* (or a fresh default)."""
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
    """ZIP container I/O.

    Reads are cursor-safe — a :meth:`BytesIO.view` owns its own cursor
    so the parent buffer position is never disturbed.
    """

    @classmethod
    def check_options(cls, options: Optional[ZipOptions], *args, **kwargs) -> ZipOptions:
        """Validate and merge caller-supplied options."""
        return ZipOptions.check_parameters(options=options, **kwargs)

    # ------------------------------------------------------------------
    # Member selection (exact or glob)
    # ------------------------------------------------------------------

    def _member_names(self, zf: zipfile.ZipFile) -> list[str]:
        """Return non-directory member names."""
        return [n for n in zf.namelist() if n and not n.endswith("/")]

    @staticmethod
    def _is_glob(s: str) -> bool:
        """Return ``True`` when *s* contains glob metacharacters."""
        return any(ch in s for ch in ("*", "?", "["))

    def _select_members(
        self,
        zf: zipfile.ZipFile,
        *,
        options: ZipOptions,
    ) -> list[str]:
        """Select members from *zf* according to *options.member*."""
        names = self._member_names(zf)

        m = options.member
        if not m:
            return names

        if not self._is_glob(m):
            if m not in names:
                raise KeyError(f"ZipIO: member not found: {m!r}. Available: {names}")
            return [m]

        matched = sorted({n for n in names if fnmatchcase(n, m)})
        if not matched:
            raise KeyError(f"ZipIO: no members matched pattern={m!r}. Available: {names}")
        return matched

    # ------------------------------------------------------------------
    # Inner media inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_inner_media_from_name(name: str) -> MediaType:
        """Infer media type from the member file name / extension."""
        try:
            return MediaType.parse(name)
        except Exception:
            return MediaType(MimeType.OCTET_STREAM)

    def _infer_inner_media(self, name: str, payload: bytes) -> MediaType:
        """Infer media type by name first, then by magic-byte sniffing."""
        mt = self._infer_inner_media_from_name(name)
        if not mt.is_octet:
            return mt

        try:
            return BytesIO(payload, config=self.buffer.config).media_type
        except Exception:
            return MediaType(MimeType.OCTET_STREAM)

    @staticmethod
    def _pick_inner_media_for_write(options: ZipOptions) -> MediaType:
        """Resolve the inner media type used when writing."""
        if options.inner_media is not None:
            mt = MediaType.parse(options.inner_media)
            if not mt.is_octet:
                return mt
        return MediaType(MimeType.PARQUET)

    @staticmethod
    def _default_member_for_write(inner_media: MediaType) -> str:
        """Generate a default member name from the media type extension."""
        ext = inner_media.full_extension
        return f"data.{ext}" if ext else "data"

    # ------------------------------------------------------------------
    # Zipfile kwargs (compression)
    # ------------------------------------------------------------------

    @staticmethod
    def _zipfile_kwargs(options: ZipOptions) -> dict:
        """Build ``**kwargs`` for :class:`zipfile.ZipFile` construction."""
        kw: dict = {"compression": zipfile.ZIP_DEFLATED}
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
    # Arrow implementation
    # ------------------------------------------------------------------

    def _read_arrow_table(self, *, options: ZipOptions) -> "pyarrow.Table":
        """Read and concatenate selected ZIP members into an Arrow table."""
        from yggdrasil.arrow.lib import pyarrow as _pa

        if self.buffer.size <= 0:
            return _pa.table({})

        tables: list[_pa.Table] = []

        with self.buffer.view(pos=0) as f:
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
        """Serialise *table* into a single ZIP member."""
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

        self.buffer._replace_with_payload(mem)

