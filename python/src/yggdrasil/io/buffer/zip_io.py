"""ZIP container I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Optional

import pyarrow as pa

from yggdrasil.io.enums import MediaType, MimeTypes, SaveMode

from .bytes_io import BytesIO
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["ZipOptions", "ZipIO"]


def _check_member_info(value: tuple[str, str] | str) -> tuple[str, str]:
    """Validate and normalize one member-info specifier."""
    if isinstance(value, str):
        value = (value, f"_zip_member_{value}")
    elif not isinstance(value, tuple):
        try:
            value = tuple(value)
        except Exception as e:
            raise TypeError(
                "ZipOptions.read_member_infos values must be str or tuple[str, str], "
                f"got {type(value).__name__}"
            ) from e

    if len(value) < 2:
        raise TypeError(
            "ZipOptions.read_member_infos values must be str or tuple[str, str], "
            f"got length {len(value)}"
        )

    key, alias = value[0], value[1]

    if key != "name":
        raise ValueError(
            "ZipOptions.read_member_infos keys must be ('name',), "
            f"got {key!r}"
        )

    if not isinstance(alias, str):
        raise TypeError(
            "ZipOptions.read_member_infos aliases must be str, "
            f"got {type(alias).__name__}"
        )

    if not alias:
        alias = f"_zip_member_{key}"

    return key, alias


@dataclass
class ZipOptions(MediaOptions):
    """Options for ZIP I/O."""

    member: str | None = None
    inner_media: MediaType | str | None = None
    force_inner_media: bool = False
    zip_compression: int = 8
    read_member_infos: list[tuple[str, str]] | None = None

    def __post_init__(self) -> None:
        """Normalize and validate ZIP-specific options."""
        super().__post_init__()

        if self.member is not None and not isinstance(self.member, str):
            raise TypeError(
                f"member must be str|None, got {type(self.member).__name__}"
            )

        if not isinstance(self.force_inner_media, bool):
            raise TypeError(
                f"force_inner_media must be bool, got {type(self.force_inner_media).__name__}"
            )

        if self.inner_media is not None:
            self.inner_media = MediaType.parse(self.inner_media)

        self.read_member_infos = self._normalize_read_member_infos(
            self.read_member_infos
        )

    @staticmethod
    def _normalize_read_member_infos(
        infos: Any,
    ) -> list[tuple[str, str]] | None:
        """Normalize ``read_member_infos`` to a validated list."""
        if not infos:
            return None

        if isinstance(infos, Mapping):
            items = infos.items()
        elif isinstance(infos, str):
            items = [infos]
        elif isinstance(infos, Iterable):
            items = [value for value in infos if value is not None]
        else:
            raise TypeError(
                "read_member_infos must be a mapping or iterable of str or tuple[str, str], "
                f"got {type(infos).__name__}"
            )

        normalized = [_check_member_info(value) for value in items]
        return normalized or None

    @property
    def is_glob(self) -> bool:
        """Return ``True`` when ``member`` contains glob wildcards."""
        member = self.member
        return bool(member and any(ch in member for ch in ("*", "?", "[")))

    @classmethod
    def resolve(cls, *, options: "ZipOptions | None" = None, **overrides: Any) -> "ZipOptions":
        """Merge overrides into ``options`` or a fresh default."""
        return cls.check_parameters(options=options, **overrides)


@dataclass(slots=True)
class ZipIO(MediaIO[ZipOptions]):
    """ZIP container I/O."""

    @classmethod
    def check_options(
        cls,
        options: Optional[ZipOptions],
        *args,
        **kwargs,
    ) -> ZipOptions:
        """Validate / merge options for this media type."""
        return ZipOptions.check_parameters(options=options, **kwargs)

    def _iter_selected_infos(
        self,
        zf: zipfile.ZipFile,
        options: ZipOptions,
    ) -> Iterator[zipfile.ZipInfo]:
        """Yield selected non-directory members in archive order."""
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]

        member = options.member
        if member is None:
            yield from infos
            return

        if not options.is_glob:
            for info in infos:
                if info.filename == member:
                    yield info
                    return
            raise KeyError(f"ZipIO: member not found: {member!r}. Available: {names}")

        matched = False
        for info in infos:
            if fnmatchcase(info.filename, member):
                matched = True
                yield info

        if not matched:
            raise KeyError(
                f"ZipIO: no members matched pattern={member!r}. Available: {names}"
            )

    @staticmethod
    def _infer_inner_media_from_name(name: str) -> MediaType:
        """Infer media type from member name / extension."""
        try:
            return MediaType.parse(name)
        except Exception:
            return MediaType(MimeTypes.OCTET_STREAM)

    def _infer_inner_media(self, name: str, payload: bytes) -> MediaType:
        """Infer media type from name first, then by sniffing payload bytes."""
        media_type = self._infer_inner_media_from_name(name)
        if not media_type.is_octet:
            return media_type

        try:
            return BytesIO(payload, config=self.buffer.config).media_type
        except Exception:
            return MediaType(MimeTypes.OCTET_STREAM)

    def _resolve_member_media_type(
        self,
        *,
        options: ZipOptions,
        name: str,
        payload: bytes,
    ) -> MediaType:
        """Resolve the media type for one member."""
        inner_media = options.inner_media
        if options.force_inner_media and inner_media is not None and not inner_media.is_octet:
            return inner_media
        return self._infer_inner_media(name, payload)

    @staticmethod
    def _pick_inner_media_for_write(options: ZipOptions) -> MediaType:
        """Resolve the inner media type used for writes."""
        inner_media = options.inner_media
        if inner_media is None or inner_media.is_octet:
            return MediaType(MimeTypes.PARQUET)
        return inner_media

    @staticmethod
    def _default_member_for_write(inner_media: MediaType) -> str:
        """Generate a default member name for the inner payload."""
        ext = inner_media.full_extension
        return f"data.{ext}" if ext else "data"

    @staticmethod
    def _zipfile_kwargs(options: ZipOptions) -> dict[str, int]:
        """Build ``zipfile.ZipFile`` kwargs for writing."""
        return {
            "compression": zipfile.ZIP_DEFLATED,
            "compresslevel": max(0, min(9, int(options.zip_compression))),
        }

    def iter_members(
        self,
        options: ZipOptions,
    ) -> Iterator[tuple[zipfile.ZipInfo, BytesIO]]:
        """Yield selected ZIP members as ``(info, buffer)``."""
        if self.buffer.size <= 0:
            return

        with self.buffer.view(pos=0) as buf_view:
            with zipfile.ZipFile(buf_view, mode="r") as zf:
                for info in self._iter_selected_infos(zf, options):
                    payload = zf.read(info.filename)
                    media_type = self._resolve_member_media_type(
                        options=options,
                        name=info.filename,
                        payload=payload,
                    )

                    buf = BytesIO(payload, config=self.buffer.config)
                    buf._media_type = media_type
                    yield info, buf

    def _apply_member_infos(
        self,
        table: "pyarrow.Table",
        *,
        name: str,
        options: ZipOptions,
    ) -> "pyarrow.Table":
        """Append requested member metadata columns to a table."""
        if not options.read_member_infos:
            return table

        from yggdrasil.arrow.lib import pyarrow as _pa

        for key, alias in options.read_member_infos:
            if key == "name":
                table = table.append_column(
                    _pa.field(alias, _pa.string(), nullable=False),
                    _pa.array([name] * table.num_rows),
                )
            else:
                raise ValueError(
                    f"ZipIO: unknown options.read_member_infos[{key!r}]. "
                    "Must be in ('name',)"
                )

        return table

    def _read_arrow_batches(
        self,
        *,
        options: ZipOptions,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Read and stream batches from selected ZIP members."""
        for info, buf in self.iter_members(options=options):
            table = buf.media_io().read_arrow_table()
            table = self._apply_member_infos(
                table,
                name=info.filename,
                options=options,
            )
            yield from table.to_batches()

    def _collect_arrow_schema(self) -> "pyarrow.Schema":
        """Return the schema of the first selected ZIP member."""
        if self.buffer.size <= 0:
            return pa.schema([])

        options = self.check_options(options=None)

        member_schema: pa.Schema | None = None
        for _info, inner_buf in self.iter_members(options=options):
            member_schema = inner_buf.media_io()._collect_arrow_schema()
            break

        if member_schema is None:
            return pa.schema([])

        if options.read_member_infos:
            extra_fields: list[pa.Field] = []
            for key, alias in options.read_member_infos:
                if key == "name":
                    extra_fields.append(pa.field(alias, pa.string(), nullable=False))
                else:
                    raise ValueError(
                        f"ZipIO: unknown options.read_member_infos[{key!r}]. "
                        "Must be in ('name',)"
                    )
            member_schema = pa.schema(list(member_schema) + extra_fields)

        return member_schema

    def _write_arrow_batches(
        self,
        *,
        batches: Iterator["pyarrow.RecordBatch"],
        schema: "pyarrow.Schema",
        options: ZipOptions,
    ) -> None:
        """Serialise batches into a single ZIP member."""
        table = pa.Table.from_batches(list(batches), schema=schema)
        inner_media = self._pick_inner_media_for_write(options)

        inner_buffer = BytesIO(config=self.buffer.config)
        inner_buffer.media_io(inner_media).write_arrow_table(
            table,
            mode=SaveMode.OVERWRITE,
        )

        member_name = options.member or self._default_member_for_write(inner_media)

        archive = io.BytesIO()
        with zipfile.ZipFile(archive, mode="w", **self._zipfile_kwargs(options)) as zf:
            zf.writestr(member_name, inner_buffer.to_bytes())

        self.buffer.seek(0)
        self.buffer.write_bytes(archive.getvalue())
        self.buffer.seek(0)