from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.arrow.lib import pyarrow as pa
from yggdrasil.io.enums import MediaType, MimeType, MimeTypes
from .bytes_io import BytesIO
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow.dataset as ds

__all__ = [
    "PathOptions", "PathIO"
]


_DEFAULT_IGNORE_PREFIXES = (".", "_")
_SUPPORTED_MIME_TYPES: tuple[MimeType, ...] = (
    MimeTypes.PARQUET,
    MimeTypes.ARROW_IPC,
    MimeTypes.CSV,
    MimeTypes.TSV,
    MimeTypes.JSON,
    MimeTypes.NDJSON,
    MimeTypes.ORC,
)


@dataclass
class PathOptions(MediaOptions):
    filter: Any = None
    recursive: bool = True
    include_hidden: bool = False
    supported_only: bool = True
    format: Any = None
    partitioning: str | Sequence[str] | None = "hive"
    partition_base_dir: str | Path | None = None
    exclude_invalid_files: bool | None = None
    ignore_prefixes: Sequence[str] | None = _DEFAULT_IGNORE_PREFIXES
    batch_readahead: int = 16
    fragment_readahead: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.recursive, bool):
            raise TypeError(f"recursive must be bool, got {type(self.recursive).__name__}")
        if not isinstance(self.include_hidden, bool):
            raise TypeError(
                f"include_hidden must be bool, got {type(self.include_hidden).__name__}"
            )
        if not isinstance(self.supported_only, bool):
            raise TypeError(
                f"supported_only must be bool, got {type(self.supported_only).__name__}"
            )

        if self.partition_base_dir is not None:
            self.partition_base_dir = Path(self.partition_base_dir)

        if self.ignore_prefixes is not None:
            if isinstance(self.ignore_prefixes, (str, bytes)):
                raise TypeError("ignore_prefixes must be a sequence of strings, not a single string")
            prefixes = list(self.ignore_prefixes)
            if not all(isinstance(item, str) for item in prefixes):
                raise TypeError("ignore_prefixes must contain only str values")
            self.ignore_prefixes = tuple(prefixes)

        self.batch_readahead = self._normalize_non_negative_int(
            "batch_readahead",
            self.batch_readahead,
        )
        self.fragment_readahead = self._normalize_non_negative_int(
            "fragment_readahead",
            self.fragment_readahead,
        )

    @staticmethod
    def _normalize_non_negative_int(name: str, value: int | None) -> int:
        if value is None:
            return 0
        if not isinstance(value, int):
            raise TypeError(f"{name} must be int|None, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
        return value

    @classmethod
    def resolve(cls, *, options: "PathOptions | None" = None, **overrides: Any) -> "PathOptions":
        return cls.check_parameters(options=options, **overrides)


@dataclass(slots=True)
class PathIO(MediaIO[PathOptions], ABC):
    path: Any = field(default_factory=Path)

    def __post_init__(self) -> None:
        if self.media_type is None:
            self.media_type = MediaType.parse(self.infer_mime_type(), default=MediaType(MimeTypes.PARQUET))

    @classmethod
    def check_options(
        cls,
        options: Optional[PathOptions],
        *args,
        **kwargs,
    ) -> PathOptions:
        del args
        return PathOptions.check_parameters(options=options, **kwargs)

    @classmethod
    @abstractmethod
    def make(
        cls,
        path: str | Path,
        media: MediaType | MimeType | str | None = None,
    ) -> "PathIO":
        raise TypeError(f"{cls.__name__} is abstract and cannot be instantiated directly")

    @abstractmethod
    def iter_files(
        self,
        recursive: bool = True,
        *,
        include_hidden: bool = False,
        supported_only: bool = True,
        mime_type: MimeType | str | None = None,
    ) -> Iterator["PathIO"]:
        raise NotImplementedError

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def is_file(self) -> bool:
        return self.path.is_file()

    @property
    def is_dir(self) -> bool:
        return self.path.is_dir()

    def infer_mime_type(
        self,
        *,
        recursive: bool = True,
        include_hidden: bool = False,
        supported_only: bool = False,
    ) -> MimeType:
        if self.media_type is not None:
            return self.media_type.mime_type

        if self.path.is_file():
            return MimeType.parse(self.path, default=MimeTypes.PARQUET)

        first_file = next(
            self.iter_files(
                recursive=recursive,
                include_hidden=include_hidden,
                supported_only=supported_only,
            ),
            None,
        )
        if first_file is None:
            return MimeTypes.PARQUET
        return MimeType.parse(first_file.path, default=MimeTypes.PARQUET)

    def read_dataset(
        self,
        *,
        options: PathOptions | None = None,
        **option_kwargs: Any,
    ) -> "ds.Dataset":
        import pyarrow.dataset as ds

        resolved = self.check_options(options=options, **option_kwargs)
        dataset_format = self._resolve_dataset_format(
            format=resolved.format,
            recursive=resolved.recursive,
            include_hidden=resolved.include_hidden,
            supported_only=resolved.supported_only,
        )
        arrow_schema = resolved.cast.target_arrow_schema

        if self.path.is_dir():
            files = [
                str(file_io.path)
                for file_io in self.iter_files(
                    recursive=resolved.recursive,
                    include_hidden=resolved.include_hidden,
                    supported_only=resolved.supported_only,
                )
            ]
            if not files:
                return ds.dataset(pa.Table.from_batches([], schema=arrow_schema or pa.schema([])))

            return ds.dataset(
                files,
                schema=arrow_schema,
                format=dataset_format,
                partitioning=resolved.partitioning,
                partition_base_dir=str(resolved.partition_base_dir or self.path),
                exclude_invalid_files=resolved.exclude_invalid_files,
                ignore_prefixes=(
                    list(resolved.ignore_prefixes) if resolved.ignore_prefixes is not None else None
                ),
            )

        return ds.dataset(
            str(self.path),
            schema=arrow_schema,
            format=dataset_format,
            partitioning=resolved.partitioning,
            partition_base_dir=str(resolved.partition_base_dir or self.path.parent),
            exclude_invalid_files=resolved.exclude_invalid_files,
            ignore_prefixes=(
                list(resolved.ignore_prefixes) if resolved.ignore_prefixes is not None else None
            ),
        )

    def to_arrow_dataset(
        self,
        *,
        options: PathOptions | None = None,
        **option_kwargs: Any,
    ) -> "ds.Dataset":
        return self.read_dataset(options=options, **option_kwargs)

    def scanner(
        self,
        *,
        options: PathOptions | None = None,
        **option_kwargs: Any,
    ) -> "ds.Scanner":
        resolved = self.check_options(options=options, **option_kwargs)
        dataset = self.read_dataset(options=resolved)
        return dataset.scanner(
            columns=list(resolved.columns) if resolved.columns is not None else None,
            filter=self._normalize_filter(resolved.filter),
            batch_size=resolved.batch_size or 131_072,
            batch_readahead=resolved.batch_readahead or 16,
            fragment_readahead=resolved.fragment_readahead or 4,
            use_threads=resolved.use_threads,
        )

    def _read_arrow_batches(self, *, options: PathOptions) -> Iterator["pa.RecordBatch"]:
        file_options = self._file_options(options=options)
        partition_base_dir = options.partition_base_dir or (
            self.path if self.path.is_dir() else getattr(self.path, "parent", self.path)
        )

        for file_io in self.iter_files(
            recursive=options.recursive,
            include_hidden=options.include_hidden,
            supported_only=options.supported_only,
            mime_type=self.media_type.mime_type if self.is_file else None,
        ):
            partition_values = self._partition_values(
                file_path=file_io.path,
                partitioning=options.partitioning,
                partition_base_dir=partition_base_dir,
            )
            yield from file_io._read_file_batches(
                options=file_options,
                partition_values=partition_values,
            )

    def _write_arrow_batches(
        self,
        *,
        batches: Iterator["pa.RecordBatch"],
        schema: "pa.Schema",
        options: PathOptions,
    ) -> None:
        del batches, schema, options
        raise NotImplementedError(f"{type(self).__name__} does not support writes yet")

    def count_rows(
        self,
        *,
        options: PathOptions | None = None,
        **option_kwargs: Any,
    ) -> int:
        resolved = self.check_options(options=options, **option_kwargs)
        return self._read_filtered_table(options=resolved).num_rows

    def select_columns(
        self,
        *columns: str,
        options: PathOptions | None = None,
        **option_kwargs: Any,
    ) -> pa.Table:
        resolved = self.check_options(options=options, columns=list(columns), **option_kwargs)
        return self.read_arrow_table(options=resolved)

    def read_arrow_table(
        self,
        *,
        options: PathOptions | None = None,
        **option_kwargs,
    ):
        resolved = self.check_options(options=options, **option_kwargs)
        table = self._read_filtered_table(options=resolved)

        bs = resolved.batch_size
        if bs and bs > 0:
            return self._iter_arrow_batches(table, bs)
        return table

    def _read_filtered_table(self, *, options: PathOptions) -> pa.Table:
        requested_columns = list(options.columns) if options.columns is not None else None
        table = self._read_table_from_batches(options=options.with_cast(None))
        if table.num_rows == 0:
            return table.select(requested_columns) if requested_columns is not None else table

        if options.filter is not None:
            import pyarrow.compute as pc

            mask = self._build_filter_mask(table=table, filter_spec=options.filter, pc=pc)
            table = table.filter(mask)

        if requested_columns is not None:
            table = table.select(requested_columns)

        return options.cast.cast_arrow(table)

    @staticmethod
    def _normalize_filter(filter: Any) -> Any:
        if filter is None:
            return None

        import pyarrow.dataset as ds

        if isinstance(filter, ds.Expression):
            return filter

        if isinstance(filter, dict):
            expr = None
            for key, value in filter.items():
                current = PathIO._normalize_filter_value(key, value)
                expr = current if expr is None else expr & current
            return expr

        if isinstance(filter, Sequence) and not isinstance(filter, (str, bytes)):
            expr = None
            for item in filter:
                if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) not in {2, 3}:
                    raise TypeError(
                        "filter sequences must contain (column, value) or (column, operator, value) items"
                    )
                current = PathIO._normalize_filter_tuple(item)
                expr = current if expr is None else expr & current
            return expr

        raise TypeError(
            "filter must be a pyarrow.dataset.Expression, mapping, or a sequence of filter tuples"
        )

    @staticmethod
    def _normalize_filter_value(column: str, value: Any):
        import pyarrow.dataset as ds

        field_expr = ds.field(column)

        if isinstance(value, range):
            value = list(value)
        if isinstance(value, set):
            value = list(value)

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values = list(value)
            if not values:
                return field_expr.is_null() & ~field_expr.is_null()
            expr = None
            for item in values:
                current = field_expr == item
                expr = current if expr is None else expr | current
            return expr

        if value is None:
            return field_expr.is_null()

        return field_expr == value

    @staticmethod
    def _normalize_filter_tuple(item: Sequence[Any]):
        import pyarrow.dataset as ds

        if len(item) == 2:
            column, value = item
            return PathIO._normalize_filter_value(column, value)

        column, operator, value = item
        field_expr = ds.field(column)
        op = str(operator).strip().lower()

        if op in {"=", "==", "eq"}:
            return PathIO._normalize_filter_value(column, value)
        if op in {"!=", "<>", "ne"}:
            return field_expr != value
        if op in {">", "gt"}:
            return field_expr > value
        if op in {">=", "gte", "ge"}:
            return field_expr >= value
        if op in {"<", "lt"}:
            return field_expr < value
        if op in {"<=", "lte", "le"}:
            return field_expr <= value
        if op == "in":
            return PathIO._normalize_filter_value(column, value)
        if op == "not in":
            return ~PathIO._normalize_filter_value(column, value)
        if op == "is":
            return field_expr.is_null() if value is None else field_expr == value
        if op == "is not":
            return field_expr.is_valid() if value is None else field_expr != value

        raise ValueError(f"Unsupported filter operator: {operator!r}")

    @classmethod
    def _build_filter_mask(cls, *, table, filter_spec: Any, pc):
        if isinstance(filter_spec, dict):
            mask = None
            for key, value in filter_spec.items():
                current = cls._build_value_mask(table=table, column=key, value=value, pc=pc)
                mask = current if mask is None else pc.and_kleene(mask, current)
            return mask

        if isinstance(filter_spec, (list, tuple)) and not isinstance(filter_spec, (str, bytes)):
            if len(filter_spec) in {2, 3} and isinstance(filter_spec[0], str):
                return cls._build_tuple_mask(table=table, item=filter_spec, pc=pc)

            mask = None
            for item in filter_spec:
                if not isinstance(item, (list, tuple)) or len(item) not in {2, 3}:
                    raise TypeError(
                        "filter sequences must contain (column, value) or "
                        "(column, operator, value) items"
                    )
                current = cls._build_tuple_mask(table=table, item=item, pc=pc)
                mask = current if mask is None else pc.and_kleene(mask, current)
            return mask

        raise TypeError(
            "filter must be a mapping or a sequence of filter tuples"
        )

    @staticmethod
    def _build_value_mask(*, table, column: str, value: Any, pc):
        array = table.column(column)

        if isinstance(value, range):
            value = list(value)
        if isinstance(value, set):
            value = list(value)

        if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes, bytearray)):
            if not value:
                return pc.is_null(array) & pc.invert(pc.is_null(array))
            mask = None
            for item in value:
                current = pc.equal(array, item)
                mask = current if mask is None else pc.or_kleene(mask, current)
            return mask

        if value is None:
            return pc.is_null(array)

        return pc.equal(array, value)

    @classmethod
    def _build_tuple_mask(cls, *, table, item, pc):
        if len(item) == 2:
            column, value = item
            return cls._build_value_mask(table=table, column=column, value=value, pc=pc)

        column, operator, value = item
        array = table.column(column)
        op = str(operator).strip().lower()

        if op in {"=", "==", "eq"}:
            return cls._build_value_mask(table=table, column=column, value=value, pc=pc)
        if op in {"!=", "<>", "ne"}:
            return pc.not_equal(array, value)
        if op in {">", "gt"}:
            return pc.greater(array, value)
        if op in {">=", "gte", "ge"}:
            return pc.greater_equal(array, value)
        if op in {"<", "lt"}:
            return pc.less(array, value)
        if op in {"<=", "lte", "le"}:
            return pc.less_equal(array, value)
        if op == "in":
            return cls._build_value_mask(table=table, column=column, value=value, pc=pc)
        if op == "not in":
            return pc.invert(cls._build_value_mask(table=table, column=column, value=value, pc=pc))
        if op == "is":
            return pc.is_null(array) if value is None else pc.equal(array, value)
        if op == "is not":
            return pc.invert(pc.is_null(array)) if value is None else pc.not_equal(array, value)

        raise ValueError(f"Unsupported filter operator: {operator!r}")

    @staticmethod
    def _path_parts(path: Any) -> tuple[str, ...]:
        parts = getattr(path, "parts", None)
        if parts is not None:
            return tuple(parts)
        return tuple(str(path).replace("\\", "/").split("/"))

    @classmethod
    def _relative_parts(cls, *, file_path: Any, partition_base_dir: Any) -> tuple[str, ...]:
        file_parts = cls._path_parts(file_path)
        base_parts = cls._path_parts(partition_base_dir)
        if base_parts and file_parts[: len(base_parts)] == base_parts:
            return file_parts[len(base_parts):]
        return file_parts

    @classmethod
    def _partition_values(
        cls,
        *,
        file_path: Any,
        partitioning: str | Sequence[str] | None,
        partition_base_dir: Any,
    ) -> dict[str, str]:
        if partitioning is None:
            return {}

        relative_parts = cls._relative_parts(
            file_path=file_path,
            partition_base_dir=partition_base_dir,
        )
        directory_parts = relative_parts[:-1]
        if not directory_parts:
            return {}

        if isinstance(partitioning, str):
            if partitioning.lower() != "hive":
                return {}

            out: dict[str, str] = {}
            for segment in directory_parts:
                if "=" not in segment:
                    continue
                key, value = segment.split("=", 1)
                if key:
                    out[key] = value
            return out

        names = list(partitioning)
        return {
            name: value
            for name, value in zip(names, directory_parts)
            if name
        }

    @staticmethod
    def _filter_columns(filter_spec: Any) -> list[str]:
        if filter_spec is None:
            return []
        if isinstance(filter_spec, dict):
            return list(filter_spec.keys())
        if isinstance(filter_spec, Sequence) and not isinstance(filter_spec, (str, bytes)):
            if len(filter_spec) in {2, 3} and isinstance(filter_spec[0], str):
                return [filter_spec[0]]

            out: list[str] = []
            for item in filter_spec:
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and item:
                    column = item[0]
                    if isinstance(column, str):
                        out.append(column)
            return out
        return []

    def _file_options(self, *, options: PathOptions) -> PathOptions:
        columns = list(options.columns) if options.columns is not None else None
        if columns is not None:
            for column in self._filter_columns(options.filter):
                if column not in columns:
                    columns.append(column)

        return self.check_options(
            options=options.with_cast(None),
            columns=columns,
        )

    def _read_file_batches(
        self,
        *,
        options: PathOptions,
        partition_values: dict[str, str] | None = None,
    ) -> Iterator["pa.RecordBatch"]:
        media = MediaType.parse(str(self.path), default=self.media_type)
        payload = self.path.read_bytes()
        with BytesIO(payload, media_type=media) as buffer:
            media_io = buffer.media_io(media)
            for batch in media_io.read_arrow_batches(
                options=media_io.check_options(options=options.with_cast(None)),
            ):
                yield from self._append_partition_values(
                    batch=batch,
                    partition_values=partition_values or {},
                )

    @staticmethod
    def _append_partition_values(
        *,
        batch: "pa.RecordBatch",
        partition_values: dict[str, str],
    ) -> Iterator["pa.RecordBatch"]:
        if not partition_values:
            yield batch
            return

        table = pa.Table.from_batches([batch])
        for name, value in partition_values.items():
            if name in table.column_names:
                continue
            table = table.append_column(name, pa.array([value] * table.num_rows))
        yield from table.to_batches()

    def _resolve_dataset_format(
        self,
        *,
        format: Any = None,
        recursive: bool = True,
        include_hidden: bool = False,
        supported_only: bool = False,
    ):
        import pyarrow.csv as pa_csv
        import pyarrow.dataset as ds

        if format is not None:
            if hasattr(format, "make_fragment"):
                return format
            normalized = str(format).strip().lower()
            if normalized in {"parquet", "pq"}:
                return ds.ParquetFileFormat()
            if normalized in {"ipc", "arrow", "feather"}:
                return ds.IpcFileFormat()
            if normalized == "csv":
                return ds.CsvFileFormat()
            if normalized == "tsv":
                return ds.CsvFileFormat(parse_options=pa_csv.ParseOptions(delimiter="\t"))
            if normalized in {"json", "ndjson"}:
                return ds.JsonFileFormat()
            if normalized == "orc":
                return ds.OrcFileFormat()
            raise ValueError(f"Unsupported dataset format: {format!r}")

        mime_type = self.infer_mime_type(
            recursive=recursive,
            include_hidden=include_hidden,
            supported_only=supported_only,
        )

        if mime_type is MimeTypes.PARQUET:
            return ds.ParquetFileFormat()
        if mime_type is MimeTypes.ARROW_IPC:
            return ds.IpcFileFormat()
        if mime_type is MimeTypes.CSV:
            return ds.CsvFileFormat()
        if mime_type is MimeTypes.TSV:
            return ds.CsvFileFormat(parse_options=pa_csv.ParseOptions(delimiter="\t"))
        if mime_type in {MimeTypes.JSON, MimeTypes.NDJSON}:
            return ds.JsonFileFormat()
        if mime_type is MimeTypes.ORC:
            return ds.OrcFileFormat()

        raise NotImplementedError(f"Unsupported dataset format for {mime_type!r}")
