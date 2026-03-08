from __future__ import annotations

from typing import ClassVar

import mongoengine
import mongoengine.queryset as queryset_pkg
import mongoengine.queryset.queryset as queryset_mod
from mongoengine.queryset.queryset import (
    QuerySet as MongoQuerySet,
    QuerySetNoCache as MongoQuerySetNoCache,
)

__all__ = [
    "QuerySet",
    "QuerySetNoCache",
]


class _QueryHelpersMixin:
    _ARROW_CACHE_BATCH_SIZE: ClassVar[int] = 4096

    @staticmethod
    def _apply_projection(
        qs,
        fields: list[str] | tuple[str, ...] | None = None,
        exclude: list[str] | tuple[str, ...] | None = None,
    ):
        if fields:
            qs = qs.only(*fields)
        if exclude:
            qs = qs.exclude(*exclude)
        return qs

    @staticmethod
    def _normalize_arrow_value(value):
        if value is None:
            return None

        t = type(value)
        name = t.__name__

        if name in {"ObjectId", "UUID"}:
            return str(value)

        if name == "DBRef":
            return str(value.id)

        if isinstance(value, dict):
            return {
                k: _QueryHelpersMixin._normalize_arrow_value(v)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                _QueryHelpersMixin._normalize_arrow_value(v)
                for v in value
            ]

        return value

    @classmethod
    def _normalize_mongo_row(cls, row: dict) -> dict:
        normalize = cls._normalize_arrow_value
        return {k: normalize(v) for k, v in row.items()}

    @classmethod
    def _iter_normalized_cache_rows(cls, cache, as_pymongo: bool):
        normalize = cls._normalize_arrow_value

        if as_pymongo:
            for row in cache:
                yield {k: normalize(v) for k, v in row.items()}
        else:
            for doc in cache:
                raw = doc.to_mongo()
                yield {k: normalize(v) for k, v in raw.items()}

    def _iter_normalized_rows(
        self,
        qs,
    ):
        cache = getattr(qs, "_result_cache", None)

        if cache:
            yield from self._iter_normalized_cache_rows(
                cache,
                as_pymongo=getattr(qs, "_as_pymongo", False),
            )
            return

        for row in qs.as_pymongo():
            yield self._normalize_mongo_row(row)

    @staticmethod
    def _recordbatch_from_rows(pa, rows, columns, schema):
        data = {name: [] for name in columns}
        for row in rows:
            for name in columns:
                data[name].append(row.get(name))
        return pa.RecordBatch.from_pydict(data, schema=schema)


class _ArrowMixin(_QueryHelpersMixin):
    def to_arrow_batches(
        self,
        batch_size: int = 1000,
        fields: list[str] | tuple[str, ...] | None = None,
        exclude: list[str] | tuple[str, ...] | None = None,
    ):
        import pyarrow as pa

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        qs = self._apply_projection(self, fields=fields, exclude=exclude)
        schema = qs._document.arrow_schema()
        columns = tuple(schema.names)

        rows = []
        append = rows.append
        size = 0

        for record in self._iter_normalized_rows(qs):
            append({name: record.get(name) for name in columns})
            size += 1

            if size >= batch_size:
                yield self._recordbatch_from_rows(pa, rows, columns, schema)
                rows = []
                append = rows.append
                size = 0

        if rows:
            yield self._recordbatch_from_rows(pa, rows, columns, schema)

    def to_arrow_table(
        self,
        batch_size: int = 1000,
        fields: list[str] | tuple[str, ...] | None = None,
        exclude: list[str] | tuple[str, ...] | None = None,
    ):
        import pyarrow as pa

        qs = self._apply_projection(self, fields=fields, exclude=exclude)
        schema = qs._document.arrow_schema()

        return pa.Table.from_batches(
            qs.to_arrow_batches(
                batch_size=batch_size,
                fields=None,
                exclude=None,
            ),
            schema=schema,
        )

    def to_pandas(
        self,
        fields: list[str] | tuple[str, ...] | None = None,
        exclude: list[str] | tuple[str, ...] | None = None,
        index: str | None = None,
        flatten: bool = False,
        use_arrow: bool = True,
    ):
        if use_arrow:
            table = self.to_arrow_table(
                fields=fields,
                exclude=exclude,
            )
            df = table.to_pandas()

            if "_id" in df.columns:
                df["_id"] = df["_id"].astype("string")

            if index is not None and index in df.columns:
                df = df.set_index(index)

            return df

        import pandas as pd

        qs = self._apply_projection(self, fields=fields, exclude=exclude)

        if flatten:
            records = list(self._iter_normalized_rows(qs))
            df = pd.json_normalize(records)
        else:
            records = list(self._iter_normalized_rows(qs))
            df = pd.DataFrame.from_records(records)

        if "_id" in df.columns:
            df["_id"] = df["_id"].astype("string")

        if index is not None and index in df.columns:
            df = df.set_index(index)

        return df


class _PickleArrowCacheMixin(_QueryHelpersMixin):
    _PICKLE_CACHE_KEY: ClassVar[str]
    _PICKLE_CACHE_COMPRESSION: ClassVar[str] = "zstd"

    def _cache_to_arrow_ipc(self, cache):
        import pyarrow as pa

        if not cache:
            return None

        schema = self._document.arrow_schema()
        columns = tuple(schema.names)
        batch_size = type(self)._ARROW_CACHE_BATCH_SIZE

        sink = pa.BufferOutputStream()
        options = pa.ipc.IpcWriteOptions(
            compression=type(self)._PICKLE_CACHE_COMPRESSION
        )

        with pa.ipc.new_stream(sink, schema, options=options) as writer:
            rows = []
            append = rows.append
            size = 0

            for row in self._iter_normalized_cache_rows(
                cache,
                as_pymongo=getattr(self, "_as_pymongo", False),
            ):
                append({name: row.get(name) for name in columns})
                size += 1

                if size >= batch_size:
                    writer.write_batch(
                        self._recordbatch_from_rows(pa, rows, columns, schema)
                    )
                    rows = []
                    append = rows.append
                    size = 0

            if rows:
                writer.write_batch(
                    self._recordbatch_from_rows(pa, rows, columns, schema)
                )

        return sink.getvalue().to_pybytes()

    def _arrow_ipc_to_cache(self, payload: bytes):
        import pyarrow as pa

        if payload is None:
            return None

        result_cache = []
        as_pymongo = getattr(self, "_as_pymongo", False)

        if as_pymongo:
            extend = result_cache.extend
            with pa.ipc.open_stream(payload) as reader:
                for batch in reader:
                    extend(batch.to_pylist())
            return result_cache

        from_son = self._document._from_son
        auto_deref = self._auto_dereference
        append = result_cache.append

        with pa.ipc.open_stream(payload) as reader:
            for batch in reader:
                for row in batch.to_pylist():
                    append(from_son(row, _auto_dereference=auto_deref))

        return result_cache

    def __getstate__(self):
        state = super().__getstate__()
        state[type(self)._PICKLE_CACHE_KEY] = self._cache_to_arrow_ipc(
            getattr(self, "_result_cache", None)
        )
        return state

    def __setstate__(self, state):
        arrow_payload = state.pop(type(self)._PICKLE_CACHE_KEY, None)

        super().__setstate__(state)

        self._cursor_obj = None
        self._iter = False

        if arrow_payload is None:
            return

        self._result_cache = self._arrow_ipc_to_cache(arrow_payload)
        self._has_more = False
        self._len = len(self._result_cache)


class QuerySet(_PickleArrowCacheMixin, _ArrowMixin, MongoQuerySet):
    _PICKLE_CACHE_KEY: ClassVar[str] = "__yggdrasil_cached_result_arrow_ipc__"
    _PICKLE_CACHE_COMPRESSION: ClassVar[str] = "zstd"


class QuerySetNoCache(_PickleArrowCacheMixin, _ArrowMixin, MongoQuerySetNoCache):
    _PICKLE_CACHE_KEY: ClassVar[str] = "__yggdrasil_nocache_result_arrow_ipc__"
    _PICKLE_CACHE_COMPRESSION: ClassVar[str] = "zstd"


queryset_mod.QuerySet = QuerySet
queryset_mod.QuerySetNoCache = QuerySetNoCache

queryset_pkg.QuerySet = QuerySet
queryset_pkg.QuerySetNoCache = QuerySetNoCache

mongoengine.QuerySet = QuerySet
mongoengine.QuerySetNoCache = QuerySetNoCache