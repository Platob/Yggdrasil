from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .cursor import HttpMongoCursor
from .results import DeleteResult, InsertManyResult, InsertOneResult, UpdateResult
from .transport import HttpTransport


class HttpCollection:
    def __init__(self, transport: HttpTransport, database: str, name: str) -> None:
        self._transport = transport
        self.database = database
        self.name = name
        self.full_name = f"{database}.{name}"

    def _rpc(self, operation: str, arguments: Dict[str, Any]) -> Any:
        return self._transport.rpc(
            database=self.database,
            collection=self.name,
            operation=operation,
            arguments=arguments,
        )

    def find(
        self,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        skip: int = 0,
        limit: int = 0,
        sort: Optional[List[Tuple[str, int]]] = None,
        **kwargs: Any,
    ) -> HttpMongoCursor:
        documents = self._rpc(
            "find",
            {
                "filter": filter or {},
                "projection": projection,
                "skip": skip,
                "limit": limit,
                "sort": sort,
                "kwargs": kwargs,
            },
        )
        return HttpMongoCursor(documents)

    def find_one(
        self,
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        return self._rpc(
            "find_one",
            {
                "filter": filter or {},
                "projection": projection,
                "kwargs": kwargs,
            },
        )

    def aggregate(self, pipeline: Sequence[Dict[str, Any]], **kwargs: Any) -> Iterator[Dict[str, Any]]:
        documents = self._rpc(
            "aggregate",
            {"pipeline": list(pipeline), "kwargs": kwargs},
        )
        return iter(documents)

    def count_documents(self, filter: Optional[Dict[str, Any]] = None, **kwargs: Any) -> int:
        return self._rpc(
            "count_documents",
            {"filter": filter or {}, "kwargs": kwargs},
        )

    def insert_one(self, document: Dict[str, Any], **_: Any) -> InsertOneResult:
        return InsertOneResult(**self._rpc("insert_one", {"document": document}))

    def insert_many(self, documents: List[Dict[str, Any]], ordered: bool = True, **_: Any) -> InsertManyResult:
        return InsertManyResult(**self._rpc("insert_many", {"documents": documents, "ordered": ordered}))

    def update_one(self, filter: Dict[str, Any], update: Dict[str, Any], upsert: bool = False, **_: Any) -> UpdateResult:
        return UpdateResult(**self._rpc("update_one", {"filter": filter, "update": update, "upsert": upsert}))

    def update_many(self, filter: Dict[str, Any], update: Dict[str, Any], upsert: bool = False, **_: Any) -> UpdateResult:
        return UpdateResult(**self._rpc("update_many", {"filter": filter, "update": update, "upsert": upsert}))

    def delete_one(self, filter: Dict[str, Any], **_: Any) -> DeleteResult:
        return DeleteResult(**self._rpc("delete_one", {"filter": filter}))

    def delete_many(self, filter: Dict[str, Any], **_: Any) -> DeleteResult:
        return DeleteResult(**self._rpc("delete_many", {"filter": filter}))

    def drop(self, **_: Any) -> bool:
        return bool(self._rpc("drop", {}))

    def create_index(self, keys: List[Tuple[str, int]], **kwargs: Any) -> str:
        return self._rpc("create_index", {"keys": keys, "kwargs": kwargs})

    def index_information(self) -> Dict[str, Any]:
        return self._rpc("index_information", {})

    def drop_index(self, name: str, **_: Any) -> bool:
        return bool(self._rpc("drop_index", {"name": name}))
