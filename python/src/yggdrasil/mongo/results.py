from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(slots=True)
class InsertOneResult:
    inserted_id: Any
    acknowledged: bool = True


@dataclass(slots=True)
class InsertManyResult:
    inserted_ids: List[Any]
    acknowledged: bool = True


@dataclass(slots=True)
class UpdateResult:
    matched_count: int
    modified_count: int
    upserted_id: Optional[Any] = None
    acknowledged: bool = True


@dataclass(slots=True)
class DeleteResult:
    deleted_count: int
    acknowledged: bool = True
