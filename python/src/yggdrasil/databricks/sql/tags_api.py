"""Thin wrapper around the Databricks ``EntityTagAssignmentsAPI``.

Documented at
https://docs.databricks.com/api/workspace/entitytagassignments. Replaces
the legacy ``ALTER ... SET TAGS (...)`` DDL path so tag writes follow
the same governance/audit pipeline as catalog/schema tag operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from databricks.sdk.errors import NotFound, DatabricksError
from databricks.sdk.service.catalog import EntityTagAssignment

from .sql_utils import _safe_str

if TYPE_CHECKING:
    pass

__all__ = [
    "apply_tags",
    "delete_tags",
    "list_tag_assignments",
]

logger = logging.getLogger(__name__)


def _tags_api(client):
    """Fetch the workspace-client ``entity_tag_assignments`` service."""
    api = getattr(client.workspace_client(), "entity_tag_assignments", None)
    if api is None:
        raise RuntimeError(
            "databricks-sdk does not expose 'entity_tag_assignments'; "
            "upgrade the SDK to use the Unity Catalog tag API."
        )
    return api


def _iter_tag_pairs(tags: Mapping[str, Any] | None) -> Iterable[tuple[str, str]]:
    """Yield ``(key, value)`` pairs, skipping empty keys/values."""
    for k, v in (tags or {}).items():
        key = _safe_str(k).strip() if k is not None else ""
        val = _safe_str(v).strip() if v is not None else ""
        if key and val:
            yield key, val


def list_tag_assignments(
    client,
    *,
    entity_type: str,
    entity_name: str,
) -> tuple[EntityTagAssignment, ...]:
    """Return every ``EntityTagAssignment`` on a given entity.

    Swallows missing-API / permission errors so the caller can treat the
    absence of tags as "none known" — matches the legacy lazy-load
    behaviour on :class:`~yggdrasil.databricks.sql.table.Table`.
    """
    api = getattr(client.workspace_client(), "entity_tag_assignments", None)
    if api is None:
        return ()
    try:
        return tuple(api.list(entity_type=entity_type, entity_name=entity_name))
    except Exception:
        logger.warning(
            "Failed to list %s tag assignments for %r",
            entity_type, entity_name, exc_info=True,
        )
        return ()


def apply_tags(
    client,
    *,
    entity_type: str,
    entity_name: str,
    tags: Mapping[str, Any] | None,
) -> list[EntityTagAssignment]:
    """Create tag assignments on *(entity_type, entity_name)*.

    Empty keys/values are skipped so legacy callers don't trip over the
    DDL-style ``"" => ""`` pattern. The API's ``create`` is idempotent
    when the (entity, tag_key) already exists — re-assigning a value is
    a no-op at the SDK level, so we don't pre-check membership.
    """
    api = _tags_api(client)
    written: list[EntityTagAssignment] = []
    for key, val in _iter_tag_pairs(tags):
        stmt = EntityTagAssignment(
            entity_type=entity_type,
            entity_name=entity_name,
            tag_key=key,
            tag_value=val,
        )
        try:
            result = api.create(tag_assignment=stmt)
        except DatabricksError:
            logger.warning(
                "Failed to apply tag %r=%r to %s %r",
                key, val, entity_type, entity_name, exc_info=True,
            )
            result = stmt  # best effort: return the intended assignment even on failure

        written.append(
            result
        )
    return written


def delete_tags(
    client,
    *,
    entity_type: str,
    entity_name: str,
    tag_keys: Iterable[str],
    if_exists: bool = True,
) -> None:
    """Delete tag assignments by key; swallow ``NotFound`` when ``if_exists``."""
    api = _tags_api(client)
    for raw_key in tag_keys:
        key = _safe_str(raw_key).strip() if raw_key is not None else ""
        if not key:
            continue
        try:
            api.delete(
                entity_type=entity_type,
                entity_name=entity_name,
                tag_key=key,
            )
        except NotFound:
            if not if_exists:
                raise
