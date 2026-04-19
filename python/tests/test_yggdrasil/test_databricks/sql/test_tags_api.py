"""Unit tests for the Unity Catalog ``entity_tag_assignments`` API wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import EntityTagAssignment

from yggdrasil.databricks.sql.tags_api import (
    apply_tags,
    delete_tags,
    list_tag_assignments,
)


class _EntityTagAssignmentsStub:
    def __init__(self, *, delete_raises: Exception | None = None) -> None:
        self.created: list[EntityTagAssignment] = []
        self.deleted: list[tuple[str, str, str]] = []
        self.listed: list[tuple[str, str]] = []
        self._delete_raises = delete_raises

    def create(self, *, tag_assignment):
        self.created.append(tag_assignment)
        return tag_assignment

    def delete(self, *, entity_type, entity_name, tag_key):
        self.deleted.append((entity_type, entity_name, tag_key))
        if self._delete_raises is not None:
            raise self._delete_raises

    def list(self, *, entity_type, entity_name, **_):
        self.listed.append((entity_type, entity_name))
        return iter([
            EntityTagAssignment(
                entity_type=entity_type,
                entity_name=entity_name,
                tag_key="pii",
                tag_value="true",
            ),
        ])


def _make_client(api):
    workspace = SimpleNamespace(entity_tag_assignments=api)
    return SimpleNamespace(workspace_client=lambda: workspace)


def test_apply_tags_skips_blank_keys_and_values():
    api = _EntityTagAssignmentsStub()
    client = _make_client(api)

    apply_tags(
        client,
        entity_type="tables",
        entity_name="main.analytics.trades",
        tags={"owner": "nika", "": "x", "domain": "", "pii": "true"},
    )

    assert len(api.created) == 2
    assert api.created[0].tag_key == "owner"
    assert api.created[1].tag_key == "pii"


def test_apply_tags_passes_entity_coordinates_through():
    api = _EntityTagAssignmentsStub()
    client = _make_client(api)

    apply_tags(
        client,
        entity_type="columns",
        entity_name="main.analytics.trades.trade_id",
        tags={"owner": "nika"},
    )

    assignment = api.created[0]
    assert assignment.entity_type == "columns"
    assert assignment.entity_name == "main.analytics.trades.trade_id"
    assert assignment.tag_key == "owner"
    assert assignment.tag_value == "nika"


def test_delete_tags_swallows_not_found_when_if_exists():
    api = _EntityTagAssignmentsStub(delete_raises=NotFound("missing"))
    client = _make_client(api)

    delete_tags(
        client,
        entity_type="tables",
        entity_name="main.analytics.trades",
        tag_keys=["owner"],
        if_exists=True,
    )

    assert api.deleted == [("tables", "main.analytics.trades", "owner")]


def test_delete_tags_propagates_not_found_when_required():
    api = _EntityTagAssignmentsStub(delete_raises=NotFound("missing"))
    client = _make_client(api)

    with pytest.raises(NotFound):
        delete_tags(
            client,
            entity_type="tables",
            entity_name="main.analytics.trades",
            tag_keys=["owner"],
            if_exists=False,
        )


def test_list_tag_assignments_returns_tuple_of_assignments():
    api = _EntityTagAssignmentsStub()
    client = _make_client(api)

    out = list_tag_assignments(
        client, entity_type="tables", entity_name="main.analytics.trades",
    )

    assert api.listed == [("tables", "main.analytics.trades")]
    assert len(out) == 1
    assert out[0].tag_key == "pii"


def test_list_tag_assignments_returns_empty_when_api_missing():
    client = SimpleNamespace(workspace_client=lambda: SimpleNamespace())

    assert list_tag_assignments(
        client, entity_type="tables", entity_name="main.analytics.trades",
    ) == ()


def test_missing_tags_api_raises_for_writes():
    client = SimpleNamespace(workspace_client=lambda: SimpleNamespace())

    with pytest.raises(RuntimeError, match="entity_tag_assignments"):
        apply_tags(
            client,
            entity_type="tables",
            entity_name="main.analytics.trades",
            tags={"owner": "nika"},
        )
