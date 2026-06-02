"""Databricks Genie service.

``client.genie.ask("which region sold the most last quarter?")`` is the
headline. The service holds a :class:`GenieDefaults` so callers set the
default ``space_id`` once and stop repeating it::

    from dataclasses import replace
    client.genie.defaults = replace(client.genie.defaults, space_id="01ef…")

    answer = client.genie.ask("top 5 customers by revenue this year")
    print(answer.text)            # natural-language summary
    print(answer.sql)             # the SQL Genie generated
    df = answer.to_polars()       # the result as a polars DataFrame

    # Drive a multi-turn conversation
    conv, first = client.genie.space().start_conversation("revenue by month")
    nxt = conv.ask("now just for EMEA")

    # Or let the agent act on its own
    run = client.genie.agent().run("explain the Q3 revenue dip")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

from yggdrasil.databricks.service import DatabricksService
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg

from .resources import (
    DEFAULT_GENIE_WAIT,
    GenieAnswer,
    GenieConversation,
    GenieDefaults,
    GenieSpace,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk.service.dashboards import GenieAPI

    from .agent import GenieAgent


__all__ = ["Genie"]


LOGGER = logging.getLogger(__name__)

#: Local cache mapping ``host|catalog|schema|title`` → default space id, so
#: :meth:`Genie.ensure_default_space` reuses the same space across processes
#: without depending on Databricks' eventually-consistent space listing.
_DEFAULT_SPACE_CACHE = Path.home() / ".ygg" / "genie.json"


def _read_default_space(key: str) -> Optional[str]:
    """Return the cached default-space id for ``key``, or ``None``."""
    try:
        return json.loads(_DEFAULT_SPACE_CACHE.read_text()).get(key)
    except (OSError, ValueError):
        return None


def _write_default_space(key: str, space_id: str) -> None:
    """Persist ``space_id`` for ``key`` (best-effort — a read-only home is fine)."""
    try:
        _DEFAULT_SPACE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        try:
            data = json.loads(_DEFAULT_SPACE_CACHE.read_text())
        except (OSError, ValueError):
            data = {}
        data[key] = space_id
        _DEFAULT_SPACE_CACHE.write_text(json.dumps(data, indent=2))
    except OSError as exc:  # pragma: no cover - depends on FS perms
        LOGGER.debug("Could not write Genie default-space cache: %s", exc)


class Genie(DatabricksService):
    """High-level wrapper around the Databricks Genie API.

    Attributes
    ----------
    defaults
        :class:`GenieDefaults` — service-wide configuration. Replace via
        ``client.genie.defaults = replace(...)`` the same way the other
        Databricks services do.
    """

    def __init__(self, client=None, defaults: Optional[GenieDefaults] = None):
        super().__init__(client=client)
        self.defaults: GenieDefaults = defaults if defaults is not None else GenieDefaults()

    # ------------------------------------------------------------------ #
    # SDK boundary
    # ------------------------------------------------------------------ #
    @property
    def api(self) -> "GenieAPI":
        return self.client.workspace_client().genie

    # ------------------------------------------------------------------ #
    # Space resolution
    # ------------------------------------------------------------------ #
    def space(self, space_id: Optional[str] = None) -> GenieSpace:
        """Return a :class:`GenieSpace` handle.

        ``space_id`` defaults to :attr:`GenieDefaults.space_id`.
        """
        sid = space_id or self.defaults.space_id
        if not sid:
            raise ValueError(
                "No space_id given and Genie.defaults.space_id is unset. "
                "Pass space_id=... or set the default."
            )
        return GenieSpace(service=self, space_id=sid)

    def list_spaces(self) -> Iterator[GenieSpace]:
        """Iterate over Genie spaces visible to the current principal."""
        token: Optional[str] = None
        while True:
            resp = self.api.list_spaces(page_token=token)
            for info in getattr(resp, "spaces", None) or []:
                sid = getattr(info, "space_id", None)
                if not sid:
                    continue
                yield GenieSpace(service=self, space_id=sid, details=info)
            token = getattr(resp, "next_page_token", None)
            if not token:
                break

    def find_space(self, *, title: str) -> Optional[GenieSpace]:
        """Return the first space whose title matches (case-insensitive)."""
        target = title.strip().lower()
        for space in self.list_spaces():
            if (space.title or "").strip().lower() == target:
                return space
        return None

    # ------------------------------------------------------------------ #
    # Space creation
    # ------------------------------------------------------------------ #
    def create_space(
        self,
        *,
        tables: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        warehouse_id: Optional[str] = None,
        description: Optional[str] = None,
        parent_path: Optional[str] = None,
        serialized_space: Optional[str] = None,
    ) -> GenieSpace:
        """Create a Genie space over a set of Unity Catalog tables.

        ``tables`` is a list of three-part ``catalog.schema.table`` names —
        they become the space's data sources. ``warehouse_id`` defaults to
        :attr:`GenieDefaults.warehouse_id`, then the workspace's default
        warehouse (:meth:`Warehouses.find_default`). ``serialized_space``
        is the raw Genie space document; pass it to override the
        ``tables``-derived one.
        """
        from yggdrasil.pickle import json as ygg_json

        if serialized_space is None:
            if not tables:
                raise ValueError(
                    "create_space needs tables= (catalog.schema.table names) "
                    "or an explicit serialized_space=."
                )
            serialized_space = ygg_json.dumps(
                {
                    "version": 2,
                    "data_sources": {
                        "tables": [{"identifier": t} for t in tables],
                    },
                },
                to_bytes=False,
            )

        wh_id = warehouse_id or self.defaults.warehouse_id
        if not wh_id:
            warehouse = self.client.warehouses.find_default()
            wh_id = warehouse.warehouse_id if warehouse is not None else None
        if not wh_id:
            raise ValueError(
                "create_space needs a warehouse_id — none given, no "
                "Genie.defaults.warehouse_id, and no default warehouse found."
            )

        LOGGER.debug(
            "Creating Genie space %r (warehouse=%s, tables=%s)",
            title, wh_id, list(tables or []),
        )
        info = self.api.create_space(
            warehouse_id=wh_id,
            serialized_space=serialized_space,
            title=title or self.defaults.default_space_title,
            description=description,
            parent_path=parent_path,
        )
        LOGGER.info("Created Genie space %s (%r)", info.space_id, info.title)
        return GenieSpace(service=self, space_id=info.space_id, details=info)

    def discover_tables(
        self,
        *,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        limit: int = 25,
    ) -> list[str]:
        """List ``catalog.schema.table`` names in a schema via ``SHOW TABLES``.

        Catalog / schema default to the client's bound ``catalog_name`` /
        ``schema_name``. Temporary tables are skipped. Capped at ``limit``
        (default 25) — a Genie space accepts at most 30 tables.
        """
        cat = catalog or getattr(self.client, "catalog_name", None)
        sch = schema or getattr(self.client, "schema_name", None)
        if not cat or not sch:
            raise ValueError(
                "discover_tables needs a catalog and schema — pass them or "
                "bind the client to a catalog/schema."
            )
        rows = self.client.sql.execute(
            f"SHOW TABLES IN `{cat}`.`{sch}`"
        ).to_arrow_table().to_pylist()
        names = [
            f"{cat}.{sch}.{r['tableName']}"
            for r in rows
            if r.get("tableName") and not r.get("isTemporary")
        ]
        return names[:limit]

    def ensure_default_space(
        self,
        *,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        title: Optional[str] = None,
        warehouse_id: Optional[str] = None,
        tables: Optional[Sequence[str]] = None,
    ) -> GenieSpace:
        """Return a default Genie space, creating it on first use.

        Resolution order: a locally cached id for this
        ``host|catalog|schema|title`` (verified to still exist), then a space
        with the default title, then create one over ``tables`` (or up to 25
        tables discovered in the catalog/schema — Genie caps a space at 30).
        The local cache (``~/.ygg/genie.json``) makes reuse stable across
        processes even though Databricks' space *listing* is eventually
        consistent. Lets ``client.genie.ask(...)`` work out of the box
        without a pre-built space.
        """
        space_title = title or self.defaults.default_space_title
        cat = catalog or getattr(self.client, "catalog_name", None)
        sch = schema or getattr(self.client, "schema_name", None)
        key = f"{getattr(self.client, 'host', '')}|{cat}|{sch}|{space_title}"

        # 1. Local cache — robust reuse across processes (verified live so a
        #    trashed space falls through to a fresh create).
        cached_id = _read_default_space(key)
        if cached_id and self.space(cached_id).exists():
            LOGGER.debug("Reusing cached default Genie space %s", cached_id)
            return self.space(cached_id)

        # 2. Title lookup (covers spaces created elsewhere).
        existing = self.find_space(title=space_title)
        if existing is not None:
            LOGGER.debug("Reusing existing default Genie space %s", existing.space_id)
            _write_default_space(key, existing.space_id)
            return existing

        # 3. Create.
        space_tables = list(tables) if tables else self.discover_tables(
            catalog=catalog, schema=schema,
        )
        if not space_tables:
            raise ValueError(
                "Cannot create a default Genie space: no tables given and none "
                "discovered. Pass tables= or a populated catalog/schema."
            )
        space = self.create_space(
            tables=space_tables, title=space_title, warehouse_id=warehouse_id,
        )
        _write_default_space(key, space.space_id)
        return space

    # ------------------------------------------------------------------ #
    # One-shot ask
    # ------------------------------------------------------------------ #
    def ask(
        self,
        question: str,
        *,
        space_id: Optional[str] = None,
        wait: WaitingConfigArg = None,
    ) -> GenieAnswer:
        """Ask a one-shot question against a space (starts a fresh conversation)."""
        return self.space(space_id).ask(question, wait=wait)

    def conversation(
        self, conversation_id: str, *, space_id: Optional[str] = None,
    ) -> GenieConversation:
        """Return a handle to an existing conversation in a space."""
        return self.space(space_id).conversation(conversation_id)

    # ------------------------------------------------------------------ #
    # Agent
    # ------------------------------------------------------------------ #
    def agent(self, *, space_id: Optional[str] = None, **kwargs) -> "GenieAgent":
        """Return a :class:`~.agent.GenieAgent` bound to a space."""
        from .agent import GenieAgent

        return GenieAgent(space=self.space(space_id), **kwargs)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_wait(self, override: WaitingConfigArg) -> WaitingConfig:
        if override is None or override is True:
            return self.defaults.wait or DEFAULT_GENIE_WAIT
        return WaitingConfig.from_(override)
