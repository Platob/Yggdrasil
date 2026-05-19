"""Filesystem-backed :class:`UnityView`.

A view's metadata records a dotted ``catalog.schema.name`` source plus
an optional SQL projection. ``_resolve_source`` walks the engine to
look the target up; reads stream the resolved :class:`Tabular`'s
batches with the projection layered on top via the project's SQL
:class:`Engine` when a ``definition`` is set.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Mapping

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.unity.fs import registry
from yggdrasil.unity.info import ViewInfo
from yggdrasil.unity.view import UnityView

if TYPE_CHECKING:
    from yggdrasil.io.path import Path
    from yggdrasil.unity.fs.engine import FSEngine
    from yggdrasil.unity.fs.schema import FSSchema


__all__ = ["FSView"]


logger = logging.getLogger(__name__)


_UNSET = ...


def _resolve_source_full_name(source: Any) -> str:
    """Project *source* into a dotted ``catalog.schema.name`` string.

    Accepts a live :class:`UnityResource` (uses its ``full_name``) or a
    plain string with exactly three dotted parts.
    """
    from yggdrasil.unity.base import UnityResource

    if isinstance(source, UnityResource):
        full = source.full_name
    elif isinstance(source, str):
        full = source
    else:
        raise TypeError(
            f"source must be a UnityResource or a dotted "
            f"'catalog.schema.name' string; got {type(source).__name__}: "
            f"{source!r}."
        )
    parts = full.split(".")
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"View source must be a fully-qualified 'catalog.schema.name' "
            f"identifier; got {full!r}."
        )
    return full


class FSView(UnityView):
    """View backed by a JSON sidecar pointing at a target table."""

    def __init__(self, *, schema: "FSSchema", name: str) -> None:
        UnityView.__init__(self)
        self._schema_handle = schema
        self._name = name

    # ── identity ───────────────────────────────────────────────────────

    @property
    def engine(self) -> "FSEngine":
        return self._schema_handle.engine

    @property
    def schema_handle(self) -> "FSSchema":
        return self._schema_handle

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> "Path":
        """Root directory of this view (``<schema>/<name>``)."""
        return self._schema_handle.child_path(self._name)

    # ── info ───────────────────────────────────────────────────────────

    def _read_info(self) -> ViewInfo:
        return registry.read_view_info(self.path)

    # ── lifecycle ──────────────────────────────────────────────────────

    def create(
        self,
        source: Any = _UNSET,
        *,
        definition: str | None = None,
        comment: str | None = None,
        owner: str | None = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> "FSView":
        if self.exists:
            if if_not_exists:
                logger.debug("View %r already exists — skipping create", self)
                return self
            raise FileExistsError(
                f"View {self.full_name!r} already exists at "
                f"{self.path.full_path()!r}."
            )
        if source is _UNSET:
            raise ValueError(
                f"View {self.full_name!r} requires a source= argument on "
                "create. Pass a UnityResource or a dotted 'catalog.schema.name'."
            )
        if not self._schema_handle.exists:
            raise FileNotFoundError(
                f"Schema {self._schema_handle.full_name!r} does not exist. "
                "Create it first via catalog.create_schema(name)."
            )
        source_full = _resolve_source_full_name(source)
        logger.debug(
            "Creating view %r (source=%r, definition=%r)",
            self, source_full, definition,
        )
        self.path.mkdir(parents=True, exist_ok=True)
        info = ViewInfo(
            catalog_name=self._schema_handle.catalog_name,
            schema_name=self._schema_handle.name,
            name=self._name,
            source_full_name=source_full,
            definition=definition,
            comment=comment,
            owner=owner,
            properties=dict(properties or {}),
        )
        registry.write_view_info(self.path, info)
        self._store_info(info)
        logger.info("Created view %r", self)
        return self

    def delete(self, *, missing_ok: bool = True) -> "FSView":
        if not self.exists:
            if missing_ok:
                logger.debug("View %r does not exist — skipping delete", self)
                return self
            raise FileNotFoundError(
                f"View {self.full_name!r} does not exist."
            )
        logger.debug("Deleting view %r", self)
        self.path.remove(recursive=True, missing_ok=missing_ok)
        self._invalidate_info()
        logger.info("Deleted view %r", self)
        return self

    # ── source resolution ──────────────────────────────────────────────

    def _resolve_source(self) -> Tabular:
        cat_name, sch_name, leaf = self.info.source_full_name.split(".")
        catalog = self.engine.catalog(cat_name)
        if not catalog.exists:
            raise FileNotFoundError(
                f"View {self.full_name!r} points at catalog {cat_name!r}, "
                f"which does not exist on {self.engine!r}."
            )
        schema = catalog.schema(sch_name)
        if not schema.exists:
            raise FileNotFoundError(
                f"View {self.full_name!r} points at schema "
                f"{cat_name}.{sch_name!r}, which does not exist."
            )
        # Tables resolve first; a view-over-a-view is allowed via the
        # fallback so chained projections compose.
        table = schema.table(leaf)
        if table.exists:
            return table
        sibling_view = schema.view(leaf)
        if sibling_view.exists:
            return sibling_view
        raise FileNotFoundError(
            f"View {self.full_name!r} points at {self.info.source_full_name!r}, "
            "which does not resolve to a table or view."
        )

    # ── Tabular contract ───────────────────────────────────────────────

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        source = self._resolve_source()
        definition = self.info.definition
        if definition is None:
            yield from source.read_arrow_batches(options)
            return
        # SQL projection: route through the project's SQL Engine with
        # the source registered under the view's qualified name AND its
        # bare leaf so a definition can reference either spelling.
        from yggdrasil.io.tabular.execution.sql.engine import Engine as _SqlEngine

        sources = {
            self.info.source_full_name: source,
            self.info.source_full_name.split(".")[-1]: source,
        }
        result = _SqlEngine(sources=sources).execute(definition)
        # The SQL engine returns a Tabular-shaped statement result — its
        # ``tabular`` attribute is the canonical row source.
        projected = getattr(result, "tabular", None) or result
        yield from projected.read_arrow_batches(options)
