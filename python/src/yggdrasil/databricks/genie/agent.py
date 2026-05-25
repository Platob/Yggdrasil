"""Local-execution agent on top of :class:`Genie`.

:class:`GenieAgent` is the orchestrator that turns Genie's per-call API into a
session: it asks Genie a question (or a sequence of them), materialises any
SQL attachment, and persists the result to a local cache directory under
``$XDG_CACHE_HOME/yggdrasil/genie`` (configurable via
:attr:`GenieDefaults.agent_output_dir`).

Two entry points cover the common shapes::

    client.genie.agent.run("How many orders last month?")
    # → GenieAnswer, file written to
    #   ~/.cache/yggdrasil/genie/<space>/<conversation>/<message>.parquet
    #   when ``agent_auto_save`` is on or ``save=True`` is passed.

    client.genie.agent.chat(
        "Show orders by region",
        "Filter to last quarter",
        "Top 5 only",
    )
    # → list[GenieAnswer], same conversation, files per step.

A small registry of safe local tools (``polars``, ``pandas``, ``arrow_table``,
``save_parquet`` / ``csv`` / ``arrow`` / ``json`` / ``text``, ``inspect``,
``url``) is wired in by default. Destructive operations (deleting spaces,
trashing conversations, dropping messages) are deliberately *not* registered;
callers who want them have to add them by hand via
:meth:`GenieAgent.register_tool`.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from .resources import GenieAnswer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..client import DatabricksClient
    from .service import Genie


__all__ = ["GenieAgent", "AGENT_SAVE_FORMATS"]

LOGGER = logging.getLogger(__name__)


#: File formats :meth:`GenieAgent.save` knows how to write.
AGENT_SAVE_FORMATS: frozenset[str] = frozenset(
    {"parquet", "csv", "arrow", "ipc", "feather", "json", "text"}
)


# ``message-9 / conv 2!`` → ``message-9_conv_2_`` — keep the on-disk layout
# resilient to whatever shape Databricks decides to return for ids.
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_segment(value: Optional[str], fallback: str) -> str:
    if not value:
        return fallback
    cleaned = _FILENAME_SAFE.sub("_", value).strip("._-")
    return cleaned or fallback


class GenieAgent:
    """Session-level orchestrator on top of :class:`Genie`.

    Holds the message history of one Python process, the local output
    directory, and a registry of callable tools. Construct via
    ``client.genie.agent`` — :class:`Genie` caches a single instance per
    service so ``agent.history`` keeps growing across calls.
    """

    def __init__(self, service: "Genie"):
        self.service: "Genie" = service
        self.history: list[GenieAnswer] = []
        self.tools: dict[str, Callable[..., Any]] = {}
        self._register_default_tools()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"history={len(self.history)}, "
            f"tools={sorted(self.tools)!r})"
        )

    # ------------------------------------------------------------------ #
    # Output directory resolution
    # ------------------------------------------------------------------ #
    @property
    def output_dir(self) -> Path:
        """Root directory under which artifacts land.

        Resolution order:

        1. :attr:`GenieDefaults.agent_output_dir`, when set.
        2. ``$XDG_CACHE_HOME/yggdrasil/genie`` (XDG base-directory spec).
        3. ``~/.cache/yggdrasil/genie`` (XDG fallback).
        """
        configured = self.service.defaults.agent_output_dir
        if configured:
            return Path(configured).expanduser()
        cache_root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        return Path(cache_root).expanduser() / "yggdrasil" / "genie"

    def resolve_path(
        self,
        answer: GenieAnswer,
        *,
        format: str = "parquet",
        filename: Optional[str] = None,
    ) -> Path:
        """Resolve where :meth:`save` would write ``answer`` for ``format``.

        Layout: ``<output_dir>/<space_id>/<conversation_id>/<message_id>.<ext>``.
        Components that come back missing from the SDK are replaced with
        readable placeholders (``unknown_space`` / ``unknown_conv`` /
        ``unknown_msg``).
        """
        ext = self._extension_for(format)
        space = _safe_segment(answer.space_id, "unknown_space")
        conv = _safe_segment(answer.conversation_id, "unknown_conv")
        message = _safe_segment(answer.message_id, "unknown_msg")
        directory = self.output_dir / space / conv
        name = filename if filename else f"{message}.{ext}"
        return directory / name

    @staticmethod
    def _extension_for(format: str) -> str:
        format = format.lower()
        if format not in AGENT_SAVE_FORMATS:
            raise ValueError(
                f"Unknown save format {format!r}; expected one of: "
                f"{sorted(AGENT_SAVE_FORMATS)!r}"
            )
        return {
            "parquet": "parquet",
            "csv": "csv",
            "arrow": "arrow",
            "ipc": "arrow",
            "feather": "arrow",
            "json": "json",
            "text": "txt",
        }[format]

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #
    def save(
        self,
        answer: GenieAnswer,
        *,
        format: Optional[str] = None,
        path: Optional[Any] = None,
    ) -> Optional[Path]:
        """Persist ``answer`` locally and return the written path.

        ``format`` defaults to :attr:`GenieDefaults.agent_auto_save_format`.
        ``path`` overrides the auto-resolved location; parent directories
        are created.

        ``"json"`` / ``"text"`` save the answer envelope (text + sql +
        ids + status) without round-tripping through the warehouse. The
        tabular formats call :meth:`GenieAnswer.arrow_table`; when the
        answer has no query attachment they return ``None`` after logging
        a debug message (no exception — the caller may have been calling
        speculatively from auto-save).
        """
        fmt = (format or self.service.defaults.agent_auto_save_format).lower()
        if fmt not in AGENT_SAVE_FORMATS:
            raise ValueError(
                f"Unknown save format {fmt!r}; expected one of: "
                f"{sorted(AGENT_SAVE_FORMATS)!r}"
            )

        target = Path(path).expanduser() if path is not None else self.resolve_path(answer, format=fmt)

        if fmt == "json":
            target.parent.mkdir(parents=True, exist_ok=True)
            from yggdrasil.pickle import json as ygg_json

            payload = {
                "space_id": answer.space_id,
                "conversation_id": answer.conversation_id,
                "message_id": answer.message_id,
                "status": getattr(answer.status, "name", None) or (
                    str(answer.status) if answer.status is not None else None
                ),
                "text": answer.text,
                "query": answer.query,
                "statement_id": answer.statement_id,
                "url": str(answer.url()),
            }
            target.write_bytes(ygg_json.dumps(payload, indent=2))
            LOGGER.info("Saved Genie answer %r (format=json, path=%s)", answer, target)
            return target

        if fmt == "text":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(answer.text or "", encoding="utf-8")
            LOGGER.info("Saved Genie answer %r (format=text, path=%s)", answer, target)
            return target

        table = answer.arrow_table()
        if table is None:
            LOGGER.debug(
                "Skipping save for %r (format=%s) — no query attachment to materialise.",
                answer, fmt,
            )
            return None

        # Create the directory only once we have something to write —
        # otherwise auto-save against a text-only answer leaves empty
        # ``<output_dir>/<space>/<conv>/`` skeletons behind.
        target.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "parquet":
            import pyarrow.parquet as pq

            pq.write_table(table, str(target))
        elif fmt == "csv":
            import pyarrow.csv as pacsv

            pacsv.write_csv(table, str(target))
        else:  # arrow / ipc / feather
            from pyarrow import feather

            feather.write_feather(table, str(target))

        LOGGER.info("Saved Genie answer %r (format=%s, path=%s)", answer, fmt, target)
        return target

    # ------------------------------------------------------------------ #
    # Running questions
    # ------------------------------------------------------------------ #
    def run(
        self,
        question: str,
        *,
        conversation_id: Optional[str] = None,
        space_id: Optional[str] = None,
        save: Optional[bool] = None,
        save_format: Optional[str] = None,
        **kwargs: Any,
    ) -> GenieAnswer:
        """Ask Genie one question, append to history, and optionally save.

        ``save`` defaults to :attr:`GenieDefaults.agent_auto_save`. When
        on, the result is written via :meth:`save` using ``save_format``
        (default :attr:`GenieDefaults.agent_auto_save_format`). When the
        answer carries no SQL attachment the save call is a no-op
        regardless.
        """
        if save is None:
            save = self.service.defaults.agent_auto_save

        answer = self.service.ask(
            question,
            space_id=space_id,
            conversation_id=conversation_id,
            **kwargs,
        )
        self.history.append(answer)

        if save:
            try:
                self.save(answer, format=save_format)
            except Exception:
                # Saving is a convenience — never let it shadow the
                # caller's actual reply. Re-raise only the deliberate
                # ValueError on unknown formats above.
                LOGGER.exception("Failed to auto-save %r", answer)

        return answer

    def chat(
        self,
        *questions: str,
        conversation_id: Optional[str] = None,
        space_id: Optional[str] = None,
        save: Optional[bool] = None,
        save_format: Optional[str] = None,
        max_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> list[GenieAnswer]:
        """Send a series of questions on the same conversation.

        The first call uses ``conversation_id`` (or starts a new
        conversation when ``None``); each subsequent question reuses the
        conversation id off the previous answer so the thread stays
        coherent.

        Stops early when ``max_steps`` (default
        :attr:`GenieDefaults.agent_max_steps`) is reached. Empty
        ``questions`` returns ``[]``.
        """
        if not questions:
            return []

        budget = max_steps if max_steps is not None else self.service.defaults.agent_max_steps
        if budget <= 0:
            raise ValueError(
                f"max_steps must be positive (got {budget!r}); pass a positive "
                "int or raise Genie.defaults.agent_max_steps."
            )

        answers: list[GenieAnswer] = []
        current_conv = conversation_id
        for index, question in enumerate(questions):
            if index >= budget:
                LOGGER.info(
                    "Stopping Genie chat after %d step(s) — max_steps=%d reached.",
                    len(answers), budget,
                )
                break
            answer = self.run(
                question,
                conversation_id=current_conv,
                space_id=space_id,
                save=save,
                save_format=save_format,
                **kwargs,
            )
            answers.append(answer)
            current_conv = answer.conversation_id or current_conv
            if answer.is_failed:
                LOGGER.warning(
                    "Stopping Genie chat — answer %r reported failure status %r.",
                    answer, answer.status,
                )
                break

        return answers

    # ------------------------------------------------------------------ #
    # Inspection helpers
    # ------------------------------------------------------------------ #
    def last(self) -> Optional[GenieAnswer]:
        """Return the most recent answer in :attr:`history`, if any."""
        return self.history[-1] if self.history else None

    def reset(self) -> None:
        """Clear :attr:`history`. Tool registry is left intact."""
        self.history.clear()

    def inspect(self, answer: GenieAnswer) -> dict[str, Any]:
        """Return a small dict summarising ``answer``.

        Useful for logging or feeding into downstream tooling that
        doesn't want to import the SDK message classes.
        """
        return {
            "space_id": answer.space_id,
            "conversation_id": answer.conversation_id,
            "message_id": answer.message_id,
            "status": getattr(answer.status, "name", None) or (
                str(answer.status) if answer.status is not None else None
            ),
            "text": answer.text,
            "query": answer.query,
            "statement_id": answer.statement_id,
            "url": str(answer.url()),
        }

    # ------------------------------------------------------------------ #
    # Tool registry
    # ------------------------------------------------------------------ #
    def register_tool(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a callable available via :meth:`run_tool`.

        Names overwrite silently — caller is responsible for not
        clobbering a default tool by accident. Use :attr:`tools` to
        inspect what's registered.
        """
        if not callable(fn):
            raise TypeError(
                f"Tool {name!r} must be callable, got {type(fn).__name__}."
            )
        self.tools[name] = fn
        LOGGER.debug("Registered Genie agent tool %r", name)

    def unregister_tool(self, name: str) -> None:
        """Drop a tool from the registry. Raises ``KeyError`` if missing."""
        del self.tools[name]

    def run_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a registered tool. Raises ``KeyError`` on unknown name."""
        try:
            fn = self.tools[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown Genie agent tool {name!r}; registered: "
                f"{sorted(self.tools)!r}"
            ) from exc
        LOGGER.debug("Invoking Genie agent tool %r (args=%d, kwargs=%d)", name, len(args), len(kwargs))
        return fn(*args, **kwargs)

    @property
    def client(self) -> "DatabricksClient":
        return self.service.client

    def _register_default_tools(self) -> None:
        # Materialisation
        self.tools["arrow_table"] = lambda ans: ans.arrow_table()
        self.tools["polars"] = lambda ans: ans.polars()
        self.tools["pandas"] = lambda ans: ans.pandas()
        self.tools["statement_result"] = lambda ans: ans.statement_result()

        # Persistence
        self.tools["save"] = self.save
        self.tools["save_parquet"] = lambda ans, **k: self.save(ans, format="parquet", **k)
        self.tools["save_csv"] = lambda ans, **k: self.save(ans, format="csv", **k)
        self.tools["save_arrow"] = lambda ans, **k: self.save(ans, format="arrow", **k)
        self.tools["save_json"] = lambda ans, **k: self.save(ans, format="json", **k)
        self.tools["save_text"] = lambda ans, **k: self.save(ans, format="text", **k)

        # Conversation / introspection
        self.tools["ask"] = self.run
        self.tools["chat"] = self.chat
        self.tools["inspect"] = self.inspect
        self.tools["url"] = lambda ans: str(ans.url())
        self.tools["refresh"] = lambda ans: ans.refresh()
        self.tools["execute_query"] = lambda ans: ans.execute_query()
        self.tools["feedback"] = lambda ans, rating, **k: ans.feedback(rating, **k)

        # Config / state
        self.tools["output_dir"] = lambda: self.output_dir
        self.tools["history"] = lambda: list(self.history)
        self.tools["last"] = self.last
        self.tools["reset"] = self.reset
        self.tools["defaults"] = lambda: asdict(self.service.defaults)

        self._register_platform_tools()

    def _register_platform_tools(self) -> None:
        """Expose the DatabricksClient surface as agent tools."""
        c = self

        # SQL
        self.tools["sql"] = lambda statement, **k: c.client.sql.execute(statement, **k)
        self.tools["sql_arrow"] = lambda statement, **k: (
            c.client.sql.execute(statement, **k).read_arrow_table()
        )

        # Tables
        self.tools["list_tables"] = lambda schema: c.client.tables.list(schema)
        self.tools["table_schema"] = lambda name: c.client.tables[name].collect_schema()
        self.tools["read_table"] = lambda name, **k: c.client.sql.execute(
            f"SELECT * FROM {name}", **k,
        ).read_arrow_table()
        self.tools["write_table"] = lambda data, name, **k: (
            c.client.tables[name].write_table(data, **k)
        )

        # Catalogs / schemas
        self.tools["list_catalogs"] = lambda: c.client.catalogs.list()
        self.tools["list_schemas"] = lambda catalog: c.client.schemas.list(catalog)

        # Secrets
        self.tools["get_secret"] = lambda scope, key: (
            c.client.secrets[f"{scope}/{key}"].svalue()
        )
        self.tools["set_secret"] = lambda scope, key, value: (
            c.client.secrets.create_secret(key=key, value=value, scope=scope)
        )
        self.tools["list_secrets"] = lambda scope: c.client.secrets.list_secrets(scope)
        self.tools["list_scopes"] = lambda: c.client.secrets.list_scopes()

        # Jobs
        self.tools["list_jobs"] = lambda **k: c.client.jobs.list(**k)
        self.tools["run_job"] = lambda job_id, **k: c.client.jobs[job_id].run(**k)
        self.tools["job_runs"] = lambda job_id, **k: c.client.jobs[job_id].runs(**k)

        # Warehouses
        self.tools["list_warehouses"] = lambda: c.client.warehouses.list()

        # Volumes
        self.tools["list_volumes"] = lambda schema: c.client.volumes.list(schema)

        # Dataset (Spark)
        self.tools["dataset"] = lambda sql_or_table, **k: c.client.dataset(sql_or_table, **k)
        self.tools["parallelize"] = lambda fn, inputs, **k: c.client.parallelize(fn, inputs, **k)
