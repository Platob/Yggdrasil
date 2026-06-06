"""Built-in Loki behaviors.

The behavior catalog. Two patterns live here:

- :class:`GenieBehavior` — guard on a detected backend, then drive a
  Databricks service endpoint through Loki's token provider
  (``agent.databricks``).
- :class:`PythonProjectBehavior` — a *local* behavior: Loki scaffolds a
  Python project, writes code into it (provided, or reasoned from a task via
  the agent's engine), and runs it — the agent authoring and executing code.

Replication, inter-agent messaging, HTTP ingestion and serving land here next.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Optional

from .behavior import LokiBehavior, register

if TYPE_CHECKING:
    from .agent import Loki

__all__ = ["AgentBehavior", "GenieBehavior", "PythonProjectBehavior", "WebBehavior",
           "TabularBehavior", "TransformBehavior"]


@register
class AgentBehavior(LokiBehavior):
    """Pursue a task autonomously inside a file tree — Loki's agentic loop.

    The headline "act on its own + modify files" behavior. Given a ``task``,
    Loki reasons against a confined toolbox (list/read/find/grep, plus
    write/edit unless ``read_only``, plus a shell when ``allow_shell``),
    taking one tool call per turn until it's done — discovering the project
    and changing files itself. Runs anywhere an engine is reachable; thin
    wrapper over :meth:`Loki.act` so code and CLI share one implementation.
    """

    name = "agent"
    description = "Autonomously discover and modify files to accomplish a task."

    def run(
        self,
        agent: Loki,
        *,
        task: str,
        root: str = ".",
        engine: Optional[str] = None,
        tier: Optional[str] = None,
        max_steps: int = 12,
        read_only: bool = False,
        allow_shell: bool = False,
        allow_web: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return agent.act(
            task,
            root=root,
            engine=engine,
            tier=tier,
            max_steps=max_steps,
            read_only=read_only,
            allow_shell=allow_shell,
            allow_web=allow_web,
        )


@register
class WebBehavior(LokiBehavior):
    """Reach the internet — browse a page, read a table, JSON, or image.

    Runs anywhere (no backend). Fetches through
    :class:`~yggdrasil.http_.HTTPSession` and parses tabular bodies through the
    io handlers (:mod:`yggdrasil.loki.web`). ``action="auto"`` infers from the
    URL (a ``.csv``/``.parquet``/… → table, an image extension → image, else
    browse as text); pass ``question=`` to have the agent reason over a
    fetched page.
    """

    name = "web"
    description = "Fetch the internet — browse pages, read tables/JSON, or images."

    def run(
        self,
        agent: Loki,
        *,
        url: str,
        action: str = "auto",
        fmt: Optional[str] = None,
        save: Optional[str] = None,
        question: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        from . import web

        if action == "auto":
            ext = url.rsplit("?", 1)[0].rsplit(".", 1)[-1].lower()
            if ext in ("csv", "tsv", "json", "ndjson", "jsonl", "parquet", "pq",
                       "arrow", "feather", "xlsx", "xls"):
                action = "json" if ext in ("json",) else "table"
            elif ext in ("png", "jpg", "jpeg", "gif", "webp"):
                action = "image"
            else:
                action = "text"

        if action == "scrape":
            return {"action": "scrape", **web.scrape(url)}
        if action == "apis":
            return {"action": "apis", **web.discover_apis(url)}
        if action == "table":
            df = web.read_table(url, fmt=fmt)
            return {"action": "table", "url": url, "shape": list(df.shape),
                    "columns": list(df.columns), "preview": str(df.head(10))}
        if action == "json":
            return {"action": "json", "url": url, "data": web.read_json(url)}
        if action == "image":
            return {"action": "image", **web.read_image(url, save_to=save)}

        page = web.read_text(url)
        out: dict[str, Any] = {"action": "text", **page}
        if question:
            eng = agent.engine()
            if eng is not None and eng.available():
                out["answer"] = agent.reason(
                    f"Using only this page, {question}\n\n{page['text']}",
                    system="Answer concisely from the page; say if it's not covered.",
                )
        return out


def _tab_slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "-", (text or "").strip().lower()).strip("-")[:48] or "data"


def _auto_key(source: str) -> str:
    """A short, stable cache key from a source — ``<domain>-<xxh32>`` for URLs."""
    import urllib.parse as _up

    import xxhash

    digest = xxhash.xxh32_hexdigest(source)[:6]
    if source.startswith(("http://", "https://")):
        host = _up.urlparse(source).netloc.split(":")[0]
        comps = [c for c in host.split(".") if c not in ("www", "api")] or [host]
        word = re.sub(r"[^a-z0-9]", "", max(comps, key=len).lower())[:16] or "data"
        return f"{word}-{digest}"
    return f"{_tab_slug(source)[:24].rstrip('-')}-{digest}"


def _json_to_frame(data: Any):
    """Normalize a decoded JSON payload into a polars frame.

    Handles the common shapes external data APIs return: a list of records, a
    nested ``{key: {date: {sym: val}}}`` time series (→ long ``date/symbol/value``),
    a flat ``{date: value}`` mapping, or a single record.
    """
    import polars as pl

    if isinstance(data, list):
        return pl.DataFrame(data)
    if isinstance(data, dict):
        for key in ("rates", "data", "results", "observations", "values", "series"):
            v = data.get(key)
            if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
                rows = [{"date": d, "symbol": s, "value": val}
                        for d, m in sorted(v.items()) for s, val in m.items()]
                return pl.DataFrame(rows)
            if isinstance(v, dict) and v:
                return pl.DataFrame({"date": list(v.keys()), "value": list(v.values())})
            if isinstance(v, list) and v:
                return pl.DataFrame(v)
        return pl.DataFrame([data])
    return pl.DataFrame({"value": [data]})


def _fetch_frame(url: str, fmt: Optional[str]):
    """Fetch *url* as a polars frame.

    JSON bodies are **normalized** into a flat/long frame (a time series →
    ``date/symbol/value``), since API JSON rarely maps 1:1 to rows. Genuinely
    tabular bodies (CSV / Parquet / Arrow / XLSX) go through the io handlers.
    """
    from . import web

    resp = web.fetch(url)
    mime = str(resp.media_type.mime_type.value)
    if not fmt and "json" in mime:
        return _json_to_frame(resp.json())
    try:
        return web.read_table(url, fmt=fmt)
    except Exception:
        return _json_to_frame(web.read_json(url))


@register
class TabularBehavior(LokiBehavior):
    """Fetch a data/timeseries source as a tabular frame, cache it, propose reuse.

    Loki's data path: when a request is data- or time-series-shaped, fetch it
    straight into a polars frame (through the io handlers for CSV/Parquet/…, or
    a normalized JSON payload), **cache it as Parquet** in the session cache via
    the io abstraction (an optimized columnar copy), and return the frame
    preview plus concrete **next steps** — reuse the cache, store it elsewhere
    (Parquet / Arrow / CSV / Delta), or load it into Databricks.
    """

    name = "tabular"
    description = "Fetch a data/timeseries source as a cached tabular frame and propose reuse/store steps."

    def run(
        self,
        agent: Loki,
        *,
        url: Optional[str] = None,
        cache: Optional[str] = None,
        store: Optional[str] = None,
        fmt: Optional[str] = None,
        key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        from yggdrasil.io.holder import IO

        if cache:                       # reuse a previously cached frame
            df, source = IO.from_(str(cache)).to_polars(), cache
        elif url:
            df, source = _fetch_frame(url, fmt), url
        else:
            raise ValueError("provide url= to fetch, or cache= to reuse a cached frame")

        cdir = pathlib.Path(cache_dir) if cache_dir else (pathlib.Path.home() / ".loki" / "cache")
        cdir.mkdir(parents=True, exist_ok=True)
        key = key or _auto_key(source)
        cached_to = cdir / f"{key}.parquet"
        IO.from_(str(cached_to)).write_polars_frame(df)   # optimized columnar copy

        stored = None
        if store:
            IO.from_(store).write_polars_frame(df)
            stored = store

        steps = [
            f"reuse the cache:  loki.run('tabular', cache={str(cached_to)!r})",
            f"transform (cast types / timezone / select):  loki.run('transform', "
            f"cache={str(cached_to)!r}, cast={{'date':'date','value':'float64'}}, tz={{'date':'UTC'}})",
            f"store as Parquet/Arrow/CSV/Delta:  loki.run('tabular', cache={str(cached_to)!r}, store='out.parquet')",
            f"read it back anywhere:  IO.from_({str(cached_to)!r}).to_polars()",
        ]
        if agent.has("databricks"):
            steps.append(
                f"load into Databricks:  upload {cached_to.name} to a UC Volume, then "
                f"loki.run('databricks-sql', query='CREATE TABLE <cat>.<sch>.{key} AS "
                f"SELECT * FROM parquet.`/Volumes/.../{cached_to.name}`')"
            )
        return {
            "source": source,
            "rows": df.height,
            "columns": list(df.columns),
            "preview": str(df.head(10)),
            "cached_to": str(cached_to),
            "stored": stored,
            "next_steps": steps,
        }


@register
class TransformBehavior(LokiBehavior):
    """Transform a cached/fetched frame before reuse — cast types, tz, rename, select.

    Closes the data loop: take a cached (or freshly fetched) tabular frame and
    reshape it using the **yggdrasil field-type casting** — cast columns to
    target types (``date``, ``float64``, ``datetime``, …, including
    **timezone**), rename, or select — then re-cache the optimized Parquet copy.
    So a fetched series can be made analysis-ready (typed dates, numeric values,
    a target timezone) and reused cleanly.
    """

    name = "transform"
    description = "Cast field types (incl. timezone), rename, or select columns of a cached frame; re-cache."

    def run(
        self,
        agent: Loki,
        *,
        cache: Optional[str] = None,
        url: Optional[str] = None,
        cast: Optional[dict] = None,
        tz: Optional[dict] = None,
        rename: Optional[dict] = None,
        select: Optional[list] = None,
        fmt: Optional[str] = None,
        cache_dir: Optional[str] = None,
        key: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        import polars as pl

        from yggdrasil.data import DataType, Field
        from yggdrasil.io.holder import IO

        if cache:
            df, source = IO.from_(str(cache)).to_polars(), cache
        elif url:
            df, source = _fetch_frame(url, fmt), url
        else:
            raise ValueError("provide cache= (or url=) to transform")

        if cast:  # field-type casting via the data layer (DataType/Field)
            cols = [
                Field(name=c, dtype=DataType.from_(t)).cast_polars_series(df[c]).alias(c)
                for c, t in cast.items() if c in df.columns
            ]
            if cols:
                df = df.with_columns(cols)
        if tz:  # timezone cast on temporal columns
            for c, target in tz.items():
                if c in df.columns and df[c].dtype == pl.Datetime:
                    try:
                        df = df.with_columns(pl.col(c).dt.convert_time_zone(target).alias(c))
                    except Exception:
                        df = df.with_columns(pl.col(c).dt.replace_time_zone(target).alias(c))
        if rename:
            df = df.rename({k: v for k, v in rename.items() if k in df.columns})
        if select:
            df = df.select([c for c in select if c in df.columns])

        cdir = pathlib.Path(cache_dir) if cache_dir else (pathlib.Path.home() / ".loki" / "cache")
        cdir.mkdir(parents=True, exist_ok=True)
        key = key or f"{_auto_key(source)}-t"
        cached_to = cdir / f"{key}.parquet"
        IO.from_(str(cached_to)).write_polars_frame(df)
        return {
            "source": source,
            "rows": df.height,
            "columns": list(df.columns),
            "schema": {c: str(t) for c, t in zip(df.columns, df.dtypes)},
            "preview": str(df.head(10)),
            "cached_to": str(cached_to),
            "next_steps": [
                f"reuse:  loki.run('tabular', cache={str(cached_to)!r})",
                f"store:  loki.run('tabular', cache={str(cached_to)!r}, store='out.parquet')",
            ],
        }


@register
class GenieBehavior(LokiBehavior):
    """Ask a Databricks Genie space a question and return its answer."""

    name = "genie"
    description = "Ask a Databricks AI/BI Genie space a question (text + SQL + rows)."
    requires = "databricks"

    def run(
        self,
        agent: Loki,
        *,
        question: str,
        space: Optional[str] = None,
        rows: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        client = agent.databricks
        if client is None:  # available() already guards, belt-and-suspenders
            raise RuntimeError("no Databricks session")

        # Autonomy: when no space is named, reason against the first space the
        # current user can reach.
        if space is None:
            spaces = client.genie.spaces()
            if not spaces:
                raise RuntimeError("no Genie spaces are accessible to this user")
            target = spaces[0]
        else:
            target = client.genie.space(space)

        answer = target.ask(question)
        out: dict[str, Any] = {
            "space_id": target.space_id,
            "conversation_id": answer.conversation_id,
            "text": answer.text,
            "query": answer.query,
            "statement_id": answer.statement_id,
        }
        if rows and answer.query:
            out["rows"] = answer.to_polars()
        return out


@register
class PythonProjectBehavior(LokiBehavior):
    """Scaffold a small Python project, write code into it, and execute it.

    Runs anywhere (no backend required). Either pass ``code`` directly, or a
    ``task`` description that the agent reasons into a script via its engine
    (``agent.reason``). Loki then writes a minimal project (``pyproject.toml``
    + a package with a ``main`` module), runs ``main.py`` in a subprocess, and
    returns where it landed plus the captured output — the agent authoring and
    executing code end-to-end.
    """

    name = "python_project"
    description = "Create a Python project, write code (given or reasoned), and run it."

    def run(
        self,
        agent: Loki,
        *,
        project: str = "ygg_demo",
        task: Optional[str] = None,
        code: Optional[str] = None,
        base_dir: Optional[str] = None,
        run: bool = True,
        timeout: float = 60.0,
        **_: Any,
    ) -> dict[str, Any]:
        # Reason the code from the task when none is supplied (needs an engine).
        if code is None and task:
            code = agent.reason(
                f"Write a single self-contained Python script that: {task}. "
                "Print its result to stdout. Output only the code — no prose, "
                "no markdown fences.",
                system="You are a senior Python engineer. Return runnable code only.",
            )
        if code is None:
            raise ValueError("provide `code=` directly or a `task=` to reason it from")
        # Strip markdown fences a reasoned reply may wrap the code in.
        code = re.sub(r"\A\s*```(?:python)?\n|\n```\s*\Z", "", code).strip() + "\n"

        pkg = re.sub(r"[^0-9A-Za-z_]+", "_", project).strip("_").lower() or "app"
        root = (
            pathlib.Path(base_dir)
            if base_dir
            else pathlib.Path(tempfile.mkdtemp(prefix="ygg-loki-"))
        )
        project = root / pkg
        package = project / pkg
        package.mkdir(parents=True, exist_ok=True)

        (project / "pyproject.toml").write_text(
            f'[project]\nname = "{pkg}"\nversion = "0.1.0"\n'
            f'requires-python = ">=3.9"\n\n'
            f'[project.scripts]\n{pkg} = "{pkg}.main:main"\n'
        )
        (project / "README.md").write_text(f"# {pkg}\n\nScaffolded by Loki.\n")
        (package / "__init__.py").write_text('__all__ = ["main"]\n')
        (package / "main.py").write_text(code)

        files = sorted(
            str(p.relative_to(project)) for p in project.rglob("*") if p.is_file()
        )
        result: dict[str, Any] = {
            "project_dir": str(project),
            "package": pkg,
            "files": files,
        }
        if run:
            proc = subprocess.run(
                [sys.executable, str(package / "main.py")],
                cwd=str(project),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONPATH": str(project)},
            )
            result.update(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        return result
