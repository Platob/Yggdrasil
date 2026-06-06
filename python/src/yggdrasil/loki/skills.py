"""Built-in Loki skills — the backend-agnostic catalog.

These run anywhere Loki runs (no cloud session required):

- :class:`WebSkill` — reach and *drive* the internet (browse, tables, forms).
- :class:`TabularSkill` / :class:`TransformSkill` — the data path: fetch a
  source into a frame through the io handlers, cache it, reshape it.
- :class:`PythonProjectSkill` — scaffold a Python project, write code into it
  (provided, or reasoned from a task via the agent's engine), and run it.
- :class:`SetupSkill` — bootstrap a free local model on demand.

Backend-specialized skills live in their own packages and register only when
their backend is reachable — Databricks (``genie``, ``databricks-*``) in
:mod:`yggdrasil.databricks.loki`, AWS (``aws-*``) in :mod:`yggdrasil.aws.loki`.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Optional

from .skill import LokiSkill, register

if TYPE_CHECKING:
    from .agent import Loki

__all__ = ["AgentSkill", "PythonProjectSkill", "SetupSkill",
           "WebSkill", "TabularSkill", "TransformSkill"]


@register
class AgentSkill(LokiSkill):
    """Pursue a task autonomously inside a file tree — Loki's agentic loop.

    The headline "act on its own + modify files" skill. Given a ``task``,
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
class SetupSkill(LokiSkill):
    """Bootstrap a free local model, and redirect heavier setup elsewhere.

    Loki's self-setup path: ready a lightweight, lazily-installed local brain
    (:meth:`Loki.bootstrap_local`) — smart enough for basic install/config —
    and surface the **redirects** for the heavier work a small model should
    hand off: configuring Databricks, pulling a bigger/smarter model, or
    escalating reasoning to a remote engine. Runs anywhere.
    """

    name = "setup"
    description = "Bootstrap a free local model (lazy-install) and point heavier setup to the right tool."

    def run(self, agent: Loki, *, model: Optional[str] = None, pull: bool = True,
            **_: Any) -> dict[str, Any]:
        res = agent.bootstrap_local(model=model, pull=pull)
        # What a lightweight model should hand heavier setup off to.
        res["redirects"] = {
            "configure databricks": "ygg databricks configure   (host + token, then `ygg databricks seed`)",
            "heavier local model": "loki.run('setup', model='qwen2.5:14b')   (or any larger ollama/HF id)",
            "stronger reasoning": "set ANTHROPIC_API_KEY / log into Claude Code, then heavy tasks escalate automatically",
        }
        return res


@register
class WebSkill(LokiSkill):
    """Reach the internet — browse, read tables/JSON/images, or drive a page.

    Runs anywhere (no backend). Fetches through
    :class:`~yggdrasil.http_.HTTPSession` and parses tabular bodies through the
    io handlers (:mod:`yggdrasil.loki.web`). ``action="auto"`` infers from the
    URL (a ``.csv``/``.parquet``/… → table, an image extension → image, else
    browse as text); pass ``question=`` to have the agent reason over a
    fetched page.

    For *interactive* pages it drives a real headless browser (Playwright):
    ``action="form"`` fills ``fields`` (selector → value) and optionally clicks
    ``submit``; ``action="interact"`` runs a list of ``steps`` (type, click,
    select, check, press, submit, wait_for) — filling forms, clicking buttons,
    and reading what the page becomes.
    """

    name = "web"
    description = "Fetch + drive the internet — browse, read tables/JSON/images, fill forms, click buttons."
    preprompt = (
        "Answer strictly from the fetched page — quote only facts present in it, "
        "give exact figures, and say plainly when the page does not cover the "
        "question. Be concise."
    )

    def run(
        self,
        agent: Loki,
        *,
        url: str,
        action: str = "auto",
        fmt: Optional[str] = None,
        save: Optional[str] = None,
        question: Optional[str] = None,
        fields: Optional[dict] = None,
        steps: Optional[list] = None,
        submit: Optional[str] = None,
        headless: bool = True,
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

        if action in ("form", "interact"):
            if not web.ensure_browser():  # auto-installs unless disabled
                return {"action": action, "url": url, "error": "browser automation unavailable",
                        "install": "pip install playwright && playwright install chromium"}
            if action == "form":
                return {"action": "form",
                        **web.fill_form(url, fields or {}, submit=submit,
                                        headless=headless, screenshot=save)}
            return {"action": "interact",
                    **web.interact(url, steps or [], headless=headless, screenshot=save)}

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
                    system=self.preprompt,
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


def _read_tabular(source: str, fmt: Optional[str] = None):
    """Read any *source* into a polars frame **through the io handlers**.

    ``IO.from_(source)`` already returns the right Tabular leaf for the scheme
    *and* format — ``http(s)://`` (over the shared ``HTTPSession``), a local
    path, ``s3://``, ``dbfs:/``, a UC Volume, CSV/JSON/Parquet/Arrow/XLSX — so
    there's nothing bespoke to do: hand it the source, take ``to_polars()``.
    """
    from yggdrasil.io.holder import IO

    return IO.from_(str(source)).to_polars()


@register
class TabularSkill(LokiSkill):
    """Fetch a data/timeseries source as a tabular frame, cache it, propose reuse.

    Loki's data path: when a request is data- or time-series-shaped, fetch it
    straight into a polars frame — through :func:`web.read_table`, i.e. the
    :class:`HTTPResponse` → io tabular handlers (CSV / JSON / Parquet / Arrow /
    XLSX, format auto-detected) — **cache it as Parquet** in the session cache
    via the io abstraction (an optimized columnar copy), and return the frame
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

        from . import dataproto

        if cache:                       # reuse a previously cached frame
            df, source = IO.from_(str(cache)).to_polars(), cache
        elif url:                       # http → session; local/s3/dbfs → io handlers
            df, source = _read_tabular(url, fmt), url
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
            "preview": dataproto.encode(df),   # token-efficient view for the LLM
            "cached_to": str(cached_to),
            "stored": stored,
            "next_steps": steps,
        }


@register
class TransformSkill(LokiSkill):
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

        from . import dataproto

        if cache:
            df, source = IO.from_(str(cache)).to_polars(), cache
        elif url:                       # http → session; local/s3/dbfs → io handlers
            df, source = _read_tabular(url, fmt), url
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
            "preview": dataproto.encode(df),   # token-efficient view for the LLM
            "cached_to": str(cached_to),
            "next_steps": [
                f"reuse:  loki.run('tabular', cache={str(cached_to)!r})",
                f"store:  loki.run('tabular', cache={str(cached_to)!r}, store='out.parquet')",
            ],
        }


@register
class PythonProjectSkill(LokiSkill):
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
    preprompt = (
        "You are a senior Python engineer writing inside the yggdrasil project. "
        "Reach for its abstractions before stdlib or third-party equivalents: "
        "IO.from_(path) for tabular IO (CSV/JSON/Parquet/Arrow/XLSX/Delta), "
        "HTTPSession for HTTP, DataType/Field for casting, dbc.<service> for "
        "Databricks, dataproto for model-facing data. Return runnable code only — "
        "no prose, no markdown fences."
    )

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
        # Ground the prompt in the relevant yggdrasil recipes so the generated
        # code uses the project's optimized path, not a hand-rolled one.
        if code is None and task:
            from . import guides

            recipes = "\n\n".join(g.as_text() for g in guides.match(task, top=2))
            grounding = f"\n\nLeverage these yggdrasil features:\n{recipes}" if recipes else ""
            code = agent.reason(
                f"Write a single self-contained Python script that: {task}. "
                f"Print its result to stdout.{grounding}",
                system=self.preprompt,
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
