"""The ``guide`` skill — point work at the most optimized yggdrasil path.

Loki knows the project it lives in. This is its *do-it-the-yggdrasil-way*
adviser: a curated map of recipes that, for a given task, names the right
abstraction to reach for (the io handlers, ``HTTPSession``, Arrow, ``Field`` /
``DataType`` casting, execution plans, the Databricks ``dbc`` accessors,
``Tabular.display`` …), shows the short idiomatic snippet, and calls out the
hand-rolled anti-pattern to avoid.

Data over code: the knowledge is the :data:`GUIDES` list; the skill just
matches a task to the relevant guides and (optionally) has the agent's engine
synthesize a concrete, grounded plan from them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .skill import LokiSkill, register

if TYPE_CHECKING:
    from .agent import Loki

__all__ = ["Guide", "GUIDES", "match", "GuideSkill"]


@dataclass(frozen=True)
class Guide:
    """One yggdrasil recipe: when it applies, what to use, and what to avoid."""

    id: str
    title: str
    signals: tuple[str, ...]          #: task keywords that select this guide
    summary: str                      #: the optimized approach, in a sentence
    use: tuple[str, ...]              #: the yggdrasil features to reach for
    example: str                      #: a short idiomatic snippet
    avoid: tuple[str, ...]            #: the hand-rolled anti-patterns

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "summary": self.summary,
            "use": list(self.use), "example": self.example, "avoid": list(self.avoid),
        }

    def as_text(self) -> str:
        """A compact, model/human-readable rendering of the guide."""
        use = "\n".join(f"  - {u}" for u in self.use)
        avoid = "\n".join(f"  - {a}" for a in self.avoid)
        return (
            f"## {self.title}\n{self.summary}\n\n"
            f"Use:\n{use}\n\nExample:\n{self.example}\n\nAvoid:\n{avoid}"
        )


#: The yggdrasil recipe book — the optimized path per concern. Extend the data,
#: not the skill: add a :class:`Guide` here and it becomes discoverable.
GUIDES: tuple[Guide, ...] = (
    Guide(
        id="tabular-io",
        title="Move tabular data through the io handlers",
        signals=("read", "write", "parse", "csv", "parquet", "xlsx", "excel",
                 "arrow", "ipc", "feather", "file", "load", "save", "tabular",
                 "dataframe", "delta", "ndjson"),
        summary="Go through `IO.from_(path)` — one handler per format (CSV / JSON / "
                "NDJSON / Parquet / Arrow IPC / XLSX / Delta), schema-aware and "
                "zero-copy where it can be. Cache as Parquet; move on the wire as Arrow IPC.",
        use=("yggdrasil.io.holder.IO — `IO.from_(path).to_polars()` / "
             "`.write_polars_frame(df)`, format chosen by extension/MediaType",
             "Parquet for at-rest/cache (columnar, compressed), Arrow IPC for hot transfer",
             "Tabular is itself a Tabular — same API across pandas/polars/Arrow/Spark"),
        example="from yggdrasil.io.holder import IO\n"
                "df = IO.from_('data.parquet').to_polars()\n"
                "IO.from_('out.arrow').write_polars_frame(df)   # any format by suffix",
        avoid=("hand-rolling csv.reader / json.loads / openpyxl per format",
               "pandas.read_*/to_* round-trips when polars+Arrow stays zero-copy"),
    ),
    Guide(
        id="http-fetch",
        title="Fetch over HTTPSession, parse with HTTPResponse",
        signals=("http", "fetch", "download", "request", "api", "url", "rest",
                 "scrape", "web", "endpoint", "get ", "post "),
        summary="Drive every HTTP call through the singleton `HTTPSession` (pooling, "
                "retry budget, response cache) and turn a tabular body straight into a "
                "frame with `HTTPResponse.to_polars()` / `.to_arrow_table()`.",
        use=("yggdrasil.http_.HTTPSession — `.get(url, json=…, timeout=…)` / `.post(...)`",
             "HTTPResponse.to_polars()/to_arrow_table()/json() — media-type auto-detected",
             "yggdrasil.loki.web for the agent-facing helpers (read_table/scrape/Browser)"),
        example="from yggdrasil.http_ import HTTPSession\n"
                "resp = HTTPSession().get('https://host/data.csv')\n"
                "df = resp.to_polars()   # CSV/JSON/Parquet/Arrow/XLSX all handled",
        avoid=("requests / urllib / httpx directly (no pooling, retry, or cache)",
               "manually decoding a body you could hand to `to_polars()`"),
    ),
    Guide(
        id="schema-cast",
        title="Type and cast through Field / DataType / Schema",
        signals=("cast", "type", "dtype", "schema", "convert", "timezone", "tz",
                 "date", "datetime", "astype", "coerce", "column type"),
        summary="Describe and convert columns with the data layer — `DataType.from_(...)`, "
                "`Field(name, dtype).cast_polars_series(...)`, `Schema` — so casting is "
                "uniform across backends and timezone-aware.",
        use=("yggdrasil.data.DataType — `DataType.from_('date'|'float64'|'datetime')`",
             "yggdrasil.data.Field — `Field(name=c, dtype=…).cast_polars_series(series)`",
             "yggdrasil.data.Schema — a typed contract for a whole frame",
             "the `transform` Loki skill wraps this (cast / tz / rename / select)"),
        example="from yggdrasil.data import DataType, Field\n"
                "typed = Field(name='ts', dtype=DataType.from_('datetime')).cast_polars_series(df['ts'])",
        avoid=("ad-hoc `df.with_columns(pl.col(c).cast(...))` for portable casts",
               "string-munging dates instead of a real temporal DataType"),
    ),
    Guide(
        id="llm-data",
        title="Share data with a model via the Tabular abstraction",
        signals=("llm", "model", "prompt", "context", "token", "share data",
                 "send data", "preview", "display", "show rows", "agent"),
        summary="Don't serialize by hand — every row set (a statement result, an "
                "IO leaf) is a Tabular: `display(n)` for an aligned preview into a "
                "prompt, `to_pylist()` for records, `to_arrow()` for the binary wire.",
        use=("Tabular.display(n) — aligned first-n-rows preview, cheap (early-stops)",
             "Tabular.to_pylist()/to_polars()/to_arrow() — records / frame / Arrow",
             "Tabular.from_(obj) to wrap, .select/.filter/.cast to transform first"),
        example="rows = dbc.sql.execute('SELECT * FROM t LIMIT 100')\n"
                "print(rows.display(10))     # aligned preview for a human/prompt\n"
                "records = rows.to_pylist()  # JSON-able records",
        avoid=("hand-rolling CSV/markdown/JSON from a frame — Tabular already converts",
               "materializing a whole result just to show a few rows (use display)"),
    ),
    Guide(
        id="databricks-compute",
        title="Pick Databricks compute the ygg way",
        signals=("databricks", "cluster", "serverless", "compute", "warehouse",
                 "wheel", "seed", "deploy", "job", "dbu", "spark"),
        summary="Prefer serverless for inner Databricks I/O, a single-user cluster for "
                "external-resource access, and always the pre-built ygg wheel "
                "environments deployed by `ygg databricks deploy`.",
        use=("yggdrasil.databricks.DatabricksClient — `dbc.<service>` accessors",
             "`ygg databricks deploy` — pre-built wheel envs (no per-run pip install)",
             "serverless for inner I/O; single-user cluster for external resources"),
        example="from yggdrasil.databricks import DatabricksClient\n"
                "dbc = DatabricksClient.current()\n"
                "rows = dbc.sql.execute('SELECT * FROM main.sales.orders LIMIT 10').to_polars()",
        avoid=("`%pip install` on every run instead of a seeded wheel env",
               "a multi-node cluster for work serverless handles cheaper"),
    ),
    Guide(
        id="databricks-query",
        title="Query Databricks via dbc.sql / genie",
        signals=("sql", "query", "genie", "unity catalog", "table", "warehouse",
                 "select", "uc ", "catalog", "ask the data"),
        summary="Run SQL through `dbc.sql.execute(...)` (results are Arrow → "
                "`.to_polars()`), or ask a question in natural language through "
                "`dbc.genie` (the `genie` Loki skill).",
        use=("dbc.sql.execute(query) — statement result, `.to_polars()` for rows",
             "dbc.genie.space(id).ask(question) — AI/BI Genie (text + SQL + rows)",
             "the `databricks-sql` / `genie` Loki skills wrap both"),
        example="loki.run('databricks-sql', query='SELECT count(*) FROM main.s.t')\n"
                "loki.run('genie', question='revenue by region last quarter')",
        avoid=("pulling whole tables to the client when SQL can aggregate first",),
    ),
    Guide(
        id="plan-sql",
        title="Build/translate SQL with execution plans",
        signals=("plan", "sql dialect", "parse sql", "emit sql", "translate",
                 "postgres", "spark sql", "optimize query", "execution plan"),
        summary="Use the plan layer to parse, emit, and execute SQL across dialects "
                "rather than string-building queries — one logical plan, many dialects.",
        use=("yggdrasil.saga.plan — execution plans + SQL parse/emit/execute",
             "yggdrasil.saga.expr — predicate/expression engine "
             "(polars / pyarrow / sql backends), e.g. `col('x') > 1`"),
        example="from yggdrasil.saga.expr import col\n"
                "predicate = (col('price') > 100) & col('active')   # → polars/pyarrow/SQL",
        avoid=("f-string SQL that only targets one dialect (and risks injection)",),
    ),
    Guide(
        id="ids-upsert",
        title="Identify with int64 xxhash IDs, upsert by default",
        signals=("id", "identifier", "primary key", "upsert", "dedupe", "merge",
                 "unique", "hash", "uuid"),
        summary="Use int64 composite IDs (`xxh32(semantic_key) << 32 | timestamp_ms`), "
                "never crypto hashes, and upsert by name (create if absent, update if "
                "present); an ID is immutable once assigned.",
        use=("xxhash.xxh32(...) for the semantic half of a composite int64 id",
             "upsert-by-name semantics; treat the id as immutable",
             "StrictModel (extra='forbid') for request/response contracts"),
        example="import xxhash\n"
                "id_ = (xxhash.xxh32_intdigest(name) << 32) | (ms & 0xFFFFFFFF)",
        avoid=("md5/sha for IDs (slow, oversized, not the convention)",
               "string UUIDs where an int64 composite is expected"),
    ),
    Guide(
        id="loki-agent",
        title="Let Loki drive — engines, skills, autonomous act",
        signals=("autonomous", "agent", "loki", "reason", "automate", "act",
                 "engine", "local model", "skill"),
        summary="Reason through `Loki.reason` (resource-aware local↔remote `select`), "
                "act on a file tree with `Loki.act`, and dispatch capabilities as "
                "skills — the small free local model handles light/setup work.",
        use=("Loki.reason / reason_stream — best or resource-selected engine",
             "Loki.act(task, root=…) — the confined reason→act→observe loop",
             "Loki.run(skill, **kw) / ygg loki run — dispatch a registered skill",
             "ygg loki setup — bootstrap a free local model on demand"),
        example="loki = Loki.current()\n"
                "loki.act('add type hints to utils.py', root='.')\n"
                "loki.run('tabular', url='https://host/data.csv')",
        avoid=("a bespoke LLM client when an engine + skill already exist",),
    ),
    Guide(
        id="energy-markets",
        title="Pull power-market data (ENTSO-E) into frames",
        signals=("entso", "entsoe", "power", "electricity", "energy", "grid",
                 "day-ahead", "day ahead", "spot price", "bidding zone", "load",
                 "generation", "mwh", "megawatt", "utility", "commodity"),
        summary="Fetch European day-ahead prices / load / generation for a bidding "
                "zone with the `entsoe` skill — it parses the Transparency Platform "
                "XML into a tidy timestamp/value polars frame and caches it through "
                "the io handlers, ready for the usual frame analytics.",
        use=("loki.run('entsoe', series='day_ahead_prices', zone='DE_LU', days=7)",
             "yggdrasil.loki.entsoe.fetch_frame / parse_timeseries_xml / build_query",
             "ZONES alias map (DE_LU, FR, NL …) or any EIC; ENTSOE_API_TOKEN to auth",
             "then the frame path: transform (cast/tz), cache, store Parquet/Delta"),
        example="loki.run('entsoe', series='day_ahead_prices', zone='FR', days=7)\n"
                "df = entsoe.fetch_frame('load', 'DE_LU', '2024-01-01', '2024-01-08')\n"
                "df.group_by_dynamic('timestamp', every='1d').agg(pl.col('value').mean())",
        avoid=("hand-rolled ENTSO-E XML parsing or a bespoke HTTP client — use the "
               "entsoe helpers over HTTPSession",),
    ),
)


def match(task: str, *, top: int = 3) -> list[Guide]:
    """The guides most relevant to *task*, best first (by signal hits)."""
    low = (task or "").lower()
    scored = [(sum(1 for s in g.signals if s in low), g) for g in GUIDES]
    hits = sorted((p for p in scored if p[0] > 0), key=lambda p: -p[0])
    return [g for _, g in hits[:top]]


@register
class GuideSkill(LokiSkill):
    """Guide a task to the most optimized yggdrasil implementation.

    Matches the task to the relevant recipes in :data:`GUIDES` — the right
    abstraction, the idiomatic snippet, the anti-pattern to avoid — and, when
    ``plan=True`` and an engine is reachable, has the agent synthesize a
    concrete plan **grounded only in those yggdrasil features**. Runs anywhere.
    """

    name = "guide"
    description = "Advise the most optimized way to build something with yggdrasil's features."
    preprompt = (
        "You are a yggdrasil expert. Recommend the idiomatic, optimized path "
        "using the project's own abstractions; never hand-roll what a yggdrasil "
        "feature already does."
    )

    def run(
        self,
        agent: Loki,
        *,
        task: Optional[str] = None,
        topic: Optional[str] = None,
        plan: bool = False,
        top: int = 3,
        **_: Any,
    ) -> dict[str, Any]:
        if topic:
            picked = [g for g in GUIDES if g.id == topic]
            if not picked:
                raise KeyError(f"unknown topic {topic!r}; topics: "
                               f"{', '.join(g.id for g in GUIDES)}")
        elif task:
            picked = match(task, top=top) or list(GUIDES[:top])
        else:
            raise ValueError("provide task= (what you want to build) or topic=")

        out: dict[str, Any] = {
            "task": task,
            "guides": [g.to_dict() for g in picked],
            "topics": [g.id for g in GUIDES],
        }
        # Ground the engine on the matched recipes for a tailored plan.
        if plan and task:
            eng = agent.engine()
            if eng is not None and eng.available():
                grounding = "\n\n".join(g.as_text() for g in picked)
                out["plan"] = agent.reason(
                    f"Task: {task}\n\nUsing ONLY these yggdrasil features, give the "
                    f"most optimized implementation as concise numbered steps with the "
                    f"exact abstractions to call:\n\n{grounding}",
                    system=self.preprompt,
                )
        return out
