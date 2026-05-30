"""AI-powered analysis and market insights via the Anthropic API."""
from __future__ import annotations

import json
from functools import partial

import polars as pl
from fastapi.concurrency import run_in_threadpool

from ...config import Settings
from .fs import FsService
from .analysis import AnalysisService


_MODEL = "claude-sonnet-4-6"
_MAX_ROWS_SAMPLE = 20
_MAX_COLS = 15


class AIInsightService:
    def __init__(self, settings: Settings, *, fs: FsService, analysis: AnalysisService) -> None:
        self.settings = settings
        self.fs = fs
        self.analysis = analysis

    async def analyze_file(self, path: str, query: str) -> dict:
        return await run_in_threadpool(partial(self._analyze_file, path, query))

    async def generate_insight(self, context: str) -> dict:
        return await run_in_threadpool(partial(self._generate_insight, context))

    def _build_client(self):
        try:
            import anthropic  # type: ignore
            return anthropic.Anthropic()
        except ImportError:
            return None

    def _analyze_file(self, path: str, query: str) -> dict:
        client = self._build_client()
        if client is None:
            return {"analysis": "anthropic SDK not installed. Run: pip install anthropic", "model": "unavailable"}

        # Build a compact data summary for the prompt
        resolved = self.fs._resolve(path)
        if not resolved.exists():
            return {"analysis": f"File not found: {path}", "model": _MODEL}

        ext = resolved.suffix.lstrip(".").lower()
        try:
            if ext in ("parquet", "pq"):
                lf = pl.scan_parquet(str(resolved))
            elif ext == "csv":
                lf = pl.scan_csv(str(resolved))
            else:
                lf = None

            if lf is not None:
                schema = lf.collect_schema()
                cols = list(schema.names())[:_MAX_COLS]
                nrows = lf.select(pl.len()).collect(engine="streaming").item()
                sample = lf.select(cols).head(_MAX_ROWS_SAMPLE).collect(engine="streaming")
                numeric = [n for n, dt in schema.items() if dt.is_numeric() and n in cols]
                stats = {}
                if numeric:
                    stats_df = lf.select(numeric).select(
                        [pl.col(c).mean().alias(f"{c}_mean") for c in numeric] +
                        [pl.col(c).std().alias(f"{c}_std") for c in numeric] +
                        [pl.col(c).min().alias(f"{c}_min") for c in numeric] +
                        [pl.col(c).max().alias(f"{c}_max") for c in numeric]
                    ).collect(engine="streaming")
                    for c in numeric:
                        stats[c] = {
                            "mean": round(float(stats_df[f"{c}_mean"][0] or 0), 4),
                            "std": round(float(stats_df[f"{c}_std"][0] or 0), 4),
                            "min": round(float(stats_df[f"{c}_min"][0] or 0), 4),
                            "max": round(float(stats_df[f"{c}_max"][0] or 0), 4),
                        }
                data_summary = (
                    f"File: {path}\nRows: {nrows:,} | Columns: {len(cols)}\n"
                    f"Schema: {dict(list(schema.items())[:_MAX_COLS])}\n"
                    f"Statistics: {json.dumps(stats, indent=2)}\n"
                    f"Sample (first {_MAX_ROWS_SAMPLE} rows):\n{sample.to_pandas().to_string(index=False)}"
                )
            else:
                data_summary = f"File: {path} (binary/unsupported format for preview)"
        except Exception as e:
            data_summary = f"File: {path} — could not read: {e}"

        prompt = f"""You are a data analyst. Analyze this dataset and answer the user's query concisely.

Dataset summary:
{data_summary}

User query: {query}

Provide a direct, insightful analysis. Focus on patterns, anomalies, and actionable insights. Be concise (max 3 paragraphs)."""

        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return {"analysis": response.content[0].text, "model": _MODEL}
        except Exception as e:
            return {"analysis": f"AI analysis failed: {e}", "model": _MODEL}

    def _generate_insight(self, context: str) -> dict:
        client = self._build_client()
        if client is None:
            return {"insight": "anthropic SDK not installed. Run: pip install anthropic", "model": "unavailable"}

        prompt = f"""You are a financial analyst. Generate a brief, insightful market commentary based on the following context.

Context:
{context}

Provide 2-3 sentences of actionable insight. Focus on key patterns, risk factors, and opportunities."""

        try:
            response = client.messages.create(
                model=_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            return {"insight": response.content[0].text, "model": _MODEL}
        except Exception as e:
            return {"insight": f"AI insight failed: {e}", "model": _MODEL}
