from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, AsyncIterator

from ...config import Settings

if TYPE_CHECKING:
    from .pyfunc import PyFuncService
    from .pyfuncrun import PyFuncRunService
    from .backend import BackendService

LOGGER = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an AI assistant embedded in Yggdrasil, a distributed Python compute framework.
You help analyze Python function code, explain system state, and suggest optimizations.
Be concise and practical. Use markdown for code blocks.
"""


class AIService:
    def __init__(
        self,
        settings: Settings,
        *,
        pyfunc_service: "PyFuncService | None" = None,
        pyfuncrun_service: "PyFuncRunService | None" = None,
        backend_service: "BackendService | None" = None,
    ) -> None:
        self.settings = settings
        self._pyfunc = pyfunc_service
        self._pyfuncrun = pyfuncrun_service
        self._backend = backend_service
        self._client = None
        self._available = False
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic()
            self._available = True
        except (ImportError, Exception) as e:
            LOGGER.warning("Anthropic SDK not available: %s — AI features disabled", e)

    @property
    def available(self) -> bool:
        return self._available

    async def analyze_func(self, func_id: int, query: str | None = None) -> dict:
        if not self._available or self._pyfunc is None:
            return {"available": False, "analysis": "AI service not configured. Install the `anthropic` package and set ANTHROPIC_API_KEY."}

        entry = await self._pyfunc.get(func_id)
        prompt = query or "Review this Python function: identify bugs, performance issues, and suggest improvements."
        user_msg = f"{prompt}\n\n```python\n{entry.code}\n```"
        if entry.dependencies:
            user_msg += f"\n\nDependencies: {', '.join(entry.dependencies)}"
        if entry.run_count > 0:
            user_msg += f"\n\nRun stats: {entry.run_count} runs, avg {entry.avg_duration_ms:.0f}ms, success rate {round(entry.success_count / max(1, entry.success_count + entry.failure_count) * 100)}%"

        resp = await self._client.messages.create(  # type: ignore[union-attr]
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text if resp.content else ""
        return {
            "available": True,
            "func_id": func_id,
            "func_name": entry.name,
            "analysis": text,
            "model": resp.model,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }

    async def query(self, question: str) -> dict:
        if not self._available:
            return {"available": False, "answer": "AI service not configured. Install the `anthropic` package and set ANTHROPIC_API_KEY."}

        context: dict = {"question": question}
        system_context = _SYSTEM_PROMPT

        if self._backend is not None:
            snap = self._backend.snapshot()
            system_context += f"\n\nNode state: CPU={snap.cpu_percent:.1f}%, RAM={snap.memory_used_mb:.0f}/{snap.memory_total_mb:.0f}MB, active_runs={snap.active_runs}, total_runs={snap.total_runs}"

        if self._pyfunc is not None:
            funcs = await self._pyfunc.list()
            func_names = [f.name for f in funcs.funcs[:20]]
            system_context += f"\n\nRegistered functions ({len(funcs.funcs)}): {', '.join(func_names)}"

        resp = await self._client.messages.create(  # type: ignore[union-attr]
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_context,
            messages=[{"role": "user", "content": question}],
        )
        text = resp.content[0].text if resp.content else ""
        return {
            "available": True,
            "answer": text,
            "context": context,
            "model": resp.model,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        if not self._available:
            yield json.dumps({"error": "AI service not configured"})
            return

        async with await self._client.messages.stream(  # type: ignore[union-attr]
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield json.dumps({"delta": text})
