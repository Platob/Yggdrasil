# ai_session.py
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, TypeVar

try:
    from openai import OpenAI
except ImportError:
    from yggdrasil.environ import PyEnv

    openai = PyEnv.runtime_import_module(module_name="openai", pip_name="OpenAI", install=True)
    from openai import OpenAI

from yggdrasil.dataclasses.expiring import Expiring

__all__ = [
    "AISession",
    "Role",
    "Message",
    "ChatResponse",
    "RetryConfig",
]

log = logging.getLogger(__name__)

T = TypeVar("T")

_CODE_FENCE_RE = re.compile(r"^```(?P<lang>\w+)?\n?(?P<body>.*?)```\s*$", re.DOTALL)
_THINK_TAG_RE  = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Enums / small value types
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        d: Dict[str, str] = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ChatResponse:
    raw: str                        # original model output
    text: str                       # cleaned text (fences / think-tags stripped)
    lang: Optional[str]             # detected fence language, e.g. "sql", "json"
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float

    # --- convenience parsers ---

    def as_json(self) -> Any:
        """Parse text as JSON; raises ValueError on failure."""
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Response is not valid JSON: {exc}\n---\n{self.text}") from exc

    def as_lines(self, *, skip_empty: bool = True) -> List[str]:
        lines = self.text.splitlines()
        return [l for l in lines if l.strip()] if skip_empty else lines

    def as_csv_rows(self, sep: str = ",") -> List[List[str]]:
        return [row.split(sep) for row in self.as_lines()]


@dataclass
class RetryConfig:
    max_attempts: int   = 3
    base_delay_s: float = 1.0
    backoff_factor: float = 2.0
    retryable_status: tuple = (429, 500, 502, 503, 504)


# ---------------------------------------------------------------------------
# Core session
# ---------------------------------------------------------------------------

@dataclass
class AISession(ABC):
    """
    Abstract base for OpenAI-compatible LLM sessions.

    Subclasses only need to implement ``system_prompt()``.
    All chat paths converge on ``_complete()``, making provider
    swapping straightforward.
    """

    api_key:  str | Expiring[str]
    base_url: str
    model:    str = "gemini-2.5-flash"
    retry:    RetryConfig = field(default_factory=RetryConfig)

    _client: OpenAI = field(default=None, init=False, repr=False)

    @property
    def client(self):
        api_key = self.api_key.value if isinstance(self.api_key, Expiring) else self.api_key

        if self._client is None:
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        elif isinstance(self.api_key, Expiring):
            if self.api_key.is_expired(now_ns=time.time_ns()):
                self._client = OpenAI(api_key=api_key, base_url=self.base_url)

        return self._client

    # ------------------------------------------------------------------
    # Abstract contract
    # ------------------------------------------------------------------

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the base system prompt for this session type."""
        ...

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def build_messages(
        self,
        user_prompt: str,
        *,
        history: Optional[List[Message]] = None,
        context_system: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> List[Message]:
        """
        Assemble the full message stack:

            [system] → [context system] → [history...] → [user]

        Parameters
        ----------
        user_prompt:
            The current user turn.
        history:
            Prior turns (assistant + user) for multi-turn conversations.
        context_system:
            Additional named system block injected after the base prompt,
            useful for RAG snippets, schema dumps, market context, etc.
        extra_instructions:
            Appended to the user message as a ``Constraints:`` block.
        """
        msgs: List[Message] = [Message(Role.SYSTEM, self.system_prompt().strip())]

        if context_system:
            msgs.append(Message(Role.SYSTEM, context_system.strip(), name="context"))

        if history:
            msgs.extend(history)

        body = user_prompt.strip()
        if extra_instructions:
            body = f"{body}\n\nConstraints:\n{extra_instructions.strip()}"

        msgs.append(Message(Role.USER, body))
        return msgs

    # ------------------------------------------------------------------
    # Low-level completion (with retry)
    # ------------------------------------------------------------------

    def _complete(
        self,
        messages: List[Message],
        *,
        temperature: float,
        max_output_tokens: int,
        stream: bool = False,
    ) -> ChatResponse:
        payload = [m.to_dict() for m in messages]
        attempt = 0
        t0 = time.monotonic()

        while True:
            attempt += 1
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=payload,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    stream=False,
                )
                latency_ms = (time.monotonic() - t0) * 1000
                raw = (resp.choices[0].message.content or "").strip()
                text, lang = _clean_output(raw)

                return ChatResponse(
                    raw=raw,
                    text=text,
                    lang=lang,
                    model=resp.model,
                    prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
                    latency_ms=latency_ms,
                )

            except Exception as exc:  # noqa: BLE001
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status in self.retry.retryable_status if status else _is_transient(exc)

                if retryable and attempt < self.retry.max_attempts:
                    delay = self.retry.base_delay_s * (self.retry.backoff_factor ** (attempt - 1))
                    log.warning("LLM attempt %d/%d failed (%s) – retrying in %.1fs",
                                attempt, self.retry.max_attempts, exc, delay)
                    time.sleep(delay)
                else:
                    raise

    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------

    def stream(
        self,
        user_prompt: str,
        *,
        history: Optional[List[Message]] = None,
        context_system: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ) -> Iterator[str]:
        """Yield raw text deltas from the model as they arrive."""
        messages = self.build_messages(
            user_prompt,
            history=history,
            context_system=context_system,
            extra_instructions=extra_instructions,
        )
        payload = [m.to_dict() for m in messages]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=temperature,
            max_tokens=max_output_tokens,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # High-level chat helpers
    # ------------------------------------------------------------------

    def chat(
        self,
        user_prompt: str,
        *,
        history: Optional[List[Message]] = None,
        context_system: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> ChatResponse:
        """Single-turn (or continued) chat returning a ``ChatResponse``."""
        messages = self.build_messages(
            user_prompt,
            history=history,
            context_system=context_system,
            extra_instructions=extra_instructions,
        )
        return self._complete(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def chat_text(self, user_prompt: str, **kwargs: Any) -> str:
        """Convenience wrapper returning plain cleaned text."""
        return self.chat(user_prompt, **kwargs).text

    def chat_json(self, user_prompt: str, **kwargs: Any) -> Any:
        """
        Ask the model to respond in JSON and parse the result.

        Injects a JSON constraint automatically.
        """
        extra = kwargs.pop("extra_instructions", None)
        json_constraint = "Respond ONLY with valid JSON. No prose, no code fences."
        if extra:
            json_constraint = f"{extra}\n{json_constraint}"

        resp = self.chat(user_prompt, extra_instructions=json_constraint, **kwargs)
        return resp.as_json()

    def chat_sql(
        self,
        user_prompt: str,
        *,
        schema: Optional[str] = None,
        dialect: str = "standard SQL",
        **kwargs: Any,
    ) -> str:
        """
        Ask the model to produce a SQL query.

        Parameters
        ----------
        schema:
            Table/column definitions injected as context (DDL, Arrow schema
            repr, etc.).
        dialect:
            SQL dialect hint, e.g. ``"DuckDB"``, ``"PostgreSQL"``.
        """
        instructions = (
            f"Respond ONLY with a {dialect} query. No explanation, no markdown."
        )
        extra = kwargs.pop("extra_instructions", None)
        if extra:
            instructions = f"{extra}\n{instructions}"

        resp = self.chat(
            user_prompt,
            context_system=schema,
            extra_instructions=instructions,
            **kwargs,
        )
        return resp.text

    def multi_turn(
        self,
        turns: List[str],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        **kwargs: Any,
    ) -> List[ChatResponse]:
        """
        Run a scripted multi-turn conversation, threading history automatically.

        Returns one ``ChatResponse`` per user turn.
        """
        history: List[Message] = []
        responses: List[ChatResponse] = []

        for user_turn in turns:
            resp = self.chat(
                user_turn,
                history=history,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                **kwargs,
            )
            history.append(Message(Role.USER, user_turn))
            history.append(Message(Role.ASSISTANT, resp.text))
            responses.append(resp)

        return responses

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars ≈ 1 token). Good enough for guards."""
        return max(1, len(text) // 4)

    def log_response(self, resp: ChatResponse, label: str = "") -> None:
        prefix = f"[{label}] " if label else ""
        log.debug(
            "%smodel=%s tokens=%d+%d latency=%.0fms lang=%s",
            prefix, resp.model,
            resp.prompt_tokens, resp.completion_tokens,
            resp.latency_ms, resp.lang or "text",
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _clean_output(raw: str) -> tuple[str, Optional[str]]:
    """
    Strip ``<think>`` tags (reasoning models) then code fences.
    Returns ``(cleaned_text, detected_language)``.
    """
    text = _THINK_TAG_RE.sub("", raw).strip()
    lang: Optional[str] = None

    m = _CODE_FENCE_RE.match(text)
    if m:
        lang = (m.group("lang") or "").lower() or None
        text = m.group("body").strip()

    return text, lang


def _is_transient(exc: Exception) -> bool:
    """Heuristic for transient network-level failures."""
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "ratelimit"))