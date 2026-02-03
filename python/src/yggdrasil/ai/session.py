# ai_session.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from yggdrasil.types.dummy_class import DummyModuleClass

try:
    from openai import OpenAI
except:
    OpenAI = DummyModuleClass

__all__ = ["AISession"]


@dataclass
class AISession(ABC):
    api_key: str
    base_url: str

    # Gemini default (via OpenAI-compatible gateway)
    model: str = "gemini-2.5-flash"

    client: OpenAI = field(init=False)

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @abstractmethod
    def system_prompt(self) -> str:
        raise NotImplementedError

    def build_messages(
        self,
        user_prompt: str,
        *,
        context_system: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        # Minimal message stack: system + optional context + user
        sys = self.system_prompt().strip()
        msgs: List[Dict[str, str]] = [{"role": "system", "content": sys}]

        if context_system:
            msgs.append({"role": "system", "name": "context", "content": context_system.strip()})

        msg = user_prompt.strip()
        if extra_instructions:
            msg = f"{msg}\n\nConstraints:\n{extra_instructions.strip()}"

        msgs.append({"role": "user", "content": msg})
        return msgs

    def chat(
        self,
        user_prompt: str,
        *,
        context_system: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 320,
        strip_code_fences: bool = True,
    ) -> str:
        messages = self.build_messages(
            user_prompt,
            context_system=context_system,
            extra_instructions=extra_instructions,
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

        out = (resp.choices[0].message.content or "").strip()

        if strip_code_fences and out.startswith("```"):
            out = out.split("```", 2)[1].strip()
            low = out.lower()
            if low.startswith("sql"):
                out = out[3:].strip()
            elif low.startswith("json"):
                out = out[4:].strip()

        return out
