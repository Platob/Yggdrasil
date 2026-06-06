"""Anthropic (Claude) -backed :class:`TokenEngine`.

Two ways to authenticate, resolved in this order:

1. **API key** — ``ANTHROPIC_API_KEY`` (or ``api_key=``). Billed per token.
2. **OAuth / subscription token** — the credential Claude Code itself logs in
   with, so Loki can reason **without a separate billed API key** when run on
   a machine where you're signed into Claude Code. Taken from
   ``ANTHROPIC_AUTH_TOKEN`` / ``CLAUDE_CODE_OAUTH_TOKEN``, or the Claude Code
   credentials file (``~/.claude/.credentials.json`` → ``claudeAiOauth``).
   The SDK sends it as a bearer token; the request carries the OAuth beta
   header and the Claude Code system identity the grant is scoped to.

An API key wins when both are present. ``available()`` is true when *either*
is resolvable, so ``ygg loki`` lights up Claude on a logged-in Claude Code
box with no extra configuration.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any, ClassVar, Optional

from ..engine import DEFAULT_MAX_TOKENS, Completion, TokenEngine

__all__ = ["ClaudeEngine"]

#: Beta header that opts a request into OAuth (subscription) authentication.
OAUTH_BETA = "oauth-2025-04-20"
#: The system identity a Claude Code OAuth grant is scoped to — required as the
#: first system block when authenticating with a subscription token.
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


def _oauth_token_from_file() -> Optional[str]:
    """The Claude Code OAuth access token from its credentials file, if any."""
    path = pathlib.Path.home() / ".claude" / ".credentials.json"
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
    token = oauth.get("accessToken") if isinstance(oauth, dict) else None
    return token or None


class ClaudeEngine(TokenEngine):
    """Reason via the Anthropic Messages API (``anthropic`` SDK)."""

    name = "claude"
    #: The most capable current Claude model (see the claude-api reference).
    default_model: ClassVar[str] = "claude-opus-4-8"
    #: Adaptive tiers, fast → capable: Haiku for light turns, Opus for the
    #: hard ones (Sonnet sits between if a caller pins ``model=``).
    MODELS: ClassVar[dict[str, str]] = {
        "fast": "claude-haiku-4-5",
        "deep": "claude-opus-4-8",
    }

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        tier: Optional[str] = None,
        api_key: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, tier=tier)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        # OAuth / subscription token: explicit arg → env → Claude Code creds file.
        self.auth_token = (
            auth_token
            or os.getenv("ANTHROPIC_AUTH_TOKEN")
            or os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
            or _oauth_token_from_file()
        )

    @property
    def uses_oauth(self) -> bool:
        """True when this engine will authenticate with a subscription token."""
        return not self.api_key and bool(self.auth_token)

    def available(self) -> bool:
        return bool(self.api_key or self.auth_token)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tier: Optional[str] = None,
        **options: Any,
    ) -> Completion:
        client, kwargs, model = self._request(messages, system, max_tokens, tier, options)
        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "input_tokens", None) if usage else None
        out_tok = getattr(usage, "output_tokens", None) if usage else None
        self._record(model, input_tokens=in_tok, output_tokens=out_tok,
                     messages=messages, system=system, text=text)
        return Completion(
            text=text,
            model=getattr(resp, "model", model),
            usage={"input_tokens": in_tok, "output_tokens": out_tok} if usage else {},
            raw=resp,
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tier: Optional[str] = None,
        **options: Any,
    ):
        client, kwargs, model = self._request(messages, system, max_tokens, tier, options)
        text_parts: list[str] = []
        with client.messages.stream(**kwargs) as stream:
            for chunk in stream.text_stream:
                text_parts.append(chunk)
                yield chunk
            final = stream.get_final_message()
        usage = getattr(final, "usage", None)
        self._record(
            model,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            messages=messages, system=system, text="".join(text_parts),
        )

    def _request(self, messages, system, max_tokens, tier, options):
        """Build the (client, kwargs, model) shared by complete + stream."""
        from ..runtime import load

        anthropic = load("anthropic")

        # Adaptive default: pick fast/deep from the request unless pinned.
        model = self.resolve_model(messages=messages, system=system, tier=tier)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": list(messages),
        }
        if self.api_key:
            client = anthropic.Anthropic(api_key=self.api_key)
            if system:
                kwargs["system"] = system
        else:
            # Subscription auth: bearer token + the OAuth beta header. The
            # grant is scoped to Claude Code, so its identity must lead the
            # system prompt; the caller's own system instruction follows it.
            client = anthropic.Anthropic(
                auth_token=self.auth_token,
                default_headers={"anthropic-beta": OAUTH_BETA},
            )
            system_blocks = [{"type": "text", "text": CLAUDE_CODE_IDENTITY}]
            if system:
                system_blocks.append({"type": "text", "text": system})
            kwargs["system"] = system_blocks
        kwargs.update(options)
        return client, kwargs, model
