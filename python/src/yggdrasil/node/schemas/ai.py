from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .common import StrictModel


class AIChatMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str


class AIChatRequest(StrictModel):
    messages: list[AIChatMessage]
    system: str | None = None
    model: str = "claude-sonnet-4-6"
    max_tokens: int = Field(default=2048, le=8192)
    stream: bool = False
    context: dict[str, Any] | None = None


class AIChatResponse(StrictModel):
    content: str
    model: str
    backend: Literal["anthropic", "mock"]


class AIAnalyzeRequest(StrictModel):
    symbol: str
    timeframe: Literal["1h", "4h", "1d", "1w"] = "1d"
    include_portfolio: bool = False


class AIAnalyzeResponse(StrictModel):
    symbol: str
    analysis: str
    signal: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    confidence: float
    key_levels: dict[str, float]
    timestamp_ms: int


class AIConversation(StrictModel):
    id: int
    title: str
    messages: list[AIChatMessage]
    created_at: int
    updated_at: int


class AIConversationCreate(StrictModel):
    title: str = "New Conversation"
    system: str | None = None


class AIConversationListResponse(StrictModel):
    conversations: list[AIConversation]


class AIConversationResponse(StrictModel):
    conversation: AIConversation


class AISuggestion(StrictModel):
    symbol: str
    title: str
    detail: str
    action: Literal["buy", "sell", "hold", "watch"]
    confidence: float


class AISuggestionsResponse(StrictModel):
    suggestions: list[AISuggestion]


class AIPortfolioAnalysis(StrictModel):
    summary: str
    risk_score: float
    diversification: str
    recommendations: list[str]
    timestamp_ms: int
