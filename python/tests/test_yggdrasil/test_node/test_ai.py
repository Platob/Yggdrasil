"""Smoke tests for the AI service (mock backend)."""
from __future__ import annotations

import pytest

from yggdrasil.node.config import Settings
from yggdrasil.node.exceptions import NotFoundError
from yggdrasil.node.schemas.ai import (
    AIChatMessage,
    AIChatRequest,
    AIConversationCreate,
)
from yggdrasil.node.services.ai import AIService
from yggdrasil.node.services.trading import TradingService


@pytest.fixture
def settings() -> Settings:
    return Settings(allow_remote=True)


@pytest.fixture
def trading(settings) -> TradingService:
    return TradingService(settings)


@pytest.fixture
def ai(settings, trading) -> AIService:
    svc = AIService(settings, trading_service=trading)
    # Force mock backend even if anthropic happens to be installed locally.
    svc._client = None  # type: ignore[attr-defined]
    return svc


@pytest.mark.asyncio
async def test_chat_mock_responds_with_indicators_for_ticker(ai):
    req = AIChatRequest(messages=[AIChatMessage(role="user", content="How is NVDA?")])
    resp = await ai.chat(req)
    assert resp.backend == "mock"
    assert "NVDA" in resp.content
    assert "RSI" in resp.content


@pytest.mark.asyncio
async def test_chat_portfolio_intent(ai):
    req = AIChatRequest(messages=[AIChatMessage(role="user", content="show me my portfolio")])
    resp = await ai.chat(req)
    assert "Portfolio" in resp.content


@pytest.mark.asyncio
async def test_analyze_symbol_returns_levels(ai):
    res = await ai.analyze_symbol("AAPL")
    assert res.symbol == "AAPL"
    assert {"support", "resistance", "target", "stop_loss"}.issubset(res.key_levels.keys())


@pytest.mark.asyncio
async def test_analyze_portfolio_empty(ai):
    res = await ai.analyze_portfolio()
    assert res.diversification == "empty"
    assert res.risk_score == 0.0


@pytest.mark.asyncio
async def test_suggestions_returns_list(ai):
    out = await ai.suggestions()
    assert isinstance(out, list)
    # Each suggestion must have a valid action.
    for s in out:
        assert s.action in {"buy", "sell", "hold", "watch"}


@pytest.mark.asyncio
async def test_conversation_crud(ai):
    conv = await ai.create_conversation(AIConversationCreate(title="test"))
    assert conv.title == "test"
    got = await ai.get_conversation(conv.id)
    assert got.id == conv.id
    appended = await ai.append_message(conv.id, AIChatMessage(role="user", content="hi"))
    assert len(appended.messages) == 1
    deleted = await ai.delete_conversation(conv.id)
    assert deleted.id == conv.id
    with pytest.raises(NotFoundError):
        await ai.get_conversation(conv.id)


@pytest.mark.asyncio
async def test_stream_chat_yields_chunks(ai):
    req = AIChatRequest(messages=[AIChatMessage(role="user", content="What about BTC?")])
    chunks = []
    async for c in ai.stream_chat(req):
        chunks.append(c)
        if len(chunks) > 5:
            break
    assert len(chunks) > 0
