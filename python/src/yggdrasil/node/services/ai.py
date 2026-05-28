"""AI assistant service.

Calls the Anthropic API when available; falls back to a rule-based mock
that uses the trading service's real indicators when ``anthropic`` is not
installed or the API key is missing. The mock path is intentionally
high-quality so the UI is useful out-of-the-box.
"""
from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import AsyncIterator

from ..config import Settings
from ..exceptions import NotFoundError
from ..ids import make_id
from ..schemas.ai import (
    AIChatMessage,
    AIChatRequest,
    AIChatResponse,
    AIConversation,
    AIConversationCreate,
    AIPortfolioAnalysis,
    AISuggestion,
    AIAnalyzeResponse,
)
from .trading import TradingService

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover — optional dep
    import anthropic  # type: ignore
    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore
    HAS_ANTHROPIC = False

_MAX_CONVERSATIONS = 50

_DEFAULT_SYSTEM = (
    "You are Yggdrasil's trading assistant — direct, analytical, and "
    "numerate. Cite indicators, flag risk, and never invent prices."
)


class AIService:
    def __init__(self, settings: Settings, *, trading_service: TradingService) -> None:
        self.settings = settings
        self._trading = trading_service
        self._conversations: "OrderedDict[int, AIConversation]" = OrderedDict()
        self._lock = Lock()
        self._client = self._init_client()

    @property
    def backend(self) -> str:
        return "anthropic" if self._client is not None else "mock"

    def _init_client(self):
        if not HAS_ANTHROPIC:
            return None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            LOGGER.info("Anthropic package installed but ANTHROPIC_API_KEY not set; using mock backend")
            return None
        try:
            return anthropic.Anthropic(api_key=api_key)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to init Anthropic client: %s; falling back to mock", exc)
            return None

    # -- chat -------------------------------------------------------------

    async def chat(self, req: AIChatRequest) -> AIChatResponse:
        system = self._compose_system(req.system, req.context)
        content = self._call_llm(req.messages, system, req.model, req.max_tokens)
        return AIChatResponse(content=content, model=req.model, backend=self.backend)

    async def stream_chat(self, req: AIChatRequest) -> AsyncIterator[str]:
        """Token-stream variant. Yields raw text chunks (no SSE framing)."""
        system = self._compose_system(req.system, req.context)
        if self._client is not None:
            try:
                with self._client.messages.stream(
                    model=req.model,
                    max_tokens=req.max_tokens,
                    system=system,
                    messages=[m.model_dump() for m in req.messages],
                ) as stream:
                    for text in stream.text_stream:
                        yield text
                return
            except Exception as exc:
                LOGGER.warning("Anthropic stream failed: %s; falling back to mock", exc)
        # Mock streaming: chunk the synthesised response so the UI sees
        # incremental tokens.
        full = self._mock_response(req.messages, system)
        for word in full.split(" "):
            yield word + " "

    def _call_llm(self, messages: list[AIChatMessage], system: str,
                  model: str, max_tokens: int) -> str:
        if self._client is not None:
            try:
                resp = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[m.model_dump() for m in messages],
                )
                # The SDK returns a list of content blocks; pull text.
                parts = []
                for block in resp.content:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(text)
                return "".join(parts) or self._mock_response(messages, system)
            except Exception as exc:
                LOGGER.warning("Anthropic call failed: %s; falling back to mock", exc)
        return self._mock_response(messages, system)

    def _compose_system(self, user_system: str | None, context: dict | None) -> str:
        base = user_system or _DEFAULT_SYSTEM
        if not context:
            return base
        bits = [base, "", "Current context:"]
        for k, v in context.items():
            bits.append(f"- {k}: {v}")
        return "\n".join(bits)

    def _mock_response(self, messages: list[AIChatMessage], system: str) -> str:
        """Heuristic responder. Looks for tickers in the last user message
        and pulls a real signal from the trading service so the answer
        contains live indicators."""
        if not messages:
            return "Ask me about a ticker (e.g. NVDA, BTC-USD) or your portfolio."
        last = messages[-1].content
        sym = _detect_symbol(last)
        if sym:
            sig = self._trading.get_signal(sym)
            ind = sig.indicators
            return (
                f"{sym} is at ${ind['price']:.2f}. "
                f"MA20={ind['ma20']:.2f}, MA50={ind['ma50']:.2f}, RSI14={ind['rsi14']:.1f}. "
                f"Signal: {sig.signal.replace('_', ' ').upper()} "
                f"({sig.confidence * 100:.0f}% confidence). {sig.reason}."
            )
        if "portfolio" in last.lower() or "position" in last.lower():
            p = self._trading.get_portfolio()
            line = (
                f"Portfolio: ${p.total_value:,.2f} total "
                f"(${p.cash:,.2f} cash, ${p.equity:,.2f} in {len(p.positions)} positions). "
                f"Unrealised P&L: ${p.total_pnl:+,.2f} ({p.total_pnl_pct:+.2f}%)."
            )
            if p.positions:
                best = max(p.positions, key=lambda x: x.pnl_pct)
                worst = min(p.positions, key=lambda x: x.pnl_pct)
                line += f" Best: {best.symbol} {best.pnl_pct:+.1f}%. Worst: {worst.symbol} {worst.pnl_pct:+.1f}%."
            return line
        return (
            "I am running on the local mock backend (Anthropic key not set). "
            "Mention a ticker symbol or ask about your portfolio for live analysis."
        )

    # -- analysis ---------------------------------------------------------

    async def analyze_symbol(self, symbol: str, timeframe: str = "1d",
                             include_portfolio: bool = False) -> AIAnalyzeResponse:
        sig = self._trading.get_signal(symbol)
        ind = sig.indicators
        price = ind["price"]
        ma20 = ind["ma20"]
        ma50 = ind["ma50"]

        support = round(min(ma20, ma50, price) * 0.98, 2)
        resistance = round(max(ma20, ma50, price) * 1.02, 2)
        if sig.signal in ("buy", "strong_buy"):
            target = round(price * 1.06, 2)
            stop_loss = round(price * 0.97, 2)
        elif sig.signal in ("sell", "strong_sell"):
            target = round(price * 0.94, 2)
            stop_loss = round(price * 1.03, 2)
        else:
            target = round(price * 1.02, 2)
            stop_loss = round(price * 0.98, 2)

        portfolio_note = ""
        if include_portfolio:
            p = self._trading.get_portfolio()
            held = next((pos for pos in p.positions if pos.symbol == symbol.upper()), None)
            if held:
                portfolio_note = (
                    f" You currently hold {held.qty:g} {held.symbol} at "
                    f"avg ${held.avg_price:.2f} (P&L {held.pnl_pct:+.2f}%)."
                )

        analysis = (
            f"{symbol.upper()} ({timeframe}): trading at ${price:.2f}. "
            f"MA20 ${ma20:.2f}, MA50 ${ma50:.2f}, RSI {ind['rsi14']:.1f}. "
            f"{sig.reason}. Signal: {sig.signal.replace('_', ' ').upper()}.{portfolio_note}"
        )

        # If a real model is wired, ask it to enrich the analysis.
        if self._client is not None:
            try:
                req = AIChatRequest(
                    messages=[AIChatMessage(role="user", content=(
                        f"Symbol {symbol} ({timeframe}). Indicators: price=${price:.2f}, "
                        f"MA20=${ma20:.2f}, MA50=${ma50:.2f}, RSI={ind['rsi14']:.1f}. "
                        f"Provide a concise 2-3 sentence analyst note."
                    ))],
                    system=_DEFAULT_SYSTEM,
                    max_tokens=300,
                )
                enriched = self._call_llm(req.messages, req.system or _DEFAULT_SYSTEM,
                                          req.model, req.max_tokens)
                if enriched.strip():
                    analysis = enriched
            except Exception as exc:
                LOGGER.warning("LLM enrich failed for analyze_symbol: %s", exc)

        return AIAnalyzeResponse(
            symbol=symbol.upper(),
            analysis=analysis,
            signal=sig.signal,
            confidence=sig.confidence,
            key_levels={
                "support": support,
                "resistance": resistance,
                "target": target,
                "stop_loss": stop_loss,
            },
            timestamp_ms=int(time.time() * 1000),
        )

    async def analyze_portfolio(self) -> AIPortfolioAnalysis:
        p = self._trading.get_portfolio()
        n = len(p.positions)
        if n == 0:
            return AIPortfolioAnalysis(
                summary="Portfolio is fully in cash — no risk exposure.",
                risk_score=0.0,
                diversification="empty",
                recommendations=["Add a position from the trading dashboard to start tracking signals."],
                timestamp_ms=int(time.time() * 1000),
            )

        # Concentration risk: largest position / total equity.
        equity = p.equity or 1.0
        weights = [(pos.symbol, (pos.current_price * pos.qty) / equity) for pos in p.positions]
        top_sym, top_w = max(weights, key=lambda x: x[1])
        risk = round(min(1.0, top_w + 0.1 * (1 - 1 / n)), 3)
        diversification = "high" if n >= 6 else "medium" if n >= 3 else "low"

        winners = [pos for pos in p.positions if pos.pnl > 0]
        losers = [pos for pos in p.positions if pos.pnl < 0]
        summary = (
            f"${p.total_value:,.0f} total ({p.cash:,.0f} cash, {n} positions, "
            f"{len(winners)} winners / {len(losers)} losers). "
            f"Unrealised P&L: {p.total_pnl:+,.2f} ({p.total_pnl_pct:+.2f}%). "
            f"Largest position: {top_sym} ({top_w * 100:.1f}% of equity)."
        )
        recs: list[str] = []
        if top_w > 0.4:
            recs.append(f"Concentration risk: trim {top_sym} below 40% of equity.")
        if diversification == "low":
            recs.append("Add 2-4 more uncorrelated tickers (e.g. mix tech, crypto, defensives).")
        if losers and any(pos.pnl_pct < -10 for pos in losers):
            bad = next(pos for pos in losers if pos.pnl_pct < -10)
            recs.append(f"Review {bad.symbol}: down {bad.pnl_pct:.1f}% — set a stop loss or close.")
        if not recs:
            recs.append("Portfolio looks balanced. Keep monitoring signals for entries.")

        return AIPortfolioAnalysis(
            summary=summary, risk_score=risk, diversification=diversification,
            recommendations=recs, timestamp_ms=int(time.time() * 1000),
        )

    async def suggestions(self) -> list[AISuggestion]:
        """Proactive ideas based on current signals."""
        sigs = self._trading.get_all_signals()
        out: list[AISuggestion] = []
        for s in sigs:
            if s.signal in ("strong_buy", "strong_sell") or s.confidence >= 0.7:
                action: str
                if s.signal in ("strong_buy", "buy"):
                    action = "buy"
                elif s.signal in ("strong_sell", "sell"):
                    action = "sell"
                else:
                    action = "watch"
                out.append(AISuggestion(
                    symbol=s.symbol,
                    title=f"{action.upper()} {s.symbol}",
                    detail=s.reason,
                    action=action,  # type: ignore[arg-type]
                    confidence=s.confidence,
                ))
        out.sort(key=lambda s: s.confidence, reverse=True)
        return out[:6]

    # -- conversations ----------------------------------------------------

    async def create_conversation(self, req: AIConversationCreate) -> AIConversation:
        cid = make_id(f"conv:{req.title}:{time.time_ns()}")
        now = int(time.time() * 1000)
        conv = AIConversation(
            id=cid, title=req.title, messages=[], created_at=now, updated_at=now,
        )
        with self._lock:
            self._conversations[cid] = conv
            while len(self._conversations) > _MAX_CONVERSATIONS:
                self._conversations.popitem(last=False)
        return conv

    async def list_conversations(self) -> list[AIConversation]:
        with self._lock:
            return list(reversed(self._conversations.values()))

    async def get_conversation(self, conv_id: int) -> AIConversation:
        with self._lock:
            conv = self._conversations.get(conv_id)
        if conv is None:
            raise NotFoundError(f"Conversation {conv_id!r} not found")
        return conv

    async def delete_conversation(self, conv_id: int) -> AIConversation:
        with self._lock:
            conv = self._conversations.pop(conv_id, None)
        if conv is None:
            raise NotFoundError(f"Conversation {conv_id!r} not found")
        return conv

    async def append_message(self, conv_id: int, msg: AIChatMessage) -> AIConversation:
        with self._lock:
            conv = self._conversations.get(conv_id)
            if conv is None:
                raise NotFoundError(f"Conversation {conv_id!r} not found")
            updated = conv.model_copy(update={
                "messages": [*conv.messages, msg],
                "updated_at": int(time.time() * 1000),
            })
            self._conversations[conv_id] = updated
            return updated


_TICKER_HINTS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "TSLA", "AMZN", "NVDA", "META",
    "BTC", "ETH", "SOL", "BTC-USD", "ETH-USD", "SOL-USD",
}


def _detect_symbol(text: str) -> str | None:
    """Best-effort ticker extraction. Returns the first match in upper text."""
    upper = text.upper()
    # Direct exact-match scan keeps the cost O(len(symbols)).
    for sym in _TICKER_HINTS:
        if sym in upper:
            return sym if "-" in sym or len(sym) > 3 else f"{sym}"
    return None
