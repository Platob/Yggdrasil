"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  aiChat, aiAnalyze, aiAnalyzePortfolio, aiSuggestions,
  aiListConversations, aiCreateConversation, aiDeleteConversation,
  getPortfolio,
  type AIChatMessage, type AIAnalyzeResponse, type AIConversation,
  type AISuggestion, type AIPortfolioAnalysis, type PortfolioSummary,
} from "@/lib/api";

const SYSTEM_PRESETS: { label: string; system: string }[] = [
  {
    label: "Trading Analyst",
    system: "You are a senior trading analyst. Cite indicators and risk-adjusted views. Be concise and numerate.",
  },
  {
    label: "Portfolio Advisor",
    system: "You are a long-only portfolio advisor. Focus on diversification, position sizing, and risk control.",
  },
  {
    label: "Market Reporter",
    system: "You are a market reporter. Summarise moves in plain language. Avoid recommendations.",
  },
];

const SIGNAL_COLOR: Record<string, string> = {
  strong_buy: "#16a34a",
  buy: "#4ade80",
  hold: "#9ca3af",
  sell: "#f97316",
  strong_sell: "#dc2626",
};

// ── Chat bubble ────────────────────────────────────────────────
function Bubble({ msg }: { msg: AIChatMessage }) {
  const user = msg.role === "user";
  return (
    <div className={`flex ${user ? "justify-end" : "justify-start"}`}>
      <div
        className="max-w-[80%] px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap"
        style={{
          background: user ? "linear-gradient(135deg, #f26b3a, #dc2626)" : "var(--card-elevated)",
          color: user ? "#fff" : "var(--foreground)",
          border: user ? "none" : "1px solid var(--border)",
        }}
      >
        {msg.content}
      </div>
    </div>
  );
}

// ── Conversations sidebar ──────────────────────────────────────
function ConversationsSidebar({
  conversations, currentId, onSelect, onNew, onDelete,
}: {
  conversations: AIConversation[];
  currentId: number | null;
  onSelect: (id: number) => void;
  onNew: () => void;
  onDelete: (id: number) => void;
}) {
  return (
    <div className="w-52 nordic-card flex flex-col">
      <div className="p-3 border-b border-border flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">Chats</h3>
        <button
          onClick={onNew}
          className="w-6 h-6 rounded-md bg-primary/10 text-primary hover:bg-primary/20 flex items-center justify-center"
          title="New conversation"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
          </svg>
        </button>
      </div>
      <div className="flex-1 overflow-auto p-2 space-y-1">
        {conversations.length === 0 && (
          <div className="text-xs text-muted text-center py-4">No saved chats</div>
        )}
        {conversations.map(c => (
          <div
            key={c.id}
            className="group flex items-center justify-between text-sm rounded-md px-2 py-1.5"
            style={{
              background: currentId === c.id ? "rgba(242,107,58,0.08)" : "transparent",
              color: currentId === c.id ? "var(--primary)" : "var(--foreground)",
              border: currentId === c.id ? "1px solid rgba(242,107,58,0.2)" : "1px solid transparent",
            }}
          >
            <button onClick={() => onSelect(c.id)} className="flex-1 text-left truncate">
              {c.title}
            </button>
            <button
              onClick={() => onDelete(c.id)}
              className="opacity-0 group-hover:opacity-100 text-muted hover:text-red-400 transition-opacity"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Symbol analysis card ───────────────────────────────────────
function AnalysisCard({ res }: { res: AIAnalyzeResponse | null }) {
  if (!res) return null;
  return (
    <div className="nordic-card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-mono font-semibold text-foreground">{res.symbol}</h3>
        <span
          className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded"
          style={{ background: (SIGNAL_COLOR[res.signal] ?? "#888") + "30", color: SIGNAL_COLOR[res.signal] ?? "#888" }}
        >
          {res.signal.replace("_", " ")} · {(res.confidence * 100).toFixed(0)}%
        </span>
      </div>
      <p className="text-sm text-foreground leading-relaxed">{res.analysis}</p>
      <div className="grid grid-cols-2 gap-2 text-xs">
        {Object.entries(res.key_levels).map(([k, v]) => (
          <div key={k} className="px-2 py-1.5 rounded bg-card-bg border border-border">
            <div className="text-[10px] uppercase tracking-wider text-muted">{k}</div>
            <div className="font-mono text-foreground">${v.toFixed(2)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────
export default function AIPage() {
  const [conversations, setConversations] = useState<AIConversation[]>([]);
  const [currentId, setCurrentId] = useState<number | null>(null);
  const [messages, setMessages] = useState<AIChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [systemIdx, setSystemIdx] = useState(0);
  const [includePortfolio, setIncludePortfolio] = useState(false);
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);

  const [symbol, setSymbol] = useState("NVDA");
  const [analysis, setAnalysis] = useState<AIAnalyzeResponse | null>(null);
  const [analysing, setAnalysing] = useState(false);

  const [suggestions, setSuggestions] = useState<AISuggestion[]>([]);
  const [portfolioAdvice, setPortfolioAdvice] = useState<AIPortfolioAnalysis | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshConversations = useCallback(async () => {
    try { setConversations(await aiListConversations()); } catch {}
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshConversations();
    aiSuggestions().then(setSuggestions).catch(() => {});
    getPortfolio().then(setPortfolio).catch(() => {});
  }, [refreshConversations]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || busy) return;
    const userMsg: AIChatMessage = { role: "user", content: input };
    const newMsgs = [...messages, userMsg];
    setMessages(newMsgs);
    setInput("");
    setBusy(true);
    try {
      const ctx: Record<string, unknown> = {};
      if (includePortfolio && portfolio) {
        ctx.portfolio = `$${portfolio.total_value.toFixed(2)} total, ${portfolio.positions.length} positions`;
        ctx.positions = portfolio.positions.map(p => `${p.symbol} qty=${p.qty} pnl=${p.pnl_pct.toFixed(1)}%`).join("; ");
      }
      const resp = await aiChat(newMsgs, {
        system: SYSTEM_PRESETS[systemIdx].system,
        context: Object.keys(ctx).length ? ctx : undefined,
      });
      setMessages([...newMsgs, { role: "assistant", content: resp.content }]);
    } catch (err) {
      setMessages([...newMsgs, { role: "assistant", content: `[error] ${err instanceof Error ? err.message : String(err)}` }]);
    } finally {
      setBusy(false);
    }
  }

  async function handleNew() {
    setMessages([]);
    setCurrentId(null);
    try {
      const conv = await aiCreateConversation();
      setCurrentId(conv.id);
      refreshConversations();
    } catch {}
  }

  async function handleSelect(id: number) {
    const conv = conversations.find(c => c.id === id);
    if (conv) {
      setCurrentId(id);
      setMessages(conv.messages);
    }
  }

  async function handleDelete(id: number) {
    await aiDeleteConversation(id);
    if (currentId === id) {
      setCurrentId(null);
      setMessages([]);
    }
    refreshConversations();
  }

  async function handleAnalyse() {
    setAnalysing(true);
    try {
      const r = await aiAnalyze(symbol, { include_portfolio: includePortfolio });
      setAnalysis(r);
    } finally {
      setAnalysing(false);
    }
  }

  async function handlePortfolioAdvice() {
    try { setPortfolioAdvice(await aiAnalyzePortfolio()); } catch {}
  }

  return (
    <div className="flex gap-4 animate-in" style={{ height: "calc(100vh - 3rem)" }}>
      <ConversationsSidebar
        conversations={conversations}
        currentId={currentId}
        onSelect={handleSelect}
        onNew={handleNew}
        onDelete={handleDelete}
      />

      {/* Chat column */}
      <div className="flex-1 flex flex-col nordic-card overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b border-border">
          <div>
            <h2 className="font-semibold text-foreground">AI Assistant</h2>
            <p className="text-xs text-muted">Ask about tickers, signals, or your portfolio</p>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={systemIdx}
              onChange={e => setSystemIdx(Number(e.target.value))}
              className="text-xs px-2 py-1 rounded outline-none"
              style={{ background: "var(--card-bg)", border: "1px solid var(--border)", color: "var(--foreground)" }}
            >
              {SYSTEM_PRESETS.map((p, i) => <option key={p.label} value={i}>{p.label}</option>)}
            </select>
            <label className="flex items-center gap-1.5 text-xs text-muted">
              <input
                type="checkbox"
                checked={includePortfolio}
                onChange={e => setIncludePortfolio(e.target.checked)}
              />
              Include portfolio
            </label>
          </div>
        </div>

        <div className="flex-1 overflow-auto p-5 space-y-3">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="mb-3 text-muted">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
              <p className="text-sm text-muted">Start by asking about a ticker or your portfolio</p>
              <p className="text-xs text-muted mt-1">e.g. &quot;How is NVDA looking?&quot;</p>
            </div>
          ) : (
            messages.map((m, i) => <Bubble key={i} msg={m} />)
          )}
          {busy && (
            <div className="flex justify-start">
              <div className="text-sm text-muted px-3 py-2 rounded-2xl bg-card-elevated border border-border animate-pulse">
                thinking…
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <form onSubmit={handleSend} className="p-3 border-t border-border flex gap-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about a ticker, signal, or your positions…"
            autoFocus
            className="flex-1 px-3 py-2 rounded-md text-sm outline-none"
            style={{ background: "var(--card-bg)", border: "1px solid var(--border)", color: "var(--foreground)" }}
          />
          <button
            type="submit"
            disabled={busy || !input.trim()}
            className="px-4 py-2 rounded-md text-sm font-semibold text-white disabled:opacity-50"
            style={{ background: "linear-gradient(135deg, #f26b3a, #dc2626)" }}
          >
            Send
          </button>
        </form>
      </div>

      {/* Side analysis column */}
      <div className="w-80 flex flex-col gap-3 overflow-y-auto">
        <div className="nordic-card p-4 space-y-2">
          <h3 className="text-xs font-medium text-muted uppercase tracking-wider">Symbol Analysis</h3>
          <div className="flex gap-2">
            <input
              value={symbol}
              onChange={e => setSymbol(e.target.value.toUpperCase())}
              className="flex-1 px-2 py-1.5 text-sm font-mono rounded outline-none"
              style={{ background: "var(--card-bg)", border: "1px solid var(--border)", color: "var(--foreground)" }}
            />
            <button
              onClick={handleAnalyse}
              disabled={analysing}
              className="px-3 py-1.5 text-sm rounded bg-primary/15 text-primary hover:bg-primary/25 disabled:opacity-50"
            >
              {analysing ? "…" : "Analyze"}
            </button>
          </div>
        </div>

        {analysis && <AnalysisCard res={analysis} />}

        <div className="nordic-card p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-medium text-muted uppercase tracking-wider">Portfolio Advice</h3>
            <button onClick={handlePortfolioAdvice} className="text-xs px-2 py-1 rounded bg-primary/15 text-primary hover:bg-primary/25">
              Refresh
            </button>
          </div>
          {portfolioAdvice ? (
            <div className="space-y-2">
              <p className="text-sm text-foreground">{portfolioAdvice.summary}</p>
              <div className="flex gap-3 text-xs text-muted">
                <span>Risk: <span className="text-foreground font-mono">{(portfolioAdvice.risk_score * 100).toFixed(0)}</span></span>
                <span>Diversification: <span className="text-foreground">{portfolioAdvice.diversification}</span></span>
              </div>
              <ul className="space-y-1.5 mt-2">
                {portfolioAdvice.recommendations.map((r, i) => (
                  <li key={i} className="text-xs text-foreground/80 flex gap-2">
                    <span className="text-primary">→</span> <span>{r}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="text-xs text-muted">Click refresh for portfolio analysis</p>
          )}
        </div>

        <div className="nordic-card p-4 space-y-2">
          <h3 className="text-xs font-medium text-muted uppercase tracking-wider">Suggestions</h3>
          {suggestions.length === 0 && <p className="text-xs text-muted">None right now</p>}
          {suggestions.map((s, i) => (
            <div key={i} className="text-xs px-2 py-1.5 rounded border border-border bg-card-bg">
              <div className="flex items-center justify-between">
                <span className="font-mono font-semibold">{s.symbol}</span>
                <span className="text-[10px] uppercase font-semibold" style={{
                  color: s.action === "buy" ? "#4ade80" : s.action === "sell" ? "#f87171" : "var(--muted)",
                }}>
                  {s.action}
                </span>
              </div>
              <p className="text-foreground/70 mt-0.5">{s.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
