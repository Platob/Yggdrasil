import type { AIAnalysis, OHLCV, PnL, Portfolio, Quote, Signal, Ticker, Trade } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "/api";

async function get<T>(path: string, params?: Record<string, string | string[]>): Promise<T> {
  const url = new URL(`${BASE}${path}`, typeof window !== "undefined" ? window.location.origin : "http://localhost:3000");
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (Array.isArray(v)) v.forEach((vi) => url.searchParams.append(k, vi));
      else url.searchParams.set(k, v);
    }
  }
  const res = await fetch(url.toString(), { next: { revalidate: 15 } });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `API ${path} → ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => get<{ status: string; ts: string }>("/health"),

  // Market
  quote: (symbol: string) => get<Quote>(`/market/quote/${symbol}`),
  quotes: (symbols: string[]) => get<Quote[]>("/market/quotes", { symbols }),
  ohlcv: (symbol: string, period = "1mo", interval = "1d") =>
    get<OHLCV[]>(`/market/ohlcv/${symbol}`, { period, interval }),
  search: (q: string) => get<Ticker[]>("/market/search", { q }),
  sectors: () => get<Record<string, string[]>>("/market/sectors"),

  // Portfolio
  portfolio: (pid = 1) => get<Portfolio>(`/portfolio/${pid}`),
  pnl: (pid = 1) => get<PnL>(`/portfolio/${pid}/pnl`),
  trade: (pid = 1, body: { symbol: string; side: "buy" | "sell"; quantity: number; price?: number }) =>
    post<Trade>(`/portfolio/${pid}/trade`, body),
  trades: (pid = 1, limit = 20) => get<Trade[]>(`/portfolio/${pid}/trades`, { limit: String(limit) }),

  // Signals
  signal: (symbol: string, period = "3mo") => get<Signal>(`/signals/${symbol}`, { period }),
  scanSignals: (symbols: string[]) => get<Signal[]>("/signals/batch/scan", { symbols }),

  // AI
  analyze: (symbol: string, context?: string) =>
    post<AIAnalysis>("/ai/analyze", { symbol, context }),
  aiScan: (symbols: string[]) => get<AIAnalysis[]>("/ai/scan", { symbols }),
};
