import type {
  BacktestResult, FinanceResult, FsListResult, IndicatorsResult,
  Message, NodeStats, SagaCatalog, ScanEntry, SqlResult,
} from "./types";

const BASE = "";  // uses Next.js rewrite proxy

async function get<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  ping: () => get<{ status: string }>("/api/ping"),
  stats: () => get<NodeStats>("/api/v2/stats"),
  health: () => get<{ status: string }>("/api/v2/health"),
  backend: () => get<{ name: string; status: string }[]>("/api/v2/backend"),

  // Files
  ls: (path = "", offset = 0, limit = 100) =>
    get<FsListResult>(`/api/v2/fs/ls?path=${encodeURIComponent(path)}&offset=${offset}&limit=${limit}`),

  // Trading
  indicators: (path: string, column: string, max_points = 2000) =>
    post<IndicatorsResult>("/api/v2/trading/indicators", { path, column, max_points }),

  signals: (path: string, column: string) =>
    post<unknown>("/api/v2/trading/signals", { path, column }),

  backtest: (
    path: string,
    column: string,
    strategy = "ema_cross",
    initial_cash = 10000,
    stop_loss_pct?: number,
    take_profit_pct?: number,
    position_sizing = "full",
  ) =>
    post<BacktestResult>("/api/v2/trading/backtest", {
      path, column, strategy, initial_cash,
      stop_loss_pct: stop_loss_pct ?? null,
      take_profit_pct: take_profit_pct ?? null,
      position_sizing,
    }),

  scan: (paths: string[], column = "close") =>
    post<{ results: ScanEntry[] }>("/api/v2/trading/scan", { paths, column }),

  strategies: () =>
    get<{ strategies: { id: string; name: string; description: string }[] }>("/api/v2/trading/strategies"),

  // Analysis
  finance: (path: string, column: string) =>
    get<FinanceResult>(`/api/v2/analysis/finance?path=${encodeURIComponent(path)}&column=${encodeURIComponent(column)}`),

  // Saga
  catalogs: () => get<{ catalogs: SagaCatalog[] }>("/api/v2/saga/catalog"),
  sql: (sql: string) => post<SqlResult>("/api/v2/saga/sql", { sql }),

  // Messenger
  sendMessage: (text: string, sender = "user", channel = "general") =>
    post<Message>("/api/messenger", { text, sender, channel }),
  messages: (channel = "general", limit = 50) =>
    get<{ messages: Message[] }>(`/api/messenger/channels/${channel}/messages?limit=${limit}`),
};
