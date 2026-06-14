const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

export interface FxRate {
  pair: string;
  rate: number | null;
  date: string;
  base: string;
  error?: string;
}

export interface FxResponse {
  rates: FxRate[];
}

export interface EnergyPoint {
  ts: string;
  value: number;
}

export interface EnergyResponse {
  zone: string;
  series: string;
  data: Record<string, unknown>[];
  error?: string;
}

export interface BackendInfo {
  name: string;
  version: string;
  status: string;
}

export interface StatsResponse {
  uptime: number;
  platform: string;
  python: string;
}

export interface HealthResponse {
  status: string;
  ts: number;
}

export interface ChatRequest {
  message: string;
  history?: { role: "user" | "assistant"; content: string }[];
}

export interface ChatResponse {
  reply: string;
  engine: string | null;
}

export interface AuditEntry {
  ts: number;
  action: string;
  resource_type: string;
  resource_id: number;
  detail: string;
}

export interface MonitorSnapshot {
  ts: number;
  cpu_percent: number;
  mem_percent: number;
  disk_percent: number;
}

export interface CryptoPrice {
  id: string;
  price: number | null;
  change_24h: number | null;
  vs: string;
}

export interface CryptoResponse {
  prices: CryptoPrice[];
  error?: string;
}

export interface MarketSummary {
  fx: FxRate[];
  crypto: { id: string; price: number | null; change_24h: number | null; vs?: string }[];
  node: { cpu_percent: number; mem_percent: number };
}

export const api = {
  ping: () => apiFetch<HealthResponse>("/api/ping"),
  health: () => apiFetch<HealthResponse>("/api/v2/health"),
  stats: () => apiFetch<StatsResponse>("/api/v2/stats"),
  backend: () => apiFetch<BackendInfo>("/api/v2/backend"),

  fx: (pairs = "EUR/USD,EUR/GBP,EUR/JPY,USD/JPY,GBP/USD", start?: string, end?: string) => {
    const q = new URLSearchParams({ pairs });
    if (start) q.set("start", start);
    if (end) q.set("end", end);
    return apiFetch<FxResponse>(`/api/v2/market/fx?${q}`);
  },

  energy: (zone = "DE_LU", series = "day_ahead_prices", start?: string, end?: string) => {
    const q = new URLSearchParams({ zone, series });
    if (start) q.set("start", start);
    if (end) q.set("end", end);
    return apiFetch<EnergyResponse>(`/api/v2/market/energy?${q}`);
  },

  crypto: (coins = "bitcoin,ethereum,solana,cardano") =>
    apiFetch<CryptoResponse>(`/api/v2/market/crypto?coins=${encodeURIComponent(coins)}`),

  marketSummary: () => apiFetch<MarketSummary>("/api/v2/market/summary"),

  chat: (req: ChatRequest) =>
    apiFetch<ChatResponse>("/api/v2/loki/chat", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  audit: (limit = 50) =>
    apiFetch<{ entries: AuditEntry[] }>(`/api/v2/audit?limit=${limit}`),

  fs: {
    ls: (path = "", offset = 0, limit = 100) =>
      apiFetch<{ entries: FsEntry[]; total: number }>(
        `/api/fs/ls?path=${encodeURIComponent(path)}&offset=${offset}&limit=${limit}`
      ),
    read: (path: string) =>
      apiFetch<{ content: string; truncated: boolean }>(
        `/api/fs/read?path=${encodeURIComponent(path)}`
      ),
  },

  analysis: {
    series: (path: string, column: string, points = 800) =>
      apiFetch<{ x: number[]; y: number[] }>("/api/v2/analysis/series", {
        method: "POST",
        body: JSON.stringify({ path, column, points }),
      }),
    ohlc: (path: string, column: string, buckets = 120) =>
      apiFetch<{
        bars: number;
        open: number[];
        high: number[];
        low: number[];
        close: number[];
        timestamps: number[];
      }>("/api/v2/analysis/ohlc", {
        method: "POST",
        body: JSON.stringify({ path, column, buckets }),
      }),
  },
};

export interface FsEntry {
  name: string;
  size: number;
  is_dir: boolean;
  mtime: string;
}
