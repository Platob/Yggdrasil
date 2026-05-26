/**
 * Yggdrasil API client
 *
 * - `bot.*` — proxied to FastAPI at /api/bot/* → :8100/api/*
 * - `api.*` — Next.js API routes
 */

// =============================================================================
// Core fetch helper
// =============================================================================

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

// =============================================================================
// Types
// =============================================================================

export interface NodeInfo {
  node_id: string;
  host: string;
  port: number;
  version: string;
  uptime: number;
  channels?: string[];
  functions?: string[];
  lat?: number | null;
  lon?: number | null;
}

export interface PeersResponse {
  node_id: string;
  peers: NodeInfo[];
}

export interface PythonResponse {
  id: string;
  node_id: string;
  returncode: number | null;
  stdout: string | null;
  stderr: string | null;
  result: unknown;
  duration: number | null;
  status: string;
}

export interface CmdResponse {
  id: string;
  node_id: string;
  command: string[];
  returncode: number | null;
  stdout: string | null;
  stderr: string | null;
  duration: number | null;
  status: string;
}

export interface Message {
  id: string;
  sender: string;
  text: string;
  channel: string;
  timestamp: string;
  node_id: string;
}

export interface ChannelInfo {
  name: string;
  created_at: string;
  last_active: string;
  message_count: number;
  members: string[];
}

export interface FunctionEntry {
  id: number;
  name: string;
  language: string;
  code: string;
  description: string;
  python_version: string | null;
  dependencies: string[];
  environment_id: number | null;
  created_at: string;
  updated_at: string;
  run_count: number;
}

export interface EnvironmentEntry {
  id: number;
  name: string;
  python_version: string;
  dependencies: string[];
  path: string;
  status: string;
  created_at: string;
  updated_at: string;
  error: string | null;
}

export interface RunEntry {
  id: number;
  function_id: number;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  duration: number | null;
  returncode: number | null;
  stdout: string | null;
  stderr: string | null;
  result: unknown;
  node_id: string;
}

export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  nextjs: boolean;
  bot: boolean;
  timestamp: string;
}

export interface AppConfig {
  version: string;
  features: Record<string, boolean>;
}

// -- Trading types --
export interface PriceQuote {
  symbol: string;
  price: number;
  currency: string;
  source: string;
  timestamp: string;
  stale: boolean;
}

export interface PricesResponse {
  prices: Record<string, PriceQuote>;
  timestamp: string;
}

export interface PriceHistoryResponse {
  symbol: string;
  prices: number[];
  timestamps: string[];
}

export interface PortfolioPositionEntry {
  symbol: string;
  quantity: number;
  avg_cost: number;
  currency: string;
  current_price: number | null;
  pnl: number | null;
  pnl_pct: number | null;
}

export interface PortfolioResponse {
  positions: PortfolioPositionEntry[];
  total_value: number;
  total_pnl: number;
  currency: string;
  timestamp: string;
}

export interface TechnicalIndicators {
  symbol: string;
  sma_20: number | null;
  sma_50: number | null;
  ema_20: number | null;
  rsi_14: number | null;
  price: number | null;
  timestamp: string;
}

export interface PriceAlert {
  id: string;
  symbol: string;
  condition: "above" | "below";
  price: number;
  created_at: string;
  triggered_at: string | null;
}

export interface AlertsResponse {
  alerts: PriceAlert[];
}

// =============================================================================
// Bot API (proxied to FastAPI via /api/bot/*)
// =============================================================================

const BOT = "/api/bot";

export const bot = {
  // Node discovery
  getNodeInfo: (): Promise<NodeInfo> => fetchJSON(`${BOT}/hello`),
  getPeers: (): Promise<PeersResponse> => fetchJSON(`${BOT}/hello/peers`),

  // Python execution
  executePython: (code: string): Promise<PythonResponse> =>
    fetchJSON(`${BOT}/python`, { method: "POST", body: JSON.stringify({ code }) }),

  // Shell commands
  executeCmd: (command: string[]): Promise<CmdResponse> =>
    fetchJSON(`${BOT}/cmd`, { method: "POST", body: JSON.stringify({ command }) }),

  // Messaging
  sendMessage: (text: string, sender: string, channel: string): Promise<Message> =>
    fetchJSON(`${BOT}/messenger`, {
      method: "POST",
      body: JSON.stringify({ text, sender, channel }),
    }),

  getChannels: async (): Promise<ChannelInfo[]> => {
    const d = await fetchJSON<{ channels: ChannelInfo[] }>(`${BOT}/messenger/channels`);
    return d.channels;
  },

  getMessages: async (channel: string, limit = 100): Promise<Message[]> => {
    const d = await fetchJSON<{ messages: Message[] }>(
      `${BOT}/messenger/channels/${encodeURIComponent(channel)}/messages?limit=${limit}`
    );
    return d.messages;
  },

  pollMessages: async (channel: string, afterId: string, timeout = 25): Promise<Message[]> => {
    const d = await fetchJSON<{ messages: Message[] }>(
      `${BOT}/messenger/channels/${encodeURIComponent(channel)}/poll?after_id=${afterId}&timeout=${timeout}`
    );
    return d.messages;
  },

  createChannel: async (name: string): Promise<ChannelInfo> => {
    const d = await fetchJSON<{ channel: ChannelInfo }>(
      `${BOT}/messenger/channels?name=${encodeURIComponent(name)}`,
      { method: "POST" }
    );
    return d.channel;
  },

  // Registry
  getRegistry: (): Promise<Record<string, string>> => fetchJSON(`${BOT}/call/registry`),

  // Functions
  listFunctions: () => fetchJSON<{ functions: FunctionEntry[] }>(`${BOT}/function`),
  createFunction: (data: { name: string; code: string; language?: string; description?: string }) =>
    fetchJSON<{ function: FunctionEntry }>(`${BOT}/function`, { method: "POST", body: JSON.stringify(data) }),
  runFunction: (id: number, args?: Record<string, unknown>) =>
    fetchJSON<{ run: RunEntry }>(`${BOT}/function/${id}/run`, { method: "POST", body: JSON.stringify(args ?? {}) }),

  // Environments
  listEnvironments: () => fetchJSON<{ environments: EnvironmentEntry[] }>(`${BOT}/environment`),

  // Runs
  listRuns: () => fetchJSON<{ runs: RunEntry[] }>(`${BOT}/run`),
  getRun: (id: number) => fetchJSON<{ run: RunEntry }>(`${BOT}/run/${id}`),

  // Trading
  getPrices: (symbols?: string[]): Promise<PricesResponse> => {
    const qs = symbols?.length ? `?symbols=${symbols.join(",")}` : "";
    return fetchJSON(`${BOT}/trading/prices${qs}`);
  },

  getPriceHistory: (symbol: string): Promise<PriceHistoryResponse> =>
    fetchJSON(`${BOT}/trading/history/${encodeURIComponent(symbol)}`),

  getPortfolio: (): Promise<PortfolioResponse> => fetchJSON(`${BOT}/trading/portfolio`),

  upsertPosition: (position: { symbol: string; quantity: number; avg_cost: number; currency?: string }): Promise<PortfolioResponse> =>
    fetchJSON(`${BOT}/trading/portfolio/position`, { method: "POST", body: JSON.stringify(position) }),

  removePosition: (symbol: string): Promise<PortfolioResponse> =>
    fetchJSON(`${BOT}/trading/portfolio/position/${encodeURIComponent(symbol)}`, { method: "DELETE" }),

  getIndicators: (symbol: string): Promise<TechnicalIndicators> =>
    fetchJSON(`${BOT}/trading/indicators/${encodeURIComponent(symbol)}`),

  getAlerts: (): Promise<AlertsResponse> => fetchJSON(`${BOT}/trading/alerts`),

  createAlert: (data: { symbol: string; condition: "above" | "below"; price: number }): Promise<PriceAlert> =>
    fetchJSON(`${BOT}/trading/alerts`, { method: "POST", body: JSON.stringify(data) }),

  removeAlert: (alertId: string): Promise<void> =>
    fetchJSON(`${BOT}/trading/alerts/${alertId}`, { method: "DELETE" }),
};

// =============================================================================
// Next.js API routes
// =============================================================================

export const api = {
  getHealth: (): Promise<HealthStatus> => fetchJSON("/api/health"),
  getConfig: (): Promise<AppConfig> => fetchJSON("/api/config"),
};

// =============================================================================
// Legacy flat exports (execute page, chat page import these directly)
// =============================================================================
export const executePython = bot.executePython;
export const executeCmd = bot.executeCmd;
export const sendMessage = bot.sendMessage;
export const getChannels = bot.getChannels;
export const getMessages = bot.getMessages;
export const pollMessages = bot.pollMessages;
export const createChannel = bot.createChannel;
export const getNodeInfo = bot.getNodeInfo;
export const getRegistry = bot.getRegistry;
