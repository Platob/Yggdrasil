// =============================================================================
// Types
// =============================================================================

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

export interface NodeInfo {
  node_id: string;
  host: string;
  port: number;
  version: string;
  uptime: number;
  channels: string[];
  functions: string[];
}

export interface AppConfig {
  version: string;
  features: {
    chat: boolean;
    execute: boolean;
    remoteCall: boolean;
  };
}

export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  nextjs: boolean;
  bot: boolean;
  timestamp: string;
}

export interface CachedDashboard {
  nodeInfo: NodeInfo | null;
  registry: Record<string, string>;
  channels: ChannelInfo[];
  cachedAt: string;
}

// =============================================================================
// Fetch with deduplication for GET requests
// =============================================================================

const inflight = new Map<string, Promise<unknown>>();

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

function deduped<T>(url: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? "GET";
  if (method !== "GET") return fetchJSON<T>(url, init);

  const existing = inflight.get(url);
  if (existing) return existing as Promise<T>;

  const promise = fetchJSON<T>(url, init);
  inflight.set(url, promise);
  promise.finally(() => inflight.delete(url));
  return promise;
}

// =============================================================================
// Bot API (proxied to FastAPI via /api/bot/*)
// =============================================================================

const BOT_BASE = "/api/bot";

export const bot = {
  executePython: (code: string): Promise<PythonResponse> =>
    fetchJSON(`${BOT_BASE}/python`, {
      method: "POST",
      body: JSON.stringify({ code }),
    }),

  executeCmd: (command: string[]): Promise<CmdResponse> =>
    fetchJSON(`${BOT_BASE}/cmd`, {
      method: "POST",
      body: JSON.stringify({ command }),
    }),

  sendMessage: (text: string, sender: string, channel: string): Promise<Message> =>
    fetchJSON(`${BOT_BASE}/messenger`, {
      method: "POST",
      body: JSON.stringify({ text, sender, channel }),
    }),

  getChannels: async (): Promise<ChannelInfo[]> => {
    const data = await deduped<{ channels: ChannelInfo[] }>(`${BOT_BASE}/messenger/channels`);
    return data.channels;
  },

  getMessages: async (channel: string, limit = 100): Promise<Message[]> => {
    const data = await fetchJSON<{ messages: Message[] }>(
      `${BOT_BASE}/messenger/channels/${channel}/messages?limit=${limit}`
    );
    return data.messages;
  },

  pollMessages: async (channel: string, afterId: string, timeout = 25): Promise<Message[]> => {
    const data = await fetchJSON<{ messages: Message[] }>(
      `${BOT_BASE}/messenger/channels/${channel}/poll?after_id=${afterId}&timeout=${timeout}`
    );
    return data.messages;
  },

  createChannel: async (name: string): Promise<ChannelInfo> => {
    const data = await fetchJSON<{ channel: ChannelInfo }>(
      `${BOT_BASE}/messenger/channels?name=${encodeURIComponent(name)}`,
      { method: "POST" }
    );
    return data.channel;
  },

  getNodeInfo: (): Promise<NodeInfo> => deduped(`${BOT_BASE}/hello`),

  getRegistry: (): Promise<Record<string, string>> => deduped(`${BOT_BASE}/call/registry`),

  callFunction: <T = unknown>(name: string, args: Record<string, unknown> = {}): Promise<T> =>
    fetchJSON(`${BOT_BASE}/call/${name}`, {
      method: "POST",
      body: JSON.stringify(args),
    }),
};

// =============================================================================
// Next.js API (local routes in /api/*)
// =============================================================================

const API_BASE = "/api";

export const api = {
  getHealth: (): Promise<HealthStatus> => deduped(`${API_BASE}/health`),
  getConfig: (): Promise<AppConfig> => deduped(`${API_BASE}/config`),
  getCachedDashboard: (): Promise<CachedDashboard> => deduped(`${API_BASE}/cache/dashboard`),
};

// =============================================================================
// Legacy exports
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
