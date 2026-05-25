/**
 * Frontend API client for Yggdrasil Dashboard
 * 
 * Two namespaces:
 * - `bot.*` - Calls proxied to FastAPI bot backend (real-time, execution)
 * - `api.*` - Calls to Next.js API routes (cached, aggregated, config)
 */

// =============================================================================
// Types
// =============================================================================

// -- Python execution --
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

// -- Shell commands --
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

// -- Messenger --
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

// -- Node info --
export interface NodeInfo {
  node_id: string;
  host: string;
  port: number;
  version: string;
  uptime: number;
  channels: string[];
  functions: string[];
}

// -- Next.js API types --
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
// Fetch Helpers
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
// Bot API (proxied to FastAPI via /api/bot/*)
// =============================================================================

const BOT_BASE = "/api/bot";

export const bot = {
  // Python execution
  executePython: (code: string): Promise<PythonResponse> =>
    fetchJSON(`${BOT_BASE}/python`, {
      method: "POST",
      body: JSON.stringify({ code }),
    }),

  // Shell commands
  executeCmd: (command: string[]): Promise<CmdResponse> =>
    fetchJSON(`${BOT_BASE}/cmd`, {
      method: "POST",
      body: JSON.stringify({ command }),
    }),

  // Messaging
  sendMessage: (text: string, sender: string, channel: string): Promise<Message> =>
    fetchJSON(`${BOT_BASE}/messenger`, {
      method: "POST",
      body: JSON.stringify({ text, sender, channel }),
    }),

  getChannels: async (): Promise<ChannelInfo[]> => {
    const data = await fetchJSON<{ channels: ChannelInfo[] }>(`${BOT_BASE}/messenger/channels`);
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

  // Node info (direct)
  getNodeInfo: (): Promise<NodeInfo> => fetchJSON(`${BOT_BASE}/hello`),

  // Registry
  getRegistry: (): Promise<Record<string, string>> => fetchJSON(`${BOT_BASE}/call/registry`),

  // Remote function call
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
  // Health check
  getHealth: (): Promise<HealthStatus> => fetchJSON(`${API_BASE}/health`),

  // App config
  getConfig: (): Promise<AppConfig> => fetchJSON(`${API_BASE}/config`),

  // Cached dashboard data (aggregated from bot)
  getCachedDashboard: (): Promise<CachedDashboard> => fetchJSON(`${API_BASE}/cache/dashboard`),
};

// =============================================================================
// Legacy exports (for backward compatibility during migration)
// =============================================================================

/** @deprecated Use bot.executePython instead */
export const executePython = bot.executePython;

/** @deprecated Use bot.executeCmd instead */
export const executeCmd = bot.executeCmd;

/** @deprecated Use bot.sendMessage instead */
export const sendMessage = bot.sendMessage;

/** @deprecated Use bot.getChannels instead */
export const getChannels = bot.getChannels;

/** @deprecated Use bot.getMessages instead */
export const getMessages = bot.getMessages;

/** @deprecated Use bot.pollMessages instead */
export const pollMessages = bot.pollMessages;

/** @deprecated Use bot.createChannel instead */
export const createChannel = bot.createChannel;

/** @deprecated Use bot.getNodeInfo instead */
export const getNodeInfo = bot.getNodeInfo;

/** @deprecated Use bot.getRegistry instead */
export const getRegistry = bot.getRegistry;
