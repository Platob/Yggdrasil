/**
 * Frontend API client for Yggdrasil Dashboard
 * 
 * Two namespaces:
 * - `node.*` - Calls proxied to FastAPI bot backend (real-time, execution)
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
  lat: number | null;
  lon: number | null;
}

// -- Peers --
export interface PeersResponse {
  node_id: string;
  peers: NodeInfo[];
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

// -- Functions --
export interface FunctionEntry {
  id: string;
  name: string;
  language: string;
  code: string;
  description: string;
  python_version: string | null;
  dependencies: string[];
  environment_id: string | null;
  creator: string;
  created_at: string;
  updated_at: string;
  run_count: number;
}

// -- Environments --
export interface EnvironmentEntry {
  id: string;
  name: string;
  python_version: string;
  dependencies: string[];
  path: string;
  status: string;
  created_at: string;
  updated_at: string;
  error: string | null;
}

// -- Runs --
export interface RunEntry {
  id: string;
  function_id: string;
  environment_id: string | null;
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
// Bot API (proxied to FastAPI via /api/node/*)
// =============================================================================

const NODE_BASE = "/api/node";

export const node = {
  // Python execution
  executePython: (code: string): Promise<PythonResponse> =>
    fetchJSON(`${NODE_BASE}/python`, {
      method: "POST",
      body: JSON.stringify({ code }),
    }),

  // Shell commands
  executeCmd: (command: string[]): Promise<CmdResponse> =>
    fetchJSON(`${NODE_BASE}/cmd`, {
      method: "POST",
      body: JSON.stringify({ command }),
    }),

  // Messaging
  sendMessage: (text: string, sender: string, channel: string): Promise<Message> =>
    fetchJSON(`${NODE_BASE}/messenger`, {
      method: "POST",
      body: JSON.stringify({ text, sender, channel }),
    }),

  getChannels: async (): Promise<ChannelInfo[]> => {
    const data = await fetchJSON<{ channels: ChannelInfo[] }>(`${NODE_BASE}/messenger/channels`);
    return data.channels;
  },

  getMessages: async (channel: string, limit = 100): Promise<Message[]> => {
    const data = await fetchJSON<{ messages: Message[] }>(
      `${NODE_BASE}/messenger/channels/${channel}/messages?limit=${limit}`
    );
    return data.messages;
  },

  pollMessages: async (channel: string, afterId: string, timeout = 25): Promise<Message[]> => {
    const data = await fetchJSON<{ messages: Message[] }>(
      `${NODE_BASE}/messenger/channels/${channel}/poll?after_id=${afterId}&timeout=${timeout}`
    );
    return data.messages;
  },

  createChannel: async (name: string): Promise<ChannelInfo> => {
    const data = await fetchJSON<{ channel: ChannelInfo }>(
      `${NODE_BASE}/messenger/channels?name=${encodeURIComponent(name)}`,
      { method: "POST" }
    );
    return data.channel;
  },

  // Node info (direct)
  getNodeInfo: (): Promise<NodeInfo> => fetchJSON(`${NODE_BASE}/hello`),

  // Peers
  getPeers: (): Promise<PeersResponse> => fetchJSON(`${NODE_BASE}/hello/peers`),

  // Registry
  getRegistry: (): Promise<Record<string, string>> => fetchJSON(`${NODE_BASE}/call/registry`),

  // Remote function call
  callFunction: <T = unknown>(name: string, args: Record<string, unknown> = {}): Promise<T> =>
    fetchJSON(`${NODE_BASE}/call/${name}`, {
      method: "POST",
      body: JSON.stringify(args),
    }),

  // Functions
  listFunctions: () => fetchJSON<{ functions: FunctionEntry[] }>(`${NODE_BASE}/function`),
  createFunction: (data: { name: string; code: string; language?: string; description?: string; python_version?: string; dependencies?: string[]; environment_id?: string }) =>
    fetchJSON<{ function: FunctionEntry }>(`${NODE_BASE}/function`, { method: "POST", body: JSON.stringify(data) }),
  getFunction: (id: string) => fetchJSON<{ function: FunctionEntry }>(`${NODE_BASE}/function/${id}`),
  updateFunction: (id: string, data: Record<string, unknown>) =>
    fetchJSON<{ function: FunctionEntry }>(`${NODE_BASE}/function/${id}`, { method: "PUT", body: JSON.stringify(data) }),
  deleteFunction: (id: string) => fetchJSON<void>(`${NODE_BASE}/function/${id}`, { method: "DELETE" }),
  runFunction: (id: string, args?: Record<string, unknown>) =>
    fetchJSON<{ run: RunEntry }>(`${NODE_BASE}/function/${id}/run`, { method: "POST", body: JSON.stringify(args || {}) }),
  listFunctionRuns: (id: string) => fetchJSON<{ runs: RunEntry[] }>(`${NODE_BASE}/function/${id}/run`),

  // Environments
  listEnvironments: () => fetchJSON<{ environments: EnvironmentEntry[] }>(`${NODE_BASE}/environment`),
  createEnvironment: (data: { name: string; python_version?: string; dependencies?: string[] }) =>
    fetchJSON<{ environment: EnvironmentEntry }>(`${NODE_BASE}/environment`, { method: "POST", body: JSON.stringify(data) }),
  getEnvironment: (id: string) => fetchJSON<{ environment: EnvironmentEntry }>(`${NODE_BASE}/environment/${id}`),
  deleteEnvironment: (id: string) => fetchJSON<void>(`${NODE_BASE}/environment/${id}`, { method: "DELETE" }),
  installPackages: (id: string, packages: string[]) =>
    fetchJSON<{ environment: EnvironmentEntry }>(`${NODE_BASE}/environment/${id}/install`, { method: "POST", body: JSON.stringify({ packages }) }),

  // Runs
  listRuns: () => fetchJSON<{ runs: RunEntry[] }>(`${NODE_BASE}/run`),
  getRun: (id: string) => fetchJSON<{ run: RunEntry }>(`${NODE_BASE}/run/${id}`),
  deleteRun: (id: string) => fetchJSON<void>(`${NODE_BASE}/run/${id}`, { method: "DELETE" }),
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

/** @deprecated Use node.executePython instead */
export const executePython = node.executePython;

/** @deprecated Use node.executeCmd instead */
export const executeCmd = node.executeCmd;

/** @deprecated Use node.sendMessage instead */
export const sendMessage = node.sendMessage;

/** @deprecated Use node.getChannels instead */
export const getChannels = node.getChannels;

/** @deprecated Use node.getMessages instead */
export const getMessages = node.getMessages;

/** @deprecated Use node.pollMessages instead */
export const pollMessages = node.pollMessages;

/** @deprecated Use node.createChannel instead */
export const createChannel = node.createChannel;

/** @deprecated Use node.getNodeInfo instead */
export const getNodeInfo = node.getNodeInfo;

/** @deprecated Use node.getRegistry instead */
export const getRegistry = node.getRegistry;
