// Browser-side bot client. All requests go through the /api/bot/* rewrite
// configured in next.config.ts, which proxies to the FastAPI node.
// Server code should use src/lib/bot-client.ts instead.

// ── Types ──────────────────────────────────────────────────────────────────

export interface NodeInfo {
  node_id: string;
  host: string;
  port: number;
  version: string;
  uptime: number;
  channels: string[];
  functions: string[];
  lat?: number | null;
  lon?: number | null;
}

export interface ChannelInfo {
  name: string;
  created_at: string;
  last_active: string;
  message_count: number;
  members: string[];
}

export interface Message {
  id: string;
  sender: string;
  text: string;
  channel: string;
  timestamp: string;
  node_id: string;
}

export interface ChannelListResponse {
  node_id: string;
  channels: ChannelInfo[];
}

export interface MessageListResponse {
  node_id: string;
  channel: string;
  messages: Message[];
}

export interface ChannelResponse {
  node_id: string;
  channel: ChannelInfo;
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

// v2 topology / stats / health / metrics ------------------------------------

export interface TopologyNode {
  node_id: string;
  host: string;
  port: number;
  role: string;
  cpu_percent: number;
  memory_percent: number;
  active_runs: number;
  gpu_count: number;
  self: boolean;
  lat?: number | null;
  lon?: number | null;
}

export interface TopologyResponse {
  nodes: TopologyNode[];
  total_cpu_percent: number;
  total_active_runs: number;
  total_gpus: number;
}

export interface StatsResponse {
  node_id: string;
  uptime: number;
  cpu_percent: number;
  memory_percent: number;
  active_runs: number;
  total_runs: number;
  env_count: number;
  func_count: number;
  dag_count: number;
  scheduled_dags: number;
  peer_count: number;
  gpu_count: number;
}

export interface HealthCheck {
  status: "ok" | "error" | string;
  [key: string]: unknown;
}

export interface HealthResponse {
  status: "healthy" | "degraded" | string;
  node_id: string;
  checks: Record<string, HealthCheck>;
}

export interface MetricsTopRun {
  id: number;
  name: string;
  runs: number;
}

export interface MetricsTopDuration {
  id: number;
  name: string;
  avg_ms: number;
}

export interface MetricsRecentRun {
  id: number;
  func_id: number;
  status: string;
  duration: number | null;
  started_at: string | null;
}

export interface MetricsResponse {
  node_id: string;
  top_by_runs: MetricsTopRun[];
  top_by_duration: MetricsTopDuration[];
  success_rate: { id: number; name: string; rate: number }[];
  recent_runs: MetricsRecentRun[];
}

// Mirrors python/src/yggdrasil/node/api/schemas/backend.py:NodeBackend
export interface GpuInfo {
  index: number;
  name: string;
  memory_used_mb: number;
  memory_total_mb: number;
  utilization_percent: number;
  temperature_c: number;
}

export interface NetworkIO {
  bytes_sent: number;
  bytes_recv: number;
  packets_sent: number;
  packets_recv: number;
}

export interface NodeBackend {
  node_id: string;
  role: string;
  hostname: string;
  platform: string;
  python_version: string;
  cpu_count: number;
  cpu_percent: number;
  memory_used_mb: number;
  memory_total_mb: number;
  disk_used_mb: number;
  disk_total_mb: number;
  gpus: GpuInfo[];
  network: NetworkIO;
  uptime_seconds: number;
  active_runs: number;
  total_runs: number;
  timestamp: string;
}

export interface BackendResponse {
  backend: NodeBackend;
}

// ── Fetch helper ───────────────────────────────────────────────────────────

const BASE = "/api/bot";

async function jget<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function jpost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`POST ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

// ── Messenger ──────────────────────────────────────────────────────────────

export async function getChannels(): Promise<ChannelInfo[]> {
  const resp = await jget<ChannelListResponse>("/messenger/channels");
  return resp.channels;
}

export async function getMessages(channel: string, limit = 200): Promise<Message[]> {
  const resp = await jget<MessageListResponse>(
    `/messenger/channels/${encodeURIComponent(channel)}/messages?limit=${limit}`,
  );
  return resp.messages;
}

export async function sendMessage(text: string, sender: string, channel: string): Promise<Message> {
  return jpost<Message>("/messenger", { text, sender, channel });
}

export async function pollMessages(
  channel: string,
  afterId: string,
  timeout = 25,
): Promise<Message[]> {
  const q = new URLSearchParams();
  if (afterId) q.set("after_id", afterId);
  q.set("timeout", String(timeout));
  const resp = await jget<MessageListResponse>(
    `/messenger/channels/${encodeURIComponent(channel)}/poll?${q.toString()}`,
  );
  return resp.messages;
}

export async function createChannel(name: string): Promise<ChannelResponse> {
  const res = await fetch(`${BASE}/messenger/channels?name=${encodeURIComponent(name)}`, {
    method: "POST",
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`createChannel failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as ChannelResponse;
}

// ── Execute ────────────────────────────────────────────────────────────────

export async function executePython(code: string): Promise<PythonResponse> {
  return jpost<PythonResponse>("/python", { code });
}

export async function executeCmd(command: string[]): Promise<CmdResponse> {
  return jpost<CmdResponse>("/cmd", { command });
}

// ── Bot namespace (v1 + v2 grouped) ────────────────────────────────────────

export const bot = {
  async getNodeInfo(): Promise<NodeInfo> {
    return jget<NodeInfo>("/hello");
  },
  async getTopology(): Promise<TopologyResponse> {
    return jget<TopologyResponse>("/v2/topology");
  },
  async getStats(): Promise<StatsResponse> {
    return jget<StatsResponse>("/v2/stats");
  },
  async getHealth(): Promise<HealthResponse> {
    return jget<HealthResponse>("/v2/health");
  },
  async getMetrics(): Promise<MetricsResponse> {
    return jget<MetricsResponse>("/v2/metrics");
  },
  async getBackend(): Promise<NodeBackend> {
    const resp = await jget<BackendResponse>("/v2/backend");
    return resp.backend;
  },
  // Opens an SSE stream of NodeBackend snapshots from /api/v2/backend/stream.
  // Returns a cleanup function. The backend router emits raw
  // NodeBackend.model_dump() per "data:" line (no wrapper envelope).
  streamBackend(
    onSnap: (snap: NodeBackend) => void,
    onError?: (e: Event) => void,
  ): () => void {
    const es = new EventSource(`${BASE}/v2/backend/stream`);
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as NodeBackend;
        onSnap(data);
      } catch {
        // Malformed line: skip.
      }
    };
    es.onerror = (e) => {
      if (onError) onError(e);
    };
    return () => es.close();
  },
};
