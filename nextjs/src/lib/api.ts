// Browser-side client for the Yggdrasil FastAPI v2 backend.
//
// All requests target relative URLs which Next.js rewrites to
// http://127.0.0.1:8100 (or BOT_API_URL) — see nextjs/next.config.ts.

import type {
  AuditEntry,
  ChannelInfo,
  ClusterStats,
  DAGEntry,
  DAGRunEntry,
  FsEntry,
  HealthResponse,
  Message,
  ExcelInfo,
  MetricsResponse,
  NodeBackend,
  NodeCard,
  NodeMeta,
  PyEnvEntry,
  PyEnvPackages,
  PyFuncEntry,
  PyFuncRunEntry,
  TopologyResponse,
  UserCard,
} from "./types";

// ── Low-level fetch helper ─────────────────────────────────────────────────

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body);
    } catch {
      try { detail = await res.text(); } catch { /* ignore */ }
    }
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${detail || url}`);
  }
  // 204 No Content
  if (res.status === 204) return undefined as unknown as T;
  return (await res.json()) as T;
}

// ── Node identity ──────────────────────────────────────────────────────────

export function getNodeCard(): Promise<NodeCard> {
  return jsonFetch<NodeCard>("/api/card");
}

// ── Aggregate endpoints ────────────────────────────────────────────────────

export function getStats(): Promise<ClusterStats> {
  return jsonFetch<ClusterStats>("/api/v2/stats");
}

export function getTopology(): Promise<TopologyResponse> {
  return jsonFetch<TopologyResponse>("/api/v2/topology");
}

export function getHealth(): Promise<HealthResponse> {
  return jsonFetch<HealthResponse>("/api/v2/health");
}

export function getMetrics(): Promise<MetricsResponse> {
  return jsonFetch<MetricsResponse>("/api/v2/metrics");
}

export function getAudit(limit = 100): Promise<{ entries: AuditEntry[] }> {
  return jsonFetch<{ entries: AuditEntry[] }>(`/api/v2/audit?limit=${limit}`);
}

// ── Backend ────────────────────────────────────────────────────────────────

export function getBackend(): Promise<{ backend: NodeBackend }> {
  return jsonFetch<{ backend: NodeBackend }>("/api/v2/backend");
}

// SSE: emits NodeBackend snapshots every ~1s. The handler reads
// `event.data` as JSON.
export function createBackendStream(): EventSource {
  return new EventSource("/api/v2/backend/stream");
}

// ── Network / peers ────────────────────────────────────────────────────────

export function getPeers(): Promise<{ node_id: string; peers: NodeMeta[] }> {
  return jsonFetch<{ node_id: string; peers: NodeMeta[] }>("/api/v2/network/peers");
}

// ── PyEnv ──────────────────────────────────────────────────────────────────

export function getEnvs(): Promise<{ node_id: string; envs: PyEnvEntry[] }> {
  return jsonFetch<{ node_id: string; envs: PyEnvEntry[] }>("/api/v2/pyenv");
}

export function getExcelInfo(): Promise<ExcelInfo> {
  return jsonFetch<ExcelInfo>("/api/v2/excel/info");
}

export function getEnvPackages(envName: string): Promise<PyEnvPackages> {
  // Keyed by name (not the int64 id, which JSON.parse can't round-trip
  // losslessly in JS). Server-side TTL-cached, so polling is cheap.
  return jsonFetch<PyEnvPackages>(`/api/v2/pyenv/by-name/${encodeURIComponent(envName)}/packages`);
}

// ── PyFunc ─────────────────────────────────────────────────────────────────

export function getFuncs(): Promise<{ node_id: string; funcs: PyFuncEntry[] }> {
  return jsonFetch<{ node_id: string; funcs: PyFuncEntry[] }>("/api/v2/pyfunc");
}

export interface CreateFuncInput {
  name: string;
  code: string;
  description?: string;
  python_version?: string;
  dependencies?: string[];
  env_id?: number | null;
}

export function createFunc(input: CreateFuncInput): Promise<{ func: PyFuncEntry }> {
  return jsonFetch<{ func: PyFuncEntry }>("/api/v2/pyfunc", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function bulkDeleteFuncs(
  ids: number[],
): Promise<{ deleted: number; failed: { id: number; error: string }[] }> {
  return jsonFetch<{ deleted: number; failed: { id: number; error: string }[] }>(
    "/api/v2/pyfunc/bulk/delete",
    { method: "POST", body: JSON.stringify({ ids }) },
  );
}

// Trigger a function by name, then wait for the result. The chat-style "Quick
// Run" expects a single returned value, not a pending run, so we block here.
export async function runFuncByName(
  name: string,
  args: unknown[] = [],
  kwargs: Record<string, unknown> = {},
): Promise<unknown> {
  const triggered = await jsonFetch<{ run: PyFuncRunEntry }>(
    `/api/v2/pyfunc/by-name/${encodeURIComponent(name)}/run`,
    { method: "POST", body: JSON.stringify({ args, kwargs }) },
  );
  // Poll the wait endpoint; default 60s timeout to keep the UI responsive.
  const final = await jsonFetch<{ run: PyFuncRunEntry }>(
    `/api/v2/pyfuncrun/${triggered.run.id}/wait?timeout=60`,
  );
  if (final.run.status === "failed" || final.run.error) {
    throw new Error(final.run.error || `Run #${final.run.id} failed`);
  }
  // Prefer the structured result; fall back to stdout for fire-and-forget funcs.
  return final.run.result ?? final.run.stdout ?? final.run;
}

// ── PyFuncRun ──────────────────────────────────────────────────────────────

export function getRuns(): Promise<{ node_id: string; runs: PyFuncRunEntry[] }> {
  return jsonFetch<{ node_id: string; runs: PyFuncRunEntry[] }>("/api/v2/pyfuncrun");
}

// ── DAG ────────────────────────────────────────────────────────────────────

export function getDags(): Promise<{ node_id: string; dags: DAGEntry[] }> {
  return jsonFetch<{ node_id: string; dags: DAGEntry[] }>("/api/v2/dag");
}

export interface CreateDagInput {
  name: string;
  description?: string;
  steps: {
    id: string;
    ref: {
      node_url: string | null;
      func_id: number | null;
      env_id: number | null;
      args: Record<string, unknown>;
    };
    depends_on: string[];
  }[];
  edges: {
    from_step: string;
    to_step: string;
    output_key: string;
    input_key: string;
  }[];
}

export function createDag(input: CreateDagInput): Promise<{ dag: DAGEntry }> {
  return jsonFetch<{ dag: DAGEntry }>("/api/v2/dag", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function deleteDag(dagId: number): Promise<{ dag: DAGEntry }> {
  return jsonFetch<{ dag: DAGEntry }>(`/api/v2/dag/${dagId}`, { method: "DELETE" });
}

export function runDag(dagId: number): Promise<{ run: DAGRunEntry }> {
  return jsonFetch<{ run: DAGRunEntry }>(`/api/v2/dag/${dagId}/run`, { method: "POST" });
}

export function getDagRuns(dagId: number): Promise<{ node_id: string; runs: DAGRunEntry[] }> {
  return jsonFetch<{ node_id: string; runs: DAGRunEntry[] }>(`/api/v2/dag/${dagId}/run`);
}

export function scheduleDag(
  dagId: number,
  intervalSeconds: number,
  maxRuns?: number,
): Promise<{ dag: DAGEntry }> {
  return jsonFetch<{ dag: DAGEntry }>(`/api/v2/dag/${dagId}/schedule`, {
    method: "POST",
    body: JSON.stringify({
      interval_seconds: intervalSeconds,
      max_runs: maxRuns ?? null,
    }),
  });
}

// ── User ───────────────────────────────────────────────────────────────────

export function getUsers(): Promise<{ node_id: string; users: UserCard[] }> {
  return jsonFetch<{ node_id: string; users: UserCard[] }>("/api/v2/user");
}

// ── Filesystem ─────────────────────────────────────────────────────────────

export function getFsListing(
  path: string,
): Promise<{ node_id: string; path: string; entries: FsEntry[] }> {
  const url = path
    ? `/api/v2/fs/ls?path=${encodeURIComponent(path)}`
    : "/api/v2/fs/ls";
  return jsonFetch<{ node_id: string; path: string; entries: FsEntry[] }>(url);
}

// Returns just the file content as a string (the route also returns encoding/size
// but the legacy file viewer only ever reads `.content`).
export async function getFsContent(path: string): Promise<string> {
  const res = await jsonFetch<{ path: string; content: string; encoding: string; size: number }>(
    `/api/v2/fs/read?path=${encodeURIComponent(path)}`,
  );
  return res.content;
}

export function getFsTail(path: string, n = 200): Promise<{ path: string; lines: string[] }> {
  return jsonFetch<{ path: string; lines: string[] }>(
    `/api/v2/fs/tail?path=${encodeURIComponent(path)}&n=${n}`,
  );
}

export function getFsHead(path: string, n = 100): Promise<{ path: string; lines: string[] }> {
  return jsonFetch<{ path: string; lines: string[] }>(
    `/api/v2/fs/head?path=${encodeURIComponent(path)}&n=${n}`,
  );
}

export function createFsWatchStream(path: string): EventSource {
  return new EventSource(`/api/v2/fs/watch?path=${encodeURIComponent(path)}`);
}

export function grepFs(
  path: string,
  pattern: string,
  opts: { regex?: boolean; case_sensitive?: boolean; max_matches?: number } = {},
): Promise<{
  path: string;
  pattern: string;
  count: number;
  matches: { path: string; line_number: number; line: string; match: string }[];
}> {
  return jsonFetch("/api/v2/fs/grep", {
    method: "POST",
    body: JSON.stringify({ path, pattern, ...opts }),
  });
}

// ── Messenger ──────────────────────────────────────────────────────────────

export function getChannels(): Promise<{ node_id: string; channels: ChannelInfo[] }> {
  return jsonFetch<{ node_id: string; channels: ChannelInfo[] }>(
    "/api/v2/messenger/channels",
  );
}

export function getMessages(
  channel: string,
  limit = 50,
): Promise<{ channel: string; messages: Message[] }> {
  return jsonFetch<{ channel: string; messages: Message[] }>(
    `/api/v2/messenger/${encodeURIComponent(channel)}?limit=${limit}`,
  );
}

export function sendMessage(channel: string, content: string): Promise<Message> {
  return jsonFetch<Message>(`/api/v2/messenger/${encodeURIComponent(channel)}`, {
    method: "POST",
    body: JSON.stringify({ channel, content }),
  });
}

export function createMessageStream(channel: string): EventSource {
  return new EventSource(`/api/v2/messenger/${encodeURIComponent(channel)}/stream`);
}
