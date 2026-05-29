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
  PyEnvEnvVars,
  PyEnvPackages,
  PyFuncEntry,
  PyFuncRunEntry,
  TopologyResponse,
  UserCard,
} from "./types";
import { cachedGet, invalidate, TTL } from "./cache";

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

export function getNodeCard(fresh = false): Promise<NodeCard> {
  return cachedGet<NodeCard>("/api/card", TTL.STRUCTURAL, jsonFetch, fresh);
}

// ── Aggregate endpoints ────────────────────────────────────────────────────

export function getStats(fresh = false): Promise<ClusterStats> {
  return cachedGet<ClusterStats>("/api/v2/stats", TTL.VITAL, jsonFetch, fresh);
}

export function getTopology(fresh = false): Promise<TopologyResponse> {
  return cachedGet<TopologyResponse>("/api/v2/topology", TTL.STRUCTURAL, jsonFetch, fresh);
}

export function getHealth(fresh = false): Promise<HealthResponse> {
  return cachedGet<HealthResponse>("/api/v2/health", TTL.VITAL, jsonFetch, fresh);
}

export function getMetrics(fresh = false): Promise<MetricsResponse> {
  return cachedGet<MetricsResponse>("/api/v2/metrics", TTL.VITAL, jsonFetch, fresh);
}

export function getAudit(limit = 100, fresh = false): Promise<{ entries: AuditEntry[] }> {
  return cachedGet<{ entries: AuditEntry[] }>(`/api/v2/audit?limit=${limit}`, TTL.VITAL, jsonFetch, fresh);
}

// ── Backend ────────────────────────────────────────────────────────────────

export function getBackend(fresh = false): Promise<{ backend: NodeBackend }> {
  return cachedGet<{ backend: NodeBackend }>("/api/v2/backend", TTL.VITAL, jsonFetch, fresh);
}

// SSE: emits NodeBackend snapshots. ``intervalSec`` sets the cadence
// (backend clamps to [0.25, 30]); the handler reads `event.data` as JSON.
export function createBackendStream(intervalSec = 1): EventSource {
  return new EventSource(`/api/v2/backend/stream?interval=${intervalSec}`);
}

// ── Network / peers ────────────────────────────────────────────────────────

export function getPeers(fresh = false): Promise<{ node_id: string; peers: NodeMeta[] }> {
  return cachedGet<{ node_id: string; peers: NodeMeta[] }>("/api/v2/network/peers", TTL.STRUCTURAL, jsonFetch, fresh);
}

// ── PyEnv ──────────────────────────────────────────────────────────────────

export function getEnvs(fresh = false): Promise<{ node_id: string; envs: PyEnvEntry[] }> {
  return cachedGet<{ node_id: string; envs: PyEnvEntry[] }>("/api/v2/pyenv", TTL.DEFINITION, jsonFetch, fresh);
}

export function getExcelInfo(): Promise<ExcelInfo> {
  return cachedGet<ExcelInfo>("/api/v2/excel/info", TTL.DEFINITION, jsonFetch);
}

export async function setEnvVars(
  envName: string,
  env_vars: Record<string, string>,
  replace = false,
): Promise<PyEnvEnvVars> {
  const res = await jsonFetch<PyEnvEnvVars>(`/api/v2/pyenv/by-name/${encodeURIComponent(envName)}/env`, {
    method: "PUT",
    body: JSON.stringify({ env_vars, replace }),
  });
  invalidate("pyenv");
  return res;
}

export async function deleteEnvVar(envName: string, key: string): Promise<PyEnvEnvVars> {
  const res = await jsonFetch<PyEnvEnvVars>(
    `/api/v2/pyenv/by-name/${encodeURIComponent(envName)}/env/${encodeURIComponent(key)}`,
    { method: "DELETE" },
  );
  invalidate("pyenv");
  return res;
}

export function getEnvPackages(envName: string): Promise<PyEnvPackages> {
  // Keyed by name (not the int64 id, which JSON.parse can't round-trip
  // losslessly in JS). Server-side TTL-cached too, so this just coalesces.
  return cachedGet<PyEnvPackages>(
    `/api/v2/pyenv/by-name/${encodeURIComponent(envName)}/packages`, TTL.DEFINITION, jsonFetch,
  );
}

// ── PyFunc ─────────────────────────────────────────────────────────────────

export function getFuncs(fresh = false): Promise<{ node_id: string; funcs: PyFuncEntry[] }> {
  return cachedGet<{ node_id: string; funcs: PyFuncEntry[] }>("/api/v2/pyfunc", TTL.DEFINITION, jsonFetch, fresh);
}

export interface CreateFuncInput {
  name: string;
  code: string;
  description?: string;
  python_version?: string;
  dependencies?: string[];
  env_id?: number | null;
}

export async function createFunc(input: CreateFuncInput): Promise<{ func: PyFuncEntry }> {
  const res = await jsonFetch<{ func: PyFuncEntry }>("/api/v2/pyfunc", {
    method: "POST",
    body: JSON.stringify(input),
  });
  invalidate("pyfunc", "stats", "metrics");
  return res;
}

export async function bulkDeleteFuncs(
  ids: number[],
): Promise<{ deleted: number; failed: { id: number; error: string }[] }> {
  const res = await jsonFetch<{ deleted: number; failed: { id: number; error: string }[] }>(
    "/api/v2/pyfunc/bulk/delete",
    { method: "POST", body: JSON.stringify({ ids }) },
  );
  invalidate("pyfunc", "stats", "metrics");
  return res;
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
  invalidate("pyfuncrun", "pyfunc", "stats", "metrics");
  if (final.run.status === "failed" || final.run.error) {
    throw new Error(final.run.error || `Run #${final.run.id} failed`);
  }
  // Prefer the structured result; fall back to stdout for fire-and-forget funcs.
  return final.run.result ?? final.run.stdout ?? final.run;
}

// ── PyFuncRun ──────────────────────────────────────────────────────────────

export function getRuns(fresh = false): Promise<{ node_id: string; runs: PyFuncRunEntry[] }> {
  return cachedGet<{ node_id: string; runs: PyFuncRunEntry[] }>("/api/v2/pyfuncrun", TTL.VITAL, jsonFetch, fresh);
}

export async function cancelRun(runId: number): Promise<{ run: PyFuncRunEntry }> {
  const res = await jsonFetch<{ run: PyFuncRunEntry }>(`/api/v2/pyfuncrun/${runId}/cancel`, { method: "POST" });
  invalidate("pyfuncrun", "stats", "metrics");
  return res;
}

export async function deleteRun(runId: number): Promise<{ run: PyFuncRunEntry }> {
  const res = await jsonFetch<{ run: PyFuncRunEntry }>(`/api/v2/pyfuncrun/${runId}`, { method: "DELETE" });
  invalidate("pyfuncrun", "stats", "metrics");
  return res;
}

// ── DAG ────────────────────────────────────────────────────────────────────

export function getDags(fresh = false): Promise<{ node_id: string; dags: DAGEntry[] }> {
  return cachedGet<{ node_id: string; dags: DAGEntry[] }>("/api/v2/dag", TTL.DEFINITION, jsonFetch, fresh);
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

export async function createDag(input: CreateDagInput): Promise<{ dag: DAGEntry }> {
  const res = await jsonFetch<{ dag: DAGEntry }>("/api/v2/dag", {
    method: "POST",
    body: JSON.stringify(input),
  });
  invalidate("dag", "stats");
  return res;
}

export async function deleteDag(dagId: number): Promise<{ dag: DAGEntry }> {
  const res = await jsonFetch<{ dag: DAGEntry }>(`/api/v2/dag/${dagId}`, { method: "DELETE" });
  invalidate("dag", "stats");
  return res;
}

export async function runDag(dagId: number): Promise<{ run: DAGRunEntry }> {
  const res = await jsonFetch<{ run: DAGRunEntry }>(`/api/v2/dag/${dagId}/run`, { method: "POST" });
  invalidate("dag", "pyfuncrun", "stats", "metrics");
  return res;
}

export function getDagRuns(dagId: number): Promise<{ node_id: string; runs: DAGRunEntry[] }> {
  return cachedGet<{ node_id: string; runs: DAGRunEntry[] }>(`/api/v2/dag/${dagId}/run`, TTL.VITAL, jsonFetch);
}

export async function scheduleDag(
  dagId: number,
  intervalSeconds: number,
  maxRuns?: number,
): Promise<{ dag: DAGEntry }> {
  const res = await jsonFetch<{ dag: DAGEntry }>(`/api/v2/dag/${dagId}/schedule`, {
    method: "POST",
    body: JSON.stringify({
      interval_seconds: intervalSeconds,
      max_runs: maxRuns ?? null,
    }),
  });
  invalidate("dag", "stats");
  return res;
}

// ── User ───────────────────────────────────────────────────────────────────

export function getUsers(fresh = false): Promise<{ node_id: string; users: UserCard[] }> {
  return cachedGet<{ node_id: string; users: UserCard[] }>("/api/v2/user", TTL.DEFINITION, jsonFetch, fresh);
}

// ── Filesystem ─────────────────────────────────────────────────────────────

// A root of the global filesystem tree: the local node + every linked peer.
export interface FsNodeRoot {
  node_id: string;
  host: string;
  port: number;
  self: boolean;
  role: string;
  cpu_percent?: number;
  memory_percent?: number;
  active_runs?: number;
}

// ``node`` selects which node's filesystem the call targets. Omitting it (or
// passing the local node id) reads the local fs; a peer id is proxied through
// the local node — see services/network.fs_proxy_*.
function nodeParam(node?: string): string {
  return node ? `&node=${encodeURIComponent(node)}` : "";
}

export function getFsNodes(fresh = false): Promise<{ node_id: string; nodes: FsNodeRoot[] }> {
  return cachedGet<{ node_id: string; nodes: FsNodeRoot[] }>(
    "/api/v2/fs/nodes", TTL.STRUCTURAL, jsonFetch, fresh,
  );
}

export function getFsListing(
  path: string,
  node?: string,
  fresh = false,
): Promise<{ node_id: string; path: string; entries: FsEntry[] }> {
  const url = `/api/v2/fs/ls?path=${encodeURIComponent(path)}${nodeParam(node)}`;
  return cachedGet<{ node_id: string; path: string; entries: FsEntry[] }>(url, TTL.VITAL, jsonFetch, fresh);
}

// Reads a bounded preview of a file. The backend caps how much it pulls into
// memory and sets `truncated` when the file is larger than the returned slice.
export function getFsRead(
  path: string,
  node?: string,
): Promise<{ path: string; content: string; encoding: string; size: number; truncated: boolean }> {
  return jsonFetch(`/api/v2/fs/read?path=${encodeURIComponent(path)}${nodeParam(node)}`);
}

export function getFsTail(path: string, n = 200, node?: string): Promise<{ path: string; lines: string[] }> {
  return jsonFetch<{ path: string; lines: string[] }>(
    `/api/v2/fs/tail?path=${encodeURIComponent(path)}&n=${n}${nodeParam(node)}`,
  );
}

export function getFsHead(path: string, n = 100, node?: string): Promise<{ path: string; lines: string[] }> {
  return jsonFetch<{ path: string; lines: string[] }>(
    `/api/v2/fs/head?path=${encodeURIComponent(path)}&n=${n}${nodeParam(node)}`,
  );
}

// Live tail (SSE) is local-node only — the watch stream is not proxied.
export function createFsWatchStream(path: string): EventSource {
  return new EventSource(`/api/v2/fs/watch?path=${encodeURIComponent(path)}`);
}

// A direct download URL (file passthrough or folder zip). Used as an <a href>
// so the browser streams it; peer downloads are proxied through the local node.
export function fsDownloadUrl(path: string, node?: string): string {
  return `/api/v2/fs/download?path=${encodeURIComponent(path)}${nodeParam(node)}`;
}

export async function deleteFsPath(path: string, node?: string): Promise<void> {
  await jsonFetch<void>(
    `/api/v2/fs/delete?path=${encodeURIComponent(path)}${nodeParam(node)}`,
    { method: "DELETE" },
  );
  invalidate("fs/ls");
}

export async function writeFsFile(
  path: string, content: string, node?: string, encoding = "utf-8",
): Promise<FsEntry> {
  const res = await jsonFetch<FsEntry>(
    `/api/v2/fs/write${node ? `?node=${encodeURIComponent(node)}` : ""}`,
    { method: "POST", body: JSON.stringify({ path, content, encoding, mkdir: true }) },
  );
  invalidate("fs/ls");
  return res;
}

export async function mkdirFs(path: string, node?: string): Promise<FsEntry> {
  const res = await jsonFetch<FsEntry>(
    `/api/v2/fs/mkdir?path=${encodeURIComponent(path)}${nodeParam(node)}`,
    { method: "POST" },
  );
  invalidate("fs/ls");
  return res;
}

// Streams one file's bytes to ``path`` on the target node. Used per-file when a
// folder is dropped, so a large upload never buffers the whole tree in memory.
export async function uploadFsFile(path: string, file: File, node?: string): Promise<FsEntry> {
  const res = await fetch(
    `/api/v2/fs/upload?path=${encodeURIComponent(path)}${nodeParam(node)}`,
    { method: "POST", body: file },
  );
  if (!res.ok) throw new Error(`Upload failed: HTTP ${res.status}`);
  invalidate("fs/ls");
  return (await res.json()) as FsEntry;
}

export function grepFs(
  path: string,
  pattern: string,
  opts: { regex?: boolean; case_sensitive?: boolean; max_matches?: number; node?: string } = {},
): Promise<{
  path: string;
  pattern: string;
  count: number;
  truncated?: boolean;
  matches: { path: string; line_number: number; line: string; match: string }[];
}> {
  const { node, ...body } = opts;
  return jsonFetch(`/api/v2/fs/grep${node ? `?node=${encodeURIComponent(node)}` : ""}`, {
    method: "POST",
    body: JSON.stringify({ path, pattern, ...body }),
  });
}

// ── Tabular ──────────────────────────────────────────────────────────────
export interface TabularColumn { name: string; type: string; }

export interface TabularInspect {
  node_id: string;
  path: string;
  source_url: string;
  media_type: string;
  is_tabular: boolean;
  columns: TabularColumn[];
  column_count: number;
  row_count: number | null;
  size_bytes: number;
  schema_hash: string;
  editable: boolean;
  schema_error: string | null;
}

export type TabularCell = string | number | boolean | null;

export interface TabularPreview {
  node_id: string;
  path: string;
  columns: TabularColumn[];
  rows: TabularCell[][];
  row_count: number;
  limit: number;
  truncated: boolean;
}

const TABULAR_EXTS = new Set(["csv", "parquet", "pq", "json", "ndjson", "arrow", "feather", "xlsx", "xls"]);
export function isTabularName(name: string): boolean {
  const idx = name.lastIndexOf(".");
  return idx >= 0 && TABULAR_EXTS.has(name.slice(idx + 1).toLowerCase());
}

export function getTabularInspect(path: string, node?: string): Promise<TabularInspect> {
  return jsonFetch(`/api/v2/tabular/inspect?path=${encodeURIComponent(path)}${nodeParam(node)}`);
}

export function getTabularPreview(path: string, limit = 100, node?: string): Promise<TabularPreview> {
  return jsonFetch(`/api/v2/tabular/preview?path=${encodeURIComponent(path)}&limit=${limit}${nodeParam(node)}`);
}

export async function writeTabular(
  path: string, columns: string[], rows: TabularCell[][], node?: string, fmt?: string,
): Promise<{ path: string; rows: number; columns: number; bytes_written: number }> {
  const res = await jsonFetch<{ path: string; rows: number; columns: number; bytes_written: number }>(
    `/api/v2/tabular/write${node ? `?node=${encodeURIComponent(node)}` : ""}`,
    { method: "POST", body: JSON.stringify({ path, columns, rows, fmt: fmt ?? null }) },
  );
  invalidate("fs/ls", "tabular");
  return res;
}

// Arrow IPC preview URL — fetched + decoded by lib/arrow.fetchArrowTable.
// ``offset`` reads a row page so the grid can stream windows of a large table.
export function tabularPreviewArrowUrl(path: string, limit = 200, offset = 0, node?: string): string {
  return `/api/v2/tabular/preview.arrow?path=${encodeURIComponent(path)}&limit=${limit}&offset=${offset}${nodeParam(node)}`;
}

export function isWorkbookName(name: string): boolean {
  const idx = name.lastIndexOf(".");
  return idx >= 0 && ["xlsx", "xls"].includes(name.slice(idx + 1).toLowerCase());
}

export interface WorkbookSheet { name: string; rows: number; cols: number; visible: boolean; }

export function getWorkbookSheets(path: string, node?: string): Promise<{ node_id: string; path: string; sheets: WorkbookSheet[] }> {
  return jsonFetch(`/api/v2/workbook/sheets?path=${encodeURIComponent(path)}${nodeParam(node)}`);
}

export function workbookReadArrowUrl(
  path: string, sheet: string, opts: { n_rows?: number; skip_rows?: number; node?: string } = {},
): string {
  const n = opts.n_rows ? `&n_rows=${opts.n_rows}` : "";
  const s = opts.skip_rows ? `&skip_rows=${opts.skip_rows}` : "";
  return `/api/v2/workbook/read?path=${encodeURIComponent(path)}&sheet=${encodeURIComponent(sheet)}${n}${s}${nodeParam(opts.node)}`;
}

// Surgical cell edits (1-based) — preserves formulas/formatting/other sheets.
export async function editWorkbook(
  path: string, sheet: string, cells: [number, number, TabularCell][], node?: string,
): Promise<{ path: string; sheet: string; cells_written: number }> {
  const res = await jsonFetch<{ path: string; sheet: string; cells_written: number }>(
    `/api/v2/workbook/edit${node ? `?node=${encodeURIComponent(node)}` : ""}`,
    { method: "POST", body: JSON.stringify({ path, sheet, cells }) },
  );
  invalidate("fs/ls", "workbook", "tabular");
  return res;
}

// ── Messenger ──────────────────────────────────────────────────────────────

export function getChannels(fresh = false): Promise<{ node_id: string; channels: ChannelInfo[] }> {
  return cachedGet<{ node_id: string; channels: ChannelInfo[] }>(
    "/api/v2/messenger/channels", TTL.DEFINITION, jsonFetch, fresh,
  );
}

export function getMessages(
  channel: string,
  limit = 50,
): Promise<{ channel: string; messages: Message[] }> {
  return cachedGet<{ channel: string; messages: Message[] }>(
    `/api/v2/messenger/${encodeURIComponent(channel)}?limit=${limit}`, TTL.VITAL, jsonFetch,
  );
}

export async function sendMessage(channel: string, content: string): Promise<Message> {
  const res = await jsonFetch<Message>(`/api/v2/messenger/${encodeURIComponent(channel)}`, {
    method: "POST",
    body: JSON.stringify({ channel, content }),
  });
  invalidate("messenger");
  return res;
}

export function createMessageStream(channel: string): EventSource {
  return new EventSource(`/api/v2/messenger/${encodeURIComponent(channel)}/stream`);
}
