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
import { cachedGet, cachedPost, invalidate, TTL } from "./cache";

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

export async function createEnv(
  name: string, python_version: string, dependencies: string[],
): Promise<{ env: PyEnvEntry }> {
  const res = await jsonFetch<{ env: PyEnvEntry }>("/api/v2/pyenv", {
    method: "POST",
    body: JSON.stringify({ name, python_version, dependencies, env_vars: {} }),
  });
  invalidate("pyenv");
  return res;
}

export async function deleteEnv(id: number): Promise<void> {
  await jsonFetch(`/api/v2/pyenv/${id}`, { method: "DELETE" });
  invalidate("pyenv");
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

export async function deleteFunc(id: number): Promise<void> {
  await jsonFetch(`/api/v2/pyfunc/${id}`, { method: "DELETE" });
  invalidate("pyfunc", "stats", "metrics");
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

export function getFsStat(path: string, node?: string): Promise<FsEntry> {
  return jsonFetch<FsEntry>(`/api/v2/fs/stat?path=${encodeURIComponent(path)}${nodeParam(node)}`);
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

// ── Analysis (pivot / describe / finance) ───────────────────────────────────
export type AggFunc = "sum" | "mean" | "min" | "max" | "count" | "median" | "std" | "var";

export interface AggregateResult {
  columns: string[];
  rows: TabularCell[][];
  group_count: number;
  source_rows: number;
  truncated: boolean;
}

export function aggregate(
  path: string,
  group_by: string[],
  measures: { column: string; agg: AggFunc }[],
  node?: string,
  limit = 500,
  filters: { column: string; op: string; value?: unknown }[] = [],
): Promise<AggregateResult> {
  const url = `/api/v2/analysis/aggregate${node ? `?node=${encodeURIComponent(node)}` : ""}`;
  const payload = { path, group_by, measures, limit, filters };
  return cachedPost(url, payload, TTL.VITAL, () => jsonFetch(url, { method: "POST", body: JSON.stringify(payload) }));
}

export interface DescribeResult {
  statistics: string[];
  columns: string[];
  rows: TabularCell[][];
  truncated: boolean;
}

export function describe(path: string, node?: string): Promise<DescribeResult> {
  return jsonFetch(`/api/v2/analysis/describe?path=${encodeURIComponent(path)}${nodeParam(node)}`);
}

export interface FinanceResult {
  column: string;
  window: number;
  index: (string | number)[];
  value: (number | null)[];
  pct_change: (number | null)[];
  cum_return: (number | null)[];
  roll_mean: (number | null)[];
  roll_vol: (number | null)[];
  truncated: boolean;
}

export function finance(
  path: string,
  column: string,
  opts: { order_by?: string; window?: number; limit?: number; node?: string } = {},
): Promise<FinanceResult> {
  const { node, ...body } = opts;
  return jsonFetch(`/api/v2/analysis/finance${node ? `?node=${encodeURIComponent(node)}` : ""}`, {
    method: "POST",
    body: JSON.stringify({ path, column, ...body }),
  });
}

export interface FilterSpec { column: string; op: string; value?: unknown; }
export interface CastSpec { column: string; dtype: string; tz?: string; }
export interface Transform { filters?: FilterSpec[]; casts?: CastSpec[]; columns?: string[]; limit?: number; }

export interface SeriesResult {
  column: string;
  x: (string | number)[];
  y: (number | null)[];
  y_min: (number | null)[];
  y_max: (number | null)[];
  source_rows: number;
  sampled: boolean;
}

// Adaptive downsample: asks the backend for ~`points` buckets over an optional
// [x_min,x_max] zoom window (predicate-pushed into the lazy scan). Cached
// client-side so zooming back to a window is instant + saves a node call.
export function analysisSeries(
  path: string, column: string,
  opts: { x?: string; points?: number; x_min?: number; x_max?: number; filters?: FilterSpec[]; node?: string } = {},
): Promise<SeriesResult> {
  const { node, ...body } = opts;
  const url = `/api/v2/analysis/series${node ? `?node=${encodeURIComponent(node)}` : ""}`;
  const payload = { path, column, ...body };
  return cachedPost(url, payload, TTL.VITAL, () => jsonFetch(url, { method: "POST", body: JSON.stringify(payload) }));
}

// POST /export: apply the transform (filters + casts incl. tz→UTC + projection)
// and download the result in any tabular media type.
export async function downloadExport(path: string, fmt: string, transform: Transform, node?: string): Promise<void> {
  const url = `/api/v2/analysis/export${node ? `?node=${encodeURIComponent(node)}` : ""}`;
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path, fmt, transform }) });
  if (!res.ok) throw new Error(`export failed: HTTP ${res.status}`);
  const blob = await res.blob();
  const base = path.split("/").pop()?.replace(/\.[^.]+$/, "") ?? "export";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${base}.${fmt}`;
  a.click();
  URL.revokeObjectURL(a.href);
}

export interface OhlcResult {
  column: string;
  x: (string | number)[];
  open: (number | null)[];
  high: (number | null)[];
  low: (number | null)[];
  close: (number | null)[];
  volume: (number | null)[] | null;
  bars: number;
  source_rows: number;
}

export function analysisOhlc(
  path: string, column: string,
  opts: { x?: string; volume?: string; buckets?: number; filters?: FilterSpec[]; node?: string } = {},
): Promise<OhlcResult> {
  const { node, ...body } = opts;
  const url = `/api/v2/analysis/ohlc${node ? `?node=${encodeURIComponent(node)}` : ""}`;
  const payload = { path, column, ...body };
  return cachedPost(url, payload, TTL.VITAL, () => jsonFetch(url, { method: "POST", body: JSON.stringify(payload) }));
}

// ── Risk analytics ──────────────────────────────────────────────────────
export interface RiskResult {
  node_id: string;
  path: string;
  column: string;
  n: number;
  periods_per_year: number;
  ann_return: number | null;
  ann_volatility: number | null;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  calmar_ratio: number | null;
  max_drawdown: number | null;
  max_drawdown_peak_i: number | null;
  max_drawdown_trough_i: number | null;
  var_95: number | null;
  var_99: number | null;
  cvar_95: number | null;
  win_rate: number | null;
  profit_factor: number | null;
  skewness: number | null;
  kurtosis: number | null;
}

export function analysisRisk(
  path: string,
  column: string,
  opts: { order_by?: string; is_returns?: boolean; periods_per_year?: number; limit?: number; filters?: FilterSpec[]; node?: string } = {},
): Promise<RiskResult> {
  const { node, ...body } = opts;
  const url = `/api/v2/analysis/risk${node ? `?node=${encodeURIComponent(node)}` : ""}`;
  const payload = { path, column, ...body };
  return cachedPost(url, payload, TTL.VITAL, () => jsonFetch(url, { method: "POST", body: JSON.stringify(payload) }));
}

// ── Technical indicators ─────────────────────────────────────────────────
export interface IndicatorsResult {
  node_id: string;
  path: string;
  column: string;
  x: (string | number)[];
  price: (number | null)[];
  indicators: Record<string, (number | null)[]>;
  n: number;
}

export function analysisIndicators(
  path: string,
  column: string,
  opts: {
    x?: string; high?: string; low?: string; volume?: string;
    sma?: number[]; ema?: number[]; rsi?: number | null;
    macd?: boolean; bollinger?: number | null; atr?: number | null;
    stoch?: number | null; obv?: boolean;
    filters?: FilterSpec[]; limit?: number; node?: string;
  } = {},
): Promise<IndicatorsResult> {
  const { node, ...body } = opts;
  const url = `/api/v2/analysis/indicators${node ? `?node=${encodeURIComponent(node)}` : ""}`;
  const payload = { path, column, ...body };
  return cachedPost(url, payload, TTL.VITAL, () => jsonFetch(url, { method: "POST", body: JSON.stringify(payload) }));
}

// ── Forecasting ─────────────────────────────────────────────────────────
export interface ForecastSeriesData {
  key: string;
  history_x: (string | number)[];
  history_y: (number | null)[];
  forecast_x: (string | number)[];
  forecast_y: (number | null)[];
  lower: (number | null)[];
  upper: (number | null)[];
  rmse: number | null;
}

export interface ForecastResult {
  node_id: string;
  path: string;
  column: string;
  model_used: string;
  horizon: number;
  period: number | null;
  series: ForecastSeriesData[];
  source_rows: number;
  sampled: boolean;
}

// POST /analysis/forecast — fit xgboost→gbr→ridge over trend/lag/seasonal
// features and project `horizon` steps with a confidence band. Cached client
// side so re-opening the panel doesn't re-fit.
export function analysisForecast(
  path: string, column: string,
  opts: {
    x?: string; group?: string; horizon?: number; model?: string;
    period?: number; agg?: string; filters?: FilterSpec[]; node?: string;
  } = {},
): Promise<ForecastResult> {
  const { node, ...body } = opts;
  const url = `/api/v2/analysis/forecast${node ? `?node=${encodeURIComponent(node)}` : ""}`;
  const payload = { path, column, ...body };
  return cachedPost(url, payload, TTL.VITAL, () => jsonFetch(url, { method: "POST", body: JSON.stringify(payload) }));
}

export interface ForecastSpec {
  source: string; column: string; x?: string | null; keys?: string[];
  horizon?: number; model?: string; period?: number | null; agg?: string;
  materialized?: boolean;
}

export interface ForecastAssetResult {
  node_id: string;
  table: TableEntry;
  model_used: string;
  rmse: number | null;
  rows: number;
  materialized_url: string | null;
  sampled: boolean;
}

// POST /saga/forecast — register a forecasting workflow as a queryable
// FORECAST catalog asset (live view, or materialised snapshot).
export function registerForecastWorkflow(
  input: { catalog?: string; schema?: string; name: string; spec: ForecastSpec; materialize?: boolean; comment?: string },
  node?: string,
): Promise<ForecastAssetResult> {
  const url = `/api/v2/saga/forecast${node ? `?node=${encodeURIComponent(node)}` : ""}`;
  return jsonFetch(url, { method: "POST", body: JSON.stringify(input) });
}

// ── Saga catalog ─────────────────────────────────────────────────────────

export interface ColumnSpec { name: string; dtype: string; nullable: boolean; comment: string; }
export interface ColumnStat {
  column: string; null_count: number | null; distinct_count: number | null;
  min: unknown; max: unknown;
}
export interface TableStatistics {
  row_count: number | null; size_bytes: number | null;
  columns: ColumnStat[]; computed_at: string | null;
}
export interface CatalogEntry {
  id: number; name: string; comment: string; owner: string; dialect: string;
  storage_location: string; node_id: string; schema_count: number;
  properties: Record<string, string>; created_at: string; updated_at: string;
}
export interface SchemaEntry {
  id: number; catalog: string; name: string; full_name: string; comment: string;
  table_count: number; properties: Record<string, string>; created_at: string; updated_at: string;
}
export interface TableEntry {
  id: number; catalog: string; schema: string; name: string; full_name: string;
  object_type: string; definition: string;
  table_type: string; format: string; source_url: string; node: string | null;
  comment: string; columns: ColumnSpec[]; statistics: TableStatistics;
  replicas: string[];
  properties: Record<string, string>; created_at: string; updated_at: string;
}
export interface SqlColumn { name: string; dtype: string; }
export interface SqlResult {
  node_id: string; columns: SqlColumn[]; rows: (string | number | boolean | null)[][];
  row_count: number; truncated: boolean; elapsed_ms: number;
  plan_sql: string; referenced_tables: string[];
}
export interface ExplainResult {
  node_id: string; dialect: string; plan: string; plan_sql: string;
  referenced_tables: string[]; statement: string;
}

const sagaNode = (node?: string) => (node ? `?node=${encodeURIComponent(node)}` : "");

export function getCatalogs(node?: string, fresh = false): Promise<{ node_id: string; catalogs: CatalogEntry[] }> {
  return cachedGet<{ node_id: string; catalogs: CatalogEntry[] }>(`/api/v2/saga/catalog${sagaNode(node)}`, TTL.DEFINITION, jsonFetch, fresh);
}

export function getSchemas(catalog: string, node?: string, fresh = false): Promise<{ node_id: string; catalog: string; schemas: SchemaEntry[] }> {
  return cachedGet<{ node_id: string; catalog: string; schemas: SchemaEntry[] }>(`/api/v2/saga/catalog/${encodeURIComponent(catalog)}/schema${sagaNode(node)}`, TTL.DEFINITION, jsonFetch, fresh);
}

export function getTables(catalog: string, schema: string, node?: string, fresh = false): Promise<{ node_id: string; catalog: string; schema: string; tables: TableEntry[] }> {
  return cachedGet<{ node_id: string; catalog: string; schema: string; tables: TableEntry[] }>(`/api/v2/saga/catalog/${encodeURIComponent(catalog)}/schema/${encodeURIComponent(schema)}/table${sagaNode(node)}`, TTL.DEFINITION, jsonFetch, fresh);
}

export function getTable(catalog: string, schema: string, name: string, node?: string): Promise<{ table: TableEntry }> {
  return jsonFetch(`/api/v2/saga/catalog/${encodeURIComponent(catalog)}/schema/${encodeURIComponent(schema)}/table/${encodeURIComponent(name)}${sagaNode(node)}`);
}

export async function createCatalog(body: { name: string; comment?: string; dialect?: string }): Promise<{ catalog: CatalogEntry }> {
  const r = await jsonFetch<{ catalog: CatalogEntry }>("/api/v2/saga/catalog", { method: "POST", body: JSON.stringify(body) });
  invalidate("saga/catalog");
  return r;
}

export async function createSchema(catalog: string, body: { name: string; comment?: string }): Promise<{ schema: SchemaEntry }> {
  const r = await jsonFetch<{ schema: SchemaEntry }>(`/api/v2/saga/catalog/${encodeURIComponent(catalog)}/schema`, { method: "POST", body: JSON.stringify(body) });
  invalidate("saga/catalog");
  return r;
}

export async function createTable(catalog: string, schema: string, body: { name: string; source_url: string; node?: string | null; table_type?: string; comment?: string; infer?: boolean }): Promise<{ table: TableEntry }> {
  const r = await jsonFetch<{ table: TableEntry }>(`/api/v2/saga/catalog/${encodeURIComponent(catalog)}/schema/${encodeURIComponent(schema)}/table`, { method: "POST", body: JSON.stringify(body) });
  invalidate("saga/catalog");
  return r;
}

export async function refreshTable(catalog: string, schema: string, name: string): Promise<{ table: TableEntry }> {
  const r = await jsonFetch<{ table: TableEntry }>(`/api/v2/saga/catalog/${encodeURIComponent(catalog)}/schema/${encodeURIComponent(schema)}/table/${encodeURIComponent(name)}/refresh`, { method: "POST", body: "{}" });
  invalidate("saga/catalog");
  return r;
}

export async function deleteCatalogEntity(path: string): Promise<void> {
  await jsonFetch(`/api/v2/saga/${path}`, { method: "DELETE" });
  invalidate("saga/catalog");
}

// One-shot: ensure catalog + schema, infer the table name from the file, and
// register + profile it. Used by the Files page's "register in Saga" action.
// The current user on this node — used to default-name a registered view.
export function getMe(): Promise<UserCard> {
  return jsonFetch<UserCard>("/api/v2/user/me");
}

export async function registerFile(body: { source_url: string; catalog?: string; schema?: string; table?: string; object_type?: string; definition?: string; node?: string | null }): Promise<{ table: TableEntry }> {
  const r = await jsonFetch<{ table: TableEntry }>("/api/v2/saga/register", { method: "POST", body: JSON.stringify(body) });
  invalidate("saga/catalog");
  return r;
}

export async function discoverTables(body: { catalog: string; schema: string; path?: string; node?: string | null; recursive?: boolean }): Promise<{ tables: TableEntry[] }> {
  const r = await jsonFetch<{ tables: TableEntry[] }>("/api/v2/saga/discover", { method: "POST", body: JSON.stringify(body) });
  invalidate("saga/catalog");
  return r;
}

export function runSql(body: { sql: string; dialect?: string; catalog?: string; schema?: string; node?: string; limit?: number }): Promise<SqlResult> {
  return jsonFetch<SqlResult>("/api/v2/saga/sql", { method: "POST", body: JSON.stringify(body) });
}

export function explainSql(body: { sql: string; dialect?: string; catalog?: string; schema?: string }): Promise<ExplainResult> {
  return jsonFetch<ExplainResult>("/api/v2/saga/explain", { method: "POST", body: JSON.stringify(body) });
}

export interface PlanOp {
  id: string; op: string; title: string; detail: string; inputs: string[];
  rows: number | null; elapsed_ms: number | null;
}
export interface PlanGraph {
  node_id: string; dialect: string; statement: string; plan_sql: string;
  ops: PlanOp[]; analyzed: boolean; total_ms: number | null; sampled: boolean;
}
export type PlanEdit = { op: string; value?: number };

export function getPlan(body: { sql: string; dialect?: string; catalog?: string; schema?: string }, analyze = false): Promise<PlanGraph> {
  return jsonFetch<PlanGraph>(`/api/v2/saga/plan${analyze ? "?analyze=true" : ""}`, { method: "POST", body: JSON.stringify(body) });
}

export function editPlan(body: { sql: string; dialect?: string; catalog?: string; schema?: string; edits: PlanEdit[] }): Promise<{ node_id: string; sql: string; plan_sql: string }> {
  return jsonFetch("/api/v2/saga/plan/edit", { method: "POST", body: JSON.stringify(body) });
}

export interface SessionResult { node_id: string; path: string; columns: SqlColumn[]; row_count: number; elapsed_ms: number }
export interface SagaFilter { column: string; op: string; value?: unknown }
export interface WindowTransform { op: "explode" | "unnest"; column: string }
// Stage a heavy result to an Arrow IPC file for lazy windowed scrolling.
export function createSession(body: { sql: string; dialect?: string; catalog?: string; schema?: string; node?: string }): Promise<SessionResult> {
  return jsonFetch<SessionResult>("/api/v2/saga/sql.session", { method: "POST", body: JSON.stringify(body) });
}
// Clear a staged session (best-effort; also fired via sendBeacon on unload).
export function closeSession(path: string, node?: string): void {
  const url = `/api/v2/saga/session/close?path=${encodeURIComponent(path)}${node ? `&node=${encodeURIComponent(node)}` : ""}`;
  try { if (navigator.sendBeacon) { navigator.sendBeacon(url); return; } } catch { /* fall through */ }
  fetch(url, { method: "POST", keepalive: true }).catch(() => {});
}

export interface MaterializeResult { node_id: string; path: string; columns: SqlColumn[]; row_count: number; elapsed_ms: number }
// Run a query once and write it to a tmp parquet, returning a node path so the
// path-based /tabular + /analysis surfaces can analyse a SQL result.
export function materializeSql(body: { sql: string; dialect?: string; catalog?: string; schema?: string; node?: string }): Promise<MaterializeResult> {
  return jsonFetch<MaterializeResult>("/api/v2/saga/sql.materialize", { method: "POST", body: JSON.stringify(body) });
}

export const SQL_EXPORT_FORMATS = ["csv", "parquet", "json", "ndjson", "arrow", "xlsx"] as const;

// Run the query on the node where the data lives and download the full result
// in any handled media type. max_rows null = the whole result.
export async function downloadSqlExport(body: { sql: string; fmt: string; dialect?: string; catalog?: string; schema?: string; node?: string; max_rows?: number | null }): Promise<void> {
  const res = await fetch("/api/v2/saga/sql.export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) {
    let detail = ""; try { detail = (await res.json()).detail; } catch { /* ignore */ }
    throw new Error(`export failed: HTTP ${res.status}${detail ? ` — ${detail}` : ""}`);
  }
  const blob = await res.blob();
  const name = res.headers.get("Content-Disposition")?.match(/filename="([^"]+)"/)?.[1] ?? `result.${body.fmt}`;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

export interface OpLogEntry {
  ts: string; op: string; user: string; node: string;
  statement: string; rows: number | null; detail: string;
}

export function getTableLog(catalog: string, schema: string, name: string, node?: string, limit = 100): Promise<{ node_id: string; asset: string; entries: OpLogEntry[] }> {
  const q = `limit=${limit}${node ? `&node=${encodeURIComponent(node)}` : ""}`;
  return jsonFetch(`/api/v2/saga/catalog/${encodeURIComponent(catalog)}/schema/${encodeURIComponent(schema)}/table/${encodeURIComponent(name)}/log?${q}`);
}

export interface ReplicateResult {
  source_node: string; target_node: string; full_name: string;
  mode: string; bytes_copied: number; target_source_url: string;
}

export async function replicateTable(body: { catalog: string; schema: string; table: string; target: string; mode: "metadata" | "data" }): Promise<ReplicateResult> {
  const r = await jsonFetch<ReplicateResult>("/api/v2/saga/replicate", { method: "POST", body: JSON.stringify(body) });
  invalidate("saga/catalog");
  return r;
}

export interface PyFuncParam { name: string; annotation: string; dtype: string; default: string | null; has_default: boolean }
export interface PyFuncInferResult {
  name: string; signature: string; params: PyFuncParam[];
  return_annotation: string; return_dtype: string;
  dependencies: string[]; python_version: string; docstring: string;
}
// Scan code → typed signature + version-pinned deps. Powers live infer in the
// function editor.
export function inferFunc(code: string, name?: string, pin_versions = true): Promise<PyFuncInferResult> {
  return jsonFetch<PyFuncInferResult>("/api/v2/pyfunc/infer", { method: "POST", body: JSON.stringify({ code, name, pin_versions }) });
}

export const OBJECT_TYPES = ["TABLE", "VIEW", "FUNCTION", "MODEL", "OTHER"] as const;

export interface SearchHit {
  kind: string; name: string; full_name: string; object_type: string;
  catalog: string; schema: string; comment: string;
}
export function searchSaga(q: string, limit = 50, node?: string): Promise<{ node_id: string; query: string; hits: SearchHit[]; total: number; truncated: boolean }> {
  const u = `q=${encodeURIComponent(q)}&limit=${limit}${node ? `&node=${encodeURIComponent(node)}` : ""}`;
  return jsonFetch(`/api/v2/saga/search?${u}`);
}

export interface ActivityResponse {
  node_id: string; asset: string; op_counts: Record<string, number>;
  total_ops: number; last_op_at: string | null; daily: number[]; recent: OpLogEntry[];
}
export function getActivity(catalog: string, schema: string, name: string, node?: string): Promise<ActivityResponse> {
  const q = node ? `?node=${encodeURIComponent(node)}` : "";
  return jsonFetch(`/api/v2/saga/catalog/${encodeURIComponent(catalog)}/schema/${encodeURIComponent(schema)}/table/${encodeURIComponent(name)}/activity${q}`);
}

export async function updateTable(catalog: string, schema: string, name: string, body: Partial<{ source_url: string; object_type: string; definition: string; comment: string; table_type: string; node: string | null; properties: Record<string, string> }>): Promise<{ table: TableEntry }> {
  const r = await jsonFetch<{ table: TableEntry }>(`/api/v2/saga/catalog/${encodeURIComponent(catalog)}/schema/${encodeURIComponent(schema)}/table/${encodeURIComponent(name)}`, { method: "PATCH", body: JSON.stringify(body) });
  invalidate("saga/catalog");
  return r;
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
