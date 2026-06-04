// TypeScript mirrors of the FastAPI v2 schemas.
// Loose where the backend response shape is unstable.

// ── Common ──────────────────────────────────────────────────────────────────

export type NodeRole = "driver" | "executor" | "hybrid";

// ── Card (GET /api/card) ────────────────────────────────────────────────────

export interface NodeCard {
  node_id: string;
  host: string;
  port: number;
  url: string;
  role: NodeRole;
  version: string;
  hostname: string;
  platform: string;
  python_version: string;
  lat: number | null;
  lon: number | null;
  cpu_count: number;
  cpu_percent: number;
  memory_used_mb: number;
  memory_total_mb: number;
  gpu_count: number;
  active_runs: number;
  total_runs: number;
  env_count: number;
  func_count: number;
  uptime_seconds: number;
  node_home: string;
  peers: string[];
  content_hash: string;
}

// ── Backend (GET /api/v2/backend, SSE /api/v2/backend/stream) ───────────────

export interface GpuInfo {
  index: number;
  name: string;
  memory_used_mb: number;
  memory_total_mb: number;
  utilization_percent: number;
  temperature_c: number;
  power_draw_w: number;
  power_limit_w: number;
}

export interface NetworkIO {
  bytes_sent: number;
  bytes_recv: number;
  packets_sent: number;
  packets_recv: number;
}

export interface NodeBackend {
  node_id: string;
  role: NodeRole;
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

// The SSE stream sends the raw snapshot, but the page checks `.backend` first;
// keep both shapes optional so both layouts work.
export interface BackendStreamEvent {
  type?: string;
  backend: NodeBackend;
}

// ── Network / Peers (GET /api/v2/network/peers) ─────────────────────────────

export interface NodeMeta {
  node_id: string;
  host: string;
  port: number;
  role: NodeRole;
  version: string;
  lat: number | null;
  lon: number | null;
  cpu_percent: number;
  memory_percent: number;
  active_runs: number;
  gpu_count: number;
}

// ── Topology (GET /api/v2/topology) ─────────────────────────────────────────

export interface TopologyNode {
  node_id: string;
  host: string;
  port: number;
  role: NodeRole;
  cpu_percent: number;
  memory_percent: number;
  active_runs: number;
  gpu_count: number;
  self: boolean;
  lat: number | null;
  lon: number | null;
}

export interface TopologyResponse {
  nodes: TopologyNode[];
  total_cpu_percent: number;
  total_active_runs: number;
  total_gpus: number;
}

// ── Cluster stats (GET /api/v2/stats) ───────────────────────────────────────

export interface ClusterStats {
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

// ── Health (GET /api/v2/health) ─────────────────────────────────────────────

export interface HealthCheck {
  status: string;
  [key: string]: unknown;
}

export interface HealthResponse {
  status: string;
  node_id: string;
  checks: Record<string, HealthCheck>;
}

// ── Metrics (GET /api/v2/metrics) ───────────────────────────────────────────

export interface MetricsTopByRuns {
  id: number;
  name: string;
  runs: number;
}

export interface MetricsTopByDuration {
  id: number;
  name: string;
  avg_ms: number;
}

export interface MetricsSuccessRate {
  id: number;
  name: string;
  rate: number;
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
  top_by_runs: MetricsTopByRuns[];
  top_by_duration: MetricsTopByDuration[];
  success_rate: MetricsSuccessRate[];
  recent_runs: MetricsRecentRun[];
}

// ── Audit (GET /api/v2/audit) ───────────────────────────────────────────────

export interface AuditEntry {
  timestamp: string;
  operation: string;
  asset_type: string;
  asset_id: number;
  detail: string;
  user_hash?: string;
  // Backend may add fields over time; allow extras.
  [key: string]: unknown;
}

// ── PyEnv ──────────────────────────────────────────────────────────────────

export interface PyEnvEntry {
  id: number;
  name: string;
  python_version: string;
  dependencies: string[];
  path: string;
  status: string;
  created_at: string;
  updated_at: string;
  error: string | null;
  last_used_at: string | null;
  content_hash: string;
  replicated_at: string | null;
  replicated_from: string | null;
  env_vars: Record<string, string>;
}

export interface PyEnvEnvVars {
  env_id: number;
  name: string;
  env_vars: Record<string, string>;
}

export interface PyEnvPackage {
  name: string;
  version: string;
}

export interface PyEnvPackages {
  env_id: number;
  name: string;
  python_version: string;
  package_count: number;
  packages: PyEnvPackage[];
  cached_at: string;
  error: string | null;
}

// ── Excel service ────────────────────────────────────────────────────────

export interface ExcelInfo {
  node_id: string;
  node_name: string;
  version: string;
  table_formats: string[];
  capabilities: string[];
}

// ── PyFunc ─────────────────────────────────────────────────────────────────

export interface PyFuncEntry {
  id: number;
  name: string;
  code: string;
  description: string;
  python_version: string | null;
  dependencies: string[];
  env_id: number | null;
  run_count: number;
  created_at: string;
  updated_at: string;
  last_run_at: string | null;
  content_hash: string;
  replicated_at: string | null;
  replicated_from: string | null;
  avg_duration_ms: number;
  last_duration_ms: number;
  success_count: number;
  failure_count: number;
}

// ── PyFuncRun ──────────────────────────────────────────────────────────────

export interface PyFuncRunEntry {
  id: number;
  func_id: number;
  env_id: number | null;
  status: string;
  args: unknown[];
  kwargs: Record<string, unknown>;
  started_at: string | null;
  completed_at: string | null;
  duration: number | null;
  returncode: number | null;
  stdout: string | null;
  stderr: string | null;
  result: unknown;
  result_type: string | null;
  error: string | null;
  node_id: string;
  progress: number;
  log_lines: number;
  pid: number | null;
  heartbeat_at: string | null;
  cancellation_requested: boolean;
  stdout_truncated: boolean;
  stderr_truncated: boolean;
}

// ── DAG ────────────────────────────────────────────────────────────────────

export interface DAGNodeRef {
  node_url: string | null;
  func_id: number | null;
  env_id: number | null;
  args: Record<string, unknown>;
}

export interface DAGEdge {
  from_step: string;
  to_step: string;
  output_key: string;
  input_key: string;
}

export interface DAGStep {
  id: string;
  ref: DAGNodeRef;
  depends_on: string[];
}

export interface DAGEntry {
  id: number;
  name: string;
  description: string;
  steps: DAGStep[];
  edges: DAGEdge[];
  created_at: string;
  updated_at: string;
  run_count: number;
  content_hash: string;
  replicated_at: string | null;
  replicated_from: string | null;
  schedule_interval: number | null;
  schedule_max_runs: number | null;
  schedule_active: boolean;
}

export interface DAGRunEntry {
  id: number;
  dag_id: number;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  duration: number | null;
  step_results: Record<string, unknown>;
  node_id: string;
}

// ── Filesystem (GET /api/v2/fs/ls, /read) ──────────────────────────────────

export interface FsEntry {
  path: string;
  name: string;
  is_dir: boolean;
  size: number;
  modified_at: string;
}

// ── Messenger ──────────────────────────────────────────────────────────────

export interface ChannelInfo {
  name: string;
  message_count: number;
  last_message_at: string | null;
  members: string[];
}

export interface Message {
  id: number;
  channel: string;
  user_id: number;
  user_key: string;
  content: string;
  timestamp: string;
  node_id: string;
}

// ── User ───────────────────────────────────────────────────────────────────

export interface UserCard {
  user_id: number;
  key: string;
  hostname: string;
  email: string | null;
  first_name: string | null;
  last_name: string | null;
  node_id: string;
  role: string;
  online: boolean;
  last_seen_at: string;
}

// ── Analysis: technical indicators / compare / AI ───────────────────────────

export interface IndicatorsResult {
  node_id: string;
  path: string;
  column: string;
  index: (string | number)[];
  value: (number | null)[];
  rsi: (number | null)[];
  macd_line: (number | null)[];
  macd_signal_line: (number | null)[];
  macd_hist: (number | null)[];
  bb_upper: (number | null)[];
  bb_mid: (number | null)[];
  bb_lower: (number | null)[];
  atr: (number | null)[] | null;
  truncated: boolean;
  source_rows: number;
}

export interface CompareSeriesInput {
  path: string;
  column: string;
  label?: string;
  order_by?: string;
}

export interface CompareResult {
  node_id: string;
  labels: string[];
  index: (string | number)[];
  values: (number | null)[][];
  correlation: (number | null)[][] | null;
}

export interface AiSummaryResult {
  node_id: string;
  path: string;
  summary: string;
  key_points: string[];
  chart_hint: string | null;
  model: string;
  error?: string | null;
}
