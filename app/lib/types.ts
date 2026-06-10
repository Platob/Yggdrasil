export interface HealthResponse {
  status: string;
}

export interface StatsResponse {
  node_id: string;
  uptime_s: number;
  messages: number;
  functions: number;
}

export interface BackendResponse {
  backend: string;
  version: string;
}

export interface AuditEntry {
  id?: string | number;
  timestamp?: string;
  ts?: string;
  action?: string;
  event?: string;
  user?: string;
  detail?: string;
  message?: string;
  level?: string;
  [key: string]: unknown;
}

export interface PyFunc {
  name: string;
  description?: string;
  args?: string[];
  [key: string]: unknown;
}

export interface Message {
  id?: string | number;
  text: string;
  sender: string;
  channel: string;
  timestamp?: string;
  ts?: string;
  [key: string]: unknown;
}

export interface Channel {
  id?: string;
  name: string;
  [key: string]: unknown;
}

export interface FsEntry {
  name: string;
  path: string;
  type: "file" | "dir" | "directory" | string;
  size?: number;
  modified?: string;
  mtime?: string;
  [key: string]: unknown;
}

export interface FsLsResponse {
  path: string;
  entries: FsEntry[];
}

export interface AggregateRequest {
  path: string;
  group_by: string;
  measure: string;
  agg: "sum" | "mean" | "count" | "min" | "max";
}

export interface AggregateResult {
  columns: string[];
  rows: (string | number | null)[][];
}

export interface OhlcRequest {
  path: string;
  column: string;
  buckets?: number;
}

export interface OhlcCandle {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface OhlcResult {
  candles: OhlcCandle[];
}

export interface SeriesRequest {
  path: string;
  column: string;
  points?: number;
}

export interface SeriesPoint {
  x: string | number;
  y: number;
}

export interface SeriesResult {
  series: SeriesPoint[];
  column: string;
}

export interface ForecastRequest {
  path: string;
  column: string;
  x_column?: string;
  horizon?: number;
  model?: string;
}

export interface ForecastPoint {
  x: string | number;
  actual?: number;
  forecast?: number;
}

export interface ForecastResult {
  points: ForecastPoint[];
  column: string;
  model: string;
}
