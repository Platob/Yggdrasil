export interface FxQuote {
  source: string;
  target: string;
  from_timestamp: string;
  to_timestamp: string;
  sampling: string;
  value: number;
}

export interface OhlcBar {
  open: number;
  high: number;
  low: number;
  close: number;
  index?: number;
}

export interface Message {
  id: string;
  text: string;
  sender: string;
  channel: string;
  timestamp: string;
}

export interface Channel {
  name: string;
  message_count: number;
  created_at: string;
}

export interface SystemStats {
  uptime: number;
  requests: number;
  memory_mb: number;
  cpu_pct?: number;
}

export interface BackendInfo {
  node_id: string;
  version: string;
  engines: string[];
  status: string;
}

export interface AuditEntry {
  id: number;
  action: string;
  resource_type: string;
  resource_id: unknown;
  detail: string;
  timestamp: string;
}

export interface AggMeasure {
  column: string;
  agg: "mean" | "sum" | "min" | "max" | "count";
}

export interface AggregateRequest {
  path: string;
  group_by: string[];
  measures: AggMeasure[];
}

export interface ForecastSeries {
  group?: string;
  rmse: number;
  predictions: number[];
}

export interface ForecastResult {
  model_used: string;
  series: ForecastSeries[];
}
