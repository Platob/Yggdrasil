import type {
  HealthResponse,
  StatsResponse,
  BackendResponse,
  AuditEntry,
  PyFunc,
  Message,
  Channel,
  FsLsResponse,
  AggregateRequest,
  AggregateResult,
  OhlcRequest,
  OhlcResult,
  SeriesRequest,
  SeriesResult,
  ForecastRequest,
  ForecastResult,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8100";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function getPing(): Promise<{ pong: boolean }> {
  return apiFetch<{ pong: boolean }>("/api/ping");
}

export async function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/v2/health");
}

export async function getStats(): Promise<StatsResponse> {
  return apiFetch<StatsResponse>("/api/v2/stats");
}

export async function getBackend(): Promise<BackendResponse> {
  return apiFetch<BackendResponse>("/api/v2/backend");
}

export async function getAudit(limit = 20): Promise<AuditEntry[]> {
  const data = await apiFetch<AuditEntry[] | { entries: AuditEntry[] }>(
    `/api/v2/audit?limit=${limit}`
  );
  return Array.isArray(data) ? data : (data.entries ?? []);
}

export async function getPyfunc(): Promise<PyFunc[]> {
  const data = await apiFetch<PyFunc[] | { functions: PyFunc[] }>(
    "/api/v2/pyfunc"
  );
  return Array.isArray(data) ? data : (data.functions ?? []);
}

export async function sendMessage(
  text: string,
  sender = "user",
  channel = "general"
): Promise<Message> {
  return apiFetch<Message>("/api/messenger", {
    method: "POST",
    body: JSON.stringify({ text, sender, channel }),
  });
}

export async function getChannels(): Promise<Channel[]> {
  const data = await apiFetch<Channel[] | { channels: Channel[] }>(
    "/api/messenger/channels"
  );
  return Array.isArray(data) ? data : (data.channels ?? []);
}

export async function getMessages(
  channel: string,
  limit = 50
): Promise<Message[]> {
  const data = await apiFetch<Message[] | { messages: Message[] }>(
    `/api/messenger/channels/${encodeURIComponent(channel)}/messages?limit=${limit}`
  );
  return Array.isArray(data) ? data : (data.messages ?? []);
}

export async function lsDir(path = ""): Promise<FsLsResponse> {
  const data = await apiFetch<FsLsResponse | { path: string; entries: unknown[] }>(
    `/fs/ls?path=${encodeURIComponent(path)}`
  );
  return data as FsLsResponse;
}

export async function aggregate(req: AggregateRequest): Promise<AggregateResult> {
  return apiFetch<AggregateResult>("/api/v2/analysis/aggregate", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function ohlc(req: OhlcRequest): Promise<OhlcResult> {
  return apiFetch<OhlcResult>("/api/v2/analysis/ohlc", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function series(req: SeriesRequest): Promise<SeriesResult> {
  return apiFetch<SeriesResult>("/api/v2/analysis/series", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function forecast(req: ForecastRequest): Promise<ForecastResult> {
  return apiFetch<ForecastResult>("/api/v2/analysis/forecast", {
    method: "POST",
    body: JSON.stringify(req),
  });
}
