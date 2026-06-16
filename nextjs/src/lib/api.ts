const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}/api${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  ping: () => get<{ pong: boolean; version: string }>("/ping"),
  stats: () => get<{ uptime: number; requests: number; memory_mb: number }>("/v2/stats"),
  health: () => get<{ status: string; uptime: number }>("/v2/health"),
  backend: () => get<{ node_id: string; version: string; engines: string[] }>("/v2/backend"),
  audit: (limit = 50) => get<{ entries: unknown[] }>(`/v2/audit?limit=${limit}`),

  // messenger
  channels: () => get<{ channels: { name: string; message_count: number }[] }>("/messenger/channels"),
  messages: (channel: string, limit = 50) =>
    get<{ messages: { id: string; text: string; sender: string; timestamp: string }[] }>(
      `/messenger/channels/${encodeURIComponent(channel)}/messages?limit=${limit}`,
    ),
  sendMessage: (text: string, sender: string, channel = "general") =>
    post<{ message: { id: string } }>("/messenger", { text, sender, channel }),

  // analysis
  aggregate: (body: unknown) => post<{ rows: unknown[]; group_count: number }>("/v2/analysis/aggregate", body),
  ohlc: (body: unknown) => post<{ bars: number; data: unknown[] }>("/v2/analysis/ohlc", body),
  series: (body: unknown) => post<{ x: unknown[]; y: unknown[] }>("/v2/analysis/series", body),
  forecast: (body: unknown) => post<{ model_used: string; series: unknown[] }>("/v2/analysis/forecast", body),
};
