// HTTP transport for the Yggdrasil node.
// Server-side calls hit BOT_API_URL directly; browser-side calls hit the
// Next.js /api proxy of the same shape so cookies / CSRF stay tidy.

const SERVER_BASE = process.env.BOT_API_URL ?? "http://127.0.0.1:8100";

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

function resolveBase(): string {
  return isBrowser() ? "" : SERVER_BASE;
}

export class BotAPIError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, detail: string, body?: unknown) {
    super(`[${status}] ${detail}`);
    this.name = "BotAPIError";
    this.status = status;
    this.body = body;
  }
}

export async function botFetch<T = unknown>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const url = `${resolveBase()}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      "Accept": "application/json",
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    let body: unknown = null;
    try { body = await res.json(); } catch {}
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : res.statusText;
    throw new BotAPIError(res.status, detail, body);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** Open an EventSource against an SSE endpoint. Browser-only. */
export function botStream(path: string): EventSource {
  if (!isBrowser()) {
    throw new Error("botStream is browser-only");
  }
  return new EventSource(`${resolveBase()}${path}`);
}

export const botAPI = {
  get: <T = unknown>(p: string) => botFetch<T>(p),
  post: <T = unknown>(p: string, body?: unknown) =>
    botFetch<T>(p, { method: "POST", body: body !== undefined ? JSON.stringify(body) : undefined }),
  delete: <T = unknown>(p: string) => botFetch<T>(p, { method: "DELETE" }),
  put: <T = unknown>(p: string, body?: unknown) =>
    botFetch<T>(p, { method: "PUT", body: body !== undefined ? JSON.stringify(body) : undefined }),
  stream: botStream,

  // Convenience aggregations used by Next.js route handlers.
  async getNodeInfo() {
    return botFetch<NodeInfoRaw>("/api/hello");
  },
  async getRegistry() {
    try { return await botFetch<Record<string, string>>("/api/call"); }
    catch { return {}; }
  },
  async getChannels() {
    const resp = await botFetch<{ channels: ChannelInfo[] }>("/api/messenger/channels");
    return resp.channels;
  },
};

// Minimal shape — full version is in lib/api.ts. Matches NodeInfo so the
// Next.js route handlers can pass the result straight through without casts.
type NodeInfoRaw = {
  node_id: string;
  host: string;
  port: number;
  version: string;
  uptime: number;
  channels: string[];
  functions: string[];
  lat?: number | null;
  lon?: number | null;
};

type ChannelInfo = {
  name: string;
  message_count: number;
  members: string[];
  last_active: string;
  created_at: string;
};
