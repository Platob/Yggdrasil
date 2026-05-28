// Server-side bot client. Only call from Next.js route handlers / RSC.
// Hits the FastAPI node directly via BOT_API_URL (no rewrite proxy involved).
//
// Browser code must use src/lib/api.ts instead, which goes through the
// /api/bot/* rewrite defined in next.config.ts.

import type { NodeInfo, ChannelInfo, ChannelListResponse } from "@/lib/api";

const BOT_API_URL = process.env.BOT_API_URL ?? "http://127.0.0.1:8100";

export class BotAPIError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, body: unknown, message?: string) {
    super(message ?? `Bot API error ${status}`);
    this.name = "BotAPIError";
    this.status = status;
    this.body = body;
  }
}

export async function botFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BOT_API_URL}${path.startsWith("/") ? path : `/${path}`}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    let body: unknown = null;
    try { body = await res.json(); } catch { try { body = await res.text(); } catch {} }
    throw new BotAPIError(res.status, body, `Bot ${res.status} on ${path}`);
  }
  return (await res.json()) as T;
}

export const botAPI = {
  async getNodeInfo(): Promise<NodeInfo> {
    return botFetch<NodeInfo>("/api/hello");
  },
  async getRegistry(): Promise<Record<string, string>> {
    return botFetch<Record<string, string>>("/api/call/registry");
  },
  async getChannels(): Promise<ChannelInfo[]> {
    const resp = await botFetch<ChannelListResponse>("/api/messenger/channels");
    return resp.channels;
  },
};
