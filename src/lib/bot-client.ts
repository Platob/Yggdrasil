/**
 * Server-side Bot API client (for Next.js route handlers only).
 * Uses absolute URL so it works in server context.
 */

const BOT_BASE = process.env.BOT_API_URL ?? "http://127.0.0.1:8100";

export class BotAPIError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: string
  ) {
    super(`Bot API error ${status}: ${body}`);
    this.name = "BotAPIError";
  }
}

export async function botFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BOT_BASE}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    next: { revalidate: 0 },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new BotAPIError(res.status, body);
  }
  return res.json();
}

import type { NodeInfo, ChannelInfo } from "./api";

export const botAPI = {
  getNodeInfo: () => botFetch<NodeInfo>("/api/hello"),
  getChannels: async () => {
    const res = await botFetch<{ channels: ChannelInfo[] }>("/api/messenger/channels");
    return res.channels;
  },
  getRegistry: () => botFetch<Record<string, string>>("/api/call/registry"),
  getPrices: () => botFetch<{ prices: Record<string, { symbol: string; price: number; currency: string; source: string; timestamp: string; stale: boolean }>; timestamp: string }>("/api/trading/prices"),
};
