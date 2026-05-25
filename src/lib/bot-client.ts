import type { NodeInfo, ChannelInfo } from "./api";

const BOT_API_URL = process.env.BOT_API_URL || "http://127.0.0.1:8100";
const BOT_API_TIMEOUT = Number(process.env.BOT_API_TIMEOUT) || 30_000;

export class BotAPIError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: string,
  ) {
    super(`Bot API ${status}: ${body}`);
    this.name = "BotAPIError";
  }
}

const inflightRequests = new Map<string, Promise<unknown>>();

export async function botFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const method = init?.method ?? "GET";
  const dedupeKey = method === "GET" ? `${method}:${path}` : null;

  if (dedupeKey) {
    const inflight = inflightRequests.get(dedupeKey);
    if (inflight) return inflight as Promise<T>;
  }

  const promise = botFetchInner<T>(path, init);

  if (dedupeKey) {
    inflightRequests.set(dedupeKey, promise);
    promise.finally(() => inflightRequests.delete(dedupeKey));
  }

  return promise;
}

async function botFetchInner<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const url = `${BOT_API_URL}${path}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), BOT_API_TIMEOUT);

  try {
    const res = await fetch(url, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        ...init?.headers,
      },
    });

    if (!res.ok) {
      const body = await res.text();
      throw new BotAPIError(res.status, body);
    }

    return res.json();
  } finally {
    clearTimeout(timeout);
  }
}

export const botAPI = {
  getNodeInfo: (): Promise<NodeInfo> => botFetch("/api/hello"),

  getRegistry: (): Promise<Record<string, string>> =>
    botFetch("/api/call/registry"),

  getChannels: async (): Promise<ChannelInfo[]> => {
    const data = await botFetch<{ channels: ChannelInfo[] }>(
      "/api/messenger/channels",
    );
    return data.channels;
  },
};
