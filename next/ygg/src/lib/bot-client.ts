import type { NodeInfo, ChannelInfo } from "./api";

const BOT_API_URL = process.env.BOT_API_URL || `http://127.0.0.1:${process.env.YGG_BOT_PORT || "8100"}`;
const BOT_API_TIMEOUT = Number(process.env.BOT_API_TIMEOUT || "30000");

export class BotAPIError extends Error {
  constructor(public status: number, public body: string) {
    super(`Bot API ${status}: ${body}`);
    this.name = "BotAPIError";
  }
}

export async function botFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BOT_API_URL}${path}`;
  const res = await fetch(url, {
    ...init,
    signal: AbortSignal.timeout(BOT_API_TIMEOUT),
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new BotAPIError(res.status, body);
  }
  return res.json();
}

export const botAPI = {
  getNodeInfo: (): Promise<NodeInfo> => botFetch("/api/hello"),

  getRegistry: (): Promise<Record<string, string>> => botFetch("/api/call/registry"),

  getChannels: async (): Promise<ChannelInfo[]> => {
    const data = await botFetch<{ channels: ChannelInfo[] }>("/api/messenger/channels");
    return data.channels;
  },
};
