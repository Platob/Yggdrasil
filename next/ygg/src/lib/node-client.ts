import type { NodeInfo, ChannelInfo } from "./api";

const NODE_API_URL = process.env.NODE_API_URL || `http://127.0.0.1:${process.env.YGG_NODE_PORT || "8100"}`;
const NODE_API_TIMEOUT = Number(process.env.NODE_API_TIMEOUT || "30000");

export class NodeAPIError extends Error {
  constructor(public status: number, public body: string) {
    super(`Bot API ${status}: ${body}`);
    this.name = "NodeAPIError";
  }
}

export async function nodeFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${NODE_API_URL}${path}`;
  const res = await fetch(url, {
    ...init,
    signal: AbortSignal.timeout(NODE_API_TIMEOUT),
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new NodeAPIError(res.status, body);
  }
  return res.json();
}

export const nodeAPI = {
  getNodeInfo: (): Promise<NodeInfo> => nodeFetch("/api/hello"),

  getRegistry: (): Promise<Record<string, string>> => nodeFetch("/api/call/registry"),

  getChannels: async (): Promise<ChannelInfo[]> => {
    const data = await nodeFetch<{ channels: ChannelInfo[] }>("/api/messenger/channels");
    return data.channels;
  },
};
