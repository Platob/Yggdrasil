const BASE = "/api";

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

// -- Python execution --
export interface PythonResponse {
  id: string;
  node_id: string;
  returncode: number | null;
  stdout: string | null;
  stderr: string | null;
  result: unknown;
  duration: number | null;
  status: string;
}

export function executePython(code: string): Promise<PythonResponse> {
  return fetchJSON("/python", {
    method: "POST",
    body: JSON.stringify({ code }),
  });
}

// -- Shell commands --
export interface CmdResponse {
  id: string;
  node_id: string;
  command: string[];
  returncode: number | null;
  stdout: string | null;
  stderr: string | null;
  duration: number | null;
  status: string;
}

export function executeCmd(command: string[]): Promise<CmdResponse> {
  return fetchJSON("/cmd", {
    method: "POST",
    body: JSON.stringify({ command }),
  });
}

// -- Messenger --
export interface Message {
  id: string;
  sender: string;
  text: string;
  channel: string;
  timestamp: string;
  node_id: string;
}

export interface ChannelInfo {
  name: string;
  created_at: string;
  last_active: string;
  message_count: number;
  members: string[];
}

export function sendMessage(text: string, sender: string, channel: string): Promise<Message> {
  return fetchJSON("/messenger", {
    method: "POST",
    body: JSON.stringify({ text, sender, channel }),
  });
}

export async function getChannels(): Promise<ChannelInfo[]> {
  const data = await fetchJSON<{ channels: ChannelInfo[] }>("/messenger/channels");
  return data.channels;
}

export async function getMessages(channel: string, limit = 100): Promise<Message[]> {
  const data = await fetchJSON<{ messages: Message[] }>(
    `/messenger/channels/${channel}/messages?limit=${limit}`
  );
  return data.messages;
}

export async function pollMessages(
  channel: string,
  afterId: string,
  timeout = 25
): Promise<Message[]> {
  const data = await fetchJSON<{ messages: Message[] }>(
    `/messenger/channels/${channel}/poll?after_id=${afterId}&timeout=${timeout}`
  );
  return data.messages;
}

export async function createChannel(name: string): Promise<ChannelInfo> {
  const data = await fetchJSON<{ channel: ChannelInfo }>(
    `/messenger/channels?name=${encodeURIComponent(name)}`,
    { method: "POST" }
  );
  return data.channel;
}

// -- Node info --
export interface NodeInfo {
  node_id: string;
  host: string;
  port: number;
  version: string;
  uptime: number;
  channels: string[];
  functions: string[];
}

export function getNodeInfo(): Promise<NodeInfo> {
  return fetchJSON("/hello");
}

export async function getRegistry(): Promise<Record<string, string>> {
  return fetchJSON("/call/registry");
}
