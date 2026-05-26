// ── API Client for Yggdrasil Bot Backend ────────────────────────────────────

export interface NodeInfo {
  node_id: string;
  host: string;
  port: number;
  version: string;
  uptime: number;
  channels: string[];
  functions: string[];
}

export interface SystemMetrics {
  cpu: number;
  ram: number;
  ramUsed: number;
  ramTotal: number;
  gpu: number;
  gpuMemUsed: number;
  gpuMemTotal: number;
  gpuName: string;
}

export interface ProcessInfo {
  pid: number;
  name: string;
  cpu: number;
  ram: number;
  status: string;
}

export interface PythonResponse {
  id: string;
  status: string;
  stdout: string | null;
  stderr: string | null;
  result: unknown;
  duration: number | null;
  returncode: number | null;
}

export interface CmdResponse {
  id: string;
  status: string;
  stdout: string | null;
  stderr: string | null;
  duration: number | null;
  returncode: number | null;
}

export interface ChannelInfo {
  name: string;
  members?: string[] | number;
  messageCount?: number;
  message_count?: number;
}

export interface Message {
  id: string;
  channel: string;
  sender: string;
  text: string;
  timestamp: number;
}

// ── Messaging API Functions ──────────────────────────────────────────────────

export async function getChannels(): Promise<ChannelInfo[]> {
  try {
    const res = await fetch("/api/bot/channels");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  } catch (err) {
    console.error("[v0] getChannels failed:", err);
    return [];
  }
}

export async function getMessages(channel: string, limit = 50): Promise<Message[]> {
  try {
    const res = await fetch(`/api/bot/channels/${channel}/messages?limit=${limit}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  } catch (err) {
    console.error("[v0] getMessages failed:", err);
    return [];
  }
}

export async function sendMessage(text: string, sender: string, channel: string): Promise<Message | null> {
  try {
    const res = await fetch(`/api/bot/channels/${channel}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, sender }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  } catch (err) {
    console.error("[v0] sendMessage failed:", err);
    return null;
  }
}

export async function pollMessages(
  channel: string,
  lastId: string,
  timeout = 30000
): Promise<Message[]> {
  try {
    const res = await fetch(
      `/api/bot/channels/${channel}/messages/poll?last_id=${lastId}&timeout=${timeout}`,
      { signal: AbortSignal.timeout(timeout + 1000) }
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  } catch (err) {
    console.error("[v0] pollMessages failed:", err);
    return [];
  }
}

export async function createChannel(name: string): Promise<ChannelInfo | null> {
  try {
    const res = await fetch("/api/bot/channels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  } catch (err) {
    console.error("[v0] createChannel failed:", err);
    return null;
  }
}

// ── Bot API Client ───────────────────────────────────────────────────────────

class BotClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = typeof window !== "undefined" 
      ? `${window.location.protocol}//${window.location.host}/api/bot`
      : "http://localhost:8100/api";
  }

  async getNodeInfo(): Promise<NodeInfo> {
    try {
      const res = await fetch(`${this.baseUrl}/node`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    } catch (err) {
      console.error("[v0] getNodeInfo failed:", err);
      throw err;
    }
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    try {
      const res = await fetch(`${this.baseUrl}/metrics`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    } catch (err) {
      console.error("[v0] getSystemMetrics failed:", err);
      throw err;
    }
  }

  async getProcesses(): Promise<ProcessInfo[]> {
    try {
      const res = await fetch(`${this.baseUrl}/processes`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    } catch (err) {
      console.error("[v0] getProcesses failed:", err);
      throw err;
    }
  }

  async executeCode(code: string): Promise<string> {
    try {
      const res = await fetch(`${this.baseUrl}/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return data.result || "";
    } catch (err) {
      console.error("[v0] executeCode failed:", err);
      throw err;
    }
  }

  async executePython(code: string): Promise<PythonResponse> {
    try {
      const res = await fetch(`${this.baseUrl}/execute/python`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    } catch (err) {
      console.error("[v0] executePython failed:", err);
      throw err;
    }
  }

  async executeCmd(args: string[]): Promise<CmdResponse> {
    try {
      const res = await fetch(`${this.baseUrl}/execute/cmd`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ args }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    } catch (err) {
      console.error("[v0] executeCmd failed:", err);
      throw err;
    }
  }
}

export const bot = new BotClient();

// Standalone execution functions for convenience
export const executePython = (code: string) => bot.executePython(code);
export const executeCmd = (args: string[]) => bot.executeCmd(args);
