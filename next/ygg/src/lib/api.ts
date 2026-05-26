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
