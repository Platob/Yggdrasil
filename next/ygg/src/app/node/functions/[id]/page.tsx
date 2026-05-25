"use client";

import { useEffect, useState, useRef, use, useCallback } from "react";
import { node as api, type FunctionEntry, type RunEntry } from "@/lib/api";
import { formatRelative, formatDuration } from "@/lib/time";
import Link from "next/link";

// -- Demo data --
const DEMO_FUNCTION: FunctionEntry = {
  id: 2,
  name: "fetch_metrics",
  language: "python",
  code: 'import psutil\nimport json\n\ndef main():\n    """Collect system metrics from the node."""\n    cpu = psutil.cpu_percent(interval=1)\n    mem = psutil.virtual_memory()\n    disk = psutil.disk_usage("/")\n    return {\n        "cpu_percent": cpu,\n        "ram_percent": mem.percent,\n        "ram_used_gb": round(mem.used / 1e9, 2),\n        "disk_percent": disk.percent,\n    }',
  description: "Collect system metrics from the node including CPU, RAM, and disk usage",
  python_version: "3.12",
  dependencies: ["psutil"],
  environment_id: 1,
  creator: "admin",
  created_at: "2025-05-18T08:30:00Z",
  updated_at: "2025-05-22T14:00:00Z",
  run_count: 156,
  deleted_at: null,
  last_used_at: "2025-05-24T10:00:00Z",
  state: "ready",
};

const DEMO_RUNS: RunEntry[] = [
  {
    id: 1,
    function_id: 2,
    environment_id: 1,
    status: "completed",
    started_at: "2025-05-24T10:00:00Z",
    completed_at: "2025-05-24T10:00:02Z",
    duration: 2.1,
    returncode: 0,
    stdout: '{"cpu_percent": 23.5, "ram_percent": 45.2, "ram_used_gb": 14.46, "disk_percent": 62.1}',
    stderr: null,
    result: { cpu_percent: 23.5, ram_percent: 45.2 },
    node_id: "ygg-node-alpha",
    max_memory_mb: 512,
    max_cpu_percent: 80,
    timeout: 30,
  },
  {
    id: 2,
    function_id: 2,
    environment_id: 1,
    status: "completed",
    started_at: "2025-05-24T09:00:00Z",
    completed_at: "2025-05-24T09:00:01Z",
    duration: 1.8,
    returncode: 0,
    stdout: '{"cpu_percent": 18.2, "ram_percent": 42.8, "ram_used_gb": 13.7, "disk_percent": 62.0}',
    stderr: null,
    result: { cpu_percent: 18.2, ram_percent: 42.8 },
    node_id: "ygg-node-alpha",
    max_memory_mb: null,
    max_cpu_percent: null,
    timeout: null,
  },
  {
    id: 3,
    function_id: 2,
    environment_id: 1,
    status: "failed",
    started_at: "2025-05-23T22:00:00Z",
    completed_at: "2025-05-23T22:00:00Z",
    duration: 0.3,
    returncode: 1,
    stdout: null,
    stderr: "ModuleNotFoundError: No module named 'psutil'",
    result: null,
    node_id: "ygg-node-alpha",
    max_memory_mb: null,
    max_cpu_percent: null,
    timeout: null,
  },
];

function statusColor(status: string): string {
  switch (status) {
    case "completed": return "var(--success)";
    case "running": return "var(--warning)";
    case "failed": case "error": return "var(--destructive)";
    default: return "var(--muted)";
  }
}

function statusDotClass(status: string): string {
  switch (status) {
    case "completed": return "status-dot online";
    case "running": return "status-dot pending";
    case "failed": case "error": return "status-dot offline";
    default: return "status-dot";
  }
}

// -- SSE Log Streaming for running functions --
function RunningLogs({ runId, onComplete }: { runId: number; onComplete: () => void }) {
  const [logs, setLogs] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const es = new EventSource(`/api/node/run/${runId}/logs`);
    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data);
        if (event.type === "stdout" || event.type === "stderr") {
          setLogs((prev) => [...prev, event.line]);
        } else if (event.type === "complete") {
          es.close();
          onComplete();
        }
      } catch {
        setLogs((prev) => [...prev, e.data]);
      }
    };
    es.onerror = () => {
      es.close();
    };
    return () => es.close();
  }, [runId, onComplete]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className="code-block p-4 max-h-64 overflow-y-auto">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-2 h-2 rounded-full bg-warning animate-pulse" />
        <span className="text-[10px] uppercase tracking-wider text-warning">Live Streaming</span>
      </div>
      <pre className="text-xs whitespace-pre-wrap font-mono">
        {logs.length > 0 ? logs.join("\n") : "Waiting for output..."}
      </pre>
      <div ref={logsEndRef} />
    </div>
  );
}

export default function FunctionDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const numericId = parseInt(id, 10);
  const [fn, setFn] = useState<FunctionEntry | null>(null);
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [expandedRun, setExpandedRun] = useState<number | null>(null);
  const [streamingRunId, setStreamingRunId] = useState<number | null>(null);

  useEffect(() => {
    loadData();
  }, [numericId]);

  async function loadData() {
    setLoading(true);
    setError(null);
    try {
      const [fnData, runsData] = await Promise.all([
        api.getFunction(numericId),
        api.listFunctionRuns(numericId),
      ]);
      setFn(fnData.function);
      setRuns(runsData.runs);
    } catch {
      setError("Node unavailable - showing demo data");
      setFn(DEMO_FUNCTION);
      setRuns(DEMO_RUNS);
    }
    setLoading(false);
  }

  const refreshRuns = useCallback(async () => {
    try {
      const runsData = await api.listFunctionRuns(numericId);
      setRuns(runsData.runs);
    } catch {
      // ignore
    }
    setStreamingRunId(null);
  }, [numericId]);

  async function handleRun() {
    setRunning(true);
    try {
      const data = await api.runFunction(numericId);
      const runId = data.run.id;
      if (data.run.status === "running") {
        setStreamingRunId(runId);
      } else {
        await loadData();
      }
    } catch (e) {
      setError(`Run failed: ${e}`);
    }
    setRunning(false);
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="text-center animate-in">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full spin-slow mx-auto mb-4" />
          <p className="text-primary font-mono text-sm pulse-primary">Loading function...</p>
        </div>
      </div>
    );
  }

  if (!fn) {
    return (
      <div className="p-6 animate-in">
        <p className="text-muted">Function not found.</p>
        <Link href="/node/functions" className="btn-ghost text-sm mt-4 inline-block">Back to Functions</Link>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 animate-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <Link href="/node/functions" className="text-muted hover:text-foreground transition-colors text-sm">
              Functions
            </Link>
            <span className="text-muted">/</span>
            <h1 className="text-xl font-bold text-foreground font-mono">{fn.name}</h1>
          </div>
          <p className="text-sm text-muted">{fn.description}</p>
        </div>
        <div className="flex items-center gap-3">
          {error && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-warning/10 border border-warning/20">
              <div className="status-dot pending" />
              <span className="text-xs font-medium text-warning">{error}</span>
            </div>
          )}
          <button onClick={handleRun} disabled={running} className="btn-primary text-sm">
            {running ? "Running..." : "Run Function"}
          </button>
        </div>
      </div>

      {/* Metadata */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Language", value: fn.language },
          { label: "Python", value: fn.python_version || "default" },
          { label: "Creator", value: fn.creator },
          { label: "Run Count", value: String(fn.run_count) },
        ].map((item) => (
          <div key={item.label} className="nordic-card p-3">
            <span className="text-[10px] uppercase tracking-wider text-muted">{item.label}</span>
            <p className="text-sm font-mono text-foreground mt-0.5">{item.value}</p>
          </div>
        ))}
      </div>

      {/* Timestamps */}
      <div className="flex gap-6 text-xs text-muted">
        <span>Created: {new Date(fn.created_at).toLocaleString()}</span>
        <span>Updated: {new Date(fn.updated_at).toLocaleString()}</span>
        {fn.environment_id && <span>Env: <span className="font-mono text-primary">{fn.environment_id}</span></span>}
      </div>

      {/* Dependencies */}
      {fn.dependencies.length > 0 && (
        <div>
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-2">Dependencies</h2>
          <div className="flex flex-wrap gap-2">
            {fn.dependencies.map((dep) => (
              <span key={dep} className="text-xs font-mono px-2 py-1 rounded bg-border/50 text-foreground-dim border border-border">
                {dep}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Code */}
      <div>
        <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-2">Source Code</h2>
        <pre className="code-block p-4 overflow-x-auto whitespace-pre-wrap">{fn.code}</pre>
      </div>

      {/* Streaming Logs for active run */}
      {streamingRunId && (
        <div>
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-2">Live Output</h2>
          <RunningLogs runId={streamingRunId} onComplete={refreshRuns} />
        </div>
      )}

      {/* Run History */}
      <div>
        <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Run History</h2>
        {runs.length === 0 ? (
          <div className="nordic-card p-6 text-center">
            <p className="text-sm text-muted">No runs yet. Click "Run Function" to execute.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {runs.map((run) => (
              <div key={run.id} className="nordic-card overflow-hidden">
                <button
                  onClick={() => setExpandedRun(expandedRun === run.id ? null : run.id)}
                  className="w-full p-4 flex items-center gap-4 text-left hover:bg-card-hover transition-colors"
                >
                  <div className={statusDotClass(run.status)} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3">
                      <span
                        className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded"
                        style={{ color: statusColor(run.status), background: `color-mix(in srgb, ${statusColor(run.status)} 15%, transparent)` }}
                      >
                        {run.status}
                      </span>
                      <span className="font-mono text-xs text-foreground-dim">#{run.id}</span>
                    </div>
                  </div>
                  <div className="text-right text-xs text-muted shrink-0 space-y-0.5">
                    <div className="font-mono">{formatDuration(run.duration)}</div>
                    <div className="text-[10px]">{formatRelative(run.started_at)}</div>
                  </div>
                  <svg
                    width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                    className="text-muted shrink-0 transition-transform"
                    style={{ transform: expandedRun === run.id ? "rotate(180deg)" : "rotate(0)" }}
                  >
                    <polyline points="6 9 12 15 18 9" />
                  </svg>
                </button>

                {/* SSE for running runs */}
                {run.status === "running" && expandedRun === run.id && (
                  <div className="border-t border-border px-4 pb-4 pt-3">
                    <RunningLogs runId={run.id} onComplete={refreshRuns} />
                  </div>
                )}

                {/* Expanded details for non-running runs */}
                {expandedRun === run.id && run.status !== "running" && (
                  <div className="border-t border-border px-4 pb-4 pt-3 space-y-3">
                    {/* stdout */}
                    {run.stdout && (
                      <div>
                        <span className="text-[10px] uppercase tracking-wider text-muted">stdout</span>
                        <pre className="bg-[#0d1117] border border-border rounded-lg p-3 mt-1 text-xs overflow-x-auto whitespace-pre-wrap font-mono text-foreground-dim">{run.stdout}</pre>
                      </div>
                    )}

                    {/* stderr */}
                    {run.stderr && (
                      <div>
                        <span className="text-[10px] uppercase tracking-wider text-destructive">stderr</span>
                        <pre className="bg-[#1a0a0a] border border-destructive/20 rounded-lg p-3 mt-1 text-xs overflow-x-auto whitespace-pre-wrap font-mono text-destructive/80">{run.stderr}</pre>
                      </div>
                    )}

                    {/* result */}
                    {run.result != null && (
                      <div>
                        <span className="text-[10px] uppercase tracking-wider text-muted">result</span>
                        <pre className="code-block p-3 mt-1 text-xs overflow-x-auto whitespace-pre-wrap font-mono">{JSON.stringify(run.result, null, 2)}</pre>
                      </div>
                    )}

                    {/* error for failed */}
                    {(run.status === "failed" || run.status === "error") && !run.stderr && (
                      <div>
                        <span className="text-[10px] uppercase tracking-wider text-destructive">error</span>
                        <p className="text-xs text-destructive mt-1">Run failed with exit code {run.returncode ?? "unknown"}</p>
                      </div>
                    )}

                    {/* Resource limits */}
                    {(run.max_memory_mb != null || run.max_cpu_percent != null || run.timeout != null) && (
                      <div>
                        <span className="text-[10px] uppercase tracking-wider text-muted">resource limits</span>
                        <div className="flex gap-4 mt-1 text-xs">
                          {run.max_memory_mb != null && (
                            <span className="font-mono text-foreground-dim">Memory: {run.max_memory_mb}MB</span>
                          )}
                          {run.max_cpu_percent != null && (
                            <span className="font-mono text-foreground-dim">CPU: {run.max_cpu_percent}%</span>
                          )}
                          {run.timeout != null && (
                            <span className="font-mono text-foreground-dim">Timeout: {run.timeout}s</span>
                          )}
                        </div>
                      </div>
                    )}

                    {/* No output */}
                    {!run.stdout && !run.stderr && run.result == null && (
                      <p className="text-xs text-muted">No output captured.</p>
                    )}

                    <div className="flex gap-4 text-[10px] text-muted pt-1">
                      <span>Node: <span className="font-mono">{run.node_id}</span></span>
                      {run.returncode != null && <span>Exit code: <span className="font-mono">{run.returncode}</span></span>}
                      {run.environment_id && <span>Env: <span className="font-mono">{run.environment_id}</span></span>}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
