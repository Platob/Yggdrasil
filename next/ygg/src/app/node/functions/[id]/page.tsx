"use client";

import { useEffect, useState, useRef, use } from "react";
import { node as api, type FunctionEntry, type RunEntry } from "@/lib/api";
import Link from "next/link";

// ── Demo data ────────────────────────────────────────────────
const DEMO_FUNCTION: FunctionEntry = {
  id: "fn-002",
  name: "fetch_metrics",
  language: "python",
  code: 'import psutil\nimport json\n\ndef main():\n    """Collect system metrics from the node."""\n    cpu = psutil.cpu_percent(interval=1)\n    mem = psutil.virtual_memory()\n    disk = psutil.disk_usage("/")\n    return {\n        "cpu_percent": cpu,\n        "ram_percent": mem.percent,\n        "ram_used_gb": round(mem.used / 1e9, 2),\n        "disk_percent": disk.percent,\n    }',
  description: "Collect system metrics from the node including CPU, RAM, and disk usage",
  python_version: "3.12",
  dependencies: ["psutil"],
  environment_id: "env-001",
  creator: "admin",
  created_at: "2025-05-18T08:30:00Z",
  updated_at: "2025-05-22T14:00:00Z",
  run_count: 156,
};

const DEMO_RUNS: RunEntry[] = [
  {
    id: "run-001",
    function_id: "fn-002",
    environment_id: "env-001",
    status: "completed",
    started_at: "2025-05-24T10:00:00Z",
    completed_at: "2025-05-24T10:00:02Z",
    duration: 2.1,
    returncode: 0,
    stdout: '{"cpu_percent": 23.5, "ram_percent": 45.2, "ram_used_gb": 14.46, "disk_percent": 62.1}',
    stderr: null,
    result: { cpu_percent: 23.5, ram_percent: 45.2 },
    node_id: "ygg-node-alpha",
  },
  {
    id: "run-002",
    function_id: "fn-002",
    environment_id: "env-001",
    status: "completed",
    started_at: "2025-05-24T09:00:00Z",
    completed_at: "2025-05-24T09:00:01Z",
    duration: 1.8,
    returncode: 0,
    stdout: '{"cpu_percent": 18.2, "ram_percent": 42.8, "ram_used_gb": 13.7, "disk_percent": 62.0}',
    stderr: null,
    result: { cpu_percent: 18.2, ram_percent: 42.8 },
    node_id: "ygg-node-alpha",
  },
  {
    id: "run-003",
    function_id: "fn-002",
    environment_id: "env-001",
    status: "failed",
    started_at: "2025-05-23T22:00:00Z",
    completed_at: "2025-05-23T22:00:00Z",
    duration: 0.3,
    returncode: 1,
    stdout: null,
    stderr: "ModuleNotFoundError: No module named 'psutil'",
    result: null,
    node_id: "ygg-node-alpha",
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

export default function FunctionDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [fn, setFn] = useState<FunctionEntry | null>(null);
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [expandedRun, setExpandedRun] = useState<string | null>(null);
  const [streamingLogs, setStreamingLogs] = useState<string>("");
  const [streamingRunId, setStreamingRunId] = useState<string | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadData();
  }, [id]);

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [streamingLogs]);

  async function loadData() {
    setLoading(true);
    setError(null);
    try {
      const [fnData, runsData] = await Promise.all([
        api.getFunction(id),
        api.listFunctionRuns(id),
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

  async function handleRun() {
    setRunning(true);
    setStreamingLogs("");
    setStreamingRunId(null);
    try {
      const data = await api.runFunction(id);
      const runId = data.run.id;
      setStreamingRunId(runId);

      // Try SSE streaming for logs
      try {
        const eventSource = new EventSource(`/api/node/run/${runId}/logs`);
        eventSource.onmessage = (event) => {
          try {
            const logData = JSON.parse(event.data);
            if (logData.line) {
              setStreamingLogs((prev) => prev + logData.line + "\n");
            }
            if (logData.done) {
              eventSource.close();
              loadData();
            }
          } catch {
            setStreamingLogs((prev) => prev + event.data + "\n");
          }
        };
        eventSource.onerror = () => {
          eventSource.close();
          // If SSE fails, just show the result from the run response
          if (data.run.stdout) {
            setStreamingLogs(data.run.stdout);
          }
          loadData();
        };
      } catch {
        // SSE not available, show result directly
        if (data.run.stdout) {
          setStreamingLogs(data.run.stdout);
        }
        loadData();
      }
    } catch (e) {
      setStreamingLogs(`Error: ${e}`);
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

      {/* Streaming Logs */}
      {(streamingLogs || streamingRunId) && (
        <div>
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-2">
            {running ? "Live Output" : "Last Run Output"}
          </h2>
          <div className="code-block p-4 max-h-64 overflow-y-auto">
            <pre className="text-xs whitespace-pre-wrap">{streamingLogs || "Waiting for output..."}</pre>
            <div ref={logsEndRef} />
          </div>
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
                      <span className="font-mono text-xs text-foreground-dim truncate">{run.id}</span>
                      <span
                        className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded"
                        style={{ color: statusColor(run.status), background: `color-mix(in srgb, ${statusColor(run.status)} 15%, transparent)` }}
                      >
                        {run.status}
                      </span>
                    </div>
                  </div>
                  <div className="text-right text-xs text-muted shrink-0">
                    {run.duration != null && <span className="font-mono">{run.duration.toFixed(1)}s</span>}
                    {run.started_at && (
                      <div className="text-[10px] mt-0.5">{new Date(run.started_at).toLocaleString()}</div>
                    )}
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

                {expandedRun === run.id && (
                  <div className="border-t border-border px-4 pb-4 pt-3 space-y-3">
                    {run.stdout && (
                      <div>
                        <span className="text-[10px] uppercase tracking-wider text-muted">stdout</span>
                        <pre className="code-block p-3 mt-1 text-xs overflow-x-auto whitespace-pre-wrap">{run.stdout}</pre>
                      </div>
                    )}
                    {run.stderr && (
                      <div>
                        <span className="text-[10px] uppercase tracking-wider text-destructive">stderr</span>
                        <pre className="code-block p-3 mt-1 text-xs overflow-x-auto whitespace-pre-wrap text-destructive/80">{run.stderr}</pre>
                      </div>
                    )}
                    {!run.stdout && !run.stderr && (
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
