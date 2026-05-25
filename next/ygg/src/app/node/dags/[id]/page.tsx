"use client";

import { useEffect, useState, use } from "react";
import { node as api, type DagEntry, type DagRunEntry } from "@/lib/api";
import { formatRelative, formatDuration } from "@/lib/time";
import Link from "next/link";

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

// -- Per-step result detail --
interface StepResult {
  status?: string;
  stdout?: string | null;
  stderr?: string | null;
  result?: unknown;
  duration?: number | null;
  error?: string | null;
}

function StepResultDetail({ stepId, data }: { stepId: string; data: StepResult }) {
  const [expanded, setExpanded] = useState(false);
  const stepStatus = data.status || "unknown";

  return (
    <div className="border border-border/50 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-3 py-2 flex items-center gap-3 text-left hover:bg-card-hover transition-colors"
      >
        <span className={statusDotClass(stepStatus)} />
        <span className="font-mono text-xs font-medium text-foreground">{stepId}</span>
        <span
          className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded"
          style={{ color: statusColor(stepStatus), background: `color-mix(in srgb, ${statusColor(stepStatus)} 15%, transparent)` }}
        >
          {stepStatus}
        </span>
        {data.duration != null && (
          <span className="ml-auto text-xs font-mono text-muted">{formatDuration(data.duration)}</span>
        )}
        <svg
          width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          className="text-muted shrink-0 transition-transform ml-1"
          style={{ transform: expanded ? "rotate(180deg)" : "rotate(0)" }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {expanded && (
        <div className="border-t border-border/50 px-3 pb-3 pt-2 space-y-2">
          {data.stdout && (
            <div>
              <span className="text-[10px] uppercase tracking-wider text-muted">stdout</span>
              <pre className="bg-[#0d1117] border border-border rounded-lg p-2 mt-1 text-xs overflow-x-auto whitespace-pre-wrap font-mono text-foreground-dim">{data.stdout}</pre>
            </div>
          )}
          {data.stderr && (
            <div>
              <span className="text-[10px] uppercase tracking-wider text-destructive">stderr</span>
              <pre className="bg-[#1a0a0a] border border-destructive/20 rounded-lg p-2 mt-1 text-xs overflow-x-auto whitespace-pre-wrap font-mono text-destructive/80">{data.stderr}</pre>
            </div>
          )}
          {data.result != null && (
            <div>
              <span className="text-[10px] uppercase tracking-wider text-muted">result</span>
              <pre className="code-block p-2 mt-1 text-xs overflow-x-auto whitespace-pre-wrap font-mono">{JSON.stringify(data.result, null, 2)}</pre>
            </div>
          )}
          {data.error && (
            <div>
              <span className="text-[10px] uppercase tracking-wider text-destructive">error</span>
              <p className="text-xs text-destructive mt-1">{data.error}</p>
            </div>
          )}
          {!data.stdout && !data.stderr && data.result == null && !data.error && (
            <p className="text-xs text-muted">No output captured for this step.</p>
          )}
        </div>
      )}
    </div>
  );
}

export default function DagDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const numericId = parseInt(id, 10);
  const [dag, setDag] = useState<DagEntry | null>(null);
  const [runs, setRuns] = useState<DagRunEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [expandedRun, setExpandedRun] = useState<number | null>(null);

  useEffect(() => {
    loadData();
  }, [numericId]);

  async function loadData() {
    setLoading(true);
    setError(null);
    try {
      const [dagData, runsData] = await Promise.all([
        api.getDag(numericId),
        api.listDagRuns(numericId),
      ]);
      setDag(dagData.dag);
      setRuns(runsData.runs);
    } catch {
      setError("Failed to load DAG");
    }
    setLoading(false);
  }

  async function handleRun() {
    setRunning(true);
    try {
      await api.runDag(numericId);
      await loadData();
    } catch (e) {
      setError(`Run failed: ${e}`);
    }
    setRunning(false);
  }

  async function handleDelete() {
    if (!confirm("Delete this DAG? This cannot be undone.")) return;
    try {
      await api.deleteDag(numericId);
      window.location.href = "/node/dags";
    } catch (e) {
      setError(`Delete failed: ${e}`);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="text-center animate-in">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full spin-slow mx-auto mb-4" />
          <p className="text-primary font-mono text-sm pulse-primary">Loading DAG...</p>
        </div>
      </div>
    );
  }

  if (!dag) {
    return (
      <div className="p-6 animate-in">
        <p className="text-muted">{error || "DAG not found."}</p>
        <Link href="/node/dags" className="btn-ghost text-sm mt-4 inline-block">Back to DAGs</Link>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 animate-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <Link href="/node/dags" className="text-muted hover:text-foreground transition-colors text-sm">
              DAGs
            </Link>
            <span className="text-muted">/</span>
            <h1 className="text-xl font-bold text-foreground font-mono">{dag.name}</h1>
          </div>
          <p className="text-sm text-muted">{dag.description}</p>
        </div>
        <div className="flex items-center gap-3">
          {error && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-warning/10 border border-warning/20">
              <div className="status-dot pending" />
              <span className="text-xs font-medium text-warning">{error}</span>
            </div>
          )}
          <button onClick={handleRun} disabled={running} className="btn-primary text-sm">
            {running ? "Running..." : "Run DAG"}
          </button>
          <button onClick={handleDelete} className="btn-ghost text-sm text-destructive hover:bg-destructive/10">
            Delete
          </button>
        </div>
      </div>

      {/* Metadata */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "ID", value: String(dag.id) },
          { label: "Steps", value: String(dag.steps.length) },
          { label: "Edges", value: String(dag.edges.length) },
          { label: "Run Count", value: String(dag.run_count) },
        ].map((item) => (
          <div key={item.label} className="nordic-card p-3">
            <span className="text-[10px] uppercase tracking-wider text-muted">{item.label}</span>
            <p className="text-sm font-mono text-foreground mt-0.5">{item.value}</p>
          </div>
        ))}
      </div>

      {/* Timestamps */}
      <div className="flex gap-6 text-xs text-muted">
        <span>Created: {new Date(dag.created_at).toLocaleString()}</span>
        <span>Updated: {new Date(dag.updated_at).toLocaleString()}</span>
      </div>

      {/* Step Flow */}
      <div>
        <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Step Flow</h2>
        {dag.steps.length === 0 ? (
          <div className="nordic-card p-6 text-center">
            <p className="text-sm text-muted">No steps defined. Add steps to build the pipeline.</p>
          </div>
        ) : (
          <div className="space-y-2">
            {dag.steps.map((step, idx) => (
              <div key={step.id} className="relative">
                {/* Arrow from previous */}
                {idx > 0 && (
                  <div className="flex justify-center py-1">
                    <svg width="16" height="20" viewBox="0 0 16 20" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-primary/50">
                      <line x1="8" y1="0" x2="8" y2="16" />
                      <polyline points="4 12 8 16 12 12" />
                    </svg>
                  </div>
                )}
                <div className="nordic-card p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                        {step.id}
                      </span>
                      {step.depends_on.length > 0 && (
                        <span className="text-[10px] text-muted">
                          depends on: {step.depends_on.join(", ")}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-x-4 gap-y-1 text-xs">
                    <div>
                      <span className="text-muted">Function</span>
                      <p className="font-mono text-foreground-dim">#{step.ref.function_id}</p>
                    </div>
                    {step.ref.environment_id != null && (
                      <div>
                        <span className="text-muted">Environment</span>
                        <p className="font-mono text-foreground-dim">#{step.ref.environment_id}</p>
                      </div>
                    )}
                    {step.ref.node_url && (
                      <div>
                        <span className="text-muted">Node</span>
                        <p className="font-mono text-foreground-dim text-[11px] truncate">{step.ref.node_url}</p>
                      </div>
                    )}
                  </div>
                  {Object.keys(step.ref.args).length > 0 && (
                    <div className="mt-2">
                      <span className="text-[10px] text-muted">Args:</span>
                      <pre className="text-[10px] font-mono text-foreground-dim mt-0.5">{JSON.stringify(step.ref.args, null, 2)}</pre>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Edges */}
      {dag.edges.length > 0 && (
        <div>
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Data Edges</h2>
          <div className="nordic-card p-4">
            <div className="space-y-2">
              {dag.edges.map((edge, idx) => (
                <div key={idx} className="flex items-center gap-2 text-xs">
                  <span className="font-mono px-1.5 py-0.5 rounded bg-border/50 text-foreground-dim">
                    {edge.from_step}
                  </span>
                  <span className="text-muted">.{edge.output_key}</span>
                  <svg width="16" height="10" viewBox="0 0 16 10" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-primary/60">
                    <line x1="0" y1="5" x2="12" y2="5" />
                    <polyline points="9 2 12 5 9 8" />
                  </svg>
                  <span className="font-mono px-1.5 py-0.5 rounded bg-border/50 text-foreground-dim">
                    {edge.to_step}
                  </span>
                  <span className="text-muted">.{edge.input_key}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Run History */}
      <div>
        <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Run History</h2>
        {runs.length === 0 ? (
          <div className="nordic-card p-6 text-center">
            <p className="text-sm text-muted">No runs yet. Click &ldquo;Run DAG&rdquo; to execute.</p>
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
                      <span className="font-mono text-xs text-foreground-dim">#{run.id}</span>
                      <span
                        className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded"
                        style={{ color: statusColor(run.status), background: `color-mix(in srgb, ${statusColor(run.status)} 15%, transparent)` }}
                      >
                        {run.status}
                      </span>
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

                {expandedRun === run.id && (
                  <div className="border-t border-border px-4 pb-4 pt-3 space-y-3">
                    <span className="text-[10px] uppercase tracking-wider text-muted">Per-Step Results</span>
                    {Object.keys(run.step_results).length > 0 ? (
                      <div className="space-y-2 mt-2">
                        {Object.entries(run.step_results).map(([stepId, stepData]) => (
                          <StepResultDetail
                            key={stepId}
                            stepId={stepId}
                            data={(stepData as StepResult) || {}}
                          />
                        ))}
                      </div>
                    ) : (
                      <p className="text-xs text-muted mt-1">No step results captured.</p>
                    )}
                    <div className="flex gap-4 text-[10px] text-muted pt-2">
                      {run.started_at && <span>Started: {formatRelative(run.started_at)}</span>}
                      {run.completed_at && <span>Completed: {formatRelative(run.completed_at)}</span>}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Back button */}
      <div>
        <Link href="/node/dags" className="btn-ghost text-sm inline-block">
          Back to DAGs
        </Link>
      </div>
    </div>
  );
}
