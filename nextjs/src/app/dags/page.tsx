"use client";

import { useEffect, useState, useCallback } from "react";
import {
  getDags,
  getDagRuns,
  createDag,
  deleteDag,
  runDag,
  scheduleDag,
  getFuncs,
  runFuncByName,
} from "@/lib/api";
import { ConfirmModal } from "@/components/ConfirmModal";
import type {
  DAGEntry,
  DAGRunEntry,
  PyFuncEntry,
} from "@/lib/types";

// ── Status badge ────────────────────────────────────────────
const RUN_STATUS_STYLES: Record<string, { bg: string; text: string }> = {
  completed: { bg: "rgba(52,211,153,0.1)",  text: "var(--emerald)" },
  success:   { bg: "rgba(52,211,153,0.1)",  text: "var(--emerald)" },
  failed:    { bg: "rgba(244,63,94,0.1)",   text: "var(--rose)" },
  error:     { bg: "rgba(244,63,94,0.1)",   text: "var(--rose)" },
  running:   { bg: "rgba(251,191,36,0.1)",  text: "var(--amber)" },
  pending:   { bg: "rgba(255,255,255,0.05)", text: "var(--muted)" },
  queued:    { bg: "rgba(255,255,255,0.05)", text: "var(--muted)" },
};

function StatusBadge({ status }: { status: string }) {
  const s = RUN_STATUS_STYLES[status] || RUN_STATUS_STYLES.pending;
  return (
    <span
      className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded inline-flex items-center gap-1"
      style={{ background: s.bg, color: s.text }}
    >
      {status === "running" && (
        <span className="relative flex h-1.5 w-1.5">
          <span className="absolute inline-flex h-full w-full rounded-full opacity-75" style={{ background: s.text, animation: "pulse-frost 1.5s ease-in-out infinite" }} />
          <span className="relative inline-flex rounded-full h-1.5 w-1.5" style={{ background: s.text }} />
        </span>
      )}
      {status}
    </span>
  );
}

function timeAgo(ts: string | null): string {
  if (!ts) return "--";
  try {
    const now = Date.now();
    const then = new Date(ts).getTime();
    const diffMs = now - then;
    if (diffMs < 0) return "just now";
    const diffSec = Math.floor(diffMs / 1000);
    if (diffSec < 60) return `${diffSec}s ago`;
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.floor(diffHr / 24);
    return `${diffDay}d ago`;
  } catch {
    return "--";
  }
}

function formatDuration(sec: number | null): string {
  if (sec == null) return "--";
  if (sec < 1) return `${(sec * 1000).toFixed(0)}ms`;
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}m ${s.toFixed(0)}s`;
}

function formatInterval(sec: number): string {
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h`;
  return `${Math.floor(sec / 86400)}d`;
}

// ── Builder step type ───────────────────────────────────────
interface BuilderStep {
  stepId: string;
  funcId: number | null;
  dependsOn: string[];
}

export default function DagsPage() {
  const [dags, setDags] = useState<DAGEntry[]>([]);
  const [funcs, setFuncs] = useState<PyFuncEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  // Expanded DAG runs
  const [expandedDag, setExpandedDag] = useState<number | null>(null);
  const [dagRuns, setDagRuns] = useState<DAGRunEntry[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);

  // Run result display
  const [runningDagId, setRunningDagId] = useState<number | null>(null);
  const [lastRunResult, setLastRunResult] = useState<DAGRunEntry | null>(null);

  // Schedule modal
  const [scheduleDagId, setScheduleDagId] = useState<number | null>(null);
  const [scheduleInterval, setScheduleInterval] = useState("60");
  const [scheduleMaxRuns, setScheduleMaxRuns] = useState("");
  const [scheduling, setScheduling] = useState(false);

  // Delete confirmation
  const [pendingDeleteId, setPendingDeleteId] = useState<number | null>(null);

  // Quick run modal
  const [quickRunOpen, setQuickRunOpen] = useState(false);
  const [quickRunName, setQuickRunName] = useState("");
  const [quickRunRunning, setQuickRunRunning] = useState(false);
  const [quickRunResult, setQuickRunResult] = useState<unknown>(null);
  const [quickRunError, setQuickRunError] = useState("");

  // Builder state
  const [builderName, setBuilderName] = useState("");
  const [builderDesc, setBuilderDesc] = useState("");
  const [builderSteps, setBuilderSteps] = useState<BuilderStep[]>([
    { stepId: "step_1", funcId: null, dependsOn: [] },
  ]);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState("");

  const fetchAll = useCallback(async (fresh = false) => {
    try {
      const [dagsRes, funcsRes] = await Promise.allSettled([
        getDags(fresh),
        getFuncs(fresh),
      ]);
      if (dagsRes.status === "fulfilled") setDags(dagsRes.value.dags);
      if (funcsRes.status === "fulfilled") setFuncs(funcsRes.value.funcs);
      const anyFailed = dagsRes.status === "rejected" && funcsRes.status === "rejected";
      if (anyFailed) setError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  // Fetch runs when a DAG is expanded
  const toggleDagExpand = async (dagId: number) => {
    if (expandedDag === dagId) {
      setExpandedDag(null);
      setDagRuns([]);
      return;
    }
    setExpandedDag(dagId);
    setLoadingRuns(true);
    try {
      const res = await getDagRuns(dagId);
      setDagRuns(res.runs);
    } catch {
      setDagRuns([]);
    } finally {
      setLoadingRuns(false);
    }
  };

  // Run a DAG
  const handleRunDag = async (dagId: number) => {
    setRunningDagId(dagId);
    setLastRunResult(null);
    try {
      const res = await runDag(dagId);
      setLastRunResult(res.run);
      // Refresh DAG list to update run_count
      fetchAll();
      // If this DAG is expanded, refresh its runs
      if (expandedDag === dagId) {
        const runsRes = await getDagRuns(dagId);
        setDagRuns(runsRes.runs);
      }
    } catch {
      // Run initiation failed
    } finally {
      setRunningDagId(null);
    }
  };

  // Delete a DAG
  const handleDeleteDag = async (dagId: number) => {
    try {
      await deleteDag(dagId);
      setDags((prev) => prev.filter((d) => d.id !== dagId));
      if (expandedDag === dagId) {
        setExpandedDag(null);
        setDagRuns([]);
      }
    } catch {
      // Delete failed silently
    }
  };

  // Schedule a DAG
  const handleSchedule = async () => {
    if (scheduleDagId == null || scheduling) return;
    const interval = parseInt(scheduleInterval, 10);
    if (isNaN(interval) || interval <= 0) return;
    const maxRuns = scheduleMaxRuns ? parseInt(scheduleMaxRuns, 10) : undefined;
    setScheduling(true);
    try {
      const res = await scheduleDag(scheduleDagId, interval, maxRuns);
      setDags((prev) => prev.map((d) => d.id === scheduleDagId ? res.dag : d));
      setScheduleDagId(null);
      setScheduleInterval("60");
      setScheduleMaxRuns("");
    } catch {
      // Schedule failed
    } finally {
      setScheduling(false);
    }
  };

  // Create a DAG
  const handleCreate = async () => {
    if (!builderName.trim()) {
      setCreateError("Name is required");
      return;
    }
    const validSteps = builderSteps.filter((s) => s.funcId != null);
    if (validSteps.length === 0) {
      setCreateError("At least one step with a function is required");
      return;
    }
    setCreating(true);
    setCreateError("");
    try {
      const steps = validSteps.map((s) => ({
        id: s.stepId,
        ref: {
          node_url: null,
          func_id: s.funcId,
          env_id: null,
          args: {},
        },
        depends_on: s.dependsOn.filter((dep) => validSteps.some((vs) => vs.stepId === dep)),
      }));
      // Build edges from depends_on relationships
      const edges: { from_step: string; to_step: string; output_key: string; input_key: string }[] = [];
      for (const step of steps) {
        for (const dep of step.depends_on) {
          edges.push({
            from_step: dep,
            to_step: step.id,
            output_key: "result",
            input_key: "input",
          });
        }
      }
      await createDag({
        name: builderName.trim(),
        description: builderDesc.trim(),
        steps,
        edges,
      });
      // Reset builder
      setBuilderName("");
      setBuilderDesc("");
      setBuilderSteps([{ stepId: "step_1", funcId: null, dependsOn: [] }]);
      // Refresh list
      fetchAll();
    } catch {
      setCreateError("Failed to create DAG");
    } finally {
      setCreating(false);
    }
  };

  // Quick run a function by name
  const handleQuickRun = async () => {
    const name = quickRunName.trim();
    if (!name || quickRunRunning) return;
    setQuickRunRunning(true);
    setQuickRunError("");
    setQuickRunResult(null);
    try {
      const result = await runFuncByName(name);
      setQuickRunResult(result);
    } catch (err) {
      setQuickRunError(err instanceof Error ? err.message : "Run failed");
    } finally {
      setQuickRunRunning(false);
    }
  };

  // Filter functions for autocomplete suggestions
  const quickRunSuggestions = quickRunName
    ? funcs
        .filter((f) => f.name.toLowerCase().includes(quickRunName.toLowerCase()))
        .slice(0, 5)
    : [];

  // Builder helpers
  const addStep = () => {
    const nextNum = builderSteps.length + 1;
    setBuilderSteps((prev) => [
      ...prev,
      { stepId: `step_${nextNum}`, funcId: null, dependsOn: [] },
    ]);
  };

  const removeStep = (index: number) => {
    if (builderSteps.length <= 1) return;
    const removedId = builderSteps[index].stepId;
    setBuilderSteps((prev) => {
      const next = prev.filter((_, i) => i !== index);
      // Clean up depends_on references
      return next.map((s) => ({
        ...s,
        dependsOn: s.dependsOn.filter((d) => d !== removedId),
      }));
    });
  };

  const updateStep = (index: number, patch: Partial<BuilderStep>) => {
    setBuilderSteps((prev) => prev.map((s, i) => i === index ? { ...s, ...patch } : s));
  };

  const toggleDependency = (stepIndex: number, depId: string) => {
    setBuilderSteps((prev) =>
      prev.map((s, i) => {
        if (i !== stepIndex) return s;
        const has = s.dependsOn.includes(depId);
        return {
          ...s,
          dependsOn: has
            ? s.dependsOn.filter((d) => d !== depId)
            : [...s.dependsOn, depId],
        };
      })
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center animate-in">
          <div className="w-8 h-8 border-2 border-frost/30 border-t-frost rounded-full spin-slow mx-auto mb-4" />
          <p className="text-sm text-muted font-mono">Loading DAGs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen animate-in">
      {/* ── Left panel: DAG list (2/3) ──────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground">DAGs</h1>
            <p className="text-sm text-muted mt-1">
              {dags.length} workflow{dags.length !== 1 ? "s" : ""} registered
            </p>
          </div>
          <button
            onClick={() => fetchAll(true)}
            className="
              px-3 py-1.5 rounded-lg text-xs font-medium
              text-frost/70 hover:text-frost
              bg-frost/5 hover:bg-frost/10
              border border-frost/10 hover:border-frost/20
              transition-all duration-150
            "
          >
            <svg className="inline-block mr-1.5 -mt-0.5" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="23 4 23 10 17 10" />
              <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" />
            </svg>
            Refresh
          </button>
        </div>

        {/* DAG cards */}
        {error && dags.length === 0 ? (
          <div className="text-center py-20">
            <div className="w-12 h-12 rounded-full bg-rose/10 flex items-center justify-center mx-auto mb-4">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--rose)" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
            </div>
            <p className="text-sm text-muted">Backend unreachable</p>
          </div>
        ) : dags.length === 0 ? (
          <div className="text-center py-20">
            <div className="w-12 h-12 rounded-full bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mx-auto mb-4">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="1.5">
                <circle cx="5" cy="6" r="3" />
                <circle cx="19" cy="6" r="3" />
                <circle cx="12" cy="18" r="3" />
                <path d="M7.5 8l3 7M16.5 8l-3 7" />
              </svg>
            </div>
            <p className="text-sm text-muted">No DAGs yet</p>
            <p className="text-xs text-muted/60 mt-1">Create one using the builder on the right</p>
          </div>
        ) : (
          <div className="space-y-4">
            {dags.map((dag) => (
              <div key={dag.id} className="glass-card overflow-hidden">
                {/* DAG card header */}
                <div className="p-5 space-y-3">
                  <div className="flex items-start justify-between">
                    <div className="min-w-0 flex-1">
                      <button
                        onClick={() => toggleDagExpand(dag.id)}
                        className="text-left group"
                      >
                        <h3 className="text-sm font-mono font-semibold text-foreground group-hover:text-frost transition-colors">
                          {dag.name}
                        </h3>
                      </button>
                      {dag.description && (
                        <p className="text-[11px] text-muted mt-0.5">{dag.description}</p>
                      )}
                    </div>
                    {/* Schedule status */}
                    {dag.schedule_active && dag.schedule_interval != null && (
                      <span
                        className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded shrink-0 ml-3"
                        style={{ background: "rgba(103,232,249,0.1)", color: "var(--frost)" }}
                      >
                        Every {formatInterval(dag.schedule_interval)}
                      </span>
                    )}
                  </div>

                  {/* Meta row */}
                  <div className="flex items-center gap-4 flex-wrap">
                    <span className="text-[10px] text-muted font-mono">
                      {dag.steps.length} step{dag.steps.length !== 1 ? "s" : ""}
                    </span>
                    <span className="text-[10px] text-muted font-mono">
                      {dag.run_count} run{dag.run_count !== 1 ? "s" : ""}
                    </span>
                    <span className="text-[10px] text-muted font-mono" title={dag.content_hash}>
                      #{dag.content_hash.slice(0, 8)}
                    </span>
                    <span className="text-[10px] text-muted font-mono">
                      {timeAgo(dag.updated_at)}
                    </span>
                  </div>

                  {/* Steps visualization */}
                  {dag.steps.length > 0 && (
                    <div className="flex items-center gap-1.5 flex-wrap">
                      {dag.steps.map((step, i) => {
                        const funcName = funcs.find((f) => f.id === step.ref.func_id)?.name ?? `func#${step.ref.func_id}`;
                        return (
                          <div key={step.id} className="flex items-center gap-1.5">
                            {i > 0 && (
                              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="2">
                                <polyline points="9 18 15 12 9 6" />
                              </svg>
                            )}
                            <span className="text-[10px] font-mono px-2 py-1 rounded bg-white/[0.04] border border-white/[0.06] text-foreground-dim">
                              {funcName}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex items-center gap-2 pt-1">
                    <button
                      onClick={() => handleRunDag(dag.id)}
                      disabled={runningDagId === dag.id}
                      className="
                        px-3 py-1.5 rounded-lg text-[11px] font-semibold
                        bg-emerald/10 text-emerald border border-emerald/20
                        hover:bg-emerald/20 hover:border-emerald/40
                        disabled:opacity-30 disabled:cursor-not-allowed
                        transition-all duration-150
                      "
                    >
                      {runningDagId === dag.id ? (
                        <span className="flex items-center gap-1.5">
                          <div className="w-3 h-3 border-2 border-emerald/30 border-t-emerald rounded-full spin-slow" />
                          Running...
                        </span>
                      ) : (
                        <span className="flex items-center gap-1.5">
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polygon points="5 3 19 12 5 21 5 3" />
                          </svg>
                          Run
                        </span>
                      )}
                    </button>
                    <button
                      onClick={() => setScheduleDagId(scheduleDagId === dag.id ? null : dag.id)}
                      className="
                        px-3 py-1.5 rounded-lg text-[11px] font-semibold
                        bg-frost/10 text-frost border border-frost/20
                        hover:bg-frost/20 hover:border-frost/40
                        transition-all duration-150
                      "
                    >
                      <span className="flex items-center gap-1.5">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <circle cx="12" cy="12" r="10" />
                          <polyline points="12 6 12 12 16 14" />
                        </svg>
                        Schedule
                      </span>
                    </button>
                    <button
                      onClick={() => toggleDagExpand(dag.id)}
                      className="
                        px-3 py-1.5 rounded-lg text-[11px] font-medium
                        text-foreground-dim hover:text-foreground
                        bg-white/[0.03] hover:bg-white/[0.06]
                        border border-white/[0.06]
                        transition-all duration-150
                      "
                    >
                      {expandedDag === dag.id ? "Hide Runs" : "Show Runs"}
                    </button>
                    <button
                      onClick={() => setPendingDeleteId(dag.id)}
                      className="
                        ml-auto px-3 py-1.5 rounded-lg text-[11px] font-medium
                        text-rose/60 hover:text-rose
                        hover:bg-rose/10
                        transition-all duration-150
                      "
                    >
                      Delete
                    </button>
                  </div>

                  {/* Schedule inline form */}
                  {scheduleDagId === dag.id && (
                    <div className="flex items-end gap-3 pt-2 border-t border-white/[0.04]">
                      <div className="space-y-1">
                        <label className="text-[10px] text-muted uppercase tracking-wider font-medium">Interval (seconds)</label>
                        <input
                          type="number"
                          value={scheduleInterval}
                          onChange={(e) => setScheduleInterval(e.target.value)}
                          min="1"
                          className="w-28 bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-1.5 text-xs font-mono text-foreground placeholder-muted/50 outline-none focus:border-frost/30 transition-colors"
                        />
                      </div>
                      <div className="space-y-1">
                        <label className="text-[10px] text-muted uppercase tracking-wider font-medium">Max Runs (optional)</label>
                        <input
                          type="number"
                          value={scheduleMaxRuns}
                          onChange={(e) => setScheduleMaxRuns(e.target.value)}
                          min="1"
                          placeholder="unlimited"
                          className="w-28 bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-1.5 text-xs font-mono text-foreground placeholder-muted/50 outline-none focus:border-frost/30 transition-colors"
                        />
                      </div>
                      <button
                        onClick={handleSchedule}
                        disabled={scheduling}
                        className="
                          px-4 py-1.5 rounded-lg text-[11px] font-semibold
                          bg-frost/10 text-frost border border-frost/20
                          hover:bg-frost/20 hover:border-frost/40
                          disabled:opacity-30 disabled:cursor-not-allowed
                          transition-all duration-150
                        "
                      >
                        {scheduling ? "Saving..." : "Save Schedule"}
                      </button>
                      <button
                        onClick={() => setScheduleDagId(null)}
                        className="px-3 py-1.5 rounded-lg text-[11px] font-medium text-muted hover:text-foreground transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                </div>

                {/* Last run result */}
                {lastRunResult && lastRunResult.dag_id === dag.id && (
                  <div className="border-t border-white/[0.04] px-5 py-3">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="text-[10px] text-muted uppercase tracking-wider font-medium">Last Run</span>
                      <StatusBadge status={lastRunResult.status} />
                      <span className="text-[10px] text-muted font-mono">{formatDuration(lastRunResult.duration)}</span>
                    </div>
                    {lastRunResult.step_results && Object.keys(lastRunResult.step_results).length > 0 && (
                      <pre className="text-[11px] font-mono text-foreground-dim bg-black/20 rounded-lg p-3 overflow-x-auto max-h-40">
                        {JSON.stringify(lastRunResult.step_results, null, 2)}
                      </pre>
                    )}
                  </div>
                )}

                {/* Expanded runs list */}
                {expandedDag === dag.id && (
                  <div className="border-t border-white/[0.04]">
                    {loadingRuns ? (
                      <div className="flex items-center gap-2 px-5 py-4">
                        <div className="w-4 h-4 border-2 border-frost/30 border-t-frost rounded-full spin-slow" />
                        <span className="text-xs text-muted font-mono">Loading runs...</span>
                      </div>
                    ) : dagRuns.length === 0 ? (
                      <div className="px-5 py-4">
                        <p className="text-xs text-muted/60 italic">No runs yet</p>
                      </div>
                    ) : (
                      <div className="divide-y divide-white/[0.03]">
                        {/* Runs header */}
                        <div className="grid grid-cols-[80px_1fr_90px_100px] gap-3 px-5 py-2 text-[10px] text-muted uppercase tracking-widest font-medium">
                          <span>Status</span>
                          <span>Run ID</span>
                          <span className="text-right">Duration</span>
                          <span className="text-right">Started</span>
                        </div>
                        {dagRuns.slice(0, 10).map((run) => (
                          <div key={run.id} className="grid grid-cols-[80px_1fr_90px_100px] gap-3 items-center px-5 py-2.5 hover:bg-white/[0.02] transition-colors">
                            <StatusBadge status={run.status} />
                            <span className="text-xs font-mono text-foreground-dim">#{run.id}</span>
                            <span className="text-xs font-mono text-muted text-right">{formatDuration(run.duration)}</span>
                            <span className="text-[11px] font-mono text-muted text-right">{timeAgo(run.started_at)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Right panel: DAG builder (1/3) ──────────────────────── */}
      <div className="w-96 border-l border-border shrink-0 overflow-y-auto bg-background-elevated/50">
        <div className="p-5 space-y-5">
          <h2 className="text-xs font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="5" cy="6" r="3" />
              <circle cx="19" cy="6" r="3" />
              <circle cx="12" cy="18" r="3" />
              <path d="M7.5 8l3 7M16.5 8l-3 7" />
            </svg>
            DAG Builder
          </h2>

          {/* Name */}
          <div className="space-y-1.5">
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium">Name</label>
            <input
              type="text"
              value={builderName}
              onChange={(e) => setBuilderName(e.target.value)}
              placeholder="my-pipeline"
              className="w-full bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-2 text-sm font-mono text-foreground placeholder-muted/50 outline-none focus:border-frost/30 transition-colors"
            />
          </div>

          {/* Description */}
          <div className="space-y-1.5">
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium">Description</label>
            <input
              type="text"
              value={builderDesc}
              onChange={(e) => setBuilderDesc(e.target.value)}
              placeholder="ETL pipeline for data processing"
              className="w-full bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-2 text-sm text-foreground placeholder-muted/50 outline-none focus:border-frost/30 transition-colors"
            />
          </div>

          {/* Steps */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-[10px] text-muted uppercase tracking-wider font-medium">Steps</label>
              <span className="text-[10px] font-mono text-foreground-dim">{builderSteps.length}</span>
            </div>

            {builderSteps.map((step, index) => {
              const otherStepIds = builderSteps.filter((_, i) => i !== index).map((s) => s.stepId);
              return (
                <div
                  key={index}
                  className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.04] space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-muted uppercase tracking-wider font-medium">Step {index + 1}</span>
                    {builderSteps.length > 1 && (
                      <button
                        onClick={() => removeStep(index)}
                        className="text-rose/40 hover:text-rose transition-colors"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <line x1="18" y1="6" x2="6" y2="18" />
                          <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                      </button>
                    )}
                  </div>

                  {/* Step ID */}
                  <input
                    type="text"
                    value={step.stepId}
                    onChange={(e) => updateStep(index, { stepId: e.target.value })}
                    placeholder="step_id"
                    className="w-full bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-1.5 text-xs font-mono text-foreground placeholder-muted/50 outline-none focus:border-frost/30 transition-colors"
                  />

                  {/* Function selector */}
                  <select
                    value={step.funcId ?? ""}
                    onChange={(e) => updateStep(index, { funcId: e.target.value ? Number(e.target.value) : null })}
                    className="w-full bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-1.5 text-xs font-mono text-foreground outline-none focus:border-frost/30 transition-colors appearance-none cursor-pointer"
                    style={{ backgroundImage: "none" }}
                  >
                    <option value="" className="bg-[#0c0c1e] text-muted">Select function...</option>
                    {funcs.map((f) => (
                      <option key={f.id} value={f.id} className="bg-[#0c0c1e] text-foreground">
                        {f.name}
                      </option>
                    ))}
                  </select>

                  {/* Dependencies */}
                  {otherStepIds.length > 0 && (
                    <div className="space-y-1">
                      <span className="text-[9px] text-muted uppercase tracking-wider">Depends on</span>
                      <div className="flex flex-wrap gap-1">
                        {otherStepIds.map((depId) => {
                          const selected = step.dependsOn.includes(depId);
                          return (
                            <button
                              key={depId}
                              onClick={() => toggleDependency(index, depId)}
                              className={`
                                text-[10px] font-mono px-2 py-0.5 rounded border transition-all
                                ${
                                  selected
                                    ? "bg-frost/10 text-frost border-frost/20"
                                    : "bg-white/[0.02] text-muted border-white/[0.06] hover:border-white/[0.12]"
                                }
                              `}
                            >
                              {depId}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}

            <button
              onClick={addStep}
              className="
                w-full px-3 py-2 rounded-lg text-[11px] font-medium
                text-frost/60 hover:text-frost
                bg-white/[0.02] hover:bg-white/[0.04]
                border border-dashed border-white/[0.08] hover:border-frost/20
                transition-all duration-150
                flex items-center justify-center gap-1.5
              "
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
              Add Step
            </button>
          </div>

          {/* Error */}
          {createError && (
            <p className="text-xs text-rose">{createError}</p>
          )}

          {/* Create button */}
          <button
            onClick={handleCreate}
            disabled={creating}
            className="
              w-full px-4 py-2.5 rounded-lg text-sm font-semibold
              bg-frost/10 text-frost border border-frost/20
              hover:bg-frost/20 hover:border-frost/40
              disabled:opacity-30 disabled:cursor-not-allowed
              transition-all duration-150
            "
          >
            {creating ? (
              <span className="flex items-center justify-center gap-2">
                <div className="w-4 h-4 border-2 border-frost/30 border-t-frost rounded-full spin-slow" />
                Creating...
              </span>
            ) : (
              "Create DAG"
            )}
          </button>
        </div>
      </div>

      {/* ── Quick Run floating action button ────────────────────── */}
      <button
        onClick={() => {
          setQuickRunOpen(true);
          setQuickRunResult(null);
          setQuickRunError("");
        }}
        className="
          fixed bottom-6 right-6 z-40
          w-14 h-14 rounded-full
          bg-frost/15 text-frost border border-frost/30
          hover:bg-frost/25 hover:border-frost/50
          backdrop-blur-md
          transition-all duration-150
          flex items-center justify-center
          glow-pulse
        "
        title="Quick Run a function"
        style={{ boxShadow: "0 4px 24px rgba(103,232,249,0.2)" }}
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="5 3 19 12 5 21 5 3" />
        </svg>
      </button>

      {/* ── Quick Run modal ─────────────────────────────────────── */}
      {quickRunOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: "rgba(5,5,16,0.7)", backdropFilter: "blur(4px)" }}
          onClick={() => setQuickRunOpen(false)}
        >
          <div
            className="glass-card p-6 w-full max-w-md space-y-4"
            onClick={(e) => e.stopPropagation()}
            style={{ background: "rgba(12,12,30,0.95)" }}
          >
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-widest text-muted flex items-center gap-2">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                Quick Run
              </h3>
              <button
                onClick={() => setQuickRunOpen(false)}
                className="text-muted hover:text-foreground transition-colors"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            {/* Name input with autocomplete */}
            <div className="space-y-1.5 relative">
              <label className="text-[10px] text-muted uppercase tracking-wider font-medium">
                Function Name
              </label>
              <input
                type="text"
                value={quickRunName}
                onChange={(e) => setQuickRunName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleQuickRun();
                }}
                placeholder="my_function"
                autoFocus
                className="w-full bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-2 text-sm font-mono text-foreground placeholder-muted/50 outline-none focus:border-frost/30 transition-colors"
              />
              {quickRunSuggestions.length > 0 && quickRunName !== quickRunSuggestions[0]?.name && (
                <div className="absolute top-full left-0 right-0 mt-1 rounded-lg bg-card border border-white/[0.08] overflow-hidden z-10">
                  {quickRunSuggestions.map((f) => (
                    <button
                      key={f.id}
                      onClick={() => setQuickRunName(f.name)}
                      className="w-full text-left px-3 py-1.5 text-xs font-mono text-foreground-dim hover:text-foreground hover:bg-white/[0.04] transition-colors"
                    >
                      {f.name}
                      {f.description && (
                        <span className="text-muted ml-2 text-[10px]">{f.description}</span>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Run button */}
            <button
              onClick={handleQuickRun}
              disabled={!quickRunName.trim() || quickRunRunning}
              className="
                w-full px-4 py-2.5 rounded-lg text-sm font-semibold
                bg-emerald/10 text-emerald border border-emerald/20
                hover:bg-emerald/20 hover:border-emerald/40
                disabled:opacity-30 disabled:cursor-not-allowed
                transition-all duration-150
              "
            >
              {quickRunRunning ? (
                <span className="flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-emerald/30 border-t-emerald rounded-full spin-slow" />
                  Running...
                </span>
              ) : (
                "Run"
              )}
            </button>

            {/* Error */}
            {quickRunError && (
              <p className="text-xs text-rose font-mono">{quickRunError}</p>
            )}

            {/* Result */}
            {quickRunResult != null && (
              <div className="space-y-1.5">
                <span className="text-[10px] text-muted uppercase tracking-wider font-medium">Result</span>
                <pre className="text-[11px] font-mono text-foreground-dim bg-black/30 rounded-lg p-3 overflow-x-auto max-h-60">
                  {typeof quickRunResult === "string"
                    ? quickRunResult
                    : JSON.stringify(quickRunResult, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Delete confirmation ─────────────────────────────── */}
      <ConfirmModal
        open={pendingDeleteId != null}
        title="Delete DAG"
        message={
          pendingDeleteId != null
            ? `Delete DAG "${dags.find((d) => d.id === pendingDeleteId)?.name ?? `#${pendingDeleteId}`}"? This cannot be undone.`
            : ""
        }
        danger
        onCancel={() => setPendingDeleteId(null)}
        onConfirm={() => {
          if (pendingDeleteId != null) {
            handleDeleteDag(pendingDeleteId);
            setPendingDeleteId(null);
          }
        }}
      />
    </div>
  );
}
