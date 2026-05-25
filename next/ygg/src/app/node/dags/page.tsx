"use client";

import { useEffect, useState } from "react";
import { node as api, type DagEntry, type DagStep, type DagEdge, type FunctionEntry, type EnvironmentEntry } from "@/lib/api";
import Link from "next/link";

// ── Custom SVG Icons ──────────────────────────────────────────
const EnvIcon = ({ size = 14, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className={className}>
    <path d="M12 2l8 4.5v9L12 20l-8-4.5v-9L12 2z"/>
    <path d="M12 7l5 2.8v5.4L12 18l-5-2.8V9.8L12 7z" strokeOpacity="0.5"/>
    <line x1="12" y1="2" x2="12" y2="7" strokeOpacity="0.3"/>
  </svg>
);

const FuncIcon = ({ size = 14, className = "" }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
    <path d="M6 4h4l2 6 4 10h4"/>
    <path d="M6 20h4l4-10"/>
  </svg>
);

// -- Demo data --
const DEMO_DAGS: DagEntry[] = [
  {
    id: 1,
    name: "metrics-pipeline",
    description: "Collect metrics, process, and alert",
    steps: [
      { id: "collect", ref: { node_url: null, function_id: 2, environment_id: 1, args: {} }, depends_on: [] },
      { id: "process", ref: { node_url: null, function_id: 3, environment_id: 2, args: {} }, depends_on: ["collect"] },
      { id: "alert", ref: { node_url: null, function_id: 4, environment_id: null, args: {} }, depends_on: ["process"] },
    ],
    edges: [
      { from_step: "collect", to_step: "process", output_key: "metrics", input_key: "data" },
      { from_step: "process", to_step: "alert", output_key: "summary", input_key: "message" },
    ],
    created_at: "2025-05-20T10:00:00Z",
    updated_at: "2025-05-24T14:00:00Z",
    run_count: 12,
  },
  {
    id: 2,
    name: "data-etl",
    description: "Extract, transform, load pipeline",
    steps: [
      { id: "extract", ref: { node_url: null, function_id: 1, environment_id: 2, args: {} }, depends_on: [] },
      { id: "transform", ref: { node_url: null, function_id: 3, environment_id: 2, args: {} }, depends_on: ["extract"] },
    ],
    edges: [
      { from_step: "extract", to_step: "transform", output_key: "raw", input_key: "source" },
    ],
    created_at: "2025-05-22T08:00:00Z",
    updated_at: "2025-05-23T16:30:00Z",
    run_count: 5,
  },
];

// -- Step builder types --
interface StepDraft {
  name: string;
  function_id: number | null;
  is_dag: boolean;
  environment_id: number | null;
  depends_on: string[];
  args_override: string;
}

interface EdgeDraft {
  from_step: string;
  to_step: string;
  output_key: string;
  input_key: string;
}

export default function DagsPage() {
  const [dags, setDags] = useState<DagEntry[]>([]);
  const [functions, setFunctions] = useState<FunctionEntry[]>([]);
  const [environments, setEnvironments] = useState<EnvironmentEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  // Form state
  const [formName, setFormName] = useState("");
  const [formDescription, setFormDescription] = useState("");
  const [formDefaultEnvId, setFormDefaultEnvId] = useState<number | null>(null);
  const [formDefaultArgs, setFormDefaultArgs] = useState("");
  const [formSteps, setFormSteps] = useState<StepDraft[]>([]);
  const [formEdges, setFormEdges] = useState<EdgeDraft[]>([]);
  const [formCron, setFormCron] = useState("");
  const [formCronEnabled, setFormCronEnabled] = useState(false);
  const [formSubmitting, setFormSubmitting] = useState(false);

  useEffect(() => {
    loadDags();
    loadResources();
  }, []);

  async function loadDags() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listDags();
      setDags(data.dags);
    } catch {
      setError("Node unavailable - showing demo data");
      setDags(DEMO_DAGS);
    }
    setLoading(false);
  }

  async function loadResources() {
    try {
      const [fnData, envData] = await Promise.allSettled([
        api.listFunctions(),
        api.listEnvironments(),
      ]);
      if (fnData.status === "fulfilled") setFunctions(fnData.value.functions);
      if (envData.status === "fulfilled") setEnvironments(envData.value.environments);
    } catch {
      // non-critical
    }
  }

  function addStep() {
    setFormSteps([...formSteps, { name: "", function_id: null, is_dag: false, environment_id: null, depends_on: [], args_override: "" }]);
  }

  function removeStep(idx: number) {
    const removed = formSteps[idx];
    const updated = formSteps.filter((_, i) => i !== idx);
    const cleaned = updated.map((s) => ({
      ...s,
      depends_on: s.depends_on.filter((d) => d !== removed.name),
    }));
    setFormSteps(cleaned);
    setFormEdges(formEdges.filter((e) => e.from_step !== removed.name && e.to_step !== removed.name));
  }

  function moveStep(idx: number, direction: "up" | "down") {
    const newIdx = direction === "up" ? idx - 1 : idx + 1;
    if (newIdx < 0 || newIdx >= formSteps.length) return;
    const updated = [...formSteps];
    [updated[idx], updated[newIdx]] = [updated[newIdx], updated[idx]];
    setFormSteps(updated);
  }

  function updateStep(idx: number, patch: Partial<StepDraft>) {
    const oldName = formSteps[idx].name;
    const updated = formSteps.map((s, i) => i === idx ? { ...s, ...patch } : s);
    if (patch.name && patch.name !== oldName && oldName) {
      const newName = patch.name;
      const fixed = updated.map((s) => ({
        ...s,
        depends_on: s.depends_on.map((d) => d === oldName ? newName : d),
      }));
      setFormSteps(fixed);
      setFormEdges(formEdges.map((e) => ({
        ...e,
        from_step: e.from_step === oldName ? newName : e.from_step,
        to_step: e.to_step === oldName ? newName : e.to_step,
      })));
    } else {
      setFormSteps(updated);
    }
  }

  function toggleDependency(stepIdx: number, depName: string) {
    const step = formSteps[stepIdx];
    const deps = step.depends_on.includes(depName)
      ? step.depends_on.filter((d) => d !== depName)
      : [...step.depends_on, depName];
    updateStep(stepIdx, { depends_on: deps });

    if (!step.depends_on.includes(depName)) {
      const existingEdge = formEdges.find((e) => e.from_step === depName && e.to_step === step.name);
      if (!existingEdge && step.name) {
        setFormEdges([...formEdges, { from_step: depName, to_step: step.name, output_key: "output", input_key: "input" }]);
      }
    } else {
      setFormEdges(formEdges.filter((e) => !(e.from_step === depName && e.to_step === step.name)));
    }
  }

  function updateEdge(idx: number, patch: Partial<EdgeDraft>) {
    setFormEdges(formEdges.map((e, i) => i === idx ? { ...e, ...patch } : e));
  }

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setFormSubmitting(true);
    try {
      let defaultArgs: Record<string, unknown> = {};
      if (formDefaultArgs.trim()) {
        try { defaultArgs = JSON.parse(formDefaultArgs); } catch { /* ignore invalid json */ }
      }

      const steps: DagStep[] = formSteps
        .filter((s) => s.name && s.function_id != null)
        .map((s) => {
          let stepArgs: Record<string, unknown> = { ...defaultArgs };
          if (s.args_override.trim()) {
            try { stepArgs = { ...stepArgs, ...JSON.parse(s.args_override) }; } catch { /* skip */ }
          }
          return {
            id: s.name,
            ref: {
              node_url: null,
              function_id: s.function_id!,
              environment_id: s.environment_id ?? formDefaultEnvId,
              args: stepArgs,
            },
            depends_on: s.depends_on,
          };
        });

      const edges: DagEdge[] = formEdges.filter(
        (e) => e.from_step && e.to_step && e.output_key && e.input_key
      );

      // Build description with cron info if enabled
      let description = formDescription;
      if (formCronEnabled && formCron.trim()) {
        description = `[cron: ${formCron.trim()}] ${description}`;
      }

      await api.createDag({
        name: formName,
        description,
        steps,
        edges,
      });
      setShowForm(false);
      setFormName("");
      setFormDescription("");
      setFormDefaultEnvId(null);
      setFormDefaultArgs("");
      setFormSteps([]);
      setFormEdges([]);
      setFormCron("");
      setFormCronEnabled(false);
      await loadDags();
    } catch (err) {
      setError(`Create failed: ${err}`);
    }
    setFormSubmitting(false);
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="text-center animate-in">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full spin-slow mx-auto mb-4" />
          <p className="text-primary font-mono text-sm pulse-primary">Loading DAGs...</p>
        </div>
      </div>
    );
  }

  const stepNames = formSteps.map((s) => s.name).filter(Boolean);

  return (
    <div className="p-6 space-y-6 animate-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-foreground">DAGs</h1>
          <p className="text-sm text-muted mt-0.5">
            {dags.length} pipeline{dags.length !== 1 ? "s" : ""} defined
          </p>
        </div>
        <div className="flex items-center gap-3">
          {error && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-warning/10 border border-warning/20">
              <div className="status-dot pending" />
              <span className="text-xs font-medium text-warning">{error}</span>
            </div>
          )}
          <button onClick={() => setShowForm(!showForm)} className="btn-primary text-sm">
            {showForm ? "Cancel" : "New DAG"}
          </button>
        </div>
      </div>

      {/* Interactive DAG Builder */}
      {showForm && (
        <form onSubmit={handleCreate} className="nordic-card p-5 space-y-5">
          <h2 className="text-sm font-semibold text-foreground">Create DAG</h2>

          {/* DAG Header: name + description */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-muted mb-1.5">Name</label>
              <input
                type="text"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
                className="input-nordic w-full text-sm"
                placeholder="my-pipeline"
                required
              />
            </div>
            <div>
              <label className="block text-xs text-muted mb-1.5">Description</label>
              <textarea
                value={formDescription}
                onChange={(e) => setFormDescription(e.target.value)}
                className="input-nordic w-full text-sm"
                rows={2}
                placeholder="What does this pipeline do?"
              />
            </div>
          </div>

          {/* Default Environment + Default Args */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-muted mb-1.5">Default Environment</label>
              <select
                value={formDefaultEnvId ?? ""}
                onChange={(e) => setFormDefaultEnvId(e.target.value ? parseInt(e.target.value, 10) : null)}
                className="input-nordic w-full text-sm"
              >
                <option value="">None</option>
                {environments.map((env) => (
                  <option key={env.id} value={env.id}>{env.name}</option>
                ))}
              </select>
              <p className="text-[10px] text-muted mt-1">Inherited by steps unless overridden</p>
            </div>
            <div>
              <label className="block text-xs text-muted mb-1.5">Default Args (JSON)</label>
              <textarea
                value={formDefaultArgs}
                onChange={(e) => setFormDefaultArgs(e.target.value)}
                className="input-nordic w-full text-sm font-mono"
                rows={2}
                placeholder='{"key": "value"}'
              />
              <p className="text-[10px] text-muted mt-1">Passed to all steps unless overridden</p>
            </div>
          </div>

          {/* Step Builder */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <label className="text-xs font-medium text-muted uppercase tracking-wider">Steps</label>
              <button type="button" onClick={addStep} className="btn-ghost text-xs">
                + Add Step
              </button>
            </div>

            {formSteps.length === 0 && (
              <div className="border border-dashed border-border rounded-lg p-4 text-center">
                <p className="text-xs text-muted">No steps yet. Click &quot;Add Step&quot; to build your pipeline.</p>
              </div>
            )}

            <div className="space-y-1">
              {formSteps.map((step, idx) => {
                const availableDeps = stepNames.filter((n) => n !== step.name);

                return (
                  <div key={idx} className="relative">
                    {/* Down-arrow connector between steps */}
                    {idx > 0 && (
                      <div className="flex justify-center py-1">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-primary/40">
                          <line x1="8" y1="0" x2="8" y2="12" />
                          <polyline points="5 9 8 12 11 9" />
                        </svg>
                      </div>
                    )}

                    {/* Step Card */}
                    <div className="bg-card border border-border rounded-lg p-4">
                      <div className="flex items-start gap-3">
                        {/* Reorder buttons */}
                        <div className="flex flex-col gap-0.5 shrink-0 mt-1">
                          <button
                            type="button"
                            onClick={() => moveStep(idx, "up")}
                            disabled={idx === 0}
                            className="text-muted hover:text-foreground disabled:opacity-20 transition-colors"
                          >
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="18 15 12 9 6 15"/></svg>
                          </button>
                          <span className="text-[9px] font-mono text-muted text-center">#{idx + 1}</span>
                          <button
                            type="button"
                            onClick={() => moveStep(idx, "down")}
                            disabled={idx === formSteps.length - 1}
                            className="text-muted hover:text-foreground disabled:opacity-20 transition-colors"
                          >
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="6 9 12 15 18 9"/></svg>
                          </button>
                        </div>

                        <div className="flex-1 space-y-3">
                          {/* Row 1: name + function/DAG selector + environment */}
                          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                            {/* Step name */}
                            <div>
                              <label className="block text-[10px] text-muted mb-1">Step Name</label>
                              <input
                                type="text"
                                value={step.name}
                                onChange={(e) => updateStep(idx, { name: e.target.value.replace(/\s/g, "_") })}
                                className="input-nordic w-full text-xs font-mono"
                                placeholder="step_name"
                                required
                              />
                            </div>

                            {/* Function / DAG selector */}
                            <div>
                              <label className="block text-[10px] text-muted mb-1">
                                {step.is_dag ? (
                                  <span className="flex items-center gap-1">Sub-DAG</span>
                                ) : (
                                  <span className="flex items-center gap-1"><FuncIcon size={10} /> Function</span>
                                )}
                              </label>
                              <select
                                value={step.is_dag ? `dag:${step.function_id ?? ""}` : String(step.function_id ?? "")}
                                onChange={(e) => {
                                  const val = e.target.value;
                                  if (val.startsWith("dag:")) {
                                    const dagId = val.replace("dag:", "");
                                    updateStep(idx, { function_id: dagId ? parseInt(dagId, 10) : null, is_dag: true });
                                  } else {
                                    updateStep(idx, { function_id: val ? parseInt(val, 10) : null, is_dag: false });
                                  }
                                }}
                                className="input-nordic w-full text-xs"
                                required
                              >
                                <option value="">Select...</option>
                                <optgroup label="Functions">
                                  {functions.map((fn) => (
                                    <option key={`fn-${fn.id}`} value={String(fn.id)}>{fn.name}</option>
                                  ))}
                                </optgroup>
                                {dags.length > 0 && (
                                  <optgroup label="Sub-DAGs">
                                    {dags.map((d) => (
                                      <option key={`dag-${d.id}`} value={`dag:${d.id}`}>dag: {d.name}</option>
                                    ))}
                                  </optgroup>
                                )}
                              </select>
                            </div>

                            {/* Environment override */}
                            <div>
                              <label className="block text-[10px] text-muted mb-1">
                                <span className="flex items-center gap-1"><EnvIcon size={10} /> Environment</span>
                              </label>
                              <select
                                value={step.environment_id ?? ""}
                                onChange={(e) => updateStep(idx, { environment_id: e.target.value ? parseInt(e.target.value, 10) : null })}
                                className="input-nordic w-full text-xs"
                              >
                                <option value="">{formDefaultEnvId ? "Inherit default" : "None"}</option>
                                {environments.map((env) => (
                                  <option key={env.id} value={env.id}>{env.name}</option>
                                ))}
                              </select>
                            </div>

                            {/* Depends on */}
                            <div>
                              <label className="block text-[10px] text-muted mb-1">Depends On</label>
                              {availableDeps.length === 0 ? (
                                <span className="text-[10px] text-muted italic">No prior steps</span>
                              ) : (
                                <div className="flex flex-wrap gap-1">
                                  {availableDeps.map((dep) => (
                                    <button
                                      key={dep}
                                      type="button"
                                      onClick={() => toggleDependency(idx, dep)}
                                      className={`text-[10px] font-mono px-1.5 py-0.5 rounded border transition-colors ${
                                        step.depends_on.includes(dep)
                                          ? "bg-primary/15 border-primary/30 text-primary"
                                          : "bg-border/30 border-border text-muted hover:border-primary/30"
                                      }`}
                                    >
                                      {dep}
                                    </button>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>

                          {/* Row 2: Args override (optional) */}
                          <div>
                            <label className="block text-[10px] text-muted mb-1">Args Override (JSON, optional)</label>
                            <input
                              type="text"
                              value={step.args_override}
                              onChange={(e) => updateStep(idx, { args_override: e.target.value })}
                              className="input-nordic w-full text-xs font-mono"
                              placeholder='{"key": "override_value"}'
                            />
                          </div>

                          {/* Step info badges */}
                          <div className="flex flex-wrap gap-2">
                            {step.function_id != null && !step.is_dag && (
                              <span className="flex items-center gap-1 text-[10px] font-mono text-primary/70 bg-primary/5 px-1.5 py-0.5 rounded border border-primary/15">
                                <FuncIcon size={12} />
                                <span className="hidden sm:inline">{functions.find((f) => f.id === step.function_id)?.name ?? `#${step.function_id}`}</span>
                              </span>
                            )}
                            {step.function_id != null && step.is_dag && (
                              <span className="flex items-center gap-1 text-[10px] font-mono text-warning/80 bg-warning/5 px-1.5 py-0.5 rounded border border-warning/15">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <circle cx="5" cy="6" r="3"/><circle cx="19" cy="6" r="3"/><circle cx="12" cy="18" r="3"/>
                                  <line x1="7.5" y1="7.5" x2="10.5" y2="16.5"/><line x1="16.5" y1="7.5" x2="13.5" y2="16.5"/>
                                </svg>
                                <span className="hidden sm:inline">{dags.find((d) => d.id === step.function_id)?.name ?? `dag #${step.function_id}`}</span>
                              </span>
                            )}
                            {(step.environment_id ?? formDefaultEnvId) != null && (
                              <span className="flex items-center gap-1 text-[10px] font-mono text-muted bg-border/30 px-1.5 py-0.5 rounded border border-border">
                                <EnvIcon size={12} />
                                <span className="hidden sm:inline">
                                  {environments.find((e) => e.id === (step.environment_id ?? formDefaultEnvId))?.name ?? `#${step.environment_id ?? formDefaultEnvId}`}
                                </span>
                              </span>
                            )}
                            {step.depends_on.map((dep) => (
                              <span key={dep} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                                {dep}
                              </span>
                            ))}
                          </div>
                        </div>

                        <button
                          type="button"
                          onClick={() => removeStep(idx)}
                          className="text-muted hover:text-destructive transition-colors shrink-0 mt-1"
                        >
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Edge configuration */}
          {formEdges.length > 0 && (
            <div>
              <label className="text-xs font-medium text-muted uppercase tracking-wider mb-3 block">Edge Mappings</label>
              <div className="space-y-2">
                {formEdges.map((edge, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-xs">
                    <span className="font-mono px-1.5 py-0.5 rounded bg-border/50 text-foreground-dim shrink-0">{edge.from_step}</span>
                    <span className="text-muted">.</span>
                    <input
                      type="text"
                      value={edge.output_key}
                      onChange={(e) => updateEdge(idx, { output_key: e.target.value })}
                      className="input-nordic text-xs font-mono w-24"
                      placeholder="output_key"
                    />
                    <svg width="16" height="10" viewBox="0 0 16 10" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-primary/60 shrink-0">
                      <line x1="0" y1="5" x2="12" y2="5" />
                      <polyline points="9 2 12 5 9 8" />
                    </svg>
                    <span className="font-mono px-1.5 py-0.5 rounded bg-border/50 text-foreground-dim shrink-0">{edge.to_step}</span>
                    <span className="text-muted">.</span>
                    <input
                      type="text"
                      value={edge.input_key}
                      onChange={(e) => updateEdge(idx, { input_key: e.target.value })}
                      className="input-nordic text-xs font-mono w-24"
                      placeholder="input_key"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Schedule Section */}
          <div>
            <label className="text-xs font-medium text-muted uppercase tracking-wider mb-3 block">Schedule</label>
            <div className="bg-card border border-border rounded-lg p-4 space-y-3">
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setFormCronEnabled(!formCronEnabled)}
                  className={`relative w-9 h-5 rounded-full transition-colors ${formCronEnabled ? "bg-primary" : "bg-border"}`}
                >
                  <span
                    className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${formCronEnabled ? "translate-x-4" : "translate-x-0"}`}
                  />
                </button>
                <span className="text-xs text-foreground">Enable scheduled runs</span>
              </div>
              {formCronEnabled && (
                <div>
                  <label className="block text-[10px] text-muted mb-1">Cron Expression</label>
                  <input
                    type="text"
                    value={formCron}
                    onChange={(e) => setFormCron(e.target.value)}
                    className="input-nordic w-full text-sm font-mono"
                    placeholder="*/5 * * * *"
                  />
                  <p className="text-[10px] text-muted mt-1">Standard cron format: minute hour day month weekday</p>
                </div>
              )}
            </div>
          </div>

          <div className="flex justify-end">
            <button type="submit" className="btn-primary text-sm" disabled={formSubmitting || formSteps.length === 0}>
              {formSubmitting ? "Creating..." : "Create DAG"}
            </button>
          </div>
        </form>
      )}

      {/* DAGs Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {dags.map((dag) => (
          <div key={dag.id} className="nordic-card p-4 flex flex-col">
            <div className="flex items-start justify-between mb-3">
              <Link
                href={`/node/dags/${dag.id}`}
                className="font-mono text-sm font-medium text-foreground hover:text-primary transition-colors truncate"
              >
                {dag.name}
              </Link>
              <span className="text-[10px] font-mono text-muted bg-border/50 px-1.5 py-0.5 rounded shrink-0 ml-2">
                #{dag.id}
              </span>
            </div>

            {dag.description && (
              <p className="text-xs text-muted mb-3 line-clamp-2">{dag.description}</p>
            )}

            {/* Step flow preview */}
            {dag.steps.length > 0 && (
              <div className="mb-3 space-y-1">
                {dag.steps.map((step, idx) => (
                  <div key={step.id} className="flex items-center gap-2">
                    {idx > 0 && (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-muted shrink-0 ml-1">
                        <polyline points="12 5 12 19" /><polyline points="19 12 12 19 5 12" />
                      </svg>
                    )}
                    <span className="flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                      <FuncIcon size={10} />
                      <span className="hidden sm:inline">{step.id}</span>
                    </span>
                  </div>
                ))}
              </div>
            )}

            <div className="mt-auto space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted">Steps</span>
                <span className="text-foreground font-mono">{dag.steps.length}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted">Runs</span>
                <span className="text-foreground font-mono">{dag.run_count}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted">Updated</span>
                <span className="text-foreground-dim text-[11px]">
                  {new Date(dag.updated_at).toLocaleDateString()}
                </span>
              </div>

              <div className="flex gap-2 pt-2">
                <Link
                  href={`/node/dags/${dag.id}`}
                  className="btn-ghost text-xs flex-1 text-center"
                >
                  Details
                </Link>
              </div>
            </div>
          </div>
        ))}
      </div>

      {dags.length === 0 && (
        <div className="nordic-card p-6 space-y-4">
          <h2 className="text-sm font-semibold text-foreground">Build Pipelines</h2>
          <p className="text-sm text-muted">Chain functions into DAGs that run across nodes.</p>

          <div className="code-block p-3 text-xs">
            <p className="text-muted mb-1"># Chain functions with &gt;&gt; operator</p>
            <p>from yggdrasil.node import function, dag</p>
            <p></p>
            <p>@function</p>
            <p>def extract(source: str) -&gt; list:</p>
            <p>    return [1, 2, 3]</p>
            <p></p>
            <p>@function</p>
            <p>def transform(data: list) -&gt; list:</p>
            <p>    return [x * 2 for x in data]</p>
            <p></p>
            <p>@function</p>
            <p>def load(data: list) -&gt; int:</p>
            <p>    return sum(data)</p>
            <p></p>
            <p># Build and run the pipeline</p>
            <p>pipeline = dag(&quot;etl&quot;, extract &gt;&gt; transform &gt;&gt; load)</p>
            <p>result = pipeline().wait()</p>
          </div>

          <button onClick={() => setShowForm(true)} className="btn-primary text-sm mt-2">
            Create your first DAG
          </button>
        </div>
      )}
    </div>
  );
}
