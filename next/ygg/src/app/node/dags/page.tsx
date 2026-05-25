"use client";

import { useEffect, useState } from "react";
import { node as api, type DagEntry } from "@/lib/api";
import Link from "next/link";

// ── Demo data ────────────────────────────────────────────────
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

export default function DagsPage() {
  const [dags, setDags] = useState<DagEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  // Form state
  const [formName, setFormName] = useState("");
  const [formDescription, setFormDescription] = useState("");
  const [formSubmitting, setFormSubmitting] = useState(false);

  useEffect(() => {
    loadDags();
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

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setFormSubmitting(true);
    try {
      await api.createDag({
        name: formName,
        description: formDescription,
        steps: [],
        edges: [],
      });
      setShowForm(false);
      setFormName("");
      setFormDescription("");
      await loadDags();
    } catch (e) {
      setError(`Create failed: ${e}`);
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

      {/* Creation Form */}
      {showForm && (
        <form onSubmit={handleCreate} className="nordic-card p-5 space-y-4">
          <h2 className="text-sm font-semibold text-foreground">Create DAG</h2>
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
            <input
              type="text"
              value={formDescription}
              onChange={(e) => setFormDescription(e.target.value)}
              className="input-nordic w-full text-sm"
              placeholder="What does this pipeline do?"
            />
          </div>
          <div className="flex justify-end">
            <button type="submit" className="btn-primary text-sm" disabled={formSubmitting}>
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
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20">
                      {step.id}
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
        <div className="nordic-card p-8 text-center">
          <p className="text-muted text-sm">No DAGs defined yet.</p>
          <button onClick={() => setShowForm(true)} className="btn-primary text-sm mt-4">
            Create your first DAG
          </button>
        </div>
      )}
    </div>
  );
}
