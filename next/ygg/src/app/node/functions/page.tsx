"use client";

import { useEffect, useState } from "react";
import { node as api, type FunctionEntry, type EnvironmentEntry } from "@/lib/api";
import Link from "next/link";

// ── Demo data ────────────────────────────────────────────────
const DEMO_FUNCTIONS: FunctionEntry[] = [
  {
    id: 1,
    name: "hello_world",
    language: "python",
    code: 'def main():\n    return "Hello, Yggdrasil!"',
    description: "Simple greeting function for testing",
    python_version: "3.12",
    dependencies: [],
    environment_id: null,
    creator: "system",
    created_at: "2025-05-20T10:00:00Z",
    updated_at: "2025-05-20T10:00:00Z",
    run_count: 42,
  },
  {
    id: 2,
    name: "fetch_metrics",
    language: "python",
    code: 'import psutil\n\ndef main():\n    return {"cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent}',
    description: "Collect system metrics from the node",
    python_version: "3.12",
    dependencies: ["psutil"],
    environment_id: 1,
    creator: "admin",
    created_at: "2025-05-18T08:30:00Z",
    updated_at: "2025-05-22T14:00:00Z",
    run_count: 156,
  },
  {
    id: 3,
    name: "data_pipeline",
    language: "python",
    code: 'import pandas as pd\n\ndef main(source: str):\n    df = pd.read_csv(source)\n    return df.describe().to_dict()',
    description: "Run data pipeline on a CSV source",
    python_version: "3.11",
    dependencies: ["pandas", "numpy"],
    environment_id: 2,
    creator: "admin",
    created_at: "2025-05-15T12:00:00Z",
    updated_at: "2025-05-23T09:15:00Z",
    run_count: 28,
  },
  {
    id: 4,
    name: "health_check",
    language: "python",
    code: 'import requests\n\ndef main(url: str):\n    r = requests.get(url, timeout=5)\n    return {"status": r.status_code, "ok": r.ok}',
    description: "Check if a URL endpoint is healthy",
    python_version: "3.12",
    dependencies: ["requests"],
    environment_id: null,
    creator: "system",
    created_at: "2025-05-10T16:45:00Z",
    updated_at: "2025-05-10T16:45:00Z",
    run_count: 312,
  },
];

export default function FunctionsPage() {
  const [functions, setFunctions] = useState<FunctionEntry[]>([]);
  const [environments, setEnvironments] = useState<EnvironmentEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [runningId, setRunningId] = useState<number | null>(null);
  const [runResult, setRunResult] = useState<{ id: number; status: string; stdout?: string | null } | null>(null);

  // Form state
  const [formName, setFormName] = useState("");
  const [formCode, setFormCode] = useState("");
  const [formDescription, setFormDescription] = useState("");
  const [formPythonVersion, setFormPythonVersion] = useState("3.12");
  const [formDeps, setFormDeps] = useState("");
  const [formEnvId, setFormEnvId] = useState<number | null>(null);
  const [formSubmitting, setFormSubmitting] = useState(false);

  useEffect(() => {
    loadFunctions();
    loadEnvironments();
  }, []);

  async function loadFunctions() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listFunctions();
      setFunctions(data.functions);
    } catch {
      setError("Node unavailable - showing demo data");
      setFunctions(DEMO_FUNCTIONS);
    }
    setLoading(false);
  }

  async function loadEnvironments() {
    try {
      const data = await api.listEnvironments();
      setEnvironments(data.environments.filter((e) => e.status === "ready"));
    } catch {
      // Environments are optional, no error shown
    }
  }

  async function handleRun(id: number) {
    setRunningId(id);
    setRunResult(null);
    try {
      const data = await api.runFunction(id);
      setRunResult({ id, status: data.run.status, stdout: data.run.stdout });
    } catch (e) {
      setRunResult({ id, status: "error", stdout: String(e) });
    }
    setRunningId(null);
  }

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setFormSubmitting(true);
    try {
      const deps = formDeps.split(",").map((d) => d.trim()).filter(Boolean);
      await api.createFunction({
        name: formName,
        code: formCode,
        language: "python",
        description: formDescription,
        python_version: formPythonVersion,
        dependencies: deps,
        environment_id: formEnvId ?? undefined,
      });
      setShowForm(false);
      setFormName("");
      setFormCode("");
      setFormDescription("");
      setFormDeps("");
      setFormEnvId(null);
      await loadFunctions();
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
          <p className="text-primary font-mono text-sm pulse-primary">Loading functions...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 animate-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-foreground">Functions</h1>
          <p className="text-sm text-muted mt-0.5">
            {functions.length} function{functions.length !== 1 ? "s" : ""} registered
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
            {showForm ? "Cancel" : "New Function"}
          </button>
        </div>
      </div>

      {/* Creation Form */}
      {showForm && (
        <form onSubmit={handleCreate} className="nordic-card p-5 space-y-4">
          <h2 className="text-sm font-semibold text-foreground">Create Function</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-muted mb-1.5">Name</label>
              <input
                type="text"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
                className="input-nordic w-full text-sm"
                placeholder="my_function"
                required
              />
            </div>
            <div>
              <label className="block text-xs text-muted mb-1.5">Python Version</label>
              <select
                value={formPythonVersion}
                onChange={(e) => setFormPythonVersion(e.target.value)}
                className="input-nordic w-full text-sm"
              >
                <option value="3.10">3.10</option>
                <option value="3.11">3.11</option>
                <option value="3.12">3.12</option>
                <option value="3.13">3.13</option>
              </select>
            </div>
          </div>
          <div>
            <label className="block text-xs text-muted mb-1.5">Description</label>
            <input
              type="text"
              value={formDescription}
              onChange={(e) => setFormDescription(e.target.value)}
              className="input-nordic w-full text-sm"
              placeholder="What does this function do?"
            />
          </div>
          <div>
            <label className="block text-xs text-muted mb-1.5">Code</label>
            <textarea
              value={formCode}
              onChange={(e) => setFormCode(e.target.value)}
              className="input-nordic w-full text-sm font-mono"
              rows={8}
              placeholder={'def main():\n    return "Hello!"'}
              required
            />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-muted mb-1.5">Dependencies (comma-separated)</label>
              <input
                type="text"
                value={formDeps}
                onChange={(e) => setFormDeps(e.target.value)}
                className="input-nordic w-full text-sm"
                placeholder="requests, pandas, numpy"
              />
            </div>
            <div>
              <label className="block text-xs text-muted mb-1.5">Environment</label>
              <select
                value={formEnvId ?? ""}
                onChange={(e) => setFormEnvId(e.target.value ? parseInt(e.target.value, 10) : null)}
                className="input-nordic w-full text-sm"
              >
                <option value="">None (default)</option>
                {environments.map((env) => (
                  <option key={env.id} value={env.id}>{env.name}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex justify-end">
            <button type="submit" className="btn-primary text-sm" disabled={formSubmitting}>
              {formSubmitting ? "Creating..." : "Create Function"}
            </button>
          </div>
        </form>
      )}

      {/* Run result toast */}
      {runResult && (
        <div
          className={`nordic-card p-4 border-l-4 ${
            runResult.status === "completed" ? "border-l-success" : "border-l-destructive"
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-foreground">
              Run {runResult.status === "completed" ? "completed" : "failed"}
            </span>
            <button onClick={() => setRunResult(null)} className="btn-ghost text-xs">Dismiss</button>
          </div>
          {runResult.stdout && (
            <pre className="code-block p-3 mt-2 text-xs overflow-x-auto">{runResult.stdout}</pre>
          )}
        </div>
      )}

      {/* Functions Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {functions.map((fn) => (
          <div key={fn.id} className="nordic-card p-4 flex flex-col">
            <div className="flex items-start justify-between mb-3">
              <Link
                href={`/node/functions/${fn.id}`}
                className="font-mono text-sm font-medium text-foreground hover:text-primary transition-colors truncate"
              >
                {fn.name}
              </Link>
              <span className="text-[10px] font-mono text-muted bg-border/50 px-1.5 py-0.5 rounded shrink-0 ml-2">
                {fn.language}
              </span>
            </div>

            {fn.description && (
              <p className="text-xs text-muted mb-3 line-clamp-2">{fn.description}</p>
            )}

            <div className="mt-auto space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted">Runs</span>
                <span className="text-foreground font-mono">{fn.run_count}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted">Created</span>
                <span className="text-foreground-dim text-[11px]">
                  {new Date(fn.created_at).toLocaleDateString()}
                </span>
              </div>
              {fn.environment_id != null && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted">Env</span>
                  <span className="text-primary font-mono text-[11px]">
                    {environments.find((e) => e.id === fn.environment_id)?.name ?? `#${fn.environment_id}`}
                  </span>
                </div>
              )}
              {fn.dependencies.length > 0 && (
                <div className="flex flex-wrap gap-1 pt-1">
                  {fn.dependencies.slice(0, 3).map((dep) => (
                    <span key={dep} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-border/50 text-muted">
                      {dep}
                    </span>
                  ))}
                  {fn.dependencies.length > 3 && (
                    <span className="text-[10px] text-muted">+{fn.dependencies.length - 3}</span>
                  )}
                </div>
              )}

              <div className="flex gap-2 pt-2">
                <Link
                  href={`/node/functions/${fn.id}`}
                  className="btn-ghost text-xs flex-1 text-center"
                >
                  Details
                </Link>
                <button
                  onClick={() => handleRun(fn.id)}
                  disabled={runningId === fn.id}
                  className="btn-primary text-xs flex-1"
                >
                  {runningId === fn.id ? "Running..." : "Run"}
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {functions.length === 0 && (
        <div className="nordic-card p-8 text-center">
          <p className="text-muted text-sm">No functions registered yet.</p>
          <button onClick={() => setShowForm(true)} className="btn-primary text-sm mt-4">
            Create your first function
          </button>
        </div>
      )}
    </div>
  );
}
