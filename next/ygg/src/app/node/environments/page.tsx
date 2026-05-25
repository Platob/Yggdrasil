"use client";

import { useEffect, useState } from "react";
import { node as api, type EnvironmentEntry } from "@/lib/api";
import Link from "next/link";

// ── Demo data ────────────────────────────────────────────────
const DEMO_ENVIRONMENTS: EnvironmentEntry[] = [
  {
    id: 1,
    name: "metrics-env",
    python_version: "3.12",
    dependencies: ["psutil", "requests"],
    path: "/var/ygg/envs/metrics-env",
    status: "ready",
    created_at: "2025-05-15T10:00:00Z",
    updated_at: "2025-05-22T14:00:00Z",
    error: null,
  },
  {
    id: 2,
    name: "data-science",
    python_version: "3.11",
    dependencies: ["pandas", "numpy", "scikit-learn", "matplotlib"],
    path: "/var/ygg/envs/data-science",
    status: "ready",
    created_at: "2025-05-12T08:00:00Z",
    updated_at: "2025-05-20T16:30:00Z",
    error: null,
  },
  {
    id: 3,
    name: "ml-pipeline",
    python_version: "3.12",
    dependencies: ["torch", "transformers", "datasets"],
    path: "/var/ygg/envs/ml-pipeline",
    status: "creating",
    created_at: "2025-05-24T12:00:00Z",
    updated_at: "2025-05-24T12:00:00Z",
    error: null,
  },
  {
    id: 4,
    name: "broken-env",
    python_version: "3.10",
    dependencies: ["nonexistent-package-xyz"],
    path: "/var/ygg/envs/broken-env",
    status: "failed",
    created_at: "2025-05-23T09:00:00Z",
    updated_at: "2025-05-23T09:01:00Z",
    error: "pip install failed: No matching distribution found for nonexistent-package-xyz",
  },
];

function statusDotClass(status: string): string {
  switch (status) {
    case "ready": return "status-dot online";
    case "creating": return "status-dot pending";
    case "failed": return "status-dot offline";
    default: return "status-dot";
  }
}

function statusLabel(status: string): string {
  switch (status) {
    case "ready": return "Ready";
    case "creating": return "Creating...";
    case "failed": return "Failed";
    default: return status;
  }
}

export default function EnvironmentsPage() {
  const [environments, setEnvironments] = useState<EnvironmentEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [installFormId, setInstallFormId] = useState<number | null>(null);
  const [installPackages, setInstallPackages] = useState("");
  const [installing, setInstalling] = useState(false);

  // Form state
  const [formName, setFormName] = useState("");
  const [formPythonVersion, setFormPythonVersion] = useState("3.12");
  const [formDeps, setFormDeps] = useState("");
  const [formSubmitting, setFormSubmitting] = useState(false);

  useEffect(() => {
    loadEnvironments();
  }, []);

  async function loadEnvironments() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listEnvironments();
      setEnvironments(data.environments);
    } catch {
      setError("Node unavailable - showing demo data");
      setEnvironments(DEMO_ENVIRONMENTS);
    }
    setLoading(false);
  }

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setFormSubmitting(true);
    try {
      const deps = formDeps.split("\n").map((d) => d.trim()).filter(Boolean);
      await api.createEnvironment({
        name: formName,
        python_version: formPythonVersion,
        dependencies: deps,
      });
      setShowForm(false);
      setFormName("");
      setFormDeps("");
      await loadEnvironments();
    } catch (e) {
      setError(`Create failed: ${e}`);
    }
    setFormSubmitting(false);
  }

  async function handleDelete(id: number) {
    if (!confirm("Delete this environment? This cannot be undone.")) return;
    try {
      await api.deleteEnvironment(id);
      await loadEnvironments();
    } catch (e) {
      setError(`Delete failed: ${e}`);
    }
  }

  async function handleInstall(id: number) {
    setInstalling(true);
    try {
      const packages = installPackages.split(",").map((p) => p.trim()).filter(Boolean);
      if (packages.length === 0) return;
      await api.installPackages(id, packages);
      setInstallFormId(null);
      setInstallPackages("");
      await loadEnvironments();
    } catch (e) {
      setError(`Install failed: ${e}`);
    }
    setInstalling(false);
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="text-center animate-in">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full spin-slow mx-auto mb-4" />
          <p className="text-primary font-mono text-sm pulse-primary">Loading environments...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 animate-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-foreground">Environments</h1>
          <p className="text-sm text-muted mt-0.5">
            {environments.length} environment{environments.length !== 1 ? "s" : ""} configured
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
            {showForm ? "Cancel" : "New Environment"}
          </button>
        </div>
      </div>

      {/* Creation Form */}
      {showForm && (
        <form onSubmit={handleCreate} className="nordic-card p-5 space-y-4">
          <h2 className="text-sm font-semibold text-foreground">Create Environment</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-muted mb-1.5">Name</label>
              <input
                type="text"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
                className="input-nordic w-full text-sm"
                placeholder="my-environment"
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
            <label className="block text-xs text-muted mb-1.5">Dependencies (one per line)</label>
            <textarea
              value={formDeps}
              onChange={(e) => setFormDeps(e.target.value)}
              className="input-nordic w-full text-sm font-mono"
              rows={5}
              placeholder={"requests\npandas\nnumpy"}
            />
          </div>
          <div className="flex justify-end">
            <button type="submit" className="btn-primary text-sm" disabled={formSubmitting}>
              {formSubmitting ? "Creating..." : "Create Environment"}
            </button>
          </div>
        </form>
      )}

      {/* Environments List */}
      <div className="space-y-3">
        {environments.map((env) => (
          <div key={env.id} className="nordic-card p-4">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-3 mb-2">
                  <div className={statusDotClass(env.status)} />
                  <Link href={`/node/environments/${env.id}`} className="font-mono text-sm font-medium text-foreground hover:text-primary transition-colors">{env.name}</Link>
                  <span
                    className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded"
                    style={{
                      color: env.status === "ready" ? "var(--success)" : env.status === "creating" ? "var(--warning)" : "var(--destructive)",
                      background: env.status === "ready" ? "rgba(74,222,128,0.1)" : env.status === "creating" ? "rgba(251,191,36,0.1)" : "rgba(239,68,68,0.1)",
                    }}
                  >
                    {statusLabel(env.status)}
                  </span>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-1 text-xs mb-3">
                  <div>
                    <span className="text-muted">Python</span>
                    <p className="font-mono text-foreground-dim">{env.python_version}</p>
                  </div>
                  <div>
                    <span className="text-muted">Packages</span>
                    <p className="text-foreground-dim">{env.dependencies.length}</p>
                  </div>
                  <div className="col-span-2">
                    <span className="text-muted">Path</span>
                    <p className="font-mono text-foreground-dim truncate text-[11px]">{env.path}</p>
                  </div>
                </div>

                {/* Dependencies pills */}
                {env.dependencies.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 mb-3">
                    {env.dependencies.map((dep) => (
                      <span key={dep} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-border/50 text-muted border border-border">
                        {dep}
                      </span>
                    ))}
                  </div>
                )}

                {/* Error message */}
                {env.error && (
                  <div className="mt-2 p-3 rounded-lg bg-destructive/5 border border-destructive/20">
                    <pre className="text-xs text-destructive/80 font-mono whitespace-pre-wrap">{env.error}</pre>
                  </div>
                )}

                {/* Install form inline */}
                {installFormId === env.id && (
                  <div className="mt-3 flex items-center gap-2">
                    <input
                      type="text"
                      value={installPackages}
                      onChange={(e) => setInstallPackages(e.target.value)}
                      className="input-nordic text-sm flex-1"
                      placeholder="package1, package2"
                      onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); handleInstall(env.id); } }}
                    />
                    <button
                      onClick={() => handleInstall(env.id)}
                      disabled={installing}
                      className="btn-primary text-xs"
                    >
                      {installing ? "..." : "Install"}
                    </button>
                    <button
                      onClick={() => { setInstallFormId(null); setInstallPackages(""); }}
                      className="btn-ghost text-xs"
                    >
                      Cancel
                    </button>
                  </div>
                )}

                {/* Timestamps */}
                <div className="flex gap-4 text-[10px] text-muted mt-2">
                  <span>Created: {new Date(env.created_at).toLocaleDateString()}</span>
                  <span>Updated: {new Date(env.updated_at).toLocaleDateString()}</span>
                </div>
              </div>

              {/* Actions */}
              <div className="flex flex-col gap-2 shrink-0">
                {env.status === "ready" && installFormId !== env.id && (
                  <button
                    onClick={() => { setInstallFormId(env.id); setInstallPackages(""); }}
                    className="btn-ghost text-xs"
                  >
                    Install Pkg
                  </button>
                )}
                <button
                  onClick={() => handleDelete(env.id)}
                  className="btn-ghost text-xs text-destructive hover:bg-destructive/10"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {environments.length === 0 && (
        <div className="nordic-card p-8 text-center">
          <p className="text-muted text-sm">No environments configured yet.</p>
          <button onClick={() => setShowForm(true)} className="btn-primary text-sm mt-4">
            Create your first environment
          </button>
        </div>
      )}
    </div>
  );
}
