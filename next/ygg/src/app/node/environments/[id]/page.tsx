"use client";

import { useEffect, useState, use } from "react";
import { node as api, type EnvironmentEntry } from "@/lib/api";
import Link from "next/link";

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

function statusColor(status: string): string {
  switch (status) {
    case "ready": return "var(--success)";
    case "creating": return "var(--warning)";
    case "failed": return "var(--destructive)";
    default: return "var(--muted)";
  }
}

export default function EnvironmentDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const numericId = parseInt(id, 10);
  const [env, setEnv] = useState<EnvironmentEntry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [installPackages, setInstallPackages] = useState("");
  const [installing, setInstalling] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    loadEnvironment();
  }, [numericId]);

  async function loadEnvironment() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getEnvironment(numericId);
      setEnv(data.environment);
    } catch {
      setError("Failed to load environment");
    }
    setLoading(false);
  }

  async function handleInstall(e: React.FormEvent) {
    e.preventDefault();
    const packages = installPackages.split(",").map((p) => p.trim()).filter(Boolean);
    if (packages.length === 0) return;
    setInstalling(true);
    try {
      await api.installPackages(numericId, packages);
      setInstallPackages("");
      await loadEnvironment();
    } catch (e) {
      setError(`Install failed: ${e}`);
    }
    setInstalling(false);
  }

  async function handleDelete() {
    if (!confirm("Delete this environment? This cannot be undone.")) return;
    setDeleting(true);
    try {
      await api.deleteEnvironment(numericId);
      window.location.href = "/node/environments";
    } catch (e) {
      setError(`Delete failed: ${e}`);
      setDeleting(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="text-center animate-in">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full spin-slow mx-auto mb-4" />
          <p className="text-primary font-mono text-sm pulse-primary">Loading environment...</p>
        </div>
      </div>
    );
  }

  if (!env) {
    return (
      <div className="p-6 animate-in">
        <p className="text-muted">{error || "Environment not found."}</p>
        <Link href="/node/environments" className="btn-ghost text-sm mt-4 inline-block">Back to Environments</Link>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 animate-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <Link href="/node/environments" className="text-muted hover:text-foreground transition-colors text-sm">
              Environments
            </Link>
            <span className="text-muted">/</span>
            <h1 className="text-xl font-bold text-foreground font-mono">{env.name}</h1>
          </div>
          <div className="flex items-center gap-2 mt-1">
            <div className={statusDotClass(env.status)} />
            <span
              className="text-xs font-semibold uppercase px-1.5 py-0.5 rounded"
              style={{
                color: statusColor(env.status),
                background: `color-mix(in srgb, ${statusColor(env.status)} 15%, transparent)`,
              }}
            >
              {statusLabel(env.status)}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {error && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-warning/10 border border-warning/20">
              <div className="status-dot pending" />
              <span className="text-xs font-medium text-warning">{error}</span>
            </div>
          )}
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="btn-ghost text-sm text-destructive hover:bg-destructive/10"
          >
            {deleting ? "Deleting..." : "Delete"}
          </button>
        </div>
      </div>

      {/* Metadata */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "ID", value: String(env.id) },
          { label: "Python", value: env.python_version },
          { label: "Packages", value: String(env.dependencies.length) },
          { label: "Status", value: statusLabel(env.status) },
        ].map((item) => (
          <div key={item.label} className="nordic-card p-3">
            <span className="text-[10px] uppercase tracking-wider text-muted">{item.label}</span>
            <p className="text-sm font-mono text-foreground mt-0.5">{item.value}</p>
          </div>
        ))}
      </div>

      {/* Path */}
      <div className="nordic-card p-4">
        <span className="text-[10px] uppercase tracking-wider text-muted">Path</span>
        <p className="text-sm font-mono text-foreground-dim mt-0.5">{env.path}</p>
      </div>

      {/* Timestamps */}
      <div className="flex gap-6 text-xs text-muted">
        <span>Created: {new Date(env.created_at).toLocaleString()}</span>
        <span>Updated: {new Date(env.updated_at).toLocaleString()}</span>
      </div>

      {/* Error message */}
      {env.error && (
        <div className="p-4 rounded-lg bg-destructive/5 border border-destructive/20">
          <span className="text-[10px] uppercase tracking-wider text-destructive font-semibold">Error</span>
          <pre className="text-xs text-destructive/80 font-mono whitespace-pre-wrap mt-1">{env.error}</pre>
        </div>
      )}

      {/* Dependencies */}
      <div>
        <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-2">Dependencies</h2>
        {env.dependencies.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {env.dependencies.map((dep) => (
              <span key={dep} className="text-xs font-mono px-2 py-1 rounded bg-border/50 text-foreground-dim border border-border">
                {dep}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted">No packages installed.</p>
        )}
      </div>

      {/* Install packages form */}
      {env.status === "ready" && (
        <div>
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-2">Install Packages</h2>
          <form onSubmit={handleInstall} className="nordic-card p-4 flex items-center gap-3">
            <input
              type="text"
              value={installPackages}
              onChange={(e) => setInstallPackages(e.target.value)}
              className="input-nordic text-sm flex-1"
              placeholder="package1, package2, package3"
            />
            <button type="submit" disabled={installing} className="btn-primary text-sm">
              {installing ? "Installing..." : "Install"}
            </button>
          </form>
        </div>
      )}

      {/* Back button */}
      <div>
        <Link href="/node/environments" className="btn-ghost text-sm inline-block">
          Back to Environments
        </Link>
      </div>
    </div>
  );
}
