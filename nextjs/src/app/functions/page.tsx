"use client";

// Functions & Environments — the accessible place to view, edit, run, and
// delete PyFuncs and manage PyEnvs. Function code comes back in the list
// (PyFuncEntry.code), so editing is a straight upsert (POST by name).

import { useCallback, useEffect, useState } from "react";
import {
  getFuncs,
  createFunc,
  deleteFunc,
  runFuncByName,
  getEnvs,
  createEnv,
  deleteEnv,
  inferFunc,
  type PyFuncInferResult,
} from "@/lib/api";
import type { PyFuncEntry, PyEnvEntry } from "@/lib/types";

const BLANK = { name: "", description: "", python_version: "", deps: "", code: "def run():\n    return 42\n" };

export default function FunctionsPage() {
  const [funcs, setFuncs] = useState<PyFuncEntry[]>([]);
  const [envs, setEnvs] = useState<PyEnvEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  // Function editor modal
  const [editing, setEditing] = useState<typeof BLANK | null>(null);
  const [inferred, setInferred] = useState<PyFuncInferResult | null>(null);
  const [isNew, setIsNew] = useState(false);

  // Live infer: scan the code as it's edited (debounced) for the signature,
  // typed params and version-pinned dependencies.
  const code = editing?.code;
  useEffect(() => {
    if (!code) { setInferred(null); return; }
    const t = setTimeout(() => { inferFunc(code).then(setInferred).catch(() => setInferred(null)); }, 500);
    return () => clearTimeout(t);
  }, [code]);
  const [saving, setSaving] = useState(false);
  const [editErr, setEditErr] = useState<string | null>(null);

  // Per-function run output
  const [running, setRunning] = useState<string | null>(null);
  const [runOut, setRunOut] = useState<Record<string, string>>({});

  // New env form
  const [envName, setEnvName] = useState("");
  const [envPy, setEnvPy] = useState("3.11");
  const [envDeps, setEnvDeps] = useState("");
  const [envBusy, setEnvBusy] = useState(false);

  const load = useCallback(async () => {
    setLoading(true); setError(false);
    try {
      const [f, e] = await Promise.allSettled([getFuncs(true), getEnvs(true)]);
      if (f.status === "fulfilled") setFuncs(f.value.funcs);
      if (e.status === "fulfilled") setEnvs(e.value.envs);
      if (f.status === "rejected" && e.status === "rejected") setError(true);
    } finally { setLoading(false); }
  }, []);
  useEffect(() => { load(); }, [load]);

  const openEdit = (fn?: PyFuncEntry) => {
    setEditErr(null);
    if (fn) {
      setIsNew(false);
      setEditing({ name: fn.name, description: fn.description, python_version: fn.python_version ?? "", deps: fn.dependencies.join(", "), code: fn.code });
    } else {
      setIsNew(true);
      setEditing({ ...BLANK });
    }
  };

  const save = async () => {
    if (!editing) return;
    setSaving(true); setEditErr(null);
    try {
      await createFunc({
        name: editing.name.trim(),
        code: editing.code,
        description: editing.description,
        python_version: editing.python_version || undefined,
        dependencies: editing.deps.split(",").map((d) => d.trim()).filter(Boolean),
      });
      setEditing(null);
      await load();
    } catch (e) {
      setEditErr(e instanceof Error ? e.message : "save failed");
    } finally { setSaving(false); }
  };

  const run = async (name: string) => {
    setRunning(name);
    try {
      const r = await runFuncByName(name);
      setRunOut((o) => ({ ...o, [name]: typeof r === "string" ? r : JSON.stringify(r) }));
    } catch (e) {
      setRunOut((o) => ({ ...o, [name]: `✗ ${e instanceof Error ? e.message : "run failed"}` }));
    } finally { setRunning(null); }
  };

  const remove = async (fn: PyFuncEntry) => {
    if (!confirm(`Delete function ${fn.name}?`)) return;
    try { await deleteFunc(fn.id); await load(); } catch { /* keep list */ }
  };

  const addEnv = async () => {
    if (!envName.trim()) return;
    setEnvBusy(true);
    try {
      await createEnv(envName.trim(), envPy, envDeps.split(",").map((d) => d.trim()).filter(Boolean));
      setEnvName(""); setEnvDeps("");
      await load();
    } catch { /* surfaced on reload */ } finally { setEnvBusy(false); }
  };

  const removeEnv = async (e: PyEnvEntry) => {
    if (!confirm(`Delete environment ${e.name}?`)) return;
    try { await deleteEnv(e.id); await load(); } catch { /* keep */ }
  };

  return (
    <div className="p-6 h-screen overflow-y-auto animate-in">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Functions &amp; Environments</h1>
          <p className="text-sm text-muted mt-1">Executable thoughts and their execution chambers — view, edit, run, manage.</p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => load()} className="px-3 py-1.5 rounded-lg text-xs font-medium text-frost/70 hover:text-frost bg-frost/5 border border-frost/10">Refresh</button>
          <button onClick={() => openEdit()} className="px-3 py-1.5 rounded-lg text-xs font-semibold text-emerald bg-emerald/15 border border-emerald/30 hover:bg-emerald/25">+ New function</button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Functions */}
        <div className="lg:col-span-2 space-y-2">
          <h2 className="text-[11px] uppercase tracking-wider text-muted/70 font-mono">Functions ({funcs.length})</h2>
          {loading ? (
            <div className="glass-card p-8 text-center"><div className="w-6 h-6 border-2 border-frost/30 border-t-frost rounded-full spin-slow mx-auto" /></div>
          ) : error ? (
            <div className="glass-card p-8 text-center text-sm text-muted">Backend unreachable.</div>
          ) : funcs.length === 0 ? (
            <div className="glass-card p-8 text-center text-sm text-muted">No functions yet — create one.</div>
          ) : funcs.map((fn) => (
            <div key={fn.id} className="glass-card p-3">
              <div className="flex items-center gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-mono font-semibold text-foreground truncate">{fn.name}</span>
                    {fn.python_version && <span className="text-[9px] font-mono text-muted bg-white/[0.05] rounded px-1.5 py-0.5">py{fn.python_version}</span>}
                    <span className="text-[9px] font-mono text-muted/70">{fn.run_count} runs</span>
                  </div>
                  {fn.description && <p className="text-[11px] text-muted truncate mt-0.5">{fn.description}</p>}
                  {fn.dependencies.length > 0 && <p className="text-[10px] font-mono text-frost/50 truncate mt-0.5">deps: {fn.dependencies.join(", ")}</p>}
                </div>
                <div className="flex items-center gap-1.5 shrink-0 text-xs font-mono">
                  <button onClick={() => run(fn.name)} disabled={running === fn.name} className="px-2.5 py-1 rounded bg-emerald/10 text-emerald border border-emerald/20 hover:bg-emerald/20 disabled:opacity-40">{running === fn.name ? "…" : "run"}</button>
                  <button onClick={() => openEdit(fn)} className="px-2.5 py-1 rounded bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20">edit</button>
                  <button onClick={() => remove(fn)} className="px-2.5 py-1 rounded bg-rose/10 text-rose border border-rose/20 hover:bg-rose/20">del</button>
                </div>
              </div>
              {runOut[fn.name] != null && (
                <pre className="mt-2 text-[11px] font-mono bg-black/30 rounded p-2 text-foreground/80 overflow-auto max-h-32 whitespace-pre-wrap break-words">{runOut[fn.name]}</pre>
              )}
            </div>
          ))}
        </div>

        {/* Environments */}
        <div className="space-y-2">
          <h2 className="text-[11px] uppercase tracking-wider text-muted/70 font-mono">Environments ({envs.length})</h2>
          <div className="glass-card p-3 space-y-2">
            <div className="text-[10px] uppercase tracking-wider text-muted/60">new env</div>
            <input value={envName} onChange={(e) => setEnvName(e.target.value)} placeholder="name" className="w-full bg-white/[0.04] border border-white/10 rounded px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
            <div className="flex gap-2">
              <select value={envPy} onChange={(e) => setEnvPy(e.target.value)} className="bg-white/[0.04] border border-white/10 rounded px-2 py-1.5 text-xs font-mono outline-none">
                {["3.11", "3.12", "3.13"].map((v) => <option key={v} value={v}>py{v}</option>)}
              </select>
              <input value={envDeps} onChange={(e) => setEnvDeps(e.target.value)} placeholder="deps (comma)" className="flex-1 bg-white/[0.04] border border-white/10 rounded px-2 py-1.5 text-xs font-mono outline-none" />
            </div>
            <button onClick={addEnv} disabled={envBusy || !envName.trim()} className="w-full px-3 py-1.5 rounded text-xs font-semibold text-emerald bg-emerald/15 border border-emerald/30 hover:bg-emerald/25 disabled:opacity-40">{envBusy ? "creating…" : "Create env"}</button>
          </div>
          {envs.map((e) => (
            <div key={e.id} className="glass-card p-3">
              <div className="flex items-center gap-2">
                <span className={`w-1.5 h-1.5 rounded-full ${e.status === "ready" ? "status-online" : "bg-amber"}`} />
                <span className="text-sm font-mono font-semibold text-foreground truncate flex-1">{e.name}</span>
                <span className="text-[9px] font-mono text-muted bg-white/[0.05] rounded px-1.5 py-0.5">py{e.python_version}</span>
                <button onClick={() => removeEnv(e)} className="text-rose/60 hover:text-rose text-xs font-mono">del</button>
              </div>
              <p className="text-[10px] font-mono text-muted/70 mt-1">{e.status}{e.dependencies.length ? ` · ${e.dependencies.length} deps` : ""}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Function code editor */}
      {editing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-6">
          <div className="absolute inset-0 bg-[var(--modal-scrim)] backdrop-blur-sm" onClick={() => setEditing(null)} />
          <div className="relative modal-surface p-5 w-full max-w-2xl max-h-[88vh] z-10 flex flex-col gap-3">
            <h3 className="text-sm font-mono font-semibold text-foreground">{isNew ? "New function" : `Edit ${editing.name}`}</h3>
            <div className="grid grid-cols-2 gap-2">
              <input value={editing.name} disabled={!isNew} onChange={(e) => setEditing({ ...editing, name: e.target.value })} placeholder="name" className="bg-white/[0.04] border border-white/10 rounded px-2 py-1.5 text-xs font-mono outline-none disabled:opacity-60" />
              <input value={editing.python_version} onChange={(e) => setEditing({ ...editing, python_version: e.target.value })} placeholder="python (e.g. 3.11)" className="bg-white/[0.04] border border-white/10 rounded px-2 py-1.5 text-xs font-mono outline-none" />
            </div>
            <input value={editing.description} onChange={(e) => setEditing({ ...editing, description: e.target.value })} placeholder="description" className="bg-white/[0.04] border border-white/10 rounded px-2 py-1.5 text-xs outline-none" />
            <input value={editing.deps} onChange={(e) => setEditing({ ...editing, deps: e.target.value })} placeholder="dependencies (comma) — inferred from imports if blank" className="bg-white/[0.04] border border-white/10 rounded px-2 py-1.5 text-xs font-mono outline-none" />
            {inferred && (
              <div className="flex items-center gap-2 text-[11px] font-mono flex-wrap rounded border border-frost/15 bg-frost/[0.04] px-2 py-1.5">
                <span className="text-frost/90 truncate" title={inferred.signature}>ƒ {inferred.signature}</span>
                {inferred.params.map((p) => p.dtype && (
                  <span key={p.name} className="px-1.5 py-0.5 rounded bg-white/[0.05] text-foreground-dim" title={`${p.name}: ${p.annotation}`}>{p.name}:{p.dtype}</span>
                ))}
                {inferred.dependencies.length > 0 && <span className="text-emerald/70" title={inferred.dependencies.join(", ")}>⬡ {inferred.dependencies.length} dep(s)</span>}
                <button onClick={() => setEditing((ed) => ed && ({
                  ...ed,
                  name: isNew && !ed.name.trim() ? inferred.name : ed.name,
                  python_version: ed.python_version.trim() || inferred.python_version,
                  deps: ed.deps.trim() || inferred.dependencies.join(", "),
                  description: ed.description.trim() || inferred.docstring,
                }))} className="ml-auto text-frost/80 hover:text-frost">use ↵</button>
              </div>
            )}
            <textarea value={editing.code} onChange={(e) => setEditing({ ...editing, code: e.target.value })} spellCheck={false} className="flex-1 min-h-[40vh] bg-black/40 border border-white/10 rounded p-3 text-xs font-mono text-foreground/90 outline-none focus:border-frost/30 resize-none" />
            {editErr && <div className="text-[11px] text-rose/90 font-mono">{editErr}</div>}
            <div className="flex items-center gap-2">
              <button onClick={save} disabled={saving || !editing.name.trim()} className="px-4 py-2 rounded-lg text-xs font-semibold bg-emerald/15 text-emerald border border-emerald/30 hover:bg-emerald/25 disabled:opacity-40">{saving ? "Saving…" : "Save"}</button>
              <button onClick={() => setEditing(null)} className="px-4 py-2 rounded-lg text-xs font-medium text-muted hover:text-foreground ml-auto">Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
