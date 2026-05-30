"use client";

import { useEffect, useState } from "react";
import { getTable, getMe, registerFile, OBJECT_TYPES, type TableEntry } from "@/lib/api";

export interface RegisterDefaults {
  catalog?: string; schema?: string; name?: string;
  objectType?: string; definition?: string;
}

interface Props {
  /** Backing data path/URL for a TABLE (omit for a VIEW). */
  source?: string;
  node?: string;
  /** Seed the form (e.g. register the current query as a VIEW). */
  defaults?: RegisterDefaults;
  onClose: () => void;
  onDone?: (t: TableEntry) => void;
}

function stem(path: string): string {
  return (path.split("/").pop() ?? path).replace(/\.[^.]+$/, "");
}

/**
 * Reusable "register in Saga" modal. Pre-fills from the asset's config (path →
 * catalog/schema/name, or seeded defaults for a view). If the target already
 * exists it warns and pre-fills with the *existing* definition so the user
 * validates/merges before saving rather than starting blank. A view defaults to
 * the `staging` catalog / `default` schema, named `<user>_<timestamp>`.
 */
export default function RegisterSagaModal({ source, node, defaults, onClose, onDone }: Props) {
  const isView = (defaults?.objectType ?? "TABLE") !== "TABLE";
  const [catalog, setCatalog] = useState(defaults?.catalog ?? (isView ? "staging" : "main"));
  const [schema, setSchema] = useState(defaults?.schema ?? "default");
  const [table, setTable] = useState(defaults?.name ?? (source ? stem(source) : ""));
  const [objectType, setObjectType] = useState(defaults?.objectType ?? "TABLE");
  const [definition, setDefinition] = useState(defaults?.definition ?? "");
  const [comment, setComment] = useState("");
  const [existing, setExisting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  // Default a view's name to <user>_<timestamp> when none was seeded.
  useEffect(() => {
    if (defaults?.name || source) return;
    getMe().then((u) => {
      const who = (u.first_name || u.hostname || "user").replace(/\W+/g, "");
      const ts = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
      setTable(`${who}_${ts}`);
    }).catch(() => setTable(`view_${Date.now()}`));
  }, [defaults?.name, source]);

  // Pre-fill from an existing registration matching the (initial) target.
  useEffect(() => {
    let live = true;
    const c = defaults?.catalog ?? (isView ? "staging" : "main");
    const s = defaults?.schema ?? "default";
    const n = defaults?.name ?? (source ? stem(source) : "");
    if (!n) return;
    getTable(c, s, n, node).then((r) => {
      if (!live) return;
      const t = r.table;
      setExisting(true);
      setCatalog(t.catalog); setSchema(t.schema); setTable(t.name);
      setObjectType(t.object_type); setComment(t.comment);
      if (t.definition) setDefinition(t.definition);
    }).catch(() => {});
    return () => { live = false; };
  }, [source, node, defaults?.catalog, defaults?.schema, defaults?.name, isView]);

  const save = async () => {
    setBusy(true); setErr("");
    try {
      const r = await registerFile({
        source_url: source ?? "", catalog: catalog.trim(), schema: schema.trim(),
        table: table.trim(), object_type: objectType,
        definition: objectType === "TABLE" ? "" : definition, node: node ?? null,
      });
      onDone?.(r.table);
      onClose();
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  const field = "w-full mt-1 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6"
      style={{ background: "var(--modal-scrim, rgba(0,0,0,0.72))" }} onClick={onClose}>
      <div className="modal-surface rounded-xl w-full max-w-md p-5 space-y-3" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold text-emerald">Register in Saga</span>
          <button onClick={onClose} className="text-muted hover:text-foreground text-sm">✕</button>
        </div>
        {existing && <div className="text-[11px] text-amber/80">⚠ Already registered — prefilled with the existing definition; saving updates it (merge your changes first).</div>}
        {source && <div className="text-[11px] text-muted font-mono truncate" title={source}>{source}{node ? ` · @${node}` : ""}</div>}
        <div className="grid grid-cols-3 gap-2">
          <label className="text-[11px] text-muted">catalog<input value={catalog} onChange={(e) => setCatalog(e.target.value)} className={field} /></label>
          <label className="text-[11px] text-muted">schema<input value={schema} onChange={(e) => setSchema(e.target.value)} className={field} /></label>
          <label className="text-[11px] text-muted">name<input value={table} onChange={(e) => setTable(e.target.value)} className={field} /></label>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <label className="text-[11px] text-muted">type
            <select value={objectType} onChange={(e) => setObjectType(e.target.value)} className={field}>
              {OBJECT_TYPES.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </label>
          <label className="text-[11px] text-muted">comment<input value={comment} onChange={(e) => setComment(e.target.value)} className={field} /></label>
        </div>
        {objectType !== "TABLE" && (
          <label className="text-[11px] text-muted block">definition
            <textarea value={definition} onChange={(e) => setDefinition(e.target.value)} rows={4}
              className="w-full mt-1 bg-[#06060f] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
          </label>
        )}
        {err && <div className="text-[11px] text-rose/90 font-mono break-words">{err}</div>}
        <div className="flex justify-end gap-2 pt-1">
          <button onClick={onClose} className="px-3 py-1.5 rounded-lg text-xs bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08]">cancel</button>
          <button onClick={save} disabled={busy}
            className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-emerald/15 text-emerald border border-emerald/30 hover:bg-emerald/25 disabled:opacity-40">
            {busy ? "Saving…" : existing ? "Update" : "Register"}
          </button>
        </div>
      </div>
    </div>
  );
}
