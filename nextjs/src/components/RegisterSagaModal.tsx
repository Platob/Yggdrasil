"use client";

import { useEffect, useState } from "react";
import { getTable, registerFile, OBJECT_TYPES, type TableEntry } from "@/lib/api";

interface Props {
  /** Backing data path/URL (node-home-relative, the rooting Saga uses). */
  source: string;
  /** Node that holds the bytes (omit for local). */
  node?: string;
  onClose: () => void;
  onDone?: (t: TableEntry) => void;
}

function stem(path: string): string {
  return (path.split("/").pop() ?? path).replace(/\.[^.]+$/, "");
}

/**
 * Reusable "register in Saga" modal. Pre-fills the form from the asset's own
 * config (catalog/schema/name inferred from the path), and if that asset is
 * already registered, pre-fills with the *existing* definition so the user
 * validates or tweaks before saving rather than starting blank.
 */
export default function RegisterSagaModal({ source, node, onClose, onDone }: Props) {
  const [catalog, setCatalog] = useState("main");
  const [schema, setSchema] = useState("default");
  const [table, setTable] = useState(stem(source));
  const [objectType, setObjectType] = useState("TABLE");
  const [comment, setComment] = useState("");
  const [existing, setExisting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  // Auto-fill from an existing registration when one matches the inferred name.
  useEffect(() => {
    let live = true;
    getTable("main", "default", stem(source), node)
      .then((r) => {
        if (!live) return;
        const t = r.table;
        setExisting(true);
        setCatalog(t.catalog); setSchema(t.schema); setTable(t.name);
        setObjectType(t.object_type); setComment(t.comment);
      })
      .catch(() => {});
    return () => { live = false; };
  }, [source, node]);

  const save = async () => {
    setBusy(true); setErr("");
    try {
      const r = await registerFile({
        source_url: source, catalog: catalog.trim(), schema: schema.trim(),
        table: table.trim(), node: node ?? null,
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
        {existing && <div className="text-[11px] text-amber/80">Already registered — prefilled with the existing definition; save to update.</div>}
        <div className="text-[11px] text-muted font-mono truncate" title={source}>{source}{node ? ` · @${node}` : ""}</div>
        <div className="grid grid-cols-3 gap-2">
          <label className="text-[11px] text-muted">catalog
            <input value={catalog} onChange={(e) => setCatalog(e.target.value)} className={field} />
          </label>
          <label className="text-[11px] text-muted">schema
            <input value={schema} onChange={(e) => setSchema(e.target.value)} className={field} />
          </label>
          <label className="text-[11px] text-muted">name
            <input value={table} onChange={(e) => setTable(e.target.value)} className={field} />
          </label>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <label className="text-[11px] text-muted">type
            <select value={objectType} onChange={(e) => setObjectType(e.target.value)} className={field}>
              {OBJECT_TYPES.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </label>
          <label className="text-[11px] text-muted">comment
            <input value={comment} onChange={(e) => setComment(e.target.value)} className={field} />
          </label>
        </div>
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
