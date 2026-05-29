"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getFsNodes,
  getCatalogs,
  getSchemas,
  getTables,
  getTable,
  createCatalog,
  createSchema,
  createTable,
  refreshTable,
  deleteCatalogEntity,
  discoverTables,
  runSql,
  explainSql,
  getTableLog,
  replicateTable,
  type CatalogEntry,
  type SchemaEntry,
  type TableEntry,
  type SqlResult,
  type ExplainResult,
  type OpLogEntry,
} from "@/lib/api";
import TabularModal from "@/components/TabularModal";

const DIALECTS = ["postgres", "sqlite", "mysql", "databricks"];

function fmtBytes(b: number | null | undefined): string {
  if (!b) return "--";
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

const chip = "px-2 py-0.5 rounded text-[10px] font-mono border border-white/[0.08] bg-white/[0.03] text-foreground-dim";

export default function SagaPage() {
  const [nodes, setNodes] = useState<{ node_id: string; self: boolean }[]>([]);
  const [node, setNode] = useState<string | undefined>(undefined);

  const [catalogs, setCatalogs] = useState<CatalogEntry[]>([]);
  const [openCat, setOpenCat] = useState<Set<string>>(new Set());
  const [openSch, setOpenSch] = useState<Set<string>>(new Set());
  const [schemas, setSchemas] = useState<Record<string, SchemaEntry[]>>({});
  const [tables, setTables] = useState<Record<string, TableEntry[]>>({});
  const [detail, setDetail] = useState<TableEntry | null>(null);
  const [log, setLog] = useState<OpLogEntry[] | null>(null);
  const [preview, setPreview] = useState<{ path: string; name: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [treeErr, setTreeErr] = useState("");

  // -- SQL editor state
  const [sql, setSql] = useState("SELECT * FROM ");
  const [dialect, setDialect] = useState("postgres");
  const [ctx, setCtx] = useState<{ catalog?: string; schema?: string }>({});
  const [result, setResult] = useState<SqlResult | null>(null);
  const [explain, setExplain] = useState<ExplainResult | null>(null);
  const [tab, setTab] = useState<"results" | "plan">("results");
  const [running, setRunning] = useState(false);
  const [sqlErr, setSqlErr] = useState("");

  const loadCatalogs = useCallback(async () => {
    setTreeErr("");
    try {
      const r = await getCatalogs(node, true);
      setCatalogs(r.catalogs);
    } catch (e) {
      setTreeErr(String(e));
    }
  }, [node]);

  useEffect(() => {
    getFsNodes().then((r) => {
      setNodes(r.nodes.map((n) => ({ node_id: n.node_id, self: n.self })));
    }).catch(() => {});
  }, []);

  useEffect(() => { loadCatalogs(); }, [loadCatalogs]);

  const toggleCat = async (c: CatalogEntry) => {
    const next = new Set(openCat);
    if (next.has(c.name)) { next.delete(c.name); setOpenCat(next); return; }
    next.add(c.name); setOpenCat(next);
    if (!schemas[c.name]) {
      try {
        const r = await getSchemas(c.name, node, true);
        setSchemas((s) => ({ ...s, [c.name]: r.schemas }));
      } catch (e) { setTreeErr(String(e)); }
    }
  };

  const toggleSch = async (cat: string, s: SchemaEntry) => {
    const key = `${cat}.${s.name}`;
    const next = new Set(openSch);
    if (next.has(key)) { next.delete(key); setOpenSch(next); return; }
    next.add(key); setOpenSch(next);
    if (!tables[key]) {
      try {
        const r = await getTables(cat, s.name, node, true);
        setTables((t) => ({ ...t, [key]: r.tables }));
      } catch (e) { setTreeErr(String(e)); }
    }
  };

  const refreshSchema = async (cat: string, schema: string) => {
    const r = await getTables(cat, schema, node, true);
    setTables((t) => ({ ...t, [`${cat}.${schema}`]: r.tables }));
  };

  // -- mutations
  const onNewCatalog = async () => {
    const name = window.prompt("New catalog name");
    if (!name) return;
    setBusy(true);
    try { await createCatalog({ name: name.trim(), dialect }); await loadCatalogs(); }
    catch (e) { setTreeErr(String(e)); } finally { setBusy(false); }
  };

  const onNewSchema = async (cat: string) => {
    const name = window.prompt(`New schema in '${cat}'`);
    if (!name) return;
    setBusy(true);
    try {
      await createSchema(cat, { name: name.trim() });
      const r = await getSchemas(cat, node, true);
      setSchemas((s) => ({ ...s, [cat]: r.schemas }));
      setOpenCat((o) => new Set(o).add(cat));
    } catch (e) { setTreeErr(String(e)); } finally { setBusy(false); }
  };

  const onRegisterTable = async (cat: string, schema: string) => {
    const name = window.prompt("Table name");
    if (!name) return;
    const source_url = window.prompt("Source file path / URL (e.g. data/trades.parquet)");
    if (!source_url) return;
    setBusy(true);
    try {
      await createTable(cat, schema, { name: name.trim(), source_url: source_url.trim() });
      await refreshSchema(cat, schema);
      setOpenSch((o) => new Set(o).add(`${cat}.${schema}`));
    } catch (e) { setTreeErr(String(e)); } finally { setBusy(false); }
  };

  const onDiscover = async (cat: string, schema: string) => {
    const path = window.prompt("Folder to scan under the node files root", "data") ?? "";
    setBusy(true);
    try {
      await discoverTables({ catalog: cat, schema, path, node });
      await refreshSchema(cat, schema);
      setOpenSch((o) => new Set(o).add(`${cat}.${schema}`));
    } catch (e) { setTreeErr(String(e)); } finally { setBusy(false); }
  };

  const onDelete = async (path: string, label: string, after: () => void) => {
    if (!window.confirm(`Drop ${label}? This removes the catalog entry (not the data file).`)) return;
    setBusy(true);
    try { await deleteCatalogEntity(path); after(); }
    catch (e) { setTreeErr(String(e)); } finally { setBusy(false); }
  };

  const openTable = async (cat: string, schema: string, name: string) => {
    try {
      const r = await getTable(cat, schema, name, node);
      setDetail(r.table);
      setLog(null);
    } catch (e) { setTreeErr(String(e)); }
  };

  const loadLog = async (t: TableEntry) => {
    try {
      const r = await getTableLog(t.catalog, t.schema, t.name, node);
      setLog(r.entries);
    } catch (e) { setTreeErr(String(e)); }
  };

  const onReplicate = async (t: TableEntry) => {
    const target = window.prompt("Replicate to which node id?");
    if (!target) return;
    const mode = window.confirm("Copy the data file too? OK = data, Cancel = metadata only")
      ? "data" : "metadata";
    setBusy(true);
    try {
      const r = await replicateTable({ catalog: t.catalog, schema: t.schema, table: t.name, target: target.trim(), mode });
      window.alert(`Replicated ${r.full_name} → ${r.target_node} (${r.mode}, ${r.bytes_copied} B)`);
      await onRefreshTable(t);
    } catch (e) { setTreeErr(String(e)); } finally { setBusy(false); }
  };

  const onRefreshTable = async (t: TableEntry) => {
    setBusy(true);
    try {
      const r = await refreshTable(t.catalog, t.schema, t.name);
      setDetail(r.table);
      await refreshSchema(t.catalog, t.schema);
    } catch (e) { setTreeErr(String(e)); } finally { setBusy(false); }
  };

  const insertRef = (full: string) => {
    setSql((s) => (s.trimEnd().endsWith("FROM") ? `${s} ${full}` : `${s}${full}`));
  };

  // -- SQL run
  const onRun = async () => {
    setRunning(true); setSqlErr(""); setExplain(null);
    try {
      const r = await runSql({ sql, dialect, catalog: ctx.catalog, schema: ctx.schema, node });
      setResult(r); setTab("results");
    } catch (e) { setSqlErr(String(e)); setResult(null); }
    finally { setRunning(false); }
  };

  const onExplain = async () => {
    setRunning(true); setSqlErr("");
    try {
      const r = await explainSql({ sql, dialect, catalog: ctx.catalog, schema: ctx.schema });
      setExplain(r); setTab("plan");
    } catch (e) { setSqlErr(String(e)); }
    finally { setRunning(false); }
  };

  const onKey = (e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); onRun(); }
  };

  const nodeLabel = useMemo(
    () => (n: { node_id: string; self: boolean }) => (n.self ? `${n.node_id} (local)` : n.node_id),
    [],
  );

  return (
    <div className="h-full flex flex-col gap-3 p-4 animate-in">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h1 className="text-lg font-semibold gradient-frost">Saga</h1>
          <p className="text-[11px] text-muted">Distributed data catalog — register sources, query across the mesh.</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={node ?? "__local"}
            onChange={(e) => setNode(e.target.value === "__local" ? undefined : e.target.value)}
            className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30"
          >
            <option value="__local">this node</option>
            {nodes.filter((n) => !n.self).map((n) => (
              <option key={n.node_id} value={n.node_id}>{nodeLabel(n)}</option>
            ))}
          </select>
          <button onClick={onNewCatalog} disabled={busy}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald/15 text-emerald border border-emerald/30 hover:bg-emerald/25 disabled:opacity-40">
            + Catalog
          </button>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-3 min-h-0">
        {/* ── Catalog tree ── */}
        <div className="glass-card p-3 overflow-auto min-h-0">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] uppercase tracking-wide text-muted">Catalogs</span>
            <button onClick={loadCatalogs} className="text-[11px] text-frost/80 hover:text-frost">refresh</button>
          </div>
          {treeErr && <div className="text-[11px] text-rose/80 font-mono mb-2 break-words">{treeErr}</div>}
          {catalogs.length === 0 && <div className="text-[11px] text-muted py-2">No catalogs yet. Create one to begin.</div>}
          <div className="space-y-0.5">
            {catalogs.map((c) => {
              const co = openCat.has(c.name);
              return (
                <div key={c.id}>
                  <div className="group flex items-center gap-1.5 py-1 rounded hover:bg-white/[0.03] cursor-pointer"
                    onClick={() => toggleCat(c)}>
                    <span className="text-frost/70 w-3 text-center text-[10px]">{co ? "▾" : "▸"}</span>
                    <span className="text-frost">🗄</span>
                    <span className="text-xs font-medium flex-1 truncate">{c.name}</span>
                    <span className={chip}>{c.schema_count}</span>
                    <button onClick={(e) => { e.stopPropagation(); onNewSchema(c.name); }}
                      className="opacity-0 group-hover:opacity-100 text-[11px] text-emerald/80 hover:text-emerald px-1">+sch</button>
                    <button onClick={(e) => { e.stopPropagation(); onDelete(`catalog/${c.name}?cascade=true`, `catalog '${c.name}'`, loadCatalogs); }}
                      className="opacity-0 group-hover:opacity-100 text-[11px] text-rose/70 hover:text-rose px-1">✕</button>
                  </div>
                  {co && (
                    <div className="ml-3 pl-2 border-l border-white/[0.06]">
                      {(schemas[c.name] ?? []).map((s) => {
                        const key = `${c.name}.${s.name}`;
                        const so = openSch.has(key);
                        return (
                          <div key={s.id}>
                            <div className="group flex items-center gap-1.5 py-1 rounded hover:bg-white/[0.03] cursor-pointer"
                              onClick={() => toggleSch(c.name, s)}>
                              <span className="text-frost/70 w-3 text-center text-[10px]">{so ? "▾" : "▸"}</span>
                              <span className="text-amber/80">▤</span>
                              <span className="text-xs flex-1 truncate">{s.name}</span>
                              <span className={chip}>{s.table_count}</span>
                              <button onClick={(e) => { e.stopPropagation(); onRegisterTable(c.name, s.name); }}
                                className="opacity-0 group-hover:opacity-100 text-[11px] text-emerald/80 hover:text-emerald px-1">+tbl</button>
                              <button onClick={(e) => { e.stopPropagation(); onDiscover(c.name, s.name); }}
                                className="opacity-0 group-hover:opacity-100 text-[11px] text-frost/70 hover:text-frost px-1">scan</button>
                            </div>
                            {so && (
                              <div className="ml-3 pl-2 border-l border-white/[0.06]">
                                {(tables[key] ?? []).map((t) => (
                                  <div key={t.id}
                                    className="group flex items-center gap-1.5 py-1 rounded hover:bg-white/[0.03] cursor-pointer"
                                    onClick={() => openTable(c.name, s.name, t.name)}
                                    onDoubleClick={() => insertRef(t.full_name)}
                                    title="click: details · double-click: insert into editor">
                                    <span className="w-3" />
                                    <span className="text-emerald/70">▦</span>
                                    <span className="text-xs flex-1 truncate">{t.name}</span>
                                    <span className={chip}>{t.format || "?"}</span>
                                  </div>
                                ))}
                                {(tables[key] ?? []).length === 0 && (
                                  <div className="text-[11px] text-muted py-1 pl-5">empty</div>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })}
                      {(schemas[c.name] ?? []).length === 0 && (
                        <div className="text-[11px] text-muted py-1 pl-5">no schemas</div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Table detail */}
          {detail && (
            <div className="mt-3 pt-3 border-t border-white/[0.08]">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-emerald truncate">{detail.full_name}</span>
                <button onClick={() => { setDetail(null); setLog(null); }} className="text-[11px] text-muted hover:text-foreground">close</button>
              </div>
              <div className="flex flex-wrap gap-2 mb-1.5 text-[11px]">
                <button onClick={() => setPreview({ path: detail.source_url, name: detail.name })}
                  className="text-frost/80 hover:text-frost">preview data</button>
                <button onClick={() => insertRef(detail.full_name)} className="text-amber/80 hover:text-amber">insert</button>
                <button onClick={() => onRefreshTable(detail)} disabled={busy} className="text-frost/80 hover:text-frost">refresh stats</button>
                <button onClick={() => onReplicate(detail)} disabled={busy} className="text-emerald/80 hover:text-emerald">replicate</button>
                <button onClick={() => loadLog(detail)} className="text-foreground-dim hover:text-foreground">history</button>
              </div>
              <div className="flex flex-wrap gap-3 text-[11px] text-foreground-dim mb-1.5 font-mono">
                <span>{detail.statistics.row_count ?? "?"} rows</span>
                <span>{fmtBytes(detail.statistics.size_bytes)}</span>
                <span className={chip}>{detail.table_type}</span>
                {detail.replicas.length > 0 && <span className="text-emerald/80">⧉ {detail.replicas.length} replica(s)</span>}
                <span className="truncate max-w-full">{detail.source_url}</span>
              </div>
              <div className="space-y-0.5 max-h-48 overflow-auto">
                {detail.columns.map((col) => {
                  const st = detail.statistics.columns.find((c) => c.column === col.name);
                  return (
                    <div key={col.name} className="flex items-center gap-2 text-[11px] py-0.5">
                      <span className="text-foreground font-mono flex-1 truncate">{col.name}</span>
                      <span className="text-frost/70">{col.dtype}</span>
                      {st && (st.min != null || st.max != null) && (
                        <span className="text-muted font-mono truncate max-w-[90px]" title={`min ${st.min} · max ${st.max}`}>
                          {String(st.min)}…{String(st.max)}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
              {log && (
                <div className="mt-2 pt-2 border-t border-white/[0.06]">
                  <div className="text-[11px] uppercase tracking-wide text-muted mb-1">History</div>
                  <div className="space-y-0.5 max-h-40 overflow-auto">
                    {log.length === 0 && <div className="text-[11px] text-muted">no operations logged</div>}
                    {log.map((e, i) => (
                      <div key={i} className="text-[11px] font-mono flex gap-2" title={e.statement}>
                        <span className="text-muted w-[88px] shrink-0 truncate">{e.ts.slice(5, 19).replace("T", " ")}</span>
                        <span className="text-frost/80 w-16 shrink-0">{e.op}</span>
                        <span className="text-foreground-dim truncate flex-1">{e.statement || e.detail}</span>
                        {e.rows != null && <span className="text-muted">{e.rows}r</span>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── SQL editor + results ── */}
        <div className="glass-card p-3 flex flex-col min-h-0">
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <select value={dialect} onChange={(e) => setDialect(e.target.value)}
              className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30">
              {DIALECTS.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
            <input placeholder="default catalog" value={ctx.catalog ?? ""}
              onChange={(e) => setCtx((c) => ({ ...c, catalog: e.target.value || undefined }))}
              className="w-28 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
            <input placeholder="default schema" value={ctx.schema ?? ""}
              onChange={(e) => setCtx((c) => ({ ...c, schema: e.target.value || undefined }))}
              className="w-28 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
            <div className="flex-1" />
            <button onClick={onExplain} disabled={running}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-white/[0.04] text-foreground-dim border border-white/[0.08] hover:bg-white/[0.08] disabled:opacity-40">
              Explain
            </button>
            <button onClick={onRun} disabled={running}
              className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-frost/15 text-frost border border-frost/30 hover:bg-frost/25 disabled:opacity-40">
              {running ? "Running…" : "Run ▸"}
            </button>
          </div>

          <textarea value={sql} onChange={(e) => setSql(e.target.value)} onKeyDown={onKey}
            spellCheck={false}
            className="w-full h-40 resize-y bg-[#06060f] border border-white/[0.08] rounded-lg p-3 text-[13px] font-mono text-foreground placeholder-muted/50 outline-none focus:border-frost/30 leading-relaxed"
            placeholder="SELECT * FROM catalog.schema.table  —  ⌘/Ctrl+Enter to run" />

          {sqlErr && <div className="mt-2 text-[12px] text-rose/90 font-mono break-words bg-rose/5 border border-rose/20 rounded-lg p-2">{sqlErr}</div>}

          {/* tabs */}
          <div className="flex items-center gap-3 mt-3 mb-1.5 text-[11px]">
            <button onClick={() => setTab("results")}
              className={tab === "results" ? "text-frost font-semibold" : "text-muted hover:text-foreground-dim"}>Results</button>
            <button onClick={() => setTab("plan")}
              className={tab === "plan" ? "text-frost font-semibold" : "text-muted hover:text-foreground-dim"}>Plan</button>
            {result && tab === "results" && (
              <span className="text-muted font-mono ml-auto">
                {result.row_count} rows · {result.elapsed_ms} ms{result.truncated ? " · truncated" : ""}
                {result.node_id ? ` · @${result.node_id}` : ""}
              </span>
            )}
          </div>

          <div className="flex-1 overflow-auto border border-white/[0.06] rounded-lg min-h-0">
            {tab === "results" && result && (
              <table className="w-full text-[12px] font-mono border-collapse">
                <thead className="sticky top-0 bg-[#0a0a1a] z-10">
                  <tr>
                    {result.columns.map((c) => (
                      <th key={c.name} className="text-left px-2 py-1.5 border-b border-white/[0.08] text-frost/80 whitespace-nowrap">
                        {c.name}<span className="text-muted ml-1 font-normal">{c.dtype}</span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row, ri) => (
                    <tr key={ri} className="hover:bg-white/[0.02]">
                      {row.map((cell, ci) => (
                        <td key={ci} className="px-2 py-1 border-b border-white/[0.04] text-foreground-dim whitespace-nowrap max-w-[280px] truncate">
                          {cell === null ? <span className="text-muted italic">null</span> : String(cell)}
                        </td>
                      ))}
                    </tr>
                  ))}
                  {result.rows.length === 0 && (
                    <tr><td className="px-2 py-3 text-muted text-center" colSpan={Math.max(1, result.columns.length)}>0 rows</td></tr>
                  )}
                </tbody>
              </table>
            )}
            {tab === "results" && !result && (
              <div className="p-4 text-[12px] text-muted">Run a query to see results.</div>
            )}
            {tab === "plan" && (
              <div className="p-3 text-[12px] font-mono space-y-3">
                {(explain ?? result) ? (
                  <>
                    <div>
                      <div className="text-[11px] uppercase tracking-wide text-muted mb-1">Emitted SQL</div>
                      <pre className="text-frost/90 whitespace-pre-wrap break-words bg-[#06060f] border border-white/[0.06] rounded p-2">{(explain?.plan_sql ?? result?.plan_sql) || "—"}</pre>
                    </div>
                    <div>
                      <div className="text-[11px] uppercase tracking-wide text-muted mb-1">Referenced tables</div>
                      <div className="flex flex-wrap gap-1.5">
                        {((explain?.referenced_tables ?? result?.referenced_tables) ?? []).map((t) => (
                          <span key={t} className={chip}>{t}</span>
                        ))}
                      </div>
                    </div>
                    {explain?.plan && (
                      <div>
                        <div className="text-[11px] uppercase tracking-wide text-muted mb-1">Plan tree</div>
                        <pre className="text-foreground-dim whitespace-pre-wrap break-words bg-[#06060f] border border-white/[0.06] rounded p-2 max-h-64 overflow-auto">{explain.plan}</pre>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-muted">Run or Explain a query to see the execution plan.</div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {preview && (
        <TabularModal
          node={node}
          nodeLabel={node ?? "local"}
          path={preview.path}
          name={preview.name}
          onClose={() => setPreview(null)}
        />
      )}
    </div>
  );
}
