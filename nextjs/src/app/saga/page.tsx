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
  getPlan,
  editPlan,
  downloadSqlExport,
  SQL_EXPORT_FORMATS,
  searchSaga,
  getTableLog,
  getActivity,
  updateTable,
  replicateTable,
  OBJECT_TYPES,
  type SearchHit,
  type ActivityResponse,
  type CatalogEntry,
  type SchemaEntry,
  type TableEntry,
  type PlanGraph,
  type PlanEdit,
  type OpLogEntry,
} from "@/lib/api";
import TabularModal from "@/components/TabularModal";
import TabularDisplay, { type QuerySpec } from "@/components/TabularDisplay";
import PlanGraphView from "@/components/PlanGraph";

const DIALECTS = ["postgres", "sqlite", "mysql", "databricks"];

function fmtBytes(b: number | null | undefined): string {
  if (!b) return "--";
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

const chip = "px-2 py-0.5 rounded text-[10px] font-mono border border-white/[0.08] bg-white/[0.03] text-foreground-dim";

// Per object-type glyph + colour so a schema's leaves read at a glance.
const OBJ: Record<string, { glyph: string; color: string }> = {
  TABLE: { glyph: "▦", color: "text-emerald/70" },
  VIEW: { glyph: "◫", color: "text-frost/70" },
  FUNCTION: { glyph: "ƒ", color: "text-amber/80" },
  MODEL: { glyph: "◈", color: "text-violet-400" },
  OTHER: { glyph: "○", color: "text-muted" },
};
const objOf = (t: string) => OBJ[t] ?? OBJ.OTHER;

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
  const [activity, setActivity] = useState<ActivityResponse | null>(null);
  const [edit, setEdit] = useState<TableEntry | null>(null);
  const [preview, setPreview] = useState<{ path: string; name: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [treeErr, setTreeErr] = useState("");
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<{ hits: SearchHit[]; total: number; truncated: boolean } | null>(null);

  // -- SQL editor state
  const [sql, setSql] = useState("SELECT * FROM ");
  const [dialect, setDialect] = useState("postgres");
  const [ctx, setCtx] = useState<{ catalog?: string; schema?: string }>({});
  const [ranQuery, setRanQuery] = useState<QuerySpec | null>(null);
  const [plan, setPlan] = useState<PlanGraph | null>(null);
  const [planBusy, setPlanBusy] = useState(false);
  const [tab, setTab] = useState<"results" | "plan">("results");
  const [running, setRunning] = useState(false);
  const [sqlErr, setSqlErr] = useState("");
  const [limit, setLimit] = useState(1000);   // rows shown in the grid
  const [dlOpen, setDlOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [expanded, setExpanded] = useState(false);   // fullscreen result viewer

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

  // Debounced asset search across the catalog (bounded, truncation-flagged).
  const onSearch = (q: string) => {
    setQuery(q);
    if (!q.trim()) { setHits(null); return; }
    searchSaga(q.trim(), 40, node).then((r) => setHits(r)).catch(() => setHits({ hits: [], total: 0, truncated: false }));
  };

  const onHit = async (h: SearchHit) => {
    if (h.kind === "table") {
      setQuery(""); setHits(null);
      const next = new Set(openCat); next.add(h.catalog); setOpenCat(next);
      if (!schemas[h.catalog]) {
        const r = await getSchemas(h.catalog, node, true);
        setSchemas((s) => ({ ...s, [h.catalog]: r.schemas }));
      }
      const key = `${h.catalog}.${h.schema}`;
      setOpenSch((o) => new Set(o).add(key));
      if (!tables[key]) {
        const r = await getTables(h.catalog, h.schema, node, true);
        setTables((t) => ({ ...t, [key]: r.tables }));
      }
      await openTable(h.catalog, h.schema, h.name);
    } else if (h.kind === "schema") {
      onSearch("");
      const c = catalogs.find((x) => x.name === h.catalog);
      if (c) await toggleCat(c);
    }
  };

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
      // Prefill the editor context from the selected table so Run "just works".
      setCtx({ catalog: cat, schema });
    } catch (e) { setTreeErr(String(e)); }
  };

  const queryTable = (t: TableEntry) => {
    setSql(`SELECT *\nFROM ${t.full_name}\nLIMIT 100`);
    setCtx({ catalog: t.catalog, schema: t.schema });
  };

  const loadLog = async (t: TableEntry) => {
    try {
      const r = await getTableLog(t.catalog, t.schema, t.name, node);
      setLog(r.entries); setActivity(null);
    } catch (e) { setTreeErr(String(e)); }
  };

  const loadActivity = async (t: TableEntry) => {
    try {
      const r = await getActivity(t.catalog, t.schema, t.name, node);
      setActivity(r); setLog(null);
    } catch (e) { setTreeErr(String(e)); }
  };

  const saveEdit = async (e: TableEntry) => {
    setBusy(true);
    try {
      const r = await updateTable(e.catalog, e.schema, e.name, {
        object_type: e.object_type, definition: e.definition, comment: e.comment,
        source_url: e.source_url, table_type: e.table_type,
      });
      setDetail(r.table); setEdit(null);
      await refreshSchema(e.catalog, e.schema);
    } catch (err) { setTreeErr(String(err)); } finally { setBusy(false); }
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

  const planBody = () => ({ sql, dialect, catalog: ctx.catalog, schema: ctx.schema });

  const queryFor = (sqlText: string): QuerySpec =>
    ({ sql: sqlText, catalog: ctx.catalog, schema: ctx.schema, node, limit });

  // -- SQL run: the typed Arrow grid fetches /sql.arrow itself; we just set the
  // query + load the plan for the strip and the Plan tab.
  const onRun = async () => {
    setRunning(true); setSqlErr("");
    try {
      setRanQuery(queryFor(sql)); setTab("results");
      getPlan(planBody()).then(setPlan).catch(() => {});
    } catch (e) { setSqlErr(String(e)); setRanQuery(null); }
    finally { setRunning(false); }
  };

  const onExport = async (fmt: string) => {
    setDlOpen(false); setExporting(true); setSqlErr("");
    try {
      await downloadSqlExport({ ...planBody(), node, fmt });  // full result, all rows
    } catch (e) { setSqlErr(String(e)); }
    finally { setExporting(false); }
  };

  const onExplain = async () => {
    setRunning(true); setSqlErr("");
    try {
      setPlan(await getPlan(planBody()));
      setTab("plan");
    } catch (e) { setSqlErr(String(e)); }
    finally { setRunning(false); }
  };

  const onAnalyze = async () => {
    setPlanBusy(true); setSqlErr("");
    try {
      setPlan(await getPlan(planBody(), true));
    } catch (e) { setSqlErr(String(e)); }
    finally { setPlanBusy(false); }
  };

  // Apply a structural plan edit live: re-emit SQL, drop it into the editor, run.
  const onPlanEdit = async (edits: PlanEdit[]) => {
    try {
      const r = await editPlan({ ...planBody(), edits });
      setSql(r.sql);
      setRanQuery(queryFor(r.sql));
      setPlan(await getPlan({ sql: r.sql, dialect, catalog: ctx.catalog, schema: ctx.schema }));
    } catch (e) { setSqlErr(String(e)); }
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
          <input value={query} onChange={(e) => onSearch(e.target.value)}
            placeholder="🔍 search assets…"
            className="w-full mb-2 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2.5 py-1.5 text-xs outline-none focus:border-frost/30" />
          {treeErr && <div className="text-[11px] text-rose/80 font-mono mb-2 break-words">{treeErr}</div>}

          {/* Search results replace the lazy tree while a query is active. */}
          {query.trim() ? (
            <div className="space-y-0.5">
              {hits && (
                <div className="text-[10px] text-muted mb-1">
                  {hits.total} match{hits.total === 1 ? "" : "es"}{hits.truncated ? ` · showing ${hits.hits.length}` : ""}
                </div>
              )}
              {hits?.hits.map((h) => (
                <div key={`${h.kind}:${h.full_name}`}
                  className="group flex items-center gap-1.5 py-1 px-1 rounded hover:bg-white/[0.03] cursor-pointer"
                  onClick={() => onHit(h)}
                  title={h.kind === "table" ? "open" : "expand"}>
                  <span className={h.kind === "table" ? objOf(h.object_type).color : "text-frost/70"}>
                    {h.kind === "catalog" ? "🗄" : h.kind === "schema" ? "▤" : objOf(h.object_type).glyph}
                  </span>
                  <span className="text-xs flex-1 truncate">{h.full_name}</span>
                  <span className={chip}>{h.kind === "table" ? h.object_type.toLowerCase() : h.kind}</span>
                </div>
              ))}
              {hits && hits.hits.length === 0 && <div className="text-[11px] text-muted py-2">no matches</div>}
            </div>
          ) : (
          <>
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
                                    <span className={objOf(t.object_type).color}>{objOf(t.object_type).glyph}</span>
                                    <span className="text-xs flex-1 truncate">{t.name}</span>
                                    <span className={chip}>{t.object_type === "TABLE" ? (t.format || "?") : t.object_type.toLowerCase()}</span>
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
          </>
          )}

          {/* Table detail */}
          {detail && (
            <div className="mt-3 pt-3 border-t border-white/[0.08]">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-emerald truncate">{detail.full_name}</span>
                <button onClick={() => { setDetail(null); setLog(null); }} className="text-[11px] text-muted hover:text-foreground">close</button>
              </div>
              <div className="flex flex-wrap gap-2 mb-1.5 text-[11px]">
                <button onClick={() => queryTable(detail)} className="text-frost/90 hover:text-frost font-semibold">query →</button>
                <button onClick={() => setPreview({ path: detail.source_url, name: detail.name })}
                  className="text-frost/80 hover:text-frost">preview data</button>
                <button onClick={() => insertRef(detail.full_name)} className="text-amber/80 hover:text-amber">insert</button>
                <button onClick={() => onRefreshTable(detail)} disabled={busy} className="text-frost/80 hover:text-frost">refresh stats</button>
                <button onClick={() => onReplicate(detail)} disabled={busy} className="text-emerald/80 hover:text-emerald">replicate</button>
                <button onClick={() => setEdit(detail)} className="text-amber/80 hover:text-amber">edit</button>
                <button onClick={() => loadActivity(detail)} className="text-violet-300 hover:text-violet-200">activity</button>
                <button onClick={() => loadLog(detail)} className="text-foreground-dim hover:text-foreground">history</button>
              </div>
              <div className="flex flex-wrap gap-3 text-[11px] text-foreground-dim mb-1.5 font-mono">
                <span className={`${objOf(detail.object_type).color}`}>{objOf(detail.object_type).glyph} {detail.object_type.toLowerCase()}</span>
                <span>{detail.statistics.row_count ?? "?"} rows</span>
                <span>{fmtBytes(detail.statistics.size_bytes)}</span>
                <span className={chip}>{detail.table_type}</span>
                {detail.replicas.length > 0 && <span className="text-emerald/80">⧉ {detail.replicas.length} replica(s)</span>}
                <span className="truncate max-w-full">{detail.source_url || "—"}</span>
              </div>
              {detail.definition && (
                <pre className="text-[11px] font-mono text-frost/80 bg-[#06060f] border border-white/[0.06] rounded p-2 mb-1.5 whitespace-pre-wrap break-words max-h-24 overflow-auto">{detail.definition}</pre>
              )}
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
              {activity && (
                <div className="mt-2 pt-2 border-t border-white/[0.06]">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[11px] uppercase tracking-wide text-muted">Activity</span>
                    <span className="text-[10px] text-muted font-mono">{activity.total_ops} ops</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5 mb-1.5">
                    {Object.entries(activity.op_counts).map(([op, n]) => (
                      <span key={op} className={chip}>{op} · {n}</span>
                    ))}
                  </div>
                  {activity.daily.length > 0 && (
                    <div className="flex items-end gap-0.5 h-10 mb-1.5" title="ops per day (last 14d)">
                      {(() => { const mx = Math.max(1, ...activity.daily); return activity.daily.map((v, i) => (
                        <div key={i} className="flex-1 bg-frost/40 rounded-t" style={{ height: `${Math.max(6, (v / mx) * 100)}%` }} title={`${v}`} />
                      )); })()}
                    </div>
                  )}
                  <div className="space-y-0.5 max-h-28 overflow-auto">
                    {activity.recent.map((e, i) => (
                      <div key={i} className="text-[11px] font-mono flex gap-2" title={e.statement}>
                        <span className="text-muted w-[88px] shrink-0 truncate">{e.ts.slice(5, 19).replace("T", " ")}</span>
                        <span className="text-frost/80 w-16 shrink-0">{e.op}</span>
                        <span className="text-foreground-dim truncate flex-1">{e.statement || e.detail}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
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
            <label className="flex items-center gap-1 text-[11px] text-muted" title="rows shown in the grid">
              limit
              <input type="number" min={1} value={limit}
                onChange={(e) => setLimit(Math.max(1, Number(e.target.value) || 1))}
                className="w-20 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
            </label>
            <div className="relative">
              <button onClick={() => setDlOpen((v) => !v)} disabled={exporting}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-white/[0.04] text-foreground-dim border border-white/[0.08] hover:bg-white/[0.08] disabled:opacity-40">
                {exporting ? "Exporting…" : "Download ▾"}
              </button>
              {dlOpen && (
                <div className="absolute right-0 mt-1 z-20 rounded-lg border border-white/[0.1] bg-[#0a0a1a] shadow-xl py-1 min-w-[120px]">
                  <div className="px-3 py-1 text-[10px] uppercase tracking-wide text-muted">full result as</div>
                  {SQL_EXPORT_FORMATS.map((f) => (
                    <button key={f} onClick={() => onExport(f)}
                      className="block w-full text-left px-3 py-1.5 text-xs font-mono text-foreground-dim hover:bg-white/[0.06] hover:text-frost">
                      {f}
                    </button>
                  ))}
                </div>
              )}
            </div>
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
            {ranQuery && tab === "results" && node && (
              <span className="text-muted font-mono ml-auto">@{node}</span>
            )}
          </div>

          <div className="flex-1 overflow-auto border border-white/[0.06] rounded-lg min-h-0">
            {tab === "results" && ranQuery && !expanded && (
              <div className="h-full p-1">
                <TabularDisplay query={ranQuery} plan={plan} maxHeight="100%" onExpand={() => setExpanded(true)} />
              </div>
            )}
            {tab === "results" && !ranQuery && (
              <div className="p-4 text-[12px] text-muted">Run a query to see results.</div>
            )}
            {tab === "plan" && (
              plan ? (
                <PlanGraphView graph={plan} busy={planBusy} onAnalyze={onAnalyze} onApply={onPlanEdit} />
              ) : (
                <div className="p-4 text-[12px] text-muted">Run or Explain a query to see the execution plan.</div>
              )
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

      {/* Full asset edit modal */}
      {edit && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-6"
          style={{ background: "var(--modal-scrim, rgba(0,0,0,0.72))" }} onClick={() => setEdit(null)}>
          <div className="modal-surface rounded-xl w-full max-w-lg p-5 space-y-3" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-emerald">{edit.full_name}</span>
              <button onClick={() => setEdit(null)} className="text-muted hover:text-foreground text-sm">✕</button>
            </div>
            <label className="block text-[11px] text-muted">type
              <select value={edit.object_type} onChange={(e) => setEdit({ ...edit, object_type: e.target.value })}
                className="w-full mt-1 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30">
                {OBJECT_TYPES.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            </label>
            {edit.object_type === "TABLE" ? (
              <label className="block text-[11px] text-muted">source_url
                <input value={edit.source_url} onChange={(e) => setEdit({ ...edit, source_url: e.target.value })}
                  className="w-full mt-1 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
              </label>
            ) : (
              <label className="block text-[11px] text-muted">definition
                <textarea value={edit.definition} onChange={(e) => setEdit({ ...edit, definition: e.target.value })} rows={4}
                  className="w-full mt-1 bg-[#06060f] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
              </label>
            )}
            <label className="block text-[11px] text-muted">comment
              <input value={edit.comment} onChange={(e) => setEdit({ ...edit, comment: e.target.value })}
                className="w-full mt-1 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs outline-none focus:border-frost/30" />
            </label>
            <div className="flex justify-end gap-2 pt-1">
              <button onClick={() => setEdit(null)} className="px-3 py-1.5 rounded-lg text-xs bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08]">cancel</button>
              <button onClick={() => saveEdit(edit)} disabled={busy}
                className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-emerald/15 text-emerald border border-emerald/30 hover:bg-emerald/25 disabled:opacity-40">save</button>
            </div>
          </div>
        </div>
      )}

      {/* The same TabularDisplay, full-screen — proving it works in a modal too. */}
      {expanded && ranQuery && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-6"
          style={{ background: "var(--modal-scrim, rgba(0,0,0,0.72))" }} onClick={() => setExpanded(false)}>
          <div className="modal-surface rounded-xl w-full max-w-6xl h-[85vh] flex flex-col p-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-frost">Query result</span>
              <button onClick={() => setExpanded(false)} className="text-muted hover:text-foreground text-sm">close ✕</button>
            </div>
            <div className="flex-1 min-h-0">
              <TabularDisplay query={ranQuery} plan={plan} maxHeight="100%" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
