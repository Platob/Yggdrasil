"use client";

import { useCallback, useEffect, useState } from "react";
import Script from "next/script";
import {
  activeSheetToGrid,
  getInfo,
  getTree,
  gridToNewSheet,
  normalizeBase,
  readFile,
  runPython,
  writeFileCsv,
  sagaCatalogs,
  sagaTables,
  sagaSql,
  type FsNode,
  type NodeInfo,
  type SagaTableRef,
} from "@/lib/excel";

type Tab = "run" | "files" | "saga" | "write";

export default function TaskPane() {
  const [officeReady, setOfficeReady] = useState(false);
  const [base, setBase] = useState("http://127.0.0.1:8100");
  const [info, setInfo] = useState<NodeInfo | null>(null);
  const [status, setStatus] = useState<{ kind: "ok" | "err" | "busy"; msg: string } | null>(null);
  const [tab, setTab] = useState<Tab>("run");

  // run-python state
  const [code, setCode] = useState(
    "import pandas as pd\ndf = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})",
  );
  const [env, setEnv] = useState("");
  const [packages, setPackages] = useState("");
  const [envVars, setEnvVars] = useState("");

  // files state
  const [cwd, setCwd] = useState("");
  const [entries, setEntries] = useState<FsNode[]>([]);

  // saga state
  const [sagaCats, setSagaCats] = useState<string[]>([]);
  const [sagaCat, setSagaCat] = useState("");
  const [sagaTbls, setSagaTbls] = useState<SagaTableRef[]>([]);
  const [sql, setSql] = useState("SELECT * FROM catalog.schema.table LIMIT 1000");

  // write state
  const [writePath, setWritePath] = useState("excel/export.csv");

  useEffect(() => {
    if (typeof window !== "undefined" && window.Office) {
      window.Office.onReady(() => setOfficeReady(true));
    }
  }, []);

  const connect = useCallback(async () => {
    setStatus({ kind: "busy", msg: "Connecting…" });
    try {
      const i = await getInfo(base);
      setInfo(i);
      setStatus({ kind: "ok", msg: `Connected to ${i.node_name} (${i.node_id})` });
    } catch (e) {
      setInfo(null);
      setStatus({ kind: "err", msg: String((e as Error).message) });
    }
  }, [base]);

  const doRun = useCallback(async () => {
    setStatus({ kind: "busy", msg: "Running…" });
    try {
      // env vars: "KEY=val, KEY2=val2"
      const env_vars: Record<string, string> = {};
      for (const pair of envVars.split(",").map((p) => p.trim()).filter(Boolean)) {
        const eq = pair.indexOf("=");
        if (eq > 0) env_vars[pair.slice(0, eq).trim()] = pair.slice(eq + 1).trim();
      }
      const grid = await runPython(base, {
        code,
        env: env.trim() || null,
        packages: packages.split(",").map((p) => p.trim()).filter(Boolean),
        env_vars,
      });
      await gridToNewSheet(grid, "Ygg Result");
      setStatus({ kind: "ok", msg: `Wrote ${grid.rows.length} rows × ${grid.headers.length} cols` });
    } catch (e) {
      setStatus({ kind: "err", msg: String((e as Error).message) });
    }
  }, [base, code, env, packages, envVars]);

  const browse = useCallback(async (path: string) => {
    setStatus({ kind: "busy", msg: "Listing…" });
    try {
      setEntries(await getTree(base, path, 1));
      setCwd(path);
      setStatus(null);
    } catch (e) {
      setStatus({ kind: "err", msg: String((e as Error).message) });
    }
  }, [base]);

  const loadFile = useCallback(async (path: string) => {
    setStatus({ kind: "busy", msg: `Loading ${path}…` });
    try {
      const grid = await readFile(base, path);
      await gridToNewSheet(grid, path.split("/").pop() || "file");
      setStatus({ kind: "ok", msg: `Loaded ${grid.rows.length} rows` });
    } catch (e) {
      setStatus({ kind: "err", msg: String((e as Error).message) });
    }
  }, [base]);

  const saveActiveSheet = useCallback(async () => {
    setStatus({ kind: "busy", msg: "Saving sheet…" });
    try {
      const grid = await activeSheetToGrid();
      if (grid.headers.length === 0) {
        setStatus({ kind: "err", msg: "Active sheet is empty — nothing to save." });
        return;
      }
      const res = await writeFileCsv(base, writePath, grid);
      setStatus({ kind: "ok", msg: `Saved ${res.rows} rows → ${writePath}` });
    } catch (e) {
      setStatus({ kind: "err", msg: String((e as Error).message) });
    }
  }, [base, writePath]);

  const loadCatalogs = useCallback(async () => {
    setStatus({ kind: "busy", msg: "Loading catalogs…" });
    try {
      const cats = await sagaCatalogs(base);
      setSagaCats(cats);
      setStatus(null);
    } catch (e) { setStatus({ kind: "err", msg: String((e as Error).message) }); }
  }, [base]);

  const pickCatalog = useCallback(async (c: string) => {
    setSagaCat(c);
    setStatus({ kind: "busy", msg: `Loading ${c}…` });
    try {
      setSagaTbls(await sagaTables(base, c));
      setStatus(null);
    } catch (e) { setStatus({ kind: "err", msg: String((e as Error).message) }); }
  }, [base]);

  const runSqlToSheet = useCallback(async (statement: string, name: string) => {
    setStatus({ kind: "busy", msg: "Querying…" });
    try {
      const grid = await sagaSql(base, statement);
      await gridToNewSheet(grid, name);
      setStatus({ kind: "ok", msg: `Loaded ${grid.rows.length} rows × ${grid.headers.length} cols` });
    } catch (e) { setStatus({ kind: "err", msg: String((e as Error).message) }); }
  }, [base]);

  const parent = cwd.includes("/") ? cwd.slice(0, cwd.lastIndexOf("/")) : "";

  return (
    <div style={S.page}>
      <Script src="https://appsforoffice.microsoft.com/lib/1/hosted/office.js" strategy="afterInteractive" />

      <h1 style={S.h1}>⌁ Yggdrasil for Excel</h1>
      {!officeReady && (
        <p style={S.note}>Office.js not detected — open this inside Excel as an add-in. (Controls still work for a connection test.)</p>
      )}

      <div style={S.row}>
        <input style={S.input} value={base} onChange={(e) => setBase(normalizeBase(e.target.value))} placeholder="node URL" />
        <button style={S.btn} onClick={connect}>Connect</button>
      </div>
      {status && (
        <div style={{ ...S.status, color: status.kind === "err" ? "#f87171" : status.kind === "ok" ? "#34d399" : "#93c5fd" }}>
          {status.msg}
        </div>
      )}
      {info && <div style={S.meta}>v{info.version} · {info.capabilities.join(", ")}</div>}

      <div style={S.tabs}>
        {(["run", "files", "saga", "write"] as Tab[]).map((t) => (
          <button key={t} style={{ ...S.tab, ...(tab === t ? S.tabActive : {}) }} onClick={() => setTab(t)}>
            {t === "run" ? "Run Python" : t === "files" ? "Files" : t === "saga" ? "Saga" : "Save"}
          </button>
        ))}
      </div>

      {tab === "run" && (
        <div style={S.panel}>
          <textarea style={S.code} value={code} onChange={(e) => setCode(e.target.value)} rows={8} />
          <input style={S.input} value={env} onChange={(e) => setEnv(e.target.value)} placeholder="PyEnv name (optional)" />
          <input style={S.input} value={packages} onChange={(e) => setPackages(e.target.value)} placeholder="packages, comma-separated (optional)" />
          <input style={S.input} value={envVars} onChange={(e) => setEnvVars(e.target.value)} placeholder="env vars: KEY=val, KEY2=val2 (optional)" />
          <button style={S.btnWide} onClick={doRun}>Run → new sheet</button>
        </div>
      )}

      {tab === "files" && (
        <div style={S.panel}>
          <div style={S.row}>
            <button style={S.btn} onClick={() => browse("")}>/ root</button>
            {cwd && <button style={S.btn} onClick={() => browse(parent)}>↑ up</button>}
            <span style={S.cwd}>/{cwd}</span>
          </div>
          <div style={S.list}>
            {entries.length === 0 && <div style={S.empty}>Click “root” to list files.</div>}
            {entries.map((e) => (
              <div key={e.path} style={S.fileRow}>
                <span>{e.is_dir ? "📁" : "📄"} {e.name}</span>
                {e.is_dir ? (
                  <button style={S.linkBtn} onClick={() => browse(e.path)}>open</button>
                ) : (
                  <button style={S.linkBtn} onClick={() => loadFile(e.path)}>→ sheet</button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === "saga" && (
        <div style={S.panel}>
          <p style={S.note}>Query the network catalog from this node — it federates across the mesh. Results land in a new sheet.</p>
          <textarea style={S.code} value={sql} onChange={(e) => setSql(e.target.value)} rows={4} />
          <button style={S.btnWide} onClick={() => runSqlToSheet(sql, "Saga Query")}>Run SQL → new sheet</button>
          <div style={{ ...S.row, marginTop: 8 }}>
            <button style={S.btn} onClick={loadCatalogs}>Catalogs</button>
            {sagaCats.map((c) => (
              <button key={c} style={{ ...S.btn, ...(c === sagaCat ? S.tabActive : {}) }} onClick={() => pickCatalog(c)}>{c}</button>
            ))}
          </div>
          <div style={S.list}>
            {sagaTbls.length === 0 && <div style={S.empty}>Pick a catalog to list its tables.</div>}
            {sagaTbls.map((t) => (
              <div key={t.full_name} style={S.fileRow}>
                <span>🗄 {t.schema}.{t.name}</span>
                <button style={S.linkBtn} onClick={() => runSqlToSheet(`SELECT * FROM ${t.full_name} LIMIT 100000`, t.name)}>→ sheet</button>
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === "write" && (
        <div style={S.panel}>
          <p style={S.note}>Write the active sheet’s used range to a file on the node (CSV).</p>
          <input style={S.input} value={writePath} onChange={(e) => setWritePath(e.target.value)} placeholder="target path" />
          <button style={S.btnWide} onClick={saveActiveSheet}>Save active sheet → node</button>
        </div>
      )}
    </div>
  );
}

const S: Record<string, React.CSSProperties> = {
  page: { fontFamily: "Segoe UI, system-ui, sans-serif", background: "#0b1020", color: "#e5e7eb", minHeight: "100vh", padding: 14, fontSize: 13 },
  h1: { fontSize: 15, fontWeight: 700, margin: "0 0 10px", color: "#a5b4fc" },
  note: { fontSize: 11, color: "#94a3b8", margin: "4px 0" },
  row: { display: "flex", gap: 6, alignItems: "center", marginBottom: 6 },
  input: { flex: 1, background: "#111827", border: "1px solid #334155", color: "#e5e7eb", borderRadius: 6, padding: "6px 8px", fontSize: 12 },
  code: { width: "100%", background: "#111827", border: "1px solid #334155", color: "#e5e7eb", borderRadius: 6, padding: 8, fontFamily: "Consolas, monospace", fontSize: 12, marginBottom: 6, boxSizing: "border-box" },
  btn: { background: "#1e293b", border: "1px solid #475569", color: "#e5e7eb", borderRadius: 6, padding: "6px 10px", cursor: "pointer", fontSize: 12 },
  btnWide: { width: "100%", background: "#4338ca", border: "none", color: "#fff", borderRadius: 6, padding: "8px 10px", cursor: "pointer", fontWeight: 600, marginTop: 4 },
  linkBtn: { background: "none", border: "none", color: "#93c5fd", cursor: "pointer", fontSize: 12 },
  status: { fontSize: 12, margin: "6px 0", minHeight: 16 },
  meta: { fontSize: 11, color: "#64748b", marginBottom: 8 },
  tabs: { display: "flex", gap: 4, borderBottom: "1px solid #1e293b", margin: "10px 0" },
  tab: { background: "none", border: "none", color: "#94a3b8", padding: "6px 8px", cursor: "pointer", fontSize: 12, borderBottom: "2px solid transparent" },
  tabActive: { color: "#a5b4fc", borderBottom: "2px solid #6366f1" },
  panel: { marginTop: 4 },
  list: { border: "1px solid #1e293b", borderRadius: 6, maxHeight: 280, overflowY: "auto" },
  fileRow: { display: "flex", justifyContent: "space-between", padding: "6px 8px", borderBottom: "1px solid #111827" },
  cwd: { fontSize: 11, color: "#64748b", fontFamily: "monospace" },
  empty: { padding: 10, color: "#64748b", fontSize: 12 },
};
