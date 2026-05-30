"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import {
  getFsNodes,
  getFsListing,
  getFsRead,
  getFsTail,
  createFsWatchStream,
  fsDownloadUrl,
  deleteFsPath,
  writeFsFile,
  uploadFsFile,
  mkdirFs,
  isTabularName,
  type FsNodeRoot,
} from "@/lib/api";
import type { FsEntry } from "@/lib/types";
import TabularModal from "@/components/TabularModal";
import RegisterSagaModal from "@/components/RegisterSagaModal";

// ── Helpers ───────────────────────────────────────────────────
function formatSize(bytes: number): string {
  if (bytes === 0) return "--";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDate(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleDateString([], { month: "short", day: "numeric" }) +
      " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "--";
  }
}

function getExt(name: string): string {
  const idx = name.lastIndexOf(".");
  return idx < 0 ? "" : name.slice(idx + 1).toLowerCase();
}

const TEXT_VIEWABLE_EXTS = new Set([
  "py", "ts", "tsx", "js", "jsx", "rs", "go", "c", "cpp", "h", "java", "rb",
  "sh", "bash", "zsh", "toml", "yaml", "yml", "json", "xml", "html", "css",
  "scss", "md", "txt", "rst", "cfg", "ini", "conf", "env", "log", "csv",
  "sql", "graphql", "proto", "dockerfile", "makefile", "gitignore",
]);
const MAX_VIEWABLE_SIZE = 4 * 1024 * 1024;

function isTextViewable(entry: FsEntry): boolean {
  if (entry.is_dir || entry.size > MAX_VIEWABLE_SIZE) return false;
  const ext = getExt(entry.name);
  const lower = entry.name.toLowerCase();
  return TEXT_VIEWABLE_EXTS.has(ext) ||
    ["makefile", "dockerfile", "readme", "license", "changelog", ".gitignore", ".env"].includes(lower);
}

function dirFirst(a: FsEntry, b: FsEntry): number {
  if (a.is_dir && !b.is_dir) return -1;
  if (!a.is_dir && b.is_dir) return 1;
  return a.name.localeCompare(b.name);
}

const dirKey = (node: string, path: string) => `${node}::${path}`;

function FileIcon({ entry }: { entry: FsEntry }) {
  if (entry.is_dir) {
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--amber)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
      </svg>
    );
  }
  const ext = getExt(entry.name);
  const code = ["py", "ts", "tsx", "js", "jsx", "rs", "go", "json", "toml", "yaml", "yml", "sh"];
  const stroke = code.includes(ext) ? "var(--frost)" : "var(--muted)";
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M13 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V9z" />
      <polyline points="13 2 13 9 20 9" />
    </svg>
  );
}

// Minimal shape of the (non-standard but universal) webkit drag-drop entry API
// — not in lib.dom, so we declare just what we use.
interface FsDropEntry {
  isFile: boolean;
  isDirectory: boolean;
  name: string;
  file(cb: (f: File) => void, err?: (e: unknown) => void): void;
  createReader(): { readEntries(cb: (e: FsDropEntry[]) => void, err?: (e: unknown) => void): void };
}

// Recurse a dropped folder/file tree into a flat list of {file, relPath}.
// Uses the webkit entry API so a whole folder can be dragged in at once.
async function gatherDropped(entry: FsDropEntry, prefix: string): Promise<{ file: File; relPath: string }[]> {
  if (entry.isFile) {
    const file: File = await new Promise((res, rej) => entry.file(res, rej));
    return [{ file, relPath: prefix + entry.name }];
  }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    const out: { file: File; relPath: string }[] = [];
    // readEntries returns at most ~100 per call; loop until it returns empty.
    for (;;) {
      const batch: FsDropEntry[] = await new Promise((res) => reader.readEntries(res, () => res([])));
      if (!batch.length) break;
      for (const e of batch) out.push(...(await gatherDropped(e, prefix + entry.name + "/")));
    }
    return out;
  }
  return [];
}

type ViewMode = "preview" | "tail";

export default function FilesPage() {
  const [nodes, setNodes] = useState<FsNodeRoot[]>([]);
  const [nodesLoading, setNodesLoading] = useState(true);
  const [nodesError, setNodesError] = useState(false);

  // Lazy tree state, lifted so refresh / mutation can target one directory
  // without remounting (and losing) the rest of the expanded tree.
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [history, setHistory] = useState<string[]>([]);   // expand/collapse trail
  const [children, setChildren] = useState<Record<string, FsEntry[]>>({});
  const [loadingKeys, setLoadingKeys] = useState<Set<string>>(new Set());
  const [errorKeys, setErrorKeys] = useState<Set<string>>(new Set());
  const [dragKey, setDragKey] = useState<string | null>(null);
  const [upload, setUpload] = useState<{ label: string; done: number; total: number } | null>(null);

  // Tabular files open the dedicated (reusable) tabular editor instead.
  const [tabular, setTabular] = useState<{ node: string; entry: FsEntry } | null>(null);
  const [sagaReg, setSagaReg] = useState<{ source: string; node?: string } | null>(null);

  // Selected file + text preview modal
  const [selected, setSelected] = useState<{ node: string; entry: FsEntry } | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [fileTruncated, setFileTruncated] = useState(false);
  const [fileEncoding, setFileEncoding] = useState("utf-8");
  const [loadingContent, setLoadingContent] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("preview");
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [tailLines, setTailLines] = useState<string[]>([]);
  const [tailLive, setTailLive] = useState(false);
  const tailEsRef = useRef<EventSource | null>(null);
  const tailEndRef = useRef<HTMLDivElement | null>(null);

  const selfNodeId = nodes.find((n) => n.self)?.node_id;

  const fetchNodes = useCallback(async (fresh = false) => {
    setNodesLoading(true);
    setNodesError(false);
    try {
      const res = await getFsNodes(fresh);
      setNodes(res.nodes);
    } catch {
      setNodesError(true);
      setNodes([]);
    } finally {
      setNodesLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchNodes();
  }, [fetchNodes]);

  const ensureLoaded = useCallback(async (node: string, path: string, fresh = false) => {
    const k = dirKey(node, path);
    setLoadingKeys((s) => new Set(s).add(k));
    setErrorKeys((s) => { const n = new Set(s); n.delete(k); return n; });
    try {
      const res = await getFsListing(path, node, fresh);
      setChildren((c) => ({ ...c, [k]: [...res.entries].sort(dirFirst) }));
    } catch {
      setErrorKeys((s) => new Set(s).add(k));
      setChildren((c) => ({ ...c, [k]: [] }));
    } finally {
      setLoadingKeys((s) => { const n = new Set(s); n.delete(k); return n; });
    }
  }, []);

  const toggleDir = useCallback((node: string, path: string, record = true) => {
    const k = dirKey(node, path);
    setExpanded((s) => {
      const n = new Set(s);
      if (n.has(k)) n.delete(k);
      else { n.add(k); ensureLoaded(node, path); }
      return n;
    });
    if (record) setHistory((h) => [...h.slice(-49), k]);
  }, [ensureLoaded]);

  // Previous-action handling: Back reverts the last expand/collapse; Collapse
  // all clears the tree. Keeps discovery from being a one-way trip.
  const back = () => {
    if (!history.length) return;
    const k = history[history.length - 1];
    const idx = k.indexOf("::");
    const node = k.slice(0, idx), path = k.slice(idx + 2);
    setExpanded((s) => {
      const n = new Set(s);
      if (n.has(k)) n.delete(k);
      else { n.add(k); ensureLoaded(node, path); }
      return n;
    });
    setHistory((h) => h.slice(0, -1));
  };
  const collapseAll = () => { setExpanded(new Set()); setHistory([]); };

  // Upload a set of {file, relPath} into basePath on a node, with a progress
  // toast and bounded concurrency. Shared by drag-drop and the Upload button.
  const runUpload = useCallback(async (
    node: string, basePath: string, files: { file: File; relPath: string }[],
  ) => {
    if (!files.length) return;
    setUpload({ label: `${node === selfNodeId ? "local" : node} ${basePath || "/"}`, done: 0, total: files.length });
    let idx = 0;
    const worker = async () => {
      while (idx < files.length) {
        const cur = files[idx++];
        const dest = basePath ? `${basePath}/${cur.relPath}` : cur.relPath;
        try { await uploadFsFile(dest, cur.file, node); } catch { /* surfaced via count */ }
        setUpload((u) => (u ? { ...u, done: u.done + 1 } : u));
      }
    };
    await Promise.all(Array.from({ length: Math.min(4, files.length) }, worker));
    setUpload(null);
    setExpanded((s) => new Set(s).add(dirKey(node, basePath)));
    await ensureLoaded(node, basePath, true);
  }, [ensureLoaded, selfNodeId]);

  const handleDrop = useCallback(async (node: string, basePath: string, e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragKey(null);
    const items = Array.from(e.dataTransfer.items)
      .map((it) => (it.webkitGetAsEntry ? (it.webkitGetAsEntry() as unknown as FsDropEntry | null) : null))
      .filter((x): x is FsDropEntry => x !== null);
    let files: { file: File; relPath: string }[] = [];
    if (items.length) {
      for (const it of items) files.push(...(await gatherDropped(it, "")));
    } else {
      files = Array.from(e.dataTransfer.files).map((f) => ({ file: f, relPath: f.name }));
    }
    await runUpload(node, basePath, files);
  }, [runUpload]);

  // Delete a file or whole folder straight from the tree, then refresh parent.
  const deleteEntry = useCallback(async (node: string, entry: FsEntry, parentPath: string) => {
    if (!confirm(`Delete ${entry.is_dir ? "folder" : "file"} ${entry.path}?`)) return;
    try {
      await deleteFsPath(entry.path, node);
      await ensureLoaded(node, parentPath, true);
    } catch { /* tree shows last good state */ }
  }, [ensureLoaded]);

  // Open the reusable Saga register modal, prefilled from the file (and from any
  // existing registration). source_url is the node-home-relative path Saga uses.
  const registerSaga = useCallback((node: string, entry: FsEntry) => {
    setSagaReg({ source: entry.path, node: node === selfNodeId ? undefined : node });
  }, [selfNodeId]);

  const newFolder = useCallback(async (node: string, basePath: string) => {
    const name = prompt("New folder name");
    if (!name) return;
    try {
      await mkdirFs(basePath ? `${basePath}/${name}` : name, node);
      setExpanded((s) => new Set(s).add(dirKey(node, basePath)));
      await ensureLoaded(node, basePath, true);
    } catch { /* ignore */ }
  }, [ensureLoaded]);

  // Hidden <input> drives the Upload button; the pending target says where to.
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const uploadTargetRef = useRef<{ node: string; path: string }>({ node: "", path: "" });
  const pickUpload = (node: string, basePath: string) => {
    uploadTargetRef.current = { node, path: basePath };
    uploadInputRef.current?.click();
  };
  const onUploadPicked = (e: React.ChangeEvent<HTMLInputElement>) => {
    const list = Array.from(e.target.files ?? []).map((f) => ({ file: f, relPath: f.name }));
    const { node, path } = uploadTargetRef.current;
    e.target.value = "";
    runUpload(node, path, list);
  };

  // ── File open / preview ─────────────────────────────────────
  const openFile = useCallback(async (node: string, entry: FsEntry) => {
    if (isTabularName(entry.name)) {
      setTabular({ node, entry });
      return;
    }
    setSelected({ node, entry });
    setFileContent(null);
    setFileTruncated(false);
    setFileEncoding("utf-8");
    setTailLines([]);
    setViewMode("preview");
    setTailLive(false);
    setEditing(false);
    if (isTextViewable(entry)) {
      setLoadingContent(true);
      try {
        const res = await getFsRead(entry.path, node);
        setFileEncoding(res.encoding);
        setFileContent(res.encoding === "base64" ? "[binary file — use Download]" : res.content);
        setFileTruncated(res.truncated);
      } catch {
        setFileContent(null);
      } finally {
        setLoadingContent(false);
      }
    }
  }, []);

  const closeModal = () => {
    setSelected(null);
    setFileContent(null);
    setTailLive(false);
    setEditing(false);
  };

  const enterTail = async () => {
    if (!selected) return;
    setViewMode("tail");
    try {
      const t = await getFsTail(selected.entry.path, 300, selected.node);
      setTailLines(t.lines);
    } catch {
      setTailLines([]);
    }
  };

  const saveEdit = async () => {
    if (!selected) return;
    setSaving(true);
    try {
      await writeFsFile(selected.entry.path, editValue, selected.node);
      setFileContent(editValue);
      setEditing(false);
      // Refresh the containing directory so size/mtime update.
      const parent = selected.entry.path.split("/").slice(0, -1).join("/");
      await ensureLoaded(selected.node, parent, true);
    } catch {
      /* keep editor open so the user can retry */
    } finally {
      setSaving(false);
    }
  };

  const deleteSelected = async () => {
    if (!selected) return;
    if (!confirm(`Delete ${selected.entry.path} on ${selected.node === selfNodeId ? "this node" : selected.node}?`)) return;
    try {
      await deleteFsPath(selected.entry.path, selected.node);
      const parent = selected.entry.path.split("/").slice(0, -1).join("/");
      closeModal();
      await ensureLoaded(selected.node, parent, true);
    } catch {
      /* leave modal open */
    }
  };

  // SSE live tail — local node only (the watch stream isn't proxied).
  useEffect(() => {
    if (!tailLive || viewMode !== "tail" || !selected || selected.node !== selfNodeId) {
      if (tailEsRef.current) { tailEsRef.current.close(); tailEsRef.current = null; }
      return;
    }
    const es = createFsWatchStream(selected.entry.path);
    tailEsRef.current = es;
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as { line: string };
        setTailLines((prev) => [...prev.slice(-500), data.line]);
        setTimeout(() => tailEndRef.current?.scrollIntoView({ behavior: "smooth" }), 0);
      } catch { /* keepalive */ }
    };
    return () => { es.close(); tailEsRef.current = null; };
  }, [tailLive, viewMode, selected, selfNodeId]);

  // ── Recursive tree render ───────────────────────────────────
  const renderDir = (node: string, path: string, depth: number) => {
    const k = dirKey(node, path);
    const isOpen = expanded.has(k);
    const rows = children[k];
    const loading = loadingKeys.has(k);
    const errored = errorKeys.has(k);
    if (!isOpen) return null;
    return (
      <div
        className={`ml-3 pl-2 border-l ${dragKey === k ? "border-emerald/60 bg-emerald/[0.04]" : "border-white/[0.06]"}`}
        onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setDragKey(k); }}
        onDragLeave={(e) => { e.stopPropagation(); setDragKey((d) => (d === k ? null : d)); }}
        onDrop={(e) => handleDrop(node, path, e)}
      >
        {loading && <div className="py-1.5 text-[11px] text-muted font-mono pl-5">loading…</div>}
        {errored && <div className="py-1.5 text-[11px] text-rose/80 font-mono pl-5">unreachable</div>}
        {rows && rows.length === 0 && !loading && (
          <div className="py-1.5 text-[11px] text-muted/60 font-mono pl-5">empty{dragKey === k ? " — drop to upload" : ""}</div>
        )}
        {rows?.map((entry) => {
          const childOpen = expanded.has(dirKey(node, entry.path));
          return (
            <div key={entry.path}>
              <div
                className="group flex items-center gap-2 px-2 py-1.5 rounded hover:bg-white/[0.03] cursor-pointer"
                onClick={() => (entry.is_dir ? toggleDir(node, entry.path) : openFile(node, entry))}
              >
                {entry.is_dir ? (
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="3"
                    style={{ transform: childOpen ? "rotate(90deg)" : "none", transition: "transform .12s" }}>
                    <polyline points="9 18 15 12 9 6" />
                  </svg>
                ) : <span className="w-[10px]" />}
                <FileIcon entry={entry} />
                <span className={`text-[13px] truncate flex-1 ${entry.is_dir ? "text-foreground/90" : "text-foreground/70"}`}>
                  {entry.name}
                </span>
                <span className="text-[10px] font-mono text-muted/70 opacity-0 group-hover:opacity-100">
                  {entry.is_dir ? "" : formatSize(entry.size)}
                </span>
                {entry.is_dir && (
                  <button
                    onClick={(e) => { e.stopPropagation(); pickUpload(node, entry.path); }}
                    className="opacity-0 group-hover:opacity-100 text-emerald/60 hover:text-emerald"
                    title="Upload into folder"
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                      <polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                  </button>
                )}
                {!entry.is_dir && isTabularName(entry.name) && (
                  <button
                    onClick={(e) => { e.stopPropagation(); registerSaga(node, entry); }}
                    className="opacity-0 group-hover:opacity-100 text-emerald/60 hover:text-emerald"
                    title="Register in Saga catalog"
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <ellipse cx="12" cy="5" rx="8" ry="3" />
                      <path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6" />
                    </svg>
                  </button>
                )}
                <a
                  href={fsDownloadUrl(entry.path, node)}
                  onClick={(e) => e.stopPropagation()}
                  className="opacity-0 group-hover:opacity-100 text-frost/60 hover:text-frost"
                  title={entry.is_dir ? "Download folder (zip)" : "Download"}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                </a>
                <button
                  onClick={(e) => { e.stopPropagation(); deleteEntry(node, entry, path); }}
                  className="opacity-0 group-hover:opacity-100 text-rose/60 hover:text-rose"
                  title={entry.is_dir ? "Delete folder" : "Delete file"}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
                  </svg>
                </button>
              </div>
              {entry.is_dir && renderDir(node, entry.path, depth + 1)}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="p-6 h-screen overflow-y-auto animate-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Memory Regions</h1>
          <p className="text-sm text-muted mt-1">
            Global filesystem across linked nodes — expand a node to stream its memory lazily, drag a folder in to replicate it.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={back}
            disabled={history.length === 0}
            title="Undo last expand/collapse"
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-foreground-dim hover:text-foreground bg-white/[0.03] border border-white/[0.06] disabled:opacity-30"
          >
            <svg className="inline-block mr-1 -mt-0.5" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="15 18 9 12 15 6" /></svg>
            Back{history.length ? ` (${history.length})` : ""}
          </button>
          <button
            onClick={collapseAll}
            disabled={expanded.size === 0}
            title="Collapse all"
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-foreground-dim hover:text-foreground bg-white/[0.03] border border-white/[0.06] disabled:opacity-30"
          >
            Collapse all
          </button>
          <button
            onClick={() => fetchNodes(true)}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-frost/70 hover:text-frost bg-frost/5 hover:bg-frost/10 border border-frost/10 hover:border-frost/20 transition-all"
          >
            <svg className="inline-block mr-1.5 -mt-0.5" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="23 4 23 10 17 10" /><path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" />
            </svg>
            Refresh nodes
          </button>
        </div>
      </div>

      {/* Node roots */}
      {nodesLoading ? (
        <div className="flex items-center justify-center py-20">
          <div className="w-8 h-8 border-2 border-frost/30 border-t-frost rounded-full spin-slow" />
        </div>
      ) : nodesError ? (
        <div className="glass-card p-8 text-center">
          <p className="text-sm text-muted">Backend unreachable — cannot list nodes.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {nodes.map((n) => {
            const rootKey = dirKey(n.node_id, "");
            const rootOpen = expanded.has(rootKey);
            return (
              <div key={n.node_id} className="glass-card overflow-hidden">
                <div
                  className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-white/[0.02]"
                  onClick={() => toggleDir(n.node_id, "")}
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--frost)" strokeWidth="3"
                    style={{ transform: rootOpen ? "rotate(90deg)" : "none", transition: "transform .12s" }}>
                    <polyline points="9 18 15 12 9 6" />
                  </svg>
                  <span className={`w-2 h-2 rounded-full ${n.self ? "bg-emerald" : "bg-frost"}`} style={{ boxShadow: `0 0 8px var(--${n.self ? "emerald" : "frost"})` }} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-foreground truncate">
                        {n.self ? "This node" : n.node_id}
                      </span>
                      <span className="text-[10px] font-mono text-muted">{n.host}:{n.port}</span>
                    </div>
                    {!n.self && (
                      <span className="text-[10px] font-mono text-muted/70">
                        cpu {Math.round(n.cpu_percent ?? 0)}% · runs {n.active_runs ?? 0} · {n.role}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
                    <button
                      onClick={() => newFolder(n.node_id, "")}
                      className="px-2 py-1 rounded text-[10px] font-mono text-frost/70 hover:text-frost hover:bg-frost/10 border border-white/[0.06]"
                      title="New folder at root"
                    >
                      + folder
                    </button>
                    <button
                      onClick={() => pickUpload(n.node_id, "")}
                      className="px-2 py-1 rounded text-[10px] font-mono text-emerald/80 hover:text-emerald hover:bg-emerald/10 border border-emerald/15"
                      title="Upload files to root"
                    >
                      ↑ upload
                    </button>
                  </div>
                </div>
                {renderDir(n.node_id, "", 0)}
              </div>
            );
          })}
          {nodes.length === 0 && (
            <div className="glass-card p-8 text-center text-sm text-muted">No nodes linked.</div>
          )}
        </div>
      )}

      {/* Hidden picker for the Upload buttons (drag-drop handles folders) */}
      <input ref={uploadInputRef} type="file" multiple className="hidden" onChange={onUploadPicked} />

      {/* Upload progress toast */}
      {upload && (
        <div className="fixed bottom-6 right-6 z-50 glass-card px-4 py-3 w-72">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-emerald">Uploading → {upload.label}</span>
            <span className="text-[10px] font-mono text-muted">{upload.done}/{upload.total}</span>
          </div>
          <div className="h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
            <div className="h-full bg-emerald transition-all" style={{ width: `${(upload.done / upload.total) * 100}%` }} />
          </div>
        </div>
      )}

      {/* Tabular editor (reusable component) */}
      {tabular && (
        <TabularModal
          node={tabular.node}
          nodeLabel={tabular.node === selfNodeId ? "local" : tabular.node}
          path={tabular.entry.path}
          name={tabular.entry.name}
          onClose={() => setTabular(null)}
        />
      )}

      {sagaReg && (
        <RegisterSagaModal source={sagaReg.source} node={sagaReg.node} onClose={() => setSagaReg(null)} />
      )}

      {/* File preview / edit modal */}
      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-6">
          <div className="absolute inset-0 bg-[var(--modal-scrim)] backdrop-blur-sm" onClick={closeModal} />
          <div className={`relative modal-surface p-6 w-full space-y-4 z-10 flex flex-col ${fileContent != null ? "max-w-3xl max-h-[85vh]" : "max-w-lg"}`}>
            <div className="flex items-start justify-between shrink-0">
              <div className="flex items-center gap-3 min-w-0">
                <FileIcon entry={selected.entry} />
                <div className="min-w-0">
                  <h3 className="text-sm font-mono font-semibold text-foreground truncate">{selected.entry.name}</h3>
                  <p className="text-[11px] text-muted font-mono truncate mt-0.5">
                    <span className="text-frost/70">{selected.node === selfNodeId ? "local" : selected.node}</span>
                    {" : "}{selected.entry.path}
                  </p>
                </div>
              </div>
              <button onClick={closeModal} className="text-muted hover:text-foreground shrink-0 ml-4">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            <div className="grid grid-cols-3 gap-4 text-sm shrink-0">
              <div>
                <span className="text-[10px] text-muted uppercase tracking-wider">Size</span>
                <p className="text-xs font-mono mt-0.5">{formatSize(selected.entry.size)}</p>
              </div>
              <div>
                <span className="text-[10px] text-muted uppercase tracking-wider">Modified</span>
                <p className="text-xs font-mono mt-0.5">{formatDate(selected.entry.modified_at)}</p>
              </div>
              <div>
                <span className="text-[10px] text-muted uppercase tracking-wider">Type</span>
                <p className="text-xs font-mono mt-0.5">{getExt(selected.entry.name) || "file"}</p>
              </div>
            </div>

            {loadingContent && (
              <div className="flex items-center gap-2 py-4 shrink-0">
                <div className="w-4 h-4 border-2 border-frost/30 border-t-frost rounded-full spin-slow" />
                <span className="text-xs text-muted font-mono">Loading…</span>
              </div>
            )}

            {/* Preview / edit */}
            {fileContent != null && viewMode === "preview" && (
              <div className="flex-1 min-h-0 overflow-hidden rounded-lg border border-white/[0.06] bg-black/30">
                <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/[0.06] bg-white/[0.02]">
                  <span className="text-[10px] text-muted uppercase tracking-wider">{editing ? "Editing" : "Content"}</span>
                  <div className="flex items-center gap-3">
                    {!editing && fileEncoding === "utf-8" && !fileTruncated && (
                      <button onClick={() => { setEditValue(fileContent); setEditing(true); }} className="text-[10px] text-emerald/80 hover:text-emerald font-mono uppercase tracking-wider">edit</button>
                    )}
                    {!editing && <button onClick={enterTail} className="text-[10px] text-amber/80 hover:text-amber font-mono uppercase tracking-wider">tail -f</button>}
                  </div>
                </div>
                {fileTruncated && !editing && (
                  <div className="px-3 py-1.5 border-b border-amber/15 bg-amber/[0.06] text-[10px] font-mono text-amber/90">
                    Preview truncated — editing disabled. Download for the full file.
                  </div>
                )}
                {editing ? (
                  <textarea
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    spellCheck={false}
                    className="w-full p-4 text-xs font-mono text-foreground/90 bg-transparent outline-none resize-none max-h-[50vh] min-h-[30vh]"
                  />
                ) : (
                  <pre className="overflow-auto p-4 text-xs font-mono text-foreground/80 leading-relaxed max-h-[50vh] whitespace-pre-wrap break-words">{fileContent}</pre>
                )}
              </div>
            )}

            {viewMode === "tail" && (
              <div className="flex-1 min-h-0 overflow-hidden rounded-lg border border-amber/20 bg-black/40">
                <div className="flex items-center justify-between px-3 py-1.5 border-b border-amber/10 bg-amber/5">
                  <div className="flex items-center gap-3">
                    <span className="text-[10px] text-amber uppercase tracking-wider">Tail</span>
                    <span className="text-[10px] text-muted font-mono">{tailLines.length} lines</span>
                    {tailLive && <span className="text-[10px] text-emerald font-mono">live</span>}
                  </div>
                  <div className="flex items-center gap-3">
                    {selected.node === selfNodeId && (
                      <button onClick={() => setTailLive((v) => !v)} className="text-[10px] font-mono uppercase tracking-wider" style={{ color: tailLive ? "var(--emerald)" : "var(--muted)" }}>
                        {tailLive ? "stop watch" : "start watch"}
                      </button>
                    )}
                    <button onClick={() => { setViewMode("preview"); setTailLive(false); }} className="text-[10px] text-frost/80 hover:text-frost font-mono uppercase tracking-wider">preview</button>
                  </div>
                </div>
                <pre className="overflow-auto p-4 text-xs font-mono text-foreground/80 leading-relaxed max-h-[50vh] whitespace-pre-wrap break-words">
                  {tailLines.join("\n")}
                  <div ref={tailEndRef} />
                </pre>
              </div>
            )}

            {/* Actions */}
            <div className="flex items-center gap-3 pt-2 border-t border-white/[0.06] shrink-0">
              {editing ? (
                <>
                  <button onClick={saveEdit} disabled={saving} className="px-4 py-2 rounded-lg text-xs font-semibold bg-emerald/15 text-emerald border border-emerald/30 hover:bg-emerald/25 disabled:opacity-40">
                    {saving ? "Saving…" : "Save"}
                  </button>
                  <button onClick={() => setEditing(false)} className="px-4 py-2 rounded-lg text-xs font-medium text-muted hover:text-foreground">Cancel</button>
                </>
              ) : (
                <a href={fsDownloadUrl(selected.entry.path, selected.node)} className="px-4 py-2 rounded-lg text-xs font-semibold bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20 transition-all">
                  Download
                </a>
              )}
              {!editing && (
                <button onClick={deleteSelected} className="px-4 py-2 rounded-lg text-xs font-semibold bg-rose/10 text-rose border border-rose/20 hover:bg-rose/20 transition-all">
                  Delete
                </button>
              )}
              <button onClick={closeModal} className="px-4 py-2 rounded-lg text-xs font-medium text-muted hover:text-foreground ml-auto">Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
