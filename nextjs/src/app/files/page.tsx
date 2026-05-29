"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import {
  getFsListing,
  getFsRead,
  getFsTail,
  createFsWatchStream,
  grepFs,
} from "@/lib/api";
import type { FsEntry } from "@/lib/types";

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
    return d.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" }) +
      " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "--";
  }
}

function getFileExtension(name: string): string {
  const idx = name.lastIndexOf(".");
  if (idx < 0) return "";
  return name.slice(idx + 1).toLowerCase();
}

const TEXT_VIEWABLE_EXTS = new Set([
  "py", "ts", "tsx", "js", "jsx", "rs", "go", "c", "cpp", "h", "java", "rb",
  "sh", "bash", "zsh", "toml", "yaml", "yml", "json", "xml", "html", "css",
  "scss", "md", "txt", "rst", "cfg", "ini", "conf", "env", "log", "csv",
  "sql", "graphql", "proto", "dockerfile", "makefile", "gitignore",
]);

// 4 MB threshold for inline viewing
const MAX_VIEWABLE_SIZE = 4 * 1024 * 1024;

function isTextViewable(entry: FsEntry): boolean {
  if (entry.is_dir || entry.size > MAX_VIEWABLE_SIZE) return false;
  const ext = getFileExtension(entry.name);
  // Also handle extensionless files that are common text files
  const lowerName = entry.name.toLowerCase();
  return TEXT_VIEWABLE_EXTS.has(ext) ||
    ["makefile", "dockerfile", "readme", "license", "changelog", ".gitignore", ".env"].includes(lowerName);
}

// Rough icon selection based on extension
function FileIcon({ entry }: { entry: FsEntry }) {
  if (entry.is_dir) {
    return (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--amber)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
      </svg>
    );
  }
  const ext = getFileExtension(entry.name);
  const codeExts = ["py", "ts", "tsx", "js", "jsx", "rs", "go", "c", "cpp", "h", "java", "rb", "sh", "toml", "yaml", "yml", "json"];
  const imgExts = ["png", "jpg", "jpeg", "gif", "svg", "webp", "ico"];
  const docExts = ["md", "txt", "rst", "pdf", "doc", "docx"];

  if (codeExts.includes(ext)) {
    return (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--frost)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="16 18 22 12 16 6" />
        <polyline points="8 6 2 12 8 18" />
      </svg>
    );
  }
  if (imgExts.includes(ext)) {
    return (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--emerald)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
        <circle cx="8.5" cy="8.5" r="1.5" />
        <polyline points="21 15 16 10 5 21" />
      </svg>
    );
  }
  if (docExts.includes(ext)) {
    return (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--foreground-dim)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
        <polyline points="10 9 9 9 8 9" />
      </svg>
    );
  }
  // Default file icon
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M13 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V9z" />
      <polyline points="13 2 13 9 20 9" />
    </svg>
  );
}

type ViewMode = "preview" | "tail";

export default function FilesPage() {
  const [currentPath, setCurrentPath] = useState("");
  const [entries, setEntries] = useState<FsEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [selectedFile, setSelectedFile] = useState<FsEntry | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [fileTruncated, setFileTruncated] = useState(false);
  const [loadingContent, setLoadingContent] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("preview");
  const [tailLines, setTailLines] = useState<string[]>([]);
  const [tailLive, setTailLive] = useState(false);
  const tailEsRef = useRef<EventSource | null>(null);
  const tailEndRef = useRef<HTMLDivElement | null>(null);

  // Grep panel state
  const [grepOpen, setGrepOpen] = useState(false);
  const [grepPattern, setGrepPattern] = useState("");
  const [grepRegex, setGrepRegex] = useState(false);
  const [grepResults, setGrepResults] = useState<
    { path: string; line_number: number; line: string; match: string }[]
  >([]);
  const [grepLoading, setGrepLoading] = useState(false);

  const runGrep = async () => {
    if (!grepPattern.trim()) return;
    setGrepLoading(true);
    try {
      const r = await grepFs(currentPath, grepPattern, { regex: grepRegex, max_matches: 200 });
      setGrepResults(r.matches);
    } catch {
      setGrepResults([]);
    } finally {
      setGrepLoading(false);
    }
  };

  const fetchListing = useCallback(async (path: string, fresh = false) => {
    setLoading(true);
    setError(false);
    try {
      const res = await getFsListing(path, fresh);
      // Sort: directories first, then files, both alphabetical
      const sorted = [...res.entries].sort((a, b) => {
        if (a.is_dir && !b.is_dir) return -1;
        if (!a.is_dir && b.is_dir) return 1;
        return a.name.localeCompare(b.name);
      });
      setEntries(sorted);
      setCurrentPath(res.path || path);
    } catch {
      setError(true);
      setEntries([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchListing(currentPath);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const navigateTo = (path: string) => {
    setSelectedFile(null);
    setFileContent(null);
    setCurrentPath(path);
    fetchListing(path);
  };

  const openFile = async (entry: FsEntry) => {
    setSelectedFile(entry);
    setFileContent(null);
    setFileTruncated(false);
    setTailLines([]);
    setViewMode("preview");
    setTailLive(false);
    if (isTextViewable(entry)) {
      setLoadingContent(true);
      try {
        const res = await getFsRead(entry.path);
        setFileContent(res.encoding === "base64" ? "[binary file — use Download]" : res.content);
        setFileTruncated(res.truncated);
      } catch {
        setFileContent(null);
      } finally {
        setLoadingContent(false);
      }
    }
  };

  // Switch into tail mode: load last N lines once, optionally subscribe to SSE.
  const enterTail = async () => {
    if (!selectedFile) return;
    setViewMode("tail");
    try {
      const t = await getFsTail(selectedFile.path, 300);
      setTailLines(t.lines);
    } catch {
      setTailLines([]);
    }
  };

  // SSE watch toggle
  useEffect(() => {
    if (!tailLive || viewMode !== "tail" || !selectedFile) {
      if (tailEsRef.current) {
        tailEsRef.current.close();
        tailEsRef.current = null;
      }
      return;
    }
    const es = createFsWatchStream(selectedFile.path);
    tailEsRef.current = es;
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as { line: string };
        setTailLines((prev) => [...prev.slice(-500), data.line]);
        setTimeout(() => tailEndRef.current?.scrollIntoView({ behavior: "smooth" }), 0);
      } catch {
        /* ignore non-JSON keepalives */
      }
    };
    es.onerror = () => { /* let close happen naturally */ };
    return () => {
      es.close();
      tailEsRef.current = null;
    };
  }, [tailLive, viewMode, selectedFile]);

  const navigateUp = () => {
    if (!currentPath) return;
    const parts = currentPath.replace(/\/$/, "").split("/");
    parts.pop();
    navigateTo(parts.join("/"));
  };

  // Build breadcrumb parts
  const breadcrumbs = currentPath
    ? currentPath.split("/").filter(Boolean)
    : [];

  return (
    <div className="p-6 h-screen overflow-y-auto animate-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Files</h1>
          <p className="text-sm text-muted mt-1">
            Browse the node filesystem
            {!loading && entries.length > 0 && (
              <span className="ml-2 text-foreground-dim font-mono">
                {entries.length} item{entries.length !== 1 ? "s" : ""}
                {" / "}
                {formatSize(entries.reduce((sum, e) => sum + (e.is_dir ? 0 : e.size), 0))} total
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setGrepOpen((v) => !v)}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-amber/70 hover:text-amber bg-amber/5 hover:bg-amber/10 border border-amber/10 hover:border-amber/20 transition-all"
          >
            <svg className="inline-block mr-1.5 -mt-0.5" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            {grepOpen ? "Hide search" : "Search"}
          </button>
          <button
            onClick={() => fetchListing(currentPath, true)}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-frost/70 hover:text-frost bg-frost/5 hover:bg-frost/10 border border-frost/10 hover:border-frost/20 transition-all"
          >
            <svg className="inline-block mr-1.5 -mt-0.5" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polyline points="23 4 23 10 17 10" />
              <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" />
            </svg>
            Refresh
          </button>
        </div>
      </div>

      {/* Grep / search-in-tree panel */}
      {grepOpen && (
        <div className="glass-card p-3 mb-4 space-y-2">
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={grepPattern}
              onChange={(e) => setGrepPattern(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") runGrep(); }}
              placeholder={`Search ${currentPath || "/"}  -  Enter to run`}
              className="flex-1 bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1.5 text-xs font-mono outline-none focus:border-amber/30"
              autoFocus
            />
            <label className="flex items-center gap-1 text-[10px] text-muted font-mono cursor-pointer">
              <input type="checkbox" checked={grepRegex} onChange={(e) => setGrepRegex(e.target.checked)} />
              regex
            </label>
            <button
              onClick={runGrep}
              disabled={grepLoading || !grepPattern.trim()}
              className="px-3 py-1.5 rounded text-xs font-semibold bg-amber/15 text-amber border border-amber/30 hover:bg-amber/25 disabled:opacity-30"
            >
              {grepLoading ? "..." : "Grep"}
            </button>
          </div>
          {grepResults.length > 0 && (
            <div className="max-h-64 overflow-y-auto divide-y divide-white/[0.04] text-[11px] font-mono">
              {grepResults.map((m, i) => (
                <button
                  key={i}
                  onClick={() => {
                    // Try to open the matching file via the listing
                    const matching = entries.find((e) => e.path === m.path);
                    if (matching) openFile(matching);
                  }}
                  className="w-full text-left flex items-baseline gap-2 py-1 hover:bg-white/[0.04]"
                >
                  <span className="text-frost/70 shrink-0">{m.path}</span>
                  <span className="text-muted shrink-0">:{m.line_number}</span>
                  <span className="text-foreground/80 truncate">{m.line}</span>
                </button>
              ))}
            </div>
          )}
          {grepResults.length === 0 && !grepLoading && grepPattern && (
            <p className="text-[11px] text-muted italic">no matches</p>
          )}
        </div>
      )}

      {/* Breadcrumb navigation */}
      <div className="glass-card px-4 py-3 mb-4 flex items-center gap-1 overflow-x-auto">
        <button
          onClick={() => navigateTo("")}
          className="text-xs font-mono text-frost/70 hover:text-frost transition-colors shrink-0"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="inline-block -mt-0.5">
            <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
          </svg>
        </button>
        {breadcrumbs.map((part, i) => {
          const path = "/" + breadcrumbs.slice(0, i + 1).join("/");
          const isLast = i === breadcrumbs.length - 1;
          return (
            <span key={i} className="flex items-center gap-1 shrink-0">
              <span className="text-muted/40 text-xs">/</span>
              {isLast ? (
                <span className="text-xs font-mono text-foreground">{part}</span>
              ) : (
                <button
                  onClick={() => navigateTo(path)}
                  className="text-xs font-mono text-frost/70 hover:text-frost transition-colors"
                >
                  {part}
                </button>
              )}
            </span>
          );
        })}
        {currentPath && (
          <button
            onClick={navigateUp}
            className="ml-auto shrink-0 text-xs text-muted hover:text-foreground-dim transition-colors flex items-center gap-1"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="15 18 9 12 15 6" />
            </svg>
            Up
          </button>
        )}
      </div>

      {/* File listing */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <div className="w-8 h-8 border-2 border-frost/30 border-t-frost rounded-full spin-slow mx-auto mb-4" />
            <p className="text-sm text-muted font-mono">Loading directory...</p>
          </div>
        </div>
      ) : error ? (
        <div className="flex items-center justify-center py-20">
          <div className="text-center space-y-2">
            <div className="w-12 h-12 rounded-full bg-rose/10 flex items-center justify-center mx-auto">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--rose)" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
            </div>
            <p className="text-sm text-muted">Backend unreachable</p>
            <p className="text-xs text-muted/60">Cannot read filesystem at this path</p>
          </div>
        </div>
      ) : entries.length === 0 ? (
        <div className="flex items-center justify-center py-20">
          <div className="text-center space-y-2">
            <div className="w-12 h-12 rounded-full bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mx-auto">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="1.5">
                <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
              </svg>
            </div>
            <p className="text-sm text-muted">Empty directory</p>
          </div>
        </div>
      ) : (
        <div className="glass-card overflow-hidden">
          {/* Table header */}
          <div className="grid grid-cols-[1fr_100px_180px] gap-4 px-4 py-2.5 border-b border-white/[0.06] text-[10px] text-muted uppercase tracking-widest font-medium">
            <span>Name</span>
            <span className="text-right">Size</span>
            <span className="text-right">Modified</span>
          </div>
          {/* Entries */}
          <div className="divide-y divide-white/[0.03]">
            {entries.map((entry) => (
              <button
                key={entry.path}
                onClick={() => {
                  if (entry.is_dir) {
                    navigateTo(entry.path);
                  } else {
                    openFile(entry);
                  }
                }}
                className="
                  w-full grid grid-cols-[1fr_100px_180px] gap-4 items-center px-4 py-3
                  text-left hover:bg-white/[0.03] transition-colors
                "
              >
                <div className="flex items-center gap-3 min-w-0">
                  <FileIcon entry={entry} />
                  <span className={`text-sm truncate ${entry.is_dir ? "font-medium text-foreground" : "text-foreground/80"}`}>
                    {entry.name}
                  </span>
                  {entry.is_dir && (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="2" className="shrink-0 ml-auto opacity-0 group-hover:opacity-100">
                      <polyline points="9 18 15 12 9 6" />
                    </svg>
                  )}
                </div>
                <span className="text-xs font-mono text-muted text-right">
                  {entry.is_dir ? "--" : formatSize(entry.size)}
                </span>
                <span className="text-[11px] font-mono text-muted text-right">
                  {formatDate(entry.modified_at)}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* File preview modal */}
      {selectedFile && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-6">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => { setSelectedFile(null); setFileContent(null); }}
          />
          {/* Modal */}
          <div className={`relative glass-card p-6 w-full space-y-4 z-10 flex flex-col ${fileContent != null ? "max-w-3xl max-h-[85vh]" : "max-w-lg"}`}>
            <div className="flex items-start justify-between shrink-0">
              <div className="flex items-center gap-3 min-w-0">
                <FileIcon entry={selectedFile} />
                <div className="min-w-0">
                  <h3 className="text-sm font-mono font-semibold text-foreground truncate">
                    {selectedFile.name}
                  </h3>
                  <p className="text-[11px] text-muted font-mono truncate mt-0.5">
                    {selectedFile.path}
                  </p>
                </div>
              </div>
              <button
                onClick={() => { setSelectedFile(null); setFileContent(null); }}
                className="text-muted hover:text-foreground transition-colors shrink-0 ml-4"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm shrink-0">
              <div>
                <span className="text-[10px] text-muted uppercase tracking-wider">Size</span>
                <p className="text-xs font-mono mt-0.5">{formatSize(selectedFile.size)}</p>
              </div>
              <div>
                <span className="text-[10px] text-muted uppercase tracking-wider">Modified</span>
                <p className="text-xs font-mono mt-0.5">{formatDate(selectedFile.modified_at)}</p>
              </div>
              <div>
                <span className="text-[10px] text-muted uppercase tracking-wider">Extension</span>
                <p className="text-xs font-mono mt-0.5">{getFileExtension(selectedFile.name) || "none"}</p>
              </div>
              <div>
                <span className="text-[10px] text-muted uppercase tracking-wider">Full Path</span>
                <p className="text-xs font-mono mt-0.5 break-all">{selectedFile.path}</p>
              </div>
            </div>

            {/* Inline text viewer */}
            {loadingContent && (
              <div className="flex items-center gap-2 py-4 shrink-0">
                <div className="w-4 h-4 border-2 border-frost/30 border-t-frost rounded-full spin-slow" />
                <span className="text-xs text-muted font-mono">Loading file content...</span>
              </div>
            )}
            {fileContent != null && viewMode === "preview" && (
              <div className="flex-1 min-h-0 overflow-hidden rounded-lg border border-white/[0.06] bg-black/30">
                <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/[0.06] bg-white/[0.02]">
                  <div className="flex items-center gap-3">
                    <span className="text-[10px] text-muted uppercase tracking-wider font-medium">Content</span>
                    <span className="text-[10px] text-muted font-mono">{fileContent.split("\n").length} lines</span>
                  </div>
                  <button
                    onClick={enterTail}
                    className="text-[10px] text-amber/80 hover:text-amber font-mono uppercase tracking-wider"
                  >
                    tail -f
                  </button>
                </div>
                {fileTruncated && (
                  <div className="px-3 py-1.5 border-b border-amber/15 bg-amber/[0.06] text-[10px] font-mono text-amber/90">
                    Preview truncated — showing the first {formatSize(fileContent?.length ?? 0)} of{" "}
                    {formatSize(selectedFile.size)}. Use Download for the full file.
                  </div>
                )}
                <pre className="overflow-auto p-4 text-xs font-mono text-foreground/80 leading-relaxed max-h-[50vh] whitespace-pre-wrap break-words">
                  {fileContent}
                </pre>
              </div>
            )}
            {viewMode === "tail" && (
              <div className="flex-1 min-h-0 overflow-hidden rounded-lg border border-amber/20 bg-black/40">
                <div className="flex items-center justify-between px-3 py-1.5 border-b border-amber/10 bg-amber/5">
                  <div className="flex items-center gap-3">
                    <span className="text-[10px] text-amber uppercase tracking-wider font-medium">Tail</span>
                    <span className="text-[10px] text-muted font-mono">{tailLines.length} lines</span>
                    {tailLive && <span className="text-[10px] text-emerald font-mono">live</span>}
                  </div>
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setTailLive((v) => !v)}
                      className="text-[10px] font-mono uppercase tracking-wider"
                      style={{ color: tailLive ? "var(--emerald)" : "var(--muted)" }}
                    >
                      {tailLive ? "stop watch" : "start watch"}
                    </button>
                    <button
                      onClick={() => { setViewMode("preview"); setTailLive(false); }}
                      className="text-[10px] text-frost/80 hover:text-frost font-mono uppercase tracking-wider"
                    >
                      preview
                    </button>
                  </div>
                </div>
                <pre className="overflow-auto p-4 text-xs font-mono text-foreground/80 leading-relaxed max-h-[50vh] whitespace-pre-wrap break-words">
                  {tailLines.join("\n")}
                  <div ref={tailEndRef} />
                </pre>
              </div>
            )}

            <div className="flex items-center gap-3 pt-2 border-t border-white/[0.06] shrink-0">
              <a
                href={`/api/v2/fs/stream?path=${encodeURIComponent(selectedFile.path)}`}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 rounded-lg text-xs font-semibold bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20 hover:border-frost/40 transition-all"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="inline-block mr-1.5 -mt-0.5">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Download
              </a>
              {viewMode === "preview" && !selectedFile.is_dir && (
                <button
                  onClick={enterTail}
                  className="px-4 py-2 rounded-lg text-xs font-semibold bg-amber/10 text-amber border border-amber/20 hover:bg-amber/20 hover:border-amber/40 transition-all"
                >
                  Tail
                </button>
              )}
              <button
                onClick={() => { setSelectedFile(null); setFileContent(null); setTailLive(false); }}
                className="px-4 py-2 rounded-lg text-xs font-medium text-muted hover:text-foreground transition-colors ml-auto"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
