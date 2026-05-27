"use client";

import { useEffect, useState, useCallback } from "react";
import { getFsListing } from "@/lib/api";
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

export default function FilesPage() {
  const [currentPath, setCurrentPath] = useState("");
  const [entries, setEntries] = useState<FsEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [selectedFile, setSelectedFile] = useState<FsEntry | null>(null);

  const fetchListing = useCallback(async (path: string) => {
    setLoading(true);
    setError(false);
    try {
      const res = await getFsListing(path);
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
    setCurrentPath(path);
    fetchListing(path);
  };

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
          <p className="text-sm text-muted mt-1">Browse the node filesystem</p>
        </div>
        <button
          onClick={() => fetchListing(currentPath)}
          className="
            px-3 py-1.5 rounded-lg text-xs font-medium
            text-frost/70 hover:text-frost
            bg-frost/5 hover:bg-frost/10
            border border-frost/10 hover:border-frost/20
            transition-all duration-150
          "
        >
          <svg className="inline-block mr-1.5 -mt-0.5" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" />
          </svg>
          Refresh
        </button>
      </div>

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
                    setSelectedFile(entry);
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
            onClick={() => setSelectedFile(null)}
          />
          {/* Modal */}
          <div className="relative glass-card p-6 max-w-lg w-full space-y-4 z-10">
            <div className="flex items-start justify-between">
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
                onClick={() => setSelectedFile(null)}
                className="text-muted hover:text-foreground transition-colors shrink-0 ml-4"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
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

            <div className="flex items-center gap-3 pt-2 border-t border-white/[0.06]">
              <a
                href={`/api/v2/fs/read?path=${encodeURIComponent(selectedFile.path)}`}
                target="_blank"
                rel="noopener noreferrer"
                className="
                  px-4 py-2 rounded-lg text-xs font-semibold
                  bg-frost/10 text-frost border border-frost/20
                  hover:bg-frost/20 hover:border-frost/40
                  transition-all duration-150
                "
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="inline-block mr-1.5 -mt-0.5">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Download
              </a>
              <button
                onClick={() => setSelectedFile(null)}
                className="px-4 py-2 rounded-lg text-xs font-medium text-muted hover:text-foreground transition-colors"
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
