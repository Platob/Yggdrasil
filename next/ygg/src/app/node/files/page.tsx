"use client";

import { useEffect, useState, useCallback } from "react";
import { node as api, type FileInfo } from "@/lib/api";
import { formatRelative } from "@/lib/time";

// -- Default directories for node_home --
const DEFAULT_DIRS = ["tmp", "downloads", "documents", "data", "logs", "cache", "mirrors"];

// -- Format file size --
function formatSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
}

// -- Build npfs URL from path --
function npfsUrl(path: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `npfs://${normalized}`;
}

// -- Icons --
const FolderIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
  </svg>
);

const FileIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" />
  </svg>
);

const CopyIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
);

export default function FilesPage() {
  const [currentPath, setCurrentPath] = useState("");
  const [entries, setEntries] = useState<FileInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewingFile, setViewingFile] = useState<{ path: string; content: string } | null>(null);
  const [showNewFolder, setShowNewFolder] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");
  const [deleting, setDeleting] = useState<string | null>(null);
  const [copiedPath, setCopiedPath] = useState<string | null>(null);

  const loadDir = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    setViewingFile(null);
    try {
      const data = await api.listDir(path);
      setEntries(data.entries);
      setCurrentPath(data.path || path);
    } catch {
      setError("Failed to load directory");
      setEntries([]);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadDir(currentPath);
  }, []);

  function navigateTo(path: string) {
    setCurrentPath(path);
    loadDir(path);
  }

  function navigateUp() {
    const parts = currentPath.split("/").filter(Boolean);
    parts.pop();
    const parent = parts.length > 0 ? "/" + parts.join("/") : "";
    navigateTo(parent);
  }

  async function handleFileClick(entry: FileInfo) {
    if (entry.is_dir) {
      navigateTo(entry.path);
    } else {
      try {
        const data = await api.readFile(entry.path);
        setViewingFile({ path: data.path, content: data.content });
      } catch {
        setError("Failed to read file");
      }
    }
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async () => {
      const content = reader.result as string;
      // Extract base64 content if it's a data URL
      const base64 = content.includes(",") ? content.split(",")[1] : content;
      const targetPath = currentPath ? `${currentPath}/${file.name}` : file.name;
      try {
        await api.writeFile(targetPath, base64);
        loadDir(currentPath);
      } catch {
        setError("Upload failed");
      }
    };
    reader.readAsDataURL(file);
    // Reset input
    e.target.value = "";
  }

  async function handleCreateFolder(e: React.FormEvent) {
    e.preventDefault();
    if (!newFolderName.trim()) return;
    const folderPath = currentPath ? `${currentPath}/${newFolderName.trim()}` : newFolderName.trim();
    try {
      await api.mkdir(folderPath);
      setShowNewFolder(false);
      setNewFolderName("");
      loadDir(currentPath);
    } catch {
      setError("Failed to create folder");
    }
  }

  async function handleDelete(entry: FileInfo) {
    if (!confirm(`Delete "${entry.name}"? This cannot be undone.`)) return;
    setDeleting(entry.path);
    try {
      await api.deleteFile(entry.path);
      loadDir(currentPath);
    } catch {
      setError("Failed to delete");
    }
    setDeleting(null);
  }

  function copyNpfsUrl(path: string) {
    const url = npfsUrl(path);
    navigator.clipboard.writeText(url);
    setCopiedPath(path);
    setTimeout(() => setCopiedPath(null), 2000);
  }

  // Build breadcrumb parts
  const breadcrumbs = currentPath
    ? currentPath.split("/").filter(Boolean)
    : [];

  // Merge default directories with entries at root level
  const displayEntries = (() => {
    if (currentPath === "" || currentPath === "/") {
      const existingNames = new Set(entries.map((e) => e.name));
      const defaultDirEntries: FileInfo[] = DEFAULT_DIRS
        .filter((name) => !existingNames.has(name))
        .map((name) => ({
          path: `/${name}`,
          name,
          is_dir: true,
          size: 0,
          modified_at: "",
          created_at: "",
        }));
      return [...entries, ...defaultDirEntries];
    }
    return entries;
  })();

  return (
    <div className="p-6 space-y-6 animate-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-foreground">Files</h1>
          <p className="text-sm text-muted mt-0.5">Browse and manage node filesystem</p>
        </div>
        <div className="flex items-center gap-2">
          {error && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-warning/10 border border-warning/20">
              <div className="status-dot pending" />
              <span className="text-xs font-medium text-warning">{error}</span>
            </div>
          )}
          <button onClick={() => setShowNewFolder(true)} className="btn-ghost text-xs">
            New Folder
          </button>
          <label className="btn-primary text-xs cursor-pointer">
            Upload
            <input type="file" className="hidden" onChange={handleUpload} />
          </label>
        </div>
      </div>

      {/* Breadcrumb with npfs:// path */}
      <div className="flex items-center gap-1 text-xs font-mono">
        <span className="text-muted select-none">npfs://</span>
        <button
          onClick={() => navigateTo("")}
          className="text-primary hover:underline px-0.5"
        >
          /
        </button>
        {breadcrumbs.map((part, idx) => {
          const pathTo = "/" + breadcrumbs.slice(0, idx + 1).join("/");
          return (
            <span key={idx} className="flex items-center gap-1">
              <span className="text-muted">/</span>
              <button
                onClick={() => navigateTo(pathTo)}
                className="text-primary hover:underline px-0.5"
              >
                {part}
              </button>
            </span>
          );
        })}
      </div>

      {/* New folder form */}
      {showNewFolder && (
        <form onSubmit={handleCreateFolder} className="nordic-card p-4 flex items-center gap-3">
          <FolderIcon />
          <input
            type="text"
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            className="input-nordic text-sm font-mono flex-1"
            placeholder="folder_name"
            autoFocus
          />
          <button type="submit" className="btn-primary text-xs">Create</button>
          <button type="button" onClick={() => { setShowNewFolder(false); setNewFolderName(""); }} className="btn-ghost text-xs">Cancel</button>
        </form>
      )}

      {/* File viewer */}
      {viewingFile && (
        <div className="nordic-card overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <div className="flex items-center gap-2 min-w-0">
              <span className="font-mono text-xs text-foreground truncate">{viewingFile.path}</span>
              <button
                onClick={() => copyNpfsUrl(viewingFile.path)}
                className="text-muted hover:text-primary transition-colors shrink-0"
                title={`Copy ${npfsUrl(viewingFile.path)}`}
              >
                <CopyIcon />
              </button>
            </div>
            <button onClick={() => setViewingFile(null)} className="btn-ghost text-xs">Close</button>
          </div>
          <pre className="p-4 text-xs font-mono text-foreground-dim overflow-auto max-h-[500px] whitespace-pre-wrap">{viewingFile.content}</pre>
        </div>
      )}

      {/* File listing */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full spin-slow mx-auto mb-4" />
            <p className="text-primary font-mono text-sm pulse-primary">Loading...</p>
          </div>
        </div>
      ) : (
        <div className="nordic-card overflow-hidden">
          {/* Go up row */}
          {currentPath && (
            <button
              onClick={navigateUp}
              className="w-full px-4 py-3 flex items-center gap-3 text-left hover:bg-card-hover transition-colors border-b border-border/50"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted">
                <polyline points="15 18 9 12 15 6" />
              </svg>
              <span className="text-xs text-muted font-mono">..</span>
            </button>
          )}

          {displayEntries.length === 0 && !currentPath && (
            <div className="p-8 text-center">
              <p className="text-sm text-muted">Empty directory</p>
            </div>
          )}

          {/* Sort: directories first, then files */}
          {[...displayEntries]
            .sort((a, b) => {
              if (a.is_dir && !b.is_dir) return -1;
              if (!a.is_dir && b.is_dir) return 1;
              return a.name.localeCompare(b.name);
            })
            .map((entry) => (
              <div
                key={entry.path}
                className="flex items-center gap-3 px-4 py-2.5 hover:bg-card-hover transition-colors border-b border-border/30 last:border-b-0 group"
              >
                <button
                  onClick={() => handleFileClick(entry)}
                  className="flex items-center gap-3 flex-1 min-w-0 text-left"
                >
                  {entry.is_dir ? <FolderIcon /> : <FileIcon />}
                  <span className="font-mono text-xs text-foreground truncate hover:text-primary transition-colors">
                    {entry.name}
                  </span>
                </button>
                <span className="text-[10px] text-muted shrink-0 w-16 text-right">
                  {entry.is_dir ? "--" : formatSize(entry.size)}
                </span>
                <span className="text-[10px] text-muted shrink-0 w-20 text-right">
                  {entry.modified_at ? formatRelative(entry.modified_at) : "--"}
                </span>
                {/* Copy npfs:// URL button */}
                <button
                  onClick={() => copyNpfsUrl(entry.path)}
                  className={`transition-colors shrink-0 ${
                    copiedPath === entry.path
                      ? "text-success"
                      : "text-muted hover:text-primary opacity-0 group-hover:opacity-100"
                  }`}
                  title={npfsUrl(entry.path)}
                >
                  {copiedPath === entry.path ? (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  ) : (
                    <CopyIcon />
                  )}
                </button>
                <button
                  onClick={() => handleDelete(entry)}
                  disabled={deleting === entry.path}
                  className="text-muted hover:text-destructive transition-colors opacity-0 group-hover:opacity-100 shrink-0"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  </svg>
                </button>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
