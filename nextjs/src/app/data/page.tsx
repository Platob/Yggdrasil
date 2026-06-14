"use client";

import { useEffect, useState, useRef } from "react";
import { api, type FsEntry } from "@/lib/api";

function FileIcon({ entry }: { entry: FsEntry }) {
  if (entry.is_dir) return <span style={{ color: "var(--accent)" }}>📁</span>;
  if (entry.name.endsWith(".parquet")) return <span style={{ color: "var(--yellow)" }}>⬡</span>;
  if (entry.name.endsWith(".csv")) return <span style={{ color: "var(--green)" }}>▤</span>;
  return <span style={{ color: "var(--text-muted)" }}>📄</span>;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function Breadcrumb({ path, onClick }: { path: string; onClick: (p: string) => void }) {
  const parts = path.split("/").filter(Boolean);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12, marginBottom: 12 }}>
      <span
        onClick={() => onClick("")}
        style={{ color: "var(--accent)", cursor: "pointer" }}
      >
        root
      </span>
      {parts.map((p, i) => {
        const subpath = parts.slice(0, i + 1).join("/");
        return (
          <span key={subpath} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ color: "var(--text-muted)" }}>/</span>
            <span
              onClick={() => onClick(subpath)}
              style={{ color: i === parts.length - 1 ? "var(--text)" : "var(--accent)", cursor: "pointer" }}
            >
              {p}
            </span>
          </span>
        );
      })}
    </div>
  );
}

function SeriesPreview({ path, column }: { path: string; column: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.analysis.series(path, column, 400)
      .then((data) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        const ys = data.y;
        const min = Math.min(...ys), max = Math.max(...ys);
        const range = max - min || 1;

        ctx.beginPath();
        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth = 1.5;
        ys.forEach((v, i) => {
          const x = (i / (ys.length - 1)) * w;
          const y = h - ((v - min) / range) * (h - 8) - 4;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
      })
      .catch((e) => setError(String(e)));
  }, [path, column]);

  if (error) return <div style={{ color: "var(--red)", fontSize: 11 }}>{error}</div>;
  return <canvas ref={canvasRef} width={500} height={100} style={{ width: "100%", height: 100 }} />;
}

function InspectPanel({ path }: { path: string }) {
  const [info, setInfo] = useState<{ row_count: number; editable: boolean; schema: Record<string, string> } | null>(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<{ column: string } | null>(null);

  useEffect(() => {
    if (!path.endsWith(".parquet")) return;
    setLoading(true);
    fetch(`/api/v2/tabular/inspect?path=${encodeURIComponent(path)}`)
      .then((r) => r.json())
      .then(setInfo)
      .catch(() => null)
      .finally(() => setLoading(false));
  }, [path]);

  if (!path.endsWith(".parquet")) return null;

  return (
    <div className="card" style={{ padding: 16, marginTop: 16 }}>
      <div style={{ fontWeight: 600, marginBottom: 10, fontSize: 13 }}>
        Inspect: {path.split("/").pop()}
      </div>
      {loading && <div style={{ color: "var(--text-muted)", fontSize: 12 }}>Loading schema…</div>}
      {info && (
        <>
          <div style={{ color: "var(--text-muted)", fontSize: 11, marginBottom: 10 }}>
            {info.row_count.toLocaleString()} rows · {Object.keys(info.schema).length} columns
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 12 }}>
            {Object.entries(info.schema).map(([col, type]) => (
              <div
                key={col}
                onClick={() => setPreview({ column: col })}
                style={{
                  background: preview?.column === col ? "rgba(59,130,246,0.15)" : "var(--bg)",
                  border: `1px solid ${preview?.column === col ? "var(--accent)" : "var(--border)"}`,
                  borderRadius: 4, padding: "3px 8px", fontSize: 11, cursor: "pointer",
                }}
              >
                <span style={{ fontWeight: 500 }}>{col}</span>
                <span style={{ color: "var(--text-muted)", marginLeft: 4 }}>{type}</span>
              </div>
            ))}
          </div>
          {preview && (
            <div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>
                Series preview: {preview.column}
              </div>
              <SeriesPreview path={path} column={preview.column} />
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function DataPage() {
  const [path, setPath] = useState("");
  const [entries, setEntries] = useState<FsEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<FsEntry | null>(null);
  const PAGE = 100;

  const navigate = (p: string) => {
    setPath(p);
    setOffset(0);
    setSelected(null);
  };

  useEffect(() => {
    setLoading(true);
    api.fs.ls(path, offset, PAGE)
      .then((r) => { setEntries(r.entries); setTotal(r.total); })
      .catch(() => { setEntries([]); setTotal(0); })
      .finally(() => setLoading(false));
  }, [path, offset]);

  const open = (entry: FsEntry) => {
    if (entry.is_dir) {
      navigate(path ? `${path}/${entry.name}` : entry.name);
    } else {
      setSelected(entry);
    }
  };

  return (
    <div style={{ padding: 28 }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>Data Browser</h1>
        <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 4 }}>
          Explore files in the Ygg Node home directory
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div>
          <div className="card" style={{ padding: 16 }}>
            <Breadcrumb path={path} onClick={navigate} />
            {loading ? (
              <div style={{ color: "var(--text-muted)", fontSize: 12 }}>Loading…</div>
            ) : entries.length === 0 ? (
              <div style={{ color: "var(--text-muted)", fontSize: 12, padding: "20px 0" }}>
                Empty directory
              </div>
            ) : (
              <div>
                {entries.map((e) => (
                  <div
                    key={e.name}
                    onClick={() => open(e)}
                    style={{
                      display: "flex", alignItems: "center", gap: 8,
                      padding: "7px 0", borderBottom: "1px solid var(--border)",
                      cursor: "pointer",
                      background: selected?.name === e.name ? "rgba(59,130,246,0.06)" : "transparent",
                    }}
                  >
                    <FileIcon entry={e} />
                    <span style={{ flex: 1, fontSize: 12, fontWeight: e.is_dir ? 500 : 400 }}>{e.name}</span>
                    {!e.is_dir && (
                      <span style={{ color: "var(--text-muted)", fontSize: 11 }}>{formatSize(e.size)}</span>
                    )}
                  </div>
                ))}
                {total > PAGE && (
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10, fontSize: 12 }}>
                    <button
                      disabled={offset === 0}
                      onClick={() => setOffset(Math.max(0, offset - PAGE))}
                      style={{ background: "var(--border)", border: "none", borderRadius: 4, color: "var(--text)", padding: "4px 10px", cursor: "pointer" }}
                    >
                      ← Prev
                    </button>
                    <span style={{ color: "var(--text-muted)" }}>
                      {offset + 1}–{Math.min(offset + PAGE, total)} of {total}
                    </span>
                    <button
                      disabled={offset + PAGE >= total}
                      onClick={() => setOffset(offset + PAGE)}
                      style={{ background: "var(--border)", border: "none", borderRadius: 4, color: "var(--text)", padding: "4px 10px", cursor: "pointer" }}
                    >
                      Next →
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div>
          {selected ? (
            <>
              <div className="card" style={{ padding: 16, marginBottom: 0 }}>
                <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 6 }}>
                  {selected.name}
                </div>
                <div style={{ color: "var(--text-muted)", fontSize: 11 }}>
                  {formatSize(selected.size)} · {selected.mtime}
                </div>
              </div>
              <InspectPanel path={path ? `${path}/${selected.name}` : selected.name} />
            </>
          ) : (
            <div className="card" style={{ padding: 20, color: "var(--text-muted)", fontSize: 12 }}>
              Select a file to inspect or click a folder to navigate.
              <br /><br />
              <strong style={{ color: "var(--text)" }}>Supported analysis:</strong>
              <ul style={{ marginTop: 8, lineHeight: 2, paddingLeft: 16 }}>
                <li>Parquet schema inspection (column types, row count)</li>
                <li>Series preview — click any numeric column after inspecting</li>
                <li>OHLC charts — available on the Market page</li>
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
