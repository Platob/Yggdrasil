"use client";

import { useState, useRef } from "react";
import { executePython, executeCmd, type PythonResponse, type CmdResponse } from "@/lib/api";

type ExecResult = {
  id: string;
  type: "python" | "cmd";
  input: string;
  status: string;
  stdout: string | null;
  stderr: string | null;
  result: unknown;
  duration: number | null;
  returncode: number | null;
};

export default function ExecutePage() {
  const [mode, setMode] = useState<"python" | "cmd">("python");
  const [code, setCode] = useState("import sys\nprint(f'Python {sys.version}')\n__result__ = {'platform': sys.platform}");
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<ExecResult[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  async function handleRun() {
    if (!code.trim()) return;
    setRunning(true);
    try {
      let r: ExecResult;
      if (mode === "python") {
        const resp = await executePython(code);
        r = { id: resp.id, type: "python", input: code, status: resp.status, stdout: resp.stdout, stderr: resp.stderr, result: resp.result, duration: resp.duration, returncode: resp.returncode };
      } else {
        const parts = code.split(/\s+/).filter(Boolean);
        const resp = await executeCmd(parts);
        r = { id: resp.id, type: "cmd", input: code, status: resp.status, stdout: resp.stdout, stderr: resp.stderr, result: null, duration: resp.duration, returncode: resp.returncode };
      }
      setResults((prev) => [r, ...prev]);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setResults((prev) => [{ id: Date.now().toString(), type: mode, input: code, status: "error", stdout: null, stderr: msg, result: null, duration: null, returncode: null }, ...prev]);
    } finally {
      setRunning(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      handleRun();
    }
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6 animate-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-foreground">Execute</h1>
        <p className="text-muted text-sm mt-1">Run Python code or shell commands remotely</p>
      </div>

      {/* Editor Card */}
      <div className="nordic-card overflow-hidden">
        {/* Mode Tabs */}
        <div className="flex border-b border-border">
          {(["python", "cmd"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`relative px-5 py-3 text-sm font-medium transition-colors ${
                mode === m
                  ? "text-primary"
                  : "text-muted hover:text-foreground"
              }`}
            >
              <div className="flex items-center gap-2">
                {m === "python" ? (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2C6.5 2 6 4 6 6v3h6v1H4c-2 0-3.5 1.5-3.5 4 0 2.5 1.5 4 3.5 4h2v-3c0-2 1.5-3.5 3.5-3.5h5c1.5 0 2.5-1 2.5-2.5V6c0-2-.5-4-5.5-4z" />
                    <path d="M12 22c5.5 0 6-2 6-4v-3h-6v-1h8c2 0 3.5-1.5 3.5-4 0-2.5-1.5-4-3.5-4h-2v3c0 2-1.5 3.5-3.5 3.5h-5c-1.5 0-2.5 1-2.5 2.5V18c0 2 .5 4 5.5 4z" />
                  </svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="4 17 10 11 4 5" />
                    <line x1="12" y1="19" x2="20" y2="19" />
                  </svg>
                )}
                {m === "python" ? "Python" : "Shell"}
              </div>
              {mode === m && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
              )}
            </button>
          ))}
        </div>

        {/* Code Editor */}
        <div className="p-4">
          <textarea
            ref={textareaRef}
            value={code}
            onChange={(e) => setCode(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={10}
            spellCheck={false}
            className="input-nordic w-full resize-none text-sm leading-relaxed"
            placeholder={mode === "python" ? "# Enter Python code..." : "# Enter shell command..."}
          />
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between px-4 pb-4">
          <div className="flex items-center gap-2 text-xs text-muted">
            <kbd className="px-1.5 py-0.5 bg-background border border-border rounded text-[10px] font-mono">Ctrl</kbd>
            <span>+</span>
            <kbd className="px-1.5 py-0.5 bg-background border border-border rounded text-[10px] font-mono">Enter</kbd>
            <span className="ml-1">to run</span>
          </div>
          <button
            onClick={handleRun}
            disabled={running || !code.trim()}
            className="btn-primary flex items-center gap-2"
          >
            {running ? (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="spin-slow">
                  <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                </svg>
                Running...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                Run Code
              </>
            )}
          </button>
        </div>
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-foreground">Results</h2>
            <button 
              onClick={() => setResults([])}
              className="btn-ghost text-xs"
            >
              Clear all
            </button>
          </div>
          <div className="space-y-3">
            {results.map((r) => (
              <ResultCard key={r.id} result={r} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ResultCard({ result: r }: { result: ExecResult }) {
  const ok = r.status === "completed";
  
  return (
    <div className={`nordic-card p-4 animate-in ${!ok ? "border-destructive/30" : ""}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`status-dot ${ok ? "online" : "offline"}`} />
          <span className="text-xs font-mono text-muted uppercase">{r.type}</span>
          <span className="text-xs text-muted">#{r.id.slice(0, 8)}</span>
        </div>
        <div className="flex items-center gap-4 text-xs text-muted">
          {r.duration !== null && (
            <span className="flex items-center gap-1">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
              {(r.duration * 1000).toFixed(0)}ms
            </span>
          )}
          {r.returncode !== null && (
            <span className={`font-mono ${r.returncode === 0 ? "text-success" : "text-destructive"}`}>
              exit {r.returncode}
            </span>
          )}
        </div>
      </div>

      {/* Output */}
      {r.stdout && (
        <pre className="code-block p-3 overflow-x-auto mb-2 max-h-48 overflow-y-auto text-foreground">
          {r.stdout}
        </pre>
      )}
      
      {r.stderr && (
        <pre className="bg-destructive/5 border border-destructive/20 rounded-lg p-3 text-xs font-mono text-destructive overflow-x-auto mb-2 max-h-32 overflow-y-auto">
          {r.stderr}
        </pre>
      )}
      
      {r.result !== null && r.result !== undefined && (
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
          <p className="text-[10px] uppercase tracking-widest text-primary-dim mb-2 flex items-center gap-1">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
              <polyline points="22 4 12 14.01 9 11.01" />
            </svg>
            Result
          </p>
          <pre className="text-xs font-mono text-foreground">{JSON.stringify(r.result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
