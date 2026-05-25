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
    <div className="max-w-4xl mx-auto space-y-5 animate-in">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Execute</h1>
        <p className="text-muted text-sm mt-1">Run Python code or shell commands</p>
      </div>

      <div className="bg-card border border-border rounded-xl overflow-hidden">
        <div className="flex border-b border-border">
          {(["python", "cmd"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-4 py-2.5 text-sm font-medium transition-colors ${
                mode === m
                  ? "text-gold border-b-2 border-gold bg-gold/5"
                  : "text-muted hover:text-foreground"
              }`}
            >
              {m === "python" ? "Python" : "Shell"}
            </button>
          ))}
        </div>

        <div className="p-4">
          <textarea
            ref={textareaRef}
            value={code}
            onChange={(e) => setCode(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={8}
            spellCheck={false}
            className="w-full bg-background border border-border rounded-lg p-3 font-mono text-sm text-foreground resize-none focus:outline-none focus:border-gold/40 transition-colors"
            placeholder={mode === "python" ? "# Python code..." : "# Shell command..."}
          />
        </div>

        <div className="flex items-center justify-between px-4 pb-4">
          <span className="text-[10px] text-muted">Ctrl+Enter to run</span>
          <button
            onClick={handleRun}
            disabled={running || !code.trim()}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all bg-gold/10 text-gold border border-gold/20 hover:bg-gold/20 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {running ? (
              <span className="spin-slow inline-block">⟳</span>
            ) : (
              <span>▶</span>
            )}
            {running ? "Running..." : "Run"}
          </button>
        </div>
      </div>

      {results.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-muted">Results</h2>
          {results.map((r) => (
            <ResultCard key={r.id} result={r} />
          ))}
        </div>
      )}
    </div>
  );
}

function ResultCard({ result: r }: { result: ExecResult }) {
  const ok = r.status === "completed";
  return (
    <div className={`bg-card border rounded-xl p-4 animate-in ${ok ? "border-border" : "border-accent-red/30"}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${ok ? "bg-green-500" : "bg-accent-red"}`} />
          <span className="text-xs font-mono text-muted">{r.type}</span>
          <span className="text-xs text-muted">#{r.id}</span>
        </div>
        <div className="flex items-center gap-3 text-xs text-muted">
          {r.duration !== null && <span>{(r.duration * 1000).toFixed(0)}ms</span>}
          {r.returncode !== null && <span>exit {r.returncode}</span>}
        </div>
      </div>

      {r.stdout && (
        <pre className="bg-background border border-border rounded-lg p-3 text-xs font-mono text-foreground overflow-x-auto mb-2 max-h-48 overflow-y-auto">
          {r.stdout}
        </pre>
      )}
      {r.stderr && (
        <pre className="bg-accent-red/5 border border-accent-red/20 rounded-lg p-3 text-xs font-mono text-accent-red overflow-x-auto mb-2 max-h-32 overflow-y-auto">
          {r.stderr}
        </pre>
      )}
      {r.result !== null && r.result !== undefined && (
        <div className="bg-gold/5 border border-gold/20 rounded-lg p-3">
          <p className="text-[10px] uppercase tracking-widest text-gold-dim mb-1">Result</p>
          <pre className="text-xs font-mono text-foreground">{JSON.stringify(r.result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
