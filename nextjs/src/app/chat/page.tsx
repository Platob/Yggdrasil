"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  ts: number;
  engine?: string | null;
}

const SUGGESTIONS = [
  "What are today's EUR/USD rates?",
  "Show me German day-ahead electricity prices",
  "What is the current market sentiment?",
  "Explain OHLC candlestick charts",
  "What is yggdrasil.loki?",
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I'm **Loki**, the Yggdrasil AI agent. I can help you with trading data, market analysis, and more. What would you like to know?",
      ts: Date.now(),
      engine: null,
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    const userMsg: Message = { role: "user", content: text, ts: Date.now() };
    setMessages((m) => [...m, userMsg]);
    setLoading(true);

    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    try {
      const res = await api.chat({ message: text, history });
      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.reply, ts: Date.now(), engine: res.engine },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Error: ${e}. Make sure the Ygg Node backend is running and Loki is configured with an LLM engine.`,
          ts: Date.now(),
          engine: null,
        },
      ]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", padding: 28, gap: 0 }}>
      <div style={{ marginBottom: 16 }}>
        <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>Loki AI</h1>
        <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 4 }}>
          Natural language queries against your data and markets
        </div>
      </div>

      <div
        style={{
          flex: 1, overflow: "auto", display: "flex", flexDirection: "column",
          gap: 16, paddingBottom: 8, minHeight: 0,
        }}
      >
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              flexDirection: m.role === "user" ? "row-reverse" : "row",
              alignItems: "flex-start",
              gap: 10,
            }}
          >
            <div
              style={{
                width: 28, height: 28, borderRadius: "50%", flexShrink: 0,
                background: m.role === "user" ? "var(--accent)" : "#1f2937",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 13,
              }}
            >
              {m.role === "user" ? "U" : "✦"}
            </div>
            <div style={{ maxWidth: "75%" }}>
              <div
                style={{
                  background: m.role === "user" ? "rgba(59,130,246,0.15)" : "var(--surface)",
                  border: "1px solid var(--border)",
                  borderRadius: m.role === "user" ? "12px 4px 12px 12px" : "4px 12px 12px 12px",
                  padding: "10px 14px",
                  lineHeight: 1.6,
                  whiteSpace: "pre-wrap",
                }}
              >
                {m.content}
              </div>
              <div style={{ color: "var(--text-muted)", fontSize: 10, marginTop: 3 }}>
                {new Date(m.ts).toLocaleTimeString()}
                {m.engine && ` · ${m.engine}`}
              </div>
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <div style={{ width: 28, height: 28, borderRadius: "50%", background: "#1f2937", display: "flex", alignItems: "center", justifyContent: "center" }}>
              ✦
            </div>
            <div className="card" style={{ padding: "10px 14px" }}>
              <span style={{ color: "var(--text-muted)" }}>Thinking…</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {messages.length <= 1 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => { setInput(s); inputRef.current?.focus(); }}
              style={{
                background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 16,
                color: "var(--text-muted)", padding: "5px 12px", fontSize: 11, cursor: "pointer",
                transition: "border-color 0.15s",
              }}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <div
        className="card"
        style={{ display: "flex", alignItems: "flex-end", gap: 8, padding: "10px 12px" }}
      >
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKey}
          placeholder="Ask anything about markets, data, or Yggdrasil…"
          rows={1}
          style={{
            flex: 1, background: "transparent", border: "none", outline: "none",
            color: "var(--text)", fontSize: 13, resize: "none", lineHeight: 1.5,
            maxHeight: 120, overflowY: "auto",
          }}
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          style={{
            background: input.trim() ? "var(--accent)" : "var(--border)",
            border: "none", borderRadius: 6, color: "#fff",
            padding: "8px 16px", fontSize: 12, cursor: input.trim() ? "pointer" : "default",
            fontWeight: 500, transition: "background 0.15s", flexShrink: 0,
          }}
        >
          Send
        </button>
      </div>
      <div style={{ color: "var(--text-muted)", fontSize: 10, textAlign: "center", marginTop: 6 }}>
        Enter to send · Shift+Enter for new line
      </div>
    </div>
  );
}
