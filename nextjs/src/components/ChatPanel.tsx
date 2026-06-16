"use client";
import { useState, useEffect, useRef } from "react";

interface Msg {
  id: string;
  text: string;
  sender: string;
  timestamp: string;
}

interface Props {
  channel?: string;
  apiBase?: string;
}

export default function ChatPanel({ channel = "general", apiBase = "" }: Props) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const [sender, setSender] = useState("trader");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const fetchMessages = async () => {
    try {
      const res = await fetch(`${apiBase}/api/messenger/channels/${encodeURIComponent(channel)}/messages?limit=50`);
      if (res.ok) {
        const data = await res.json() as { messages: Msg[] };
        setMessages(data.messages ?? []);
      }
    } catch {
      // api not available
    }
  };

  useEffect(() => { void fetchMessages(); }, [channel]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const send = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      await fetch(`${apiBase}/api/messenger`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim(), sender, channel }),
      });
      setText("");
      await fetchMessages();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 0 }}>
      <div style={{
        flex: 1,
        overflowY: "auto",
        padding: "12px 0",
        display: "flex",
        flexDirection: "column",
        gap: 2,
        minHeight: 0,
      }}>
        {messages.length === 0 && (
          <div style={{ color: "var(--muted)", textAlign: "center", padding: 32, fontSize: 13 }}>
            No messages yet in #{channel}
          </div>
        )}
        {messages.map(m => (
          <div key={m.id} style={{
            padding: "6px 12px",
            borderRadius: 6,
            transition: "background 0.1s",
          }}
            onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.03)")}
            onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
          >
            <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 2 }}>
              <span style={{ fontWeight: 600, color: "var(--accent)", fontSize: 12 }}>{m.sender}</span>
              <span style={{ color: "var(--muted)", fontSize: 10 }}>
                {new Date(m.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <div style={{ color: "var(--text)", fontSize: 13, lineHeight: 1.5 }}>{m.text}</div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div style={{
        borderTop: "1px solid var(--border)",
        padding: "12px",
        display: "flex",
        gap: 8,
        alignItems: "center",
      }}>
        <input
          value={sender}
          onChange={e => setSender(e.target.value)}
          style={{ width: 90, fontSize: 12, padding: "6px 10px" }}
          placeholder="name"
        />
        <input
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => e.key === "Enter" && !e.shiftKey && void send()}
          style={{ flex: 1, fontSize: 13, padding: "6px 12px" }}
          placeholder={`Message #${channel}`}
          disabled={loading}
        />
        <button
          onClick={() => void send()}
          disabled={loading || !text.trim()}
          style={{
            background: "var(--accent)",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            padding: "6px 16px",
            fontSize: 13,
            fontWeight: 600,
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.6 : 1,
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
