"use client";
import { useState, useEffect } from "react";
import ChatPanel from "@/components/ChatPanel";

export default function ChatPage() {
  const [channels, setChannels] = useState<{ name: string; message_count: number }[]>([]);
  const [active, setActive] = useState("general");
  const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "";

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${apiBase}/api/messenger/channels`);
        if (res.ok) {
          const data = await res.json() as { channels: { name: string; message_count: number }[] };
          setChannels(data.channels ?? []);
          if (data.channels?.length > 0 && !data.channels.find(c => c.name === active)) {
            setActive(data.channels[0].name);
          }
        }
      } catch {
        setChannels([{ name: "general", message_count: 0 }]);
      }
    };
    void load();
  }, [apiBase, active]);

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", height: "calc(100vh - 96px)", display: "flex", flexDirection: "column" }}>
      <div style={{ marginBottom: 16 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, letterSpacing: "-0.02em", margin: 0 }}>Messenger</h1>
        <p style={{ color: "var(--muted)", marginTop: 4, fontSize: 12 }}>In-node chat channels</p>
      </div>

      <div className="card" style={{ flex: 1, display: "flex", gap: 0, padding: 0, overflow: "hidden" }}>
        {/* sidebar */}
        <div style={{
          width: 180,
          borderRight: "1px solid var(--border)",
          padding: "12px",
          display: "flex",
          flexDirection: "column",
          gap: 2,
          overflowY: "auto",
        }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", padding: "4px 8px", marginBottom: 4 }}>
            Channels
          </div>
          {(channels.length ? channels : [{ name: "general", message_count: 0 }]).map(ch => (
            <button key={ch.name} onClick={() => setActive(ch.name)} style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "6px 10px",
              borderRadius: 7,
              border: "none",
              background: active === ch.name ? "rgba(59,130,246,0.15)" : "transparent",
              color: active === ch.name ? "var(--text)" : "var(--muted)",
              cursor: "pointer",
              fontSize: 13,
              fontWeight: active === ch.name ? 600 : 400,
              width: "100%",
              textAlign: "left",
            }}>
              <span># {ch.name}</span>
              {ch.message_count > 0 && (
                <span style={{ fontSize: 10, background: "var(--accent)", color: "#fff", borderRadius: 9999, padding: "1px 5px" }}>
                  {ch.message_count > 99 ? "99+" : ch.message_count}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* chat area */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
          <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", fontSize: 13, fontWeight: 600 }}>
            # {active}
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            <ChatPanel channel={active} apiBase={apiBase} />
          </div>
        </div>
      </div>
    </div>
  );
}
