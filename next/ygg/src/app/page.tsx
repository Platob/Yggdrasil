"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

// ─── Canvas Interstellar Scene ───────────────────────────────────────────────

function InterstellarCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let W = 0, H = 0;

    // Stars
    type Star = { x: number; y: number; z: number; px: number; py: number };
    const STAR_COUNT = 220;
    const stars: Star[] = [];

    function initStars() {
      stars.length = 0;
      for (let i = 0; i < STAR_COUNT; i++) {
        stars.push({
          x: (Math.random() - 0.5) * W * 2,
          y: (Math.random() - 0.5) * H * 2,
          z: Math.random() * W,
          px: 0,
          py: 0,
        });
      }
    }

    function resize() {
      W = canvas.width = canvas.offsetWidth;
      H = canvas.height = canvas.offsetHeight;
      initStars();
    }

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();

    // Tree branch definition (normalised coords, center = 0,0)
    type Branch = { x1: number; y1: number; x2: number; y2: number; w: number; delay: number };
    const TREE_BRANCHES: Branch[] = [
      // trunk
      { x1: 0, y1: 120, x2: 0, y2: -60, w: 7, delay: 0 },
      // roots
      { x1: 0, y1: 120, x2: -55, y2: 165, w: 4, delay: 0.1 },
      { x1: 0, y1: 120, x2: 55, y2: 165, w: 4, delay: 0.1 },
      // major left branch
      { x1: 0, y1: 20, x2: -90, y2: -40, w: 5, delay: 0.3 },
      { x1: -90, y1: -40, x2: -130, y2: -80, w: 3, delay: 0.5 },
      { x1: -90, y1: -40, x2: -110, y2: -10, w: 2.5, delay: 0.55 },
      // major right branch
      { x1: 0, y1: 20, x2: 90, y2: -40, w: 5, delay: 0.3 },
      { x1: 90, y1: -40, x2: 130, y2: -80, w: 3, delay: 0.5 },
      { x1: 90, y1: -40, x2: 110, y2: -10, w: 2.5, delay: 0.55 },
      // mid-left
      { x1: 0, y1: -20, x2: -60, y2: -70, w: 3.5, delay: 0.4 },
      { x1: -60, y1: -70, x2: -85, y2: -100, w: 2, delay: 0.6 },
      // mid-right
      { x1: 0, y1: -20, x2: 60, y2: -70, w: 3.5, delay: 0.4 },
      { x1: 60, y1: -70, x2: 85, y2: -100, w: 2, delay: 0.6 },
      // crown
      { x1: 0, y1: -60, x2: 0, y2: -130, w: 4, delay: 0.5 },
      { x1: 0, y1: -130, x2: -30, y2: -165, w: 2.5, delay: 0.7 },
      { x1: 0, y1: -130, x2: 30, y2: -165, w: 2.5, delay: 0.7 },
      { x1: 0, y1: -130, x2: 0, y2: -175, w: 2, delay: 0.75 },
    ];

    const SPEED = 1.2;
    let t = 0; // elapsed seconds
    const GROW_DURATION = 3.5; // seconds for tree to fully draw

    function drawTree(cx: number, cy: number, progress: number) {
      for (const b of TREE_BRANCHES) {
        const branchStart = b.delay / 1.0;
        const branchEnd = branchStart + 0.5;
        const p = Math.max(0, Math.min(1, (progress - branchStart) / (branchEnd - branchStart)));
        if (p <= 0) continue;

        const ex = b.x1 + (b.x2 - b.x1) * p;
        const ey = b.y1 + (b.y2 - b.y1) * p;

        // Glow pass
        ctx.save();
        ctx.strokeStyle = "rgba(242,107,58,0.18)";
        ctx.lineWidth = (b.w + 6) * Math.min(W / 900, 1);
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(cx + b.x1 * Math.min(W / 900, 1), cy + b.y1 * Math.min(H / 700, 1));
        ctx.lineTo(cx + ex * Math.min(W / 900, 1), cy + ey * Math.min(H / 700, 1));
        ctx.stroke();
        ctx.restore();

        // Main line
        ctx.save();
        ctx.strokeStyle = "rgba(242,107,58,0.85)";
        ctx.lineWidth = b.w * Math.min(W / 900, 1);
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(cx + b.x1 * Math.min(W / 900, 1), cy + b.y1 * Math.min(H / 700, 1));
        ctx.lineTo(cx + ex * Math.min(W / 900, 1), cy + ey * Math.min(H / 700, 1));
        ctx.stroke();
        ctx.restore();
      }

      // Floating ember particles on branches
      if (progress > 0.8) {
        const particleCount = 6;
        for (let i = 0; i < particleCount; i++) {
          const angle = (t * 0.6 + i * (Math.PI * 2 / particleCount));
          const r = (90 + Math.sin(t * 0.4 + i) * 20) * Math.min(W / 900, 1);
          const px = cx + Math.cos(angle) * r;
          const py = cy - 30 * Math.min(H / 700, 1) + Math.sin(angle * 1.3) * r * 0.5;
          const alpha = 0.3 + 0.3 * Math.sin(t * 1.5 + i);
          ctx.beginPath();
          ctx.arc(px, py, 1.5, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(251,146,60,${alpha})`;
          ctx.fill();
        }
      }
    }

    let last = 0;
    function animate(now: number) {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      t += dt;

      ctx.clearRect(0, 0, W, H);

      // Deep space background gradient
      const bg = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.8);
      bg.addColorStop(0, "#0d0a12");
      bg.addColorStop(0.5, "#080508");
      bg.addColorStop(1, "#000000");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      // Nebula glow behind tree
      const nebula = ctx.createRadialGradient(W / 2, H / 2.2, 0, W / 2, H / 2.2, W * 0.3);
      nebula.addColorStop(0, "rgba(180,60,20,0.07)");
      nebula.addColorStop(0.5, "rgba(120,30,10,0.04)");
      nebula.addColorStop(1, "transparent");
      ctx.fillStyle = nebula;
      ctx.fillRect(0, 0, W, H);

      // Warp stars
      ctx.save();
      ctx.translate(W / 2, H / 2);
      for (const s of stars) {
        s.px = (s.x / s.z) * W;
        s.py = (s.y / s.z) * H;

        s.z -= SPEED;
        if (s.z <= 0) {
          s.x = (Math.random() - 0.5) * W * 2;
          s.y = (Math.random() - 0.5) * H * 2;
          s.z = W;
          s.px = (s.x / s.z) * W;
          s.py = (s.y / s.z) * H;
        }

        const sx = (s.x / s.z) * W;
        const sy = (s.y / s.z) * H;
        const size = Math.max(0.5, (1 - s.z / W) * 3);
        const alpha = Math.min(1, (1 - s.z / W) * 1.4);

        // Star trail
        ctx.beginPath();
        ctx.moveTo(s.px, s.py);
        ctx.lineTo(sx, sy);
        ctx.strokeStyle = `rgba(255,255,255,${alpha * 0.5})`;
        ctx.lineWidth = size * 0.5;
        ctx.stroke();

        // Star dot
        ctx.beginPath();
        ctx.arc(sx, sy, size * 0.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${alpha})`;
        ctx.fill();
      }
      ctx.restore();

      // Draw Yggdrasil tree
      const treeProgress = Math.min(t / GROW_DURATION, 1);
      // After grown, add gentle pulse
      const pulse = treeProgress >= 1 ? 1 + 0.03 * Math.sin(t * 1.5) : 1;
      ctx.save();
      ctx.scale(pulse, pulse);
      ctx.translate(W / 2 * (1 - pulse), H / 2 * (1 - pulse));
      drawTree(W / 2, H / 2 + (H * 0.05), treeProgress);
      ctx.restore();

      animId = requestAnimationFrame(animate);
    }

    animId = requestAnimationFrame(animate);
    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{ display: "block" }}
    />
  );
}

// ─── Services ──────────────────────────────────────────���─────────────────────

const SERVICES = [
  {
    id: "bot",
    name: "Bot Control",
    tag: "Active",
    href: "/bot",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <path d="M9 9h.01M15 9h.01M9 15h6" />
      </svg>
    ),
  },
  {
    id: "msg",
    name: "Messaging",
    tag: "Active",
    href: "/msg",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    ),
  },
  {
    id: "trading",
    name: "Trading",
    tag: "Soon",
    href: "#",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
        <polyline points="16 7 22 7 22 13" />
      </svg>
    ),
    comingSoon: true,
  },
  {
    id: "data",
    name: "Data Streams",
    tag: "Soon",
    href: "#",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
      </svg>
    ),
    comingSoon: true,
  },
  {
    id: "agents",
    name: "AI Agents",
    tag: "Soon",
    href: "#",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
      </svg>
    ),
    comingSoon: true,
  },
];

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function WelcomePage() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const id = setTimeout(() => setReady(true), 100);
    return () => clearTimeout(id);
  }, []);

  return (
    <div className="relative min-h-screen bg-black overflow-hidden font-sans">

      {/* Canvas background */}
      <InterstellarCanvas />

      {/* Content layer */}
      <div
        className="relative z-10 flex flex-col min-h-screen"
        style={{
          opacity: ready ? 1 : 0,
          transition: "opacity 1s ease",
        }}
      >

        {/* Nav */}
        <nav className="flex items-center justify-between px-8 py-5">
          <div className="flex items-center gap-2.5">
            <svg width="28" height="28" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M24 44V28M24 28L19 33M24 28L29 33M24 28V18M24 18L14 8M14 8L10 4M14 8L10 12M24 18L34 8M34 8L38 4M34 8L38 12M24 18V8M24 8L20 4M24 8L28 4" stroke="#f26b3a" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span className="font-bold text-white tracking-widest text-sm uppercase">Yggdrasil</span>
          </div>
          <Link
            href="/bot"
            className="text-sm font-medium px-4 py-1.5 rounded-full border border-white/20 text-white/70 hover:text-white hover:border-white/40 transition-all"
          >
            Dashboard
          </Link>
        </nav>

        {/* Hero - pushed to lower half since tree is center-canvas */}
        <main className="flex-1 flex flex-col items-center justify-end pb-24 px-8 text-center">
          <div className="space-y-4 mb-12">
            <h1 className="text-5xl md:text-6xl font-bold text-white tracking-tight text-balance">
              The World Tree
            </h1>
            <p className="text-lg text-white/40 tracking-widest uppercase">
              Distributed Systems Framework
            </p>
            <div className="pt-2">
              <Link
                href="/bot"
                className="inline-flex items-center gap-2 px-7 py-3 rounded-full text-sm font-semibold transition-all"
                style={{
                  background: "linear-gradient(135deg, #f26b3a, #dc2626)",
                  color: "#fff",
                  boxShadow: "0 0 32px rgba(242,107,58,0.35)",
                }}
              >
                Enter
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </Link>
            </div>
          </div>

          {/* Service constellation */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 w-full max-w-3xl">
            {SERVICES.map((s) => (
              <Link
                key={s.id}
                href={s.href}
                onClick={(e) => s.comingSoon && e.preventDefault()}
                className="group flex flex-col items-center gap-2 py-4 px-3 rounded-xl border transition-all cursor-pointer"
                style={{
                  background: s.comingSoon ? "rgba(255,255,255,0.02)" : "rgba(242,107,58,0.06)",
                  borderColor: s.comingSoon ? "rgba(255,255,255,0.07)" : "rgba(242,107,58,0.25)",
                }}
              >
                <span
                  className="transition-colors"
                  style={{ color: s.comingSoon ? "rgba(255,255,255,0.25)" : "#f26b3a" }}
                >
                  {s.icon}
                </span>
                <span
                  className="text-xs font-medium transition-colors"
                  style={{ color: s.comingSoon ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.85)" }}
                >
                  {s.name}
                </span>
                <span
                  className="text-[10px] px-2 py-0.5 rounded-full"
                  style={{
                    background: s.comingSoon ? "rgba(255,255,255,0.05)" : "rgba(242,107,58,0.15)",
                    color: s.comingSoon ? "rgba(255,255,255,0.25)" : "#fb923c",
                  }}
                >
                  {s.tag}
                </span>
              </Link>
            ))}
          </div>
        </main>

      </div>
    </div>
  );
}
