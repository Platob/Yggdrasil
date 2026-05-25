"use client";

/**
 * Yggdrasil Logo — accurate SVG recreation of the brand mark.
 *
 * The real logo has:
 *   - A Y-shaped trunk/base (two angled roots meeting a central stem)
 *   - A dense radial crown of ~10 thick blunt-tipped branches fanning outward
 *   - Solid fill, coral-orange (#f26b3a)
 *
 * For animation we convert fills to strokes and use stroke-dasharray draw-on.
 */

import { useEffect, useRef } from "react";

// ─── Static logo (filled, matches brand exactly) ─────────────────────────────

export function YggdrasilLogo({
  className = "",
  size = 40,
  color = "#f26b3a",
}: {
  className?: string;
  size?: number;
  color?: string;
}) {
  // Viewbox 100x120, origin top-left
  // Crown center: (50, 45), trunk Y-fork at (50, 85)
  return (
    <svg
      width={size}
      height={Math.round(size * 1.2)}
      viewBox="0 0 100 120"
      fill={color}
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Yggdrasil tree logo"
    >
      {/* Y-trunk: left root, right root, center stem */}
      <path d="M50 120 L50 85 L32 105 Z" />
      <path d="M50 120 L50 85 L68 105 Z" />
      {/* Stem from fork up to crown base */}
      <rect x="44" y="55" width="12" height="35" rx="6" />
      {/* Crown — 9 radial branches fanning from center (50,45) */}
      {/* Top */}
      <rect x="44" y="10" width="12" height="40" rx="6" />
      {/* Top-left */}
      <rect
        x="44" y="10" width="12" height="40" rx="6"
        transform="rotate(-40 50 55)"
      />
      {/* Left */}
      <rect
        x="44" y="10" width="12" height="40" rx="6"
        transform="rotate(-80 50 55)"
      />
      {/* Bottom-left */}
      <rect
        x="44" y="10" width="12" height="40" rx="6"
        transform="rotate(-120 50 55)"
      />
      {/* Far bottom-left (root fill) */}
      <rect
        x="44" y="10" width="12" height="38" rx="6"
        transform="rotate(-150 50 55)"
      />
      {/* Top-right */}
      <rect
        x="44" y="10" width="12" height="40" rx="6"
        transform="rotate(40 50 55)"
      />
      {/* Right */}
      <rect
        x="44" y="10" width="12" height="40" rx="6"
        transform="rotate(80 50 55)"
      />
      {/* Bottom-right */}
      <rect
        x="44" y="10" width="12" height="40" rx="6"
        transform="rotate(120 50 55)"
      />
      {/* Far bottom-right */}
      <rect
        x="44" y="10" width="12" height="38" rx="6"
        transform="rotate(150 50 55)"
      />
    </svg>
  );
}

// ─── Animated logo (canvas draw-on) ──────────────────────────────────────────

/**
 * Draws the Yggdrasil tree branch-by-branch on a canvas using
 * requestAnimationFrame. Each branch grows from its base outward.
 * After fully drawn the tree gently pulses with a glow.
 */
export function AnimatedYggdrasilTree({
  className = "",
  color = "#f26b3a",
}: {
  className?: string;
  color?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let t = 0;

    // Each segment: [x1,y1, x2,y2, width, startFraction, endFraction]
    // Fractions are 0-1 across the whole grow animation
    type Seg = {
      x1: number; y1: number;
      x2: number; y2: number;
      w: number;
      s: number; // start fraction
      e: number; // end fraction
    };

    // Crown center at (0,0), branches radiate outward.
    // Y-fork below center.
    const L = 130; // branch length reference

    const segments: Seg[] = [
      // stem (center down to fork)
      { x1: 0, y1: 0,    x2: 0,   y2: 70,   w: 11, s: 0,    e: 0.18 },
      // Y left root
      { x1: 0, y1: 70,   x2: -40, y2: 110,  w: 9,  s: 0.15, e: 0.30 },
      // Y right root
      { x1: 0, y1: 70,   x2: 40,  y2: 110,  w: 9,  s: 0.15, e: 0.30 },

      // Crown branches — radiating from (0,0)
      // top
      { x1: 0, y1: 0,    x2: 0,   y2: -L,   w: 11, s: 0.05, e: 0.40 },
      // top-left (−40°)
      { x1: 0, y1: 0,    x2: -L * Math.sin(Math.PI * 40/180), y2: -L * Math.cos(Math.PI * 40/180), w: 11, s: 0.08, e: 0.45 },
      // left (−80°)
      { x1: 0, y1: 0,    x2: -L * Math.sin(Math.PI * 80/180), y2: -L * Math.cos(Math.PI * 80/180), w: 10, s: 0.12, e: 0.50 },
      // lower-left (−120°)
      { x1: 0, y1: 0,    x2: -L * Math.sin(Math.PI * 120/180), y2: -L * Math.cos(Math.PI * 120/180), w: 9,  s: 0.16, e: 0.55 },
      // far lower-left (−150°)
      { x1: 0, y1: 0,    x2: -L * Math.sin(Math.PI * 150/180), y2: -L * Math.cos(Math.PI * 150/180), w: 8,  s: 0.20, e: 0.58 },
      // top-right (+40°)
      { x1: 0, y1: 0,    x2:  L * Math.sin(Math.PI * 40/180),  y2: -L * Math.cos(Math.PI * 40/180), w: 11, s: 0.08, e: 0.45 },
      // right (+80°)
      { x1: 0, y1: 0,    x2:  L * Math.sin(Math.PI * 80/180),  y2: -L * Math.cos(Math.PI * 80/180), w: 10, s: 0.12, e: 0.50 },
      // lower-right (+120°)
      { x1: 0, y1: 0,    x2:  L * Math.sin(Math.PI * 120/180), y2: -L * Math.cos(Math.PI * 120/180), w: 9,  s: 0.16, e: 0.55 },
      // far lower-right (+150°)
      { x1: 0, y1: 0,    x2:  L * Math.sin(Math.PI * 150/180), y2: -L * Math.cos(Math.PI * 150/180), w: 8,  s: 0.20, e: 0.58 },
    ];

    const GROW_DURATION = 2.8; // seconds

    function drawFrame() {
      const W = canvas.width;
      const H = canvas.height;
      const scale = Math.min(W, H) / 340;
      const cx = W / 2;
      const cy = H / 2 - 20 * scale;

      ctx.clearRect(0, 0, W, H);

      const progress = Math.min(t / GROW_DURATION, 1);
      // After fully drawn, gentle breathing glow
      const glow = progress >= 1 ? 4 + 3 * Math.sin(t * 1.8) : 2;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.scale(scale, scale);

      for (const seg of segments) {
        const p = Math.max(0, Math.min(1, (progress - seg.s) / (seg.e - seg.s)));
        if (p <= 0) continue;
        const ex = seg.x1 + (seg.x2 - seg.x1) * p;
        const ey = seg.y1 + (seg.y2 - seg.y1) * p;

        // Outer glow
        ctx.save();
        ctx.strokeStyle = `rgba(242,107,58,0.15)`;
        ctx.lineWidth = seg.w + glow * 3;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(seg.x1, seg.y1);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.restore();

        // Inner glow
        ctx.save();
        ctx.strokeStyle = `rgba(242,107,58,0.35)`;
        ctx.lineWidth = seg.w + glow;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(seg.x1, seg.y1);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.restore();

        // Core stroke
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = seg.w;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(seg.x1, seg.y1);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.restore();
      }

      ctx.restore();
    }

    let last = 0;
    function loop(now: number) {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      t += dt;
      drawFrame();
      animId = requestAnimationFrame(loop);
    }

    // Handle resize
    const ro = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
      canvas.height = canvas.offsetHeight * (window.devicePixelRatio || 1);
      drawFrame();
    });
    ro.observe(canvas);
    canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
    canvas.height = canvas.offsetHeight * (window.devicePixelRatio || 1);

    animId = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
    };
  }, [color]);

  return (
    <canvas
      ref={canvasRef}
      className={`w-full h-full ${className}`}
      style={{ display: "block" }}
      aria-label="Animated Yggdrasil tree"
    />
  );
}

// ─── Brand (logo + wordmark) ──────────────────────────────────────────────────

export function YggdrasilBrand({
  className = "",
  size = 32,
}: {
  className?: string;
  size?: number;
}) {
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <YggdrasilLogo size={size} />
      <span
        className="font-bold tracking-widest uppercase text-white"
        style={{ fontSize: size * 0.5 }}
      >
        Yggdrasil
      </span>
    </div>
  );
}
