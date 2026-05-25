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
  color = "#fa6432",
}: {
  className?: string;
  size?: number;
  color?: string;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 150 170"
      fill={color}
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Yggdrasil tree logo"
    >
      <path
        d="M57.2,36.22c-2.41-6.61-4.38-13.26-5.36-20.21-.29-1.94-.25-2,1.62-2.65,3.77-1.31,7.62-2.34,11.57-2.95,1.83-.28,1.92-.22,2.17,1.65,1.04,7.89,3.26,15.44,6.62,22.65,.38,.82,.59,2.15,1.46,2.19,1.11,.05,1.26-1.36,1.66-2.21,3.35-7.13,5.55-14.6,6.56-22.42,.27-2.11,.33-2.16,2.46-1.85,3.96,.59,7.78,1.71,11.56,2.99,1.72,.59,1.69,.66,1.42,2.53-.9,6.25-2.53,12.31-4.67,18.24-.24,.67-.98,1.46-.26,2.09,.57,.5,1.29-.15,1.9-.4,6.45-2.67,12.07-6.64,16.97-11.6q1.74-1.72,3.68-.1c2.89,2.41,5.66,4.94,8.09,7.82,1.22,1.46,1.21,1.48-.05,2.81-9.64,10.14-21.57,16.63-35.2,19.67-1.4,.42-1.5,.73-.43,1.82,3.55,3.63,7.48,6.81,11.7,9.62,1.32,.88,1.37,.84,2.9-.6,6.5-6.08,13.83-10.9,21.92-14.58,2.37-1.08,4.81-2,7.26-2.9,1.66-.61,1.93-.47,2.61,1.18,1.53,3.69,2.84,7.45,3.67,11.36,.38,1.76,.35,1.84-1.44,2.48-3.91,1.38-7.68,3.09-11.27,5.17-2.32,1.35-4.56,2.83-6.71,4.43-.6,.44-1.55,.8-1.45,1.63,.12,1.02,1.28,.95,2.02,1.19,5.5,1.74,11.13,2.83,16.89,3.28,4.33,.33,3.49,.2,3.15,3.87-.32,3.44-.99,6.83-1.96,10.14-.52,1.77-.57,1.84-2.4,1.7-22.06-1.63-42.83-11.44-58.39-27.1-2.21-2.2-1.8-2.45-4.18-.06-11.99,12.09-26.24,20.29-42.71,24.59-4.98,1.3-10.04,2.16-15.17,2.55-2.37,.18-2.33,.12-3-2.09-1.07-3.54-1.6-7.18-2.01-10.84-.28-2.44-.23-2.52,2.35-2.7,5.61-.39,11.11-1.32,16.49-2.93,.95-.29,1.9-.6,2.84-.93,.81-.29,.9-.82,.27-1.36-5.7-4.56-12.18-8.03-19.05-10.49-2.03-.71-2.05-.74-1.57-2.77,.88-3.74,2.01-7.4,3.57-10.91,.87-1.96,.9-2.01,2.89-1.32,10.82,3.72,20.74,9.8,29.16,17.53,1.47,1.33,1.46,1.3,3.09,.22,4.2-2.83,8.13-6.04,11.6-9.73,1.03-1.24-.9-1.55-1.76-1.74-13.5-3.1-25.07-9.78-34.6-19.81-.57-.58-.68-1.16-.1-1.81,3.01-3.39,6.23-6.55,9.8-9.36,.74-.58,1.24-.24,1.77,.31,5.17,5.46,11.41,9.9,18.42,12.66,.41,.16,.83,.42,1.61,.04"
        fillRule="evenodd"
      />
      <path
        d="M75.16,140.03c-1.91,0-3.82,.01-5.73-.02-1.88-.02-1.95,.1-2,.96-.13,3.04-.41,6.08-1.07,9.07-.33,1.48-.26,1.5,1.17,1.54,16.55,.42,32.32-3.08,47.19-10.32,6.51-3.25,12.6-7.36,17.99-12.62,1.37-1.33,1.36-1.43-.12-2.59-3.84-3.05-7.88-5.91-12.23-8.42-1.37-.81-1.38-.82-.72-2.26,1.23-2.58,2.16-5.27,2.76-8.03,.3-1.42,.36-1.41,1.74-.81,12.02,5.28,23.1,12.3,32.62,21.4,.78,.74,.76,.75-.48,1.97-8.45,8.42-18.51,14.45-29.63,18.61-15.37,5.66-31.22,6.77-47.4,3.99-2.46-.43-2.44-.41-2.53,2.03-.12,2.64-.46,5.28-1.1,7.88-.31,1.24-.29,1.3,.95,1.41,1.51,.14,3.03,.11,4.54,.12Z"
        fillRule="evenodd"
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
