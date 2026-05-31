// Client-side rendering of tabular data and SVG charts to downloadable images
// (png/jpg). Dependency-free: a grid is painted onto a <canvas>; charts
// (already SVG) are serialized — with the theme's CSS custom properties
// (var(--emerald)…) resolved to concrete colors so a detached SVG keeps its
// Nordic palette — then rasterized through an <img> onto a canvas. Drives the
// TabularModal "Download as → image" options.

const MIME: Record<string, string> = { png: "image/png", jpg: "image/jpeg" };

// Resolve var(--x) references against the document root's computed theme.
function themeColor(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function downloadBlob(blob: Blob, filename: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// Rasterize an on-screen SVG to a png/jpg blob, painted at `scale`× over a
// themed background. Reproduces exactly what's rendered (same viewBox +
// rendered px box).
export async function svgToImage(svg: SVGSVGElement, fmt: string, scale = 2): Promise<Blob> {
  const rect = svg.getBoundingClientRect();
  const w = Math.ceil(rect.width) || svg.viewBox.baseVal.width || 720;
  const h = Math.ceil(rect.height) || svg.viewBox.baseVal.height || 220;
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(w));
  clone.setAttribute("height", String(h));
  const cs = getComputedStyle(document.documentElement);
  const str = new XMLSerializer()
    .serializeToString(clone)
    .replace(/var\((--[\w-]+)\)/g, (_, n) => cs.getPropertyValue(n).trim() || "#888");

  const img = new Image();
  await new Promise<void>((res, rej) => {
    img.onload = () => res();
    img.onerror = () => rej(new Error("svg render failed"));
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(str);
  });
  const canvas = document.createElement("canvas");
  canvas.width = w * scale;
  canvas.height = h * scale;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(scale, scale);
  ctx.fillStyle = themeColor("--background", "#050510");
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
  return await new Promise<Blob>((res, rej) =>
    canvas.toBlob((b) => (b ? res(b) : rej(new Error("encode failed"))), MIME[fmt] ?? "image/png"),
  );
}

// Paint a bounded grid (header + rows + row index) onto a canvas in the Nordic
// palette and return it as png | jpg.
export async function tableToImage(columns: string[], rows: string[][], fmt: string, scale = 2): Promise<Blob> {
  const bg = themeColor("--background", "#050510");
  const fg = themeColor("--foreground", "#e4e2df");
  const head = themeColor("--frost", "#67e8f9");
  const muted = themeColor("--muted", "#5a5856");
  const font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
  const padX = 10, rowH = 22, idxW = 44, maxColW = 320;

  const probe = document.createElement("canvas").getContext("2d")!;
  probe.font = font;
  const clip = (s: string) => (s.length > 80 ? s.slice(0, 79) + "…" : s);
  const widths = columns.map((c, ci) => {
    let w = probe.measureText(c).width;
    for (const r of rows) w = Math.max(w, probe.measureText(clip(r[ci] ?? "")).width);
    return Math.min(maxColW, Math.ceil(w) + padX * 2);
  });
  const totalW = idxW + widths.reduce((s, w) => s + w, 0);
  const totalH = rowH * (rows.length + 1);

  const canvas = document.createElement("canvas");
  canvas.width = totalW * scale;
  canvas.height = totalH * scale;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(scale, scale);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, totalW, totalH);
  ctx.font = font;
  ctx.textBaseline = "middle";

  // Header row.
  ctx.fillStyle = head;
  let x = idxW;
  columns.forEach((c, ci) => {
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, 0, widths[ci], rowH);
    ctx.clip();
    ctx.fillText(c, x + padX, rowH / 2 + 1);
    ctx.restore();
    x += widths[ci];
  });
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.beginPath();
  ctx.moveTo(0, rowH);
  ctx.lineTo(totalW, rowH);
  ctx.stroke();

  // Data rows.
  rows.forEach((r, ri) => {
    const y = rowH * (ri + 1);
    ctx.fillStyle = muted;
    ctx.textAlign = "right";
    ctx.fillText(String(ri), idxW - 8, y + rowH / 2 + 1);
    ctx.textAlign = "left";
    ctx.fillStyle = fg;
    let cx = idxW;
    columns.forEach((_, ci) => {
      ctx.save();
      ctx.beginPath();
      ctx.rect(cx, y, widths[ci], rowH);
      ctx.clip();
      ctx.fillText(clip(r[ci] ?? ""), cx + padX, y + rowH / 2 + 1);
      ctx.restore();
      cx += widths[ci];
    });
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.beginPath();
    ctx.moveTo(0, y + rowH);
    ctx.lineTo(totalW, y + rowH);
    ctx.stroke();
  });

  return await new Promise<Blob>((res, rej) =>
    canvas.toBlob((b) => (b ? res(b) : rej(new Error("encode failed"))), MIME[fmt] ?? "image/png"),
  );
}
