# Yggdrasil Frontend — Agent Instructions

Read `node_modules/next/dist/docs/` before writing Next.js code — this version has breaking changes.

## Architecture

- **React 19 + Next.js 16 + Tailwind v4**
- Bot API proxied via `/api/bot/*` → FastAPI at `BOT_API_URL` (see `next.config.ts`)
- Next.js API routes at `/api/*` for caching, aggregation, config
- `GlobalSidebar` in root layout — all pages get sidebar + dark/light toggle

## Key Files

| File | Purpose |
|------|---------|
| `src/lib/api.ts` | Client API: `bot.*` (proxied) and `api.*` (local) |
| `src/lib/bot-client.ts` | Server-side bot API client for route handlers |
| `src/components/global-sidebar.tsx` | Sidebar with nav, theme toggle, status |
| `src/components/service-layout.tsx` | Sidebar + content wrapper |
| `src/app/globals.css` | Design tokens (CSS vars), dark/light themes |
| `next.config.ts` | Bot API proxy rewrite config |

## Design System

- **Brand**: `#f26b3a` (coral/orange) — `var(--primary)`
- **Dark default**, light via `.light` class on `<html>`
- **CSS classes**: `.nordic-card`, `.btn-primary`, `.btn-ghost`, `.input-nordic`, `.status-dot`
- **Logo**: `import { YggdrasilLogo } from "@/components/logo"`

## Routes

| Route | Description |
|-------|-------------|
| `/` | Welcome page with interactive 3D globe |
| `/bot` | Bot dashboard — metrics, processes, system info |
| `/bot/execute` | Python/shell code execution |
| `/bot/network` | 3D network visualization |
| `/msg` | Real-time messaging channels |

## Adding a Service

1. Add to `SERVICES` in `src/components/global-sidebar.tsx`
2. Create route at `src/app/[service-name]/`
3. Layout is inherited from root — no need to wrap in `ServiceLayout`
