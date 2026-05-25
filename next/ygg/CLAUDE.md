# Yggdrasil Frontend

Next.js 16 + React 19 + Tailwind v4. Read `node_modules/next/dist/docs/` — breaking changes from training data.

## API

- **Bot proxy**: `/api/bot/*` → FastAPI (configured in `next.config.ts`)
- **Client**: `import { bot } from "@/lib/api"` — `bot.getNodeInfo()`, `bot.getPeers()`, `bot.executePython()`, etc.
- **Server-side**: `import { botFetch } from "@/lib/bot-client"` for route handlers

## Routes

| Route | Page |
|-------|------|
| `/` | 3D globe welcome |
| `/bot` | Network overview — node grid, closest neighbors |
| `/bot/[id]` | Node detail — resource graphs, processes |
| `/bot/network` | 3D network visualization |
| `/bot/execute` | Code execution |
| `/msg` | Messaging channels |

## Design

- Brand: `#f26b3a` (coral) = `var(--primary)`
- Dark default, `.light` class toggles theme
- Cards: `.nordic-card` or `bg-card border border-border rounded-xl`
- Sidebar: `GlobalSidebar` in root layout, all pages inherit it
