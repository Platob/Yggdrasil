# Yggdrasil Frontend

Next.js 16 + React 19 + Tailwind v4. Read `node_modules/next/dist/docs/` — breaking changes from training data.

## API

- **Node proxy**: `/api/node/*` → FastAPI (configured in `next.config.ts`)
- **Client**: `import { node } from "@/lib/api"` — `node.getNodeInfo()`, `node.getPeers()`, `node.executePython()`, etc.
- **Server-side**: `import { nodeFetch } from "@/lib/node-client"` for route handlers

## Routes

| Route | Page |
|-------|------|
| `/` | 3D globe welcome |
| `/node` | Network overview — node grid, closest neighbors |
| `/node/[id]` | Node detail — resource graphs, processes |
| `/node/network` | 3D network visualization |
| `/node/execute` | Code execution |
| `/msg` | Messaging channels |

## Design

- Brand: `#f26b3a` (coral) = `var(--primary)`
- Dark default, `.light` class toggles theme
- Cards: `.nordic-card` or `bg-card border border-border rounded-xl`
- Sidebar: `GlobalSidebar` in root layout, all pages inherit it
