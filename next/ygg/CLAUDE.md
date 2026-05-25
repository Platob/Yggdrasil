# Yggdrasil Frontend

Next.js 16 + React 19 + Tailwind v4. Read `node_modules/next/dist/docs/` — breaking changes from training data.

## API

- **Node proxy**: `/api/node/*` → FastAPI (configured in `next.config.ts`)
- **Client**: `import { node } from "@/lib/api"` — functions, environments, runs, DAGs
- **Server-side**: `import { nodeFetch } from "@/lib/node-client"` for route handlers
- **IDs are int64** — all resource IDs are numbers, not strings

## Routes

| Route | Page |
|-------|------|
| `/` | 3D globe welcome |
| `/node` | Network overview — node grid, closest neighbors |
| `/node/{id}` | Node detail — resource graphs, processes |
| `/node/functions` | Functions CRUD — create, edit, run |
| `/node/functions/{id}` | Function detail — code, runs, SSE logs |
| `/node/environments` | Python venv management |
| `/node/environments/{id}` | Environment detail — packages, status |
| `/node/dags` | DAG definitions — function pipelines |
| `/node/dags/{id}` | DAG detail — step flow, runs |
| `/node/network` | 3D network visualization |
| `/node/execute` | Direct code execution |
| `/msg` | Messaging channels |

## Design

- Brand: `#f26b3a` (coral) = `var(--primary)`
- Dark default, `.light` class toggles theme
- Cards: `.nordic-card` or `bg-card border border-border rounded-xl`
- Sidebar: `GlobalSidebar` in root layout, all pages inherit it
- IDs: displayed as mono numbers, URLs use numeric paths
