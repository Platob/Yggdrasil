# Yggdrasil Next.js Frontend - AI Agent Instructions

## Design System: Nordic Mythic Theme

This frontend uses a **mythic Nordic aesthetic** inspired by the Yggdrasil World Tree.

### Brand Colors (Black, White, Grey, Red, Orange, Coral)
- **Primary**: `#f26b3a` (Coral/Orange) - from the Yggdrasil logo
- **Secondary**: `#dc2626` (Deep Red) - for emphasis
- **Accent**: `#fb923c` (Warm Coral) - highlights
- **Background**: `#050507` (Pure black void)
- **Foreground**: `#ffffff` (Pure white)
- **Muted**: `#525252` / `#737373` (Grey tones)

### Design Principles
1. **Dark, atmospheric** - deep blacks and charcoals evoke Norse mythology
2. **Orange/coral accents** - the World Tree color, used sparingly for emphasis
3. **Clean and functional** - modern dashboard UI, not decorative fantasy
4. **Subtle glows** - primary color has soft glow effects for status/emphasis

### Component Patterns
- Use `.nordic-card` for card containers
- Use `.btn-primary` for primary actions (orange gradient with glow)
- Use `.btn-ghost` for secondary actions
- Use `.input-nordic` for form inputs
- Use `.status-dot` with `.online/.offline/.pending` for status indicators

### Logo
- Import from `@/components/logo` 
- `<YggdrasilLogo />` - Static SVG tree icon
- `<AnimatedYggdrasilTree />` - Large animated tree with draw-on effects
- `<YggdrasilBrand />` - logo + wordmark combo

### CSS Classes in `globals.css`
- `.nordic-card` - Card with border, rounded corners, hover state
- `.constellation-card` - Service card with hover glow and gradient
- `.btn-primary` - Primary button with gradient and glow
- `.btn-secondary` - Border button with hover fill
- `.btn-ghost` - Ghost button for secondary actions  
- `.input-nordic` - Styled input with focus ring
- `.code-block` - Code/pre block styling
- `.glow-intense` - Intense animated glow for hero elements
- `.pulse-glow` - Subtle pulsing glow for background orbs
- `.float` - Gentle floating animation
- `.twinkle` - Star twinkling animation
- `.constellation-line` - Animated SVG line for connections
- `.gradient-text` - Multi-color gradient text
- `.star-field`, `.star` - Star background elements

## Architecture Overview

This is a **hybrid architecture** with two backends:

1. **FastAPI Bot Backend** (`BOT_API_URL`, default: `http://127.0.0.1:8100`)
   - Runs on the Yggdrasil bot instance
   - Handles: Python execution, shell commands, messaging, remote function calls
   - Real-time bot state, node info, channel management
   - **Owner maintains this** - do not suggest changes to FastAPI code

2. **Next.js API Routes** (`/api/...` in `src/app/api/`)
   - Handles: Caching, aggregation, auth, frontend-optimized endpoints
   - SSR data fetching, session management, preferences
   - **You (v0) maintain this** - implement features here when optimal

## When to Use Each Backend

### Use FastAPI (Bot) Backend For:
- Python/shell code execution (`/api/bot/python`, `/api/bot/cmd`)
- Real-time messaging and long-polling (`/api/bot/messenger/*`)
- Node info and bot state (`/api/bot/hello`)
- Remote function registry and calls (`/api/bot/call/*`)
- Anything requiring direct bot process access

### Use Next.js API Routes For:
- **Caching/Aggregation**: Combine multiple bot API calls into one
- **Auth & Sessions**: User authentication, API key management
- **Preferences**: User settings, theme, layout preferences
- **Data Transformation**: Reshape bot responses for frontend needs
- **Rate Limiting**: Protect bot from excessive requests
- **Webhooks**: External service integrations
- **Static Data**: Feature flags, config, UI metadata

## API Structure

```
/api/
├── bot/           → Proxied to FastAPI (rewrites in next.config.ts)
│   ├── hello
│   ├── python
│   ├── cmd
│   ├── messenger/
│   └── call/
│
└── [next routes]  → Next.js API handlers
    ├── health/    → Health check, version info
    ├── config/    → Frontend config, feature flags
    ├── cache/     → Cached bot data (node info, registry)
    └── prefs/     → User preferences (future)
```

## Environment Variables

```env
# Bot backend URL (default: http://127.0.0.1:8100)
BOT_API_URL=http://127.0.0.1:8100

# Optional: API timeout in ms
BOT_API_TIMEOUT=30000
```

## Code Patterns

### Calling Bot API from Next.js Route Handlers

```typescript
// src/lib/bot-client.ts
import { botFetch } from "@/lib/bot-client";

// In a route handler:
export async function GET() {
  const data = await botFetch("/api/hello");
  return Response.json(data);
}
```

### Frontend API Client Usage

```typescript
// For bot-proxied endpoints (real-time, execution)
import { bot } from "@/lib/api";
await bot.executePython(code);
await bot.pollMessages(channel, afterId);

// For Next.js endpoints (cached, aggregated)
import { api } from "@/lib/api";
await api.getConfig();
await api.getCachedNodeInfo();
```

## Translation Guidelines: FastAPI → Next.js

When the owner asks to "translate" or "optimize" a FastAPI endpoint:

1. **Identify if it benefits from Next.js**:
   - Does it need caching? → Next.js with `unstable_cache` or `revalidateTag`
   - Does it aggregate data? → Next.js to reduce client requests
   - Is it read-only + cacheable? → Next.js
   - Does it need bot process access? → Keep in FastAPI

2. **Create Next.js route** in `src/app/api/[feature]/route.ts`

3. **Update frontend API client** in `src/lib/api.ts`

4. **Document the change** - note what was moved and why

## File Structure

```
src/
├── app/
│   ├── api/           # Next.js API routes
│   │   ├── health/
│   │   ├── config/
│   │   └── cache/
│   ├── bot/           # Bot control dashboard (with sidebar)
│   │   ├── layout.tsx # Sidebar wrapper
│   │   ├── page.tsx   # Dashboard
│   │   ├── execute/   # Code execution
│   │   └── chat/      # Messaging
│   ├── layout.tsx     # Root layout (minimal)
│   ├── page.tsx       # Welcome/landing page (animated tree, constellations)
│   └── globals.css
├── components/
│   ├── logo.tsx       # AnimatedYggdrasilTree, YggdrasilLogo, YggdrasilBrand
│   └── sidebar.tsx    # Bot dashboard sidebar
└── lib/
    ├── api.ts         # Frontend API client (both backends)
    ├── bot-client.ts  # Server-side bot API client
    └── utils.ts       # Shared utilities
```

## Route Structure

| Route | Description |
|-------|-------------|
| `/` | Welcome landing page with animated tree and service constellation |
| `/bot` | Bot dashboard - node overview, stats |
| `/bot/execute` | Python/shell code execution |
| `/bot/chat` | Real-time messaging channels |
| `/trading` | Trading service (coming soon) |
| `/data` | Data streams service (coming soon) |
| `/agents` | AI Agents service (coming soon) |

## Services Constellation

The welcome page displays services as a "constellation" - interconnected nodes representing different Yggdrasil capabilities:

1. **Bot Control** (`/bot`) - Active, the main dashboard
2. **Trading** - Coming soon, market data & execution
3. **Data Streams** - Coming soon, real-time feeds & analytics
4. **AI Agents** - Coming soon, autonomous workflows

To add a new service:
1. Add to `SERVICES` array in `src/app/page.tsx`
2. Add connections in `CONNECTIONS` array
3. Create route at `src/app/[service-name]/`
4. Set `comingSoon: false` when ready

## Feature Implementation Checklist

When adding a new feature:

- [ ] Determine which backend handles it (see "When to Use" above)
- [ ] If Next.js: create route in `src/app/api/`
- [ ] If Bot proxy: ensure rewrite exists in `next.config.ts`
- [ ] Update `src/lib/api.ts` with typed client function
- [ ] Add TypeScript interfaces for request/response
- [ ] Handle errors with proper status codes
- [ ] Add loading states in UI

## Current Bot API Endpoints (FastAPI)

These are proxied via `/api/bot/*`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/hello` | GET | Node info, uptime, version |
| `/api/python` | POST | Execute Python code |
| `/api/cmd` | POST | Execute shell command |
| `/api/messenger` | POST | Send message |
| `/api/messenger/channels` | GET/POST | List/create channels |
| `/api/messenger/channels/{name}/messages` | GET | Get channel messages |
| `/api/messenger/channels/{name}/poll` | GET | Long-poll for new messages |
| `/api/call/registry` | GET | List registered @remote functions |
| `/api/call/{function}` | POST | Call a remote function |
