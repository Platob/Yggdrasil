# Yggdrasil Next.js Frontend - Agent Instructions

This is the Next.js frontend for the Yggdrasil distributed bot framework.

**For detailed architecture and implementation guidelines, see [CLAUDE.md](./CLAUDE.md).**

## Quick Reference

- **Bot API**: Proxied via `/api/bot/*` → FastAPI at `BOT_API_URL`
- **Next.js API**: Local routes at `/api/*` (health, config, cache)
- **Frontend**: React 19 + Next.js 16 + Tailwind v4

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Full architecture docs and AI agent instructions |
| `src/lib/api.ts` | Frontend API client (`bot.*` and `api.*` namespaces) |
| `src/lib/bot-client.ts` | Server-side bot API client for route handlers |
| `next.config.ts` | Bot API proxy rewrites |
