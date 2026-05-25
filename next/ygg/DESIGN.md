# YGG Frontend Design Guidelines

## Color Palette

| Token | Hex | Usage |
|---|---|---|
| `background` | `#0a0a0a` | Page background |
| `foreground` | `#e5e5e5` | Primary text |
| `muted` | `#737373` | Secondary text, labels |
| `border` | `#262626` | Card borders, dividers |
| `card` | `#141414` | Card backgrounds |
| `card-hover` | `#1a1a1a` | Card hover state |
| `gold` | `#d4a843` | Primary accent — active states, CTAs, highlights |
| `gold-dim` | `#a68632` | Gold secondary — labels, dimmed accent |
| `accent-red` | `#dc2626` | Errors, destructive actions |
| `accent-red-dim` | `#991b1b` | Error backgrounds |

## Typography

- **Sans:** Geist Sans (Inter fallback)
- **Mono:** Geist Mono (JetBrains Mono fallback)
- Use mono for code, IDs, timestamps, technical values
- Use sans for headings, labels, body text

## Layout

- Fixed 56px sidebar on the left (black with border)
- Main content area with 24px padding
- Max content width: 4xl (896px) for readability
- Cards: rounded-xl, bg-card, border border-border

## Components

- **Stat cards:** uppercase tracking-widest 10px label, mono value below
- **Code editor:** textarea with mono font, bg-background, gold focus ring
- **Chat messages:** grouped by sender, colored usernames (6-color hash), dim timestamps
- **Buttons:** bg-gold/10, text-gold, border-gold/20, hover:bg-gold/20
- **Tabs:** bottom-border gold active indicator

## Animations

- `animate-in`: fadeIn 0.3s with translateY(8px)
- `pulse-gold`: opacity pulse for loading states
- `spin-slow`: 1.5s rotation for spinners

## v0.dev Integration

This project is designed for v0.dev. When generating new components:

1. Use the color tokens above (not arbitrary Tailwind colors)
2. Keep the dark theme — no light mode toggle
3. Use Geist font family
4. Follow the card + border pattern for containers
5. Gold is the primary accent, red is for errors only
6. Animations should be subtle and purposeful

## API Endpoints

The frontend proxies all `/api/*` requests to the YGGBOT backend
(default port 8100). See `next.config.ts` for the rewrite rule.

Key endpoints used by the UI:
- `GET /api/hello` — node info for dashboard
- `GET /api/call/registry` — registered functions
- `POST /api/python` — execute Python code
- `POST /api/cmd` — execute shell commands
- `POST /api/messenger` — send chat message
- `GET /api/messenger/channels` — list channels
- `GET /api/messenger/channels/{name}/messages` — fetch messages
- `GET /api/messenger/channels/{name}/poll` — long-poll for real-time
