# vercel/

Home for Next.js frontends and Vercel-hosted apps in the Yggdrasil monorepo.

This folder centralizes any web UI that ships to Vercel — dashboards, docs
companions, internal tools, demo apps — so they live alongside the Python core
(`python/`), Rust kernels (`rust/`), and Power Query connectors (`powerquery/`)
without polluting the repo root.

## Layout

Each app lives in its own subfolder with its own `package.json` and Vercel
project, e.g.:

```
vercel/
  <app-name>/
    package.json
    next.config.mjs
    app/ or pages/
    ...
```

No workspace tooling is enforced at this level yet — apps are independent and
can pick their own package manager (pnpm recommended) and Next.js version.

## Conventions

- **One Vercel project per subfolder.** Set the project's Root Directory to
  `vercel/<app-name>` in the Vercel dashboard.
- **Next.js App Router by default** unless an app has a specific reason to
  stay on Pages Router.
- **Talk to the Python service over HTTP.** Frontends call the FastAPI service
  exposed from `python/` (same surface the Power Query connector uses) — they
  do not import Python or Rust code directly.
- **No secrets in the repo.** Use Vercel project env vars; mirror required
  keys in each app's `README.md`.

## Adding a new app

1. `cd vercel && pnpm create next-app@latest <app-name>`
2. Commit on a feature branch (never push directly to `main`).
3. In Vercel, create a new project pointing at this repo with Root Directory
   `vercel/<app-name>`.
4. Document required env vars and the upstream FastAPI endpoint in the app's
   own `README.md`.
