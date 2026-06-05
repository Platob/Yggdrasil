# Frontend Enhancement Proposal — surfacing the new Saga capabilities

_Static review (no browser available in this environment); evidence is file:line._

## Why

Recent backend work added three capabilities that the Next.js UI does **not**
expose yet:

1. **Saga mounts** — named aliases over a base path/URL (Databricks volume, S3,
   `npfs://` node, node folder, **or a live-DB connection URI**).
   `GET/POST /api/v2/saga/mount`, `GET /api/v2/saga/mount/{alias}/ls`
   (entries flag `is_tabular`); mounts also ride in `GET /api/v2/fs/nodes`
   under a new `mounts` key.
2. **Database sources** — `SELECT … FROM 'alias/schema.table'` over postgres /
   mysql / sqlite / mssql via a `database`-kind mount.
3. **Finance risk metrics** — `/api/v2/analysis/finance` now returns `ema[]`,
   `drawdown[]`, and a `metrics{}` object (total_return, cagr, ann_return,
   ann_volatility, sharpe, sortino, max_drawdown, calmar).

The headline finding: **`finance()` already exists in the API client
(`nextjs/src/lib/api.ts:581`) but is never called anywhere**, and mounts have
zero UI representation. The plumbing is closer than expected; this is mostly
wiring + small components.

## Current state (evidence)

| Area | File:line | State |
|------|-----------|-------|
| Saga catalog tree | `app/saga/page.tsx:412–476` | Catalog→Schema→Table only; no mounts |
| SQL editor + Run | `app/saga/page.tsx:577–618` | dialect/catalog/schema pickers exist |
| Double-click → insert ref | `app/saga/page.tsx:452` | reusable for mount entries |
| Files roots | `app/files/page.tsx:537–584` | node→folders→files; ignores `mounts` key |
| Register-in-Saga action | `app/files/page.tsx:446`, `RegisterSagaModal.tsx` | exists; forkable for mounts |
| Chart | `components/Chart.tsx:1–120` | bar/line/area/candle + band + one overlay |
| Analyze panel | `components/TabularModal.tsx:532–717` | pivot/series/candles; no finance tab |
| `finance()` client | `lib/api.ts:581–591` | **defined, never called** |
| `FinanceResult` type | `lib/types.ts` | missing the new fields |
| `FsNodeRoot` type | `lib/api.ts:315–324` | missing `mounts` field |

## Proposed enhancements (additive, ordered by ROI)

### Phase 1 — highest ROI, low risk

**1. Mounts in the Files tree** · `app/files/page.tsx`, `lib/api.ts`
- Add `mounts?: {alias,target,kind,read_only,comment}[]` to `FsNodeRoot` and
  read the existing `mounts` key from `getFsNodes()`.
- Render a collapsible "Mounts" section per node (before the fs tree). Each
  mount expands via `GET /api/v2/saga/mount/{alias}/ls`; `is_tabular` entries
  open the existing `TabularDisplay`. ~40 lines, reuses the modal.

**2. Finance metrics panel** · new `components/FinancePanel.tsx` + `TabularModal.tsx`
- Add a 4th analyze kind `"finance"`; call the already-present `finance()`.
- Render `metrics{}` as a compact KPI grid (Sharpe, Sortino, CAGR, ann vol,
  max drawdown, Calmar, total return) + a `cum_return` sparkline.
- Mirror the response in a `FinanceResult` type in `lib/types.ts`. ~120 lines.

### Phase 2 — polish

**3. Mounts in the Saga tree + click-to-query** · `app/saga/page.tsx`
- Fetch `GET /api/v2/saga/mount`; render a "Mounts" section above catalogs.
- Click a tabular entry / DB table → pre-fill the SQL editor with
  `SELECT * FROM 'alias/sub' LIMIT 100` (reuse `insertRef`/`onRun` at :452/:291).
  ~80 lines, mirrors the catalog tree.

**4. EMA + drawdown overlays** · `components/Chart.tsx`, `TabularModal.tsx`
- Accept optional `emaLine[]` / `drawdownLine[]`; draw as secondary polylines
  (amber EMA on the price axis, rose drawdown on a small lower panel).
- Feed from `finance().ema` / `.drawdown`. ~30 lines in Chart.

### Phase 3 — advanced, optional

**5. Mount registration form** · fork `RegisterSagaModal.tsx` → `RegisterMountModal.tsx`
- Fields: alias, target, kind (auto-sniffed from target), read-only. POST
  `/api/v2/saga/mount`. Trigger from a Files context action. ~80 lines.

**6. Database-mount connection wizard** · new `components/MountWizard.tsx`
- URI input + dialect hint + "list tables" preview (calls `/mount/{alias}/ls`).
  POST a `database`-kind mount. ~100 lines.

## Notes

- Every change is **additive** — no edits to existing tree/chart branches.
- Phases 1–2 cover the bulk of the value (~270 lines) and unblock the demo:
  browse a mount, query a DB table, and read Sharpe/drawdown off a price file.
- A live screenshot pass should follow once a node + `next dev` can run together
  with seeded data (not possible in the current sandbox: no browser, and the
  dev server needs a backing node).
