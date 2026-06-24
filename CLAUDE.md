# ygg

Schema-aware data interchange, built **Rust-first**. The Rust crate
(`rust/`) is the single source of truth; the Python (`python/`) and JS/TS
(`js/`) packages are **thin binding layers that wrap the same Rust core** —
not independent reimplementations.

> **Clean restart.** The repo was reset to a pure-Rust foundation. Build the
> core in Rust, then expose it to Python and JS through bindings. Logic that
> matters lives in Rust exactly once.

## ⚠️ Cross-language model (IMPORTANT — global rule)

There is **one implementation**: the Rust crate. Python and JS do not
re-derive behavior — they call into the compiled Rust core and present a
language-idiomatic surface over it.

```
rust/ygg/  ── the engine. All real logic: identifiers, schema, IO, …
python/    ── PyO3/maturin bindings → import ygg's Rust core from Python.
js/        ── napi-rs (native) or wasm-bindgen (wasm) → same core from JS/TS.
```

Rules, in order of importance:

1. **Logic lives in Rust, once.** A new feature is designed and implemented
   in `rust/` first. If you find yourself writing data-transform or
   validation logic in Python or JS, it belongs in Rust — move it down and
   bind to it.
2. **Bindings only marshal + adapt.** The Python/JS layers convert types at
   the boundary (PyObject/JsValue ↔ Rust), name things idiomatically
   (`snake_case` in Python, `camelCase` in JS), and add per-language ergonomics
   (context managers, async, iterators). They contain **no business rules**.
3. **The public API is one contract.** A method added to the Rust surface is
   exposed in *both* bindings in the same change, with matching semantics and
   parallel names (Rust `Schema::cast` ↔ Python `Schema.cast` ↔ JS
   `Schema.cast`). Same inputs → same outputs across all three.
4. **Never let the surfaces drift silently.** If you can only wire up one
   binding now, leave an explicit `// PARITY:` (Rust/JS) or `# PARITY:`
   (Python) note at every relevant site describing the gap, and call it out.
   Treat a cross-language divergence as a bug.
5. **Data crosses as Arrow.** Tabular data moves over the boundary as Apache
   Arrow (zero-copy where possible), not bespoke per-language structs.

## Architecture & tooling

- **Core** — `rust/` is a Cargo workspace; the engine is the `ygg`
  crate (`rust/ygg/`). Plain Rust library, no dependencies, no Python/JS
  knowledge leaks into it. Keep the binding-facing API explicit and stable.
- **Python** — `python/`, package `ygg`. A dedicated `ygg-python` crate
  (`python/Cargo.toml`, `crate-type = ["cdylib"]`, lib `ygg_python`) builds a
  **PyO3** extension module depending on the `ygg` engine by path (aliased
  `ygg_engine` to avoid clashing with the `#[pymodule] fn ygg`); **maturin**
  (`pyproject.toml`, `[tool.maturin]`) packages it into wheels named `ygg`.
  There is **no pure-Python implementation** — Python is a binding over the
  Rust core.
- **JS/TS** — `js/`, package `@platob/ygg`. A `ygg-js` cdylib crate binds the
  `ygg` engine via **wasm-bindgen** (`crate-type = ["cdylib", "rlib"]`),
  packaged by **wasm-pack** into the npm package; the TS types are generated
  from the Rust surface. (napi-rs remains an option for native Node addons.)

## Coding style

1. **Real logic in Rust, thin glue elsewhere.** A 30-line Rust function with
   two 5-line binding shims beats the same logic copied into three languages.
2. **No logic forks across languages.** Three nearly-identical algorithms in
   Rust/Python/JS is the failure mode this layout exists to prevent.
3. **Flat call stacks.** Engine code does the work directly; avoid
   `service → helper → util` chains. Two levels max on the common path.
4. **No premature abstraction.** No traits/generics/registries until a third
   concrete user exists. Bindings stay boring on purpose.
5. **Errors are typed at the core.** Define error types in Rust; map them to
   Python exceptions and JS errors at the boundary. No ad-hoc error strings
   per language.
6. **Idiomatic at the edges only.** Rust stays Rust; Python feels like Python;
   JS feels like JS — achieved by the binding layer, never by changing core
   behavior.
7. **Delete dead code.** No commented-out blocks, no `TODO: maybe`, no unused
   exports. If it isn't called, it doesn't exist.

## Layout

```
rust/                 Cargo workspace
  Cargo.toml          workspace manifest
  ygg/                the core engine (single source of truth)
    src/lib.rs        re-exports
    src/uri.rs        Uri  — RFC 3986 component split
    src/url.rs        Url  — located URI (scheme + host + port + …)
python/               PyO3/maturin bindings → package `ygg` (PyPI)
  Cargo.toml          ygg-python cdylib crate (depends on ../rust/ygg)
  pyproject.toml      maturin build config
  src/lib.rs          #[pymodule] exposing Uri, Url
js/                   wasm-bindgen bindings → package `@platob/ygg` (npm)
  Cargo.toml          ygg-js cdylib crate (wasm-bindgen, depends on ../rust/ygg)
  src/lib.rs          #[wasm_bindgen] Uri, Url
.github/workflows/    publish-rust.yml · publish-python.yml · publish-js.yml
LICENSE               Apache-2.0
```

## Build

```bash
cd rust && cargo test                    # the core (ygg)
cd python && maturin develop             # build + install the Python binding
cd js && wasm-pack build --target nodejs --out-dir pkg --release   # JS binding
```

## Publishing

One workflow per language, named for its target:

| Language | Dir | Package | Workflow | Trigger |
|----------|-----|---------|----------|---------|
| Rust → crates.io | `rust/ygg/` | `ygg` | `publish-rust.yml` | tag `ygg-rust-v*` |
| Python → PyPI | `python/` | `ygg` | `publish-python.yml` | tag `ygg-python-v*` |
| JS/TS → npm | `js/` | `@platob/ygg` | `publish-js.yml` | tag `ygg-js-v*` |

Each release ships the **same core** — the `ygg` crate plus the bindings that
wrap it. Python ships as maturin-built wheels (compiled extension), **never** a
pure-Python package.

## Versioning (IMPORTANT — global rule)

**All three packages share one uniform version.** Rust (`ygg`), Python (`ygg`),
and JS (`@platob/ygg`) are released in lockstep at the *same* version number —
never let them drift. The current version is **`0.9.0`**.

The version lives in three manifests; bump **all** of them together:

- `rust/Cargo.toml` → `[workspace.package] version` (the `ygg` engine inherits it)
- `python/Cargo.toml` → `version` (maturin reads it; `pyproject.toml` is `dynamic`)
- `js/Cargo.toml` → `version` (wasm-pack writes it into the npm `package.json`)

When you change the version, change it in all three in the same commit, and tag
each release from the matching value (`ygg-rust-v0.9.0`, `ygg-python-v0.9.0`,
`ygg-js-v0.9.0`).
