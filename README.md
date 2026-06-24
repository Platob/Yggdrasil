# Yggdrasil

Schema-aware data interchange, built **Rust-first**. The engine lives in the
Rust crate `ygg`; the Python and JS/TS packages are thin bindings that wrap
the same core — not separate reimplementations.

> Reset to a clean Rust foundation. Logic that matters lives in Rust once;
> the other languages extend outward from it. See [CLAUDE.md](CLAUDE.md).

## Layout

```
rust/                Cargo workspace
  ygg/               the engine — single source of truth (Uri, Url, …)
python/              PyO3/maturin bindings → package `ygg` (PyPI)
js/                  napi-rs/wasm bindings → package `@platob/yggdrasil` (npm)
```

## Build

```bash
# Rust core
cd rust && cargo test

# Python bindings (wraps the Rust core via maturin)
cd python && pip install maturin && maturin develop
python -c "from ygg import Url; print(Url.parse('https://example.com:443/p'))"
```

## Publishing

One workflow per language, each named for its target:

| Language | Directory | Package | Workflow | Trigger |
|----------|-----------|---------|----------|---------|
| Rust → crates.io | `rust/ygg/` | `ygg` | `publish-rust.yml` | tag `ygg-rust-v*` |
| Python → PyPI | `python/` | `ygg` | `publish-python.yml` | tag `ygg-python-v*` |
| JS/TS → npm | `js/` | `@platob/yggdrasil` | `publish-js.yml` | tag `ygg-js-v*` |

## License

Apache-2.0 — see [LICENSE](LICENSE).
