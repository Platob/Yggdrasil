# Yggdrasil

Schema-aware data interchange, rebuilt from scratch in **pure Rust**.

> This repository was reset to a clean Rust foundation. The previous
> Python / JS-TS / Databricks implementation was removed. Build up from here.

## Layout

Each language implementation lives in its own top-level directory, kept
cleanly separated:

```
rust/      Rust crate (reference implementation, started here)
python/    Python package (to be (re)added)
js/        JS/TS package (to be (re)added)
```

## Build

```bash
cd rust
cargo build
cargo test
```

## Publishing

Releases are driven by GitHub Actions, one workflow per language:

| Language | Directory | Crate/Package | Workflow | Trigger |
|----------|-----------|---------------|----------|---------|
| Rust (crates.io) | `rust/` | `ygg` | `publish-rust.yml` | tag `ygg-rust-v*` |
| Python (PyPI) | `python/` | `ygg` | `publish-python.yml` | tag `v*` / `python/**` push |
| JS/TS (npm) | `js/` | `@platob/yggdrasil` | `publish-js.yml` | tag `yggdrasil-js-v*` |

## License

Apache-2.0 — see [LICENSE](LICENSE).
