# Yggdrasil

Schema-aware data interchange, rebuilt from scratch in **pure Rust**.

> This repository was reset to a clean Rust foundation. The previous
> Python / JS-TS / Databricks implementation was removed. Build up from here.

## Build

```bash
cargo build
cargo test
```

## Publishing

Releases are driven by GitHub Actions:

| Language | Workflow | Trigger |
|----------|----------|---------|
| Rust (crates.io) | `.github/workflows/publish-rust.yml` | tag `yggdrasil-rust-v*` |
| Python (PyPI) | `.github/workflows/publish.yml` | tag `v*` / `python/**` push |
| JS/TS (npm) | `.github/workflows/publish-yggdrasil-npm.yml` | tag `yggdrasil-js-v*` |

## License

Apache-2.0 — see [LICENSE](LICENSE).
