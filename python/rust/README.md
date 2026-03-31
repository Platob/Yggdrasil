# Rust acceleration (experimental)

This folder contains optional Rust extensions for hot-path operations used by
`yggdrasil`.

## Why

- Keep Python APIs stable while accelerating tight loops.
- Reuse Arrow-compatible memory ideas for future schema and metadata kernels.
- Allow graceful fallback to pure Python when native builds are unavailable.

## Current module

- `yggdrasil_rust`: exposes `utf8_len(values)` to count Unicode characters.

## Build locally

From `python/rust/yggdrasil_rust`:

```bash
maturin develop --release
```

After building, Python imports the extension automatically through
`yggdrasil.rust_accel`.
