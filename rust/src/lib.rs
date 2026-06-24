//! ygg — schema-aware data interchange, rebuilt in pure Rust.
//!
//! This is a fresh start. The crate is intentionally empty beyond a
//! version probe; build the project up from here.

/// The crate version, sourced from `Cargo.toml` at compile time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_exposed() {
        assert!(!VERSION.is_empty());
    }
}
