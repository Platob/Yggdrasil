//! ygg — the Rust engine (single source of truth).
//!
//! This crate holds the real logic; the Python (`python/`) and JS (`js/`)
//! packages are thin bindings that wrap it. Keep the public surface here
//! explicit and stable — it is the cross-language contract.
//!
//! Today it provides the base identifier types:
//!
//! - [`Uri`] — a generic RFC 3986 identifier split into its components.
//! - [`Url`] — a locator: a URI with a scheme and host, with the authority
//!   decomposed into userinfo / host / port.

mod uri;
mod url;

pub use uri::{Uri, UriError};
pub use url::{Url, UrlError};

/// The crate version, sourced from `Cargo.toml` at compile time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
