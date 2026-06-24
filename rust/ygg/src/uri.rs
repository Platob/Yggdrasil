//! Generic URI parsing, following the component grammar of RFC 3986:
//!
//! ```text
//! URI = scheme ":" ["//" authority] path ["?" query] ["#" fragment]
//! ```
//!
//! [`Uri::parse`] splits a string into those components without validating
//! or decoding them — it is a structural split, the foundation [`crate::Url`]
//! builds on.

use std::fmt;
use std::str::FromStr;

/// Error returned when a string cannot be parsed as a [`Uri`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UriError {
    /// The input was empty.
    Empty,
}

impl fmt::Display for UriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UriError::Empty => write!(f, "empty URI"),
        }
    }
}

impl std::error::Error for UriError {}

/// A URI split into its RFC 3986 components.
///
/// Components are stored exactly as they appear in the source (no
/// percent-decoding); `scheme` is lowercased since schemes are
/// case-insensitive. [`Uri::to_string`] reconstructs the URI.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Uri {
    /// Scheme without the trailing `:` (lowercased), e.g. `https`.
    pub scheme: Option<String>,
    /// Authority between `//` and the path, e.g. `user@host:443`.
    pub authority: Option<String>,
    /// Path component; may be empty.
    pub path: String,
    /// Query without the leading `?`.
    pub query: Option<String>,
    /// Fragment without the leading `#`.
    pub fragment: Option<String>,
}

impl Uri {
    /// Parse a string into its URI components.
    ///
    /// The only failure is an empty input; every other string parses into
    /// some arrangement of components (a bare `"foo"` is a path-only URI).
    pub fn parse(input: &str) -> Result<Uri, UriError> {
        if input.is_empty() {
            return Err(UriError::Empty);
        }

        let mut rest = input;

        // Peel the fragment, then the query, off the right.
        let fragment = rest.find('#').map(|i| {
            let frag = rest[i + 1..].to_string();
            rest = &rest[..i];
            frag
        });
        let query = rest.find('?').map(|i| {
            let q = rest[i + 1..].to_string();
            rest = &rest[..i];
            q
        });

        // Scheme: a leading `name:` where `name` is a valid scheme and the
        // `:` precedes the first `/` (otherwise the `:` belongs to the path,
        // e.g. a relative reference like `foo/bar:baz`).
        let scheme = rest.find(':').and_then(|i| {
            let candidate = &rest[..i];
            let colon_before_slash = rest.find('/').map_or(true, |s| i < s);
            if colon_before_slash && is_valid_scheme(candidate) {
                rest = &rest[i + 1..];
                Some(candidate.to_ascii_lowercase())
            } else {
                None
            }
        });

        // Authority is present iff the hier-part starts with `//`; it runs
        // up to the next `/` (the query/fragment are already removed).
        let (authority, path) = match rest.strip_prefix("//") {
            Some(after) => {
                let end = after.find('/').unwrap_or(after.len());
                (Some(after[..end].to_string()), after[end..].to_string())
            }
            None => (None, rest.to_string()),
        };

        Ok(Uri { scheme, authority, path, query, fragment })
    }
}

/// A scheme is `ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )`.
fn is_valid_scheme(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
}

impl fmt::Display for Uri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(scheme) = &self.scheme {
            write!(f, "{scheme}:")?;
        }
        if let Some(authority) = &self.authority {
            write!(f, "//{authority}")?;
        }
        f.write_str(&self.path)?;
        if let Some(query) = &self.query {
            write!(f, "?{query}")?;
        }
        if let Some(fragment) = &self.fragment {
            write!(f, "#{fragment}")?;
        }
        Ok(())
    }
}

impl FromStr for Uri {
    type Err = UriError;
    fn from_str(s: &str) -> Result<Uri, UriError> {
        Uri::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_uri() {
        let uri = Uri::parse("https://user@example.com:443/a/b?q=1#frag").unwrap();
        assert_eq!(uri.scheme.as_deref(), Some("https"));
        assert_eq!(uri.authority.as_deref(), Some("user@example.com:443"));
        assert_eq!(uri.path, "/a/b");
        assert_eq!(uri.query.as_deref(), Some("q=1"));
        assert_eq!(uri.fragment.as_deref(), Some("frag"));
    }

    #[test]
    fn lowercases_scheme() {
        assert_eq!(Uri::parse("HTTPS://x").unwrap().scheme.as_deref(), Some("https"));
    }

    #[test]
    fn scheme_only_when_colon_precedes_slash() {
        // `urn:isbn:0451450523` — authority absent, path carries the rest.
        let uri = Uri::parse("urn:isbn:0451450523").unwrap();
        assert_eq!(uri.scheme.as_deref(), Some("urn"));
        assert_eq!(uri.authority, None);
        assert_eq!(uri.path, "isbn:0451450523");
    }

    #[test]
    fn relative_reference_has_no_scheme() {
        let uri = Uri::parse("foo/bar:baz").unwrap();
        assert_eq!(uri.scheme, None);
        assert_eq!(uri.path, "foo/bar:baz");
    }

    #[test]
    fn mailto_has_no_authority() {
        let uri = Uri::parse("mailto:a@b.com").unwrap();
        assert_eq!(uri.scheme.as_deref(), Some("mailto"));
        assert_eq!(uri.authority, None);
        assert_eq!(uri.path, "a@b.com");
    }

    #[test]
    fn round_trips() {
        for s in [
            "https://example.com/p?q#f",
            "mailto:a@b.com",
            "urn:isbn:0451450523",
            "foo/bar",
            "//host/path",
            "https://example.com",
        ] {
            assert_eq!(Uri::parse(s).unwrap().to_string(), s, "round-trip {s}");
        }
    }

    #[test]
    fn empty_is_error() {
        assert_eq!(Uri::parse(""), Err(UriError::Empty));
    }
}
