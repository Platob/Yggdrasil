//! URL parsing — a [`Url`] is a [`crate::Uri`] that locates a resource: it
//! must have a scheme and an authority, and the authority is decomposed into
//! optional userinfo, a host, and an optional port.

use std::fmt;
use std::str::FromStr;

use crate::uri::{Uri, UriError};

/// Error returned when a string cannot be parsed as a [`Url`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UrlError {
    /// The underlying URI parse failed (e.g. empty input).
    Uri(UriError),
    /// No scheme was present (`example.com/path` has none).
    MissingScheme,
    /// No authority/host was present (`mailto:a@b.com` has none).
    MissingHost,
    /// The port was present but not a valid `u16`.
    InvalidPort(String),
}

impl fmt::Display for UrlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UrlError::Uri(e) => write!(f, "{e}"),
            UrlError::MissingScheme => write!(f, "URL has no scheme"),
            UrlError::MissingHost => write!(f, "URL has no host"),
            UrlError::InvalidPort(p) => write!(f, "invalid port: {p:?}"),
        }
    }
}

impl std::error::Error for UrlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            UrlError::Uri(e) => Some(e),
            _ => None,
        }
    }
}

impl From<UriError> for UrlError {
    fn from(e: UriError) -> UrlError {
        UrlError::Uri(e)
    }
}

/// A parsed URL: a located URI with its authority decomposed.
///
/// `host` keeps the bracketed form for IPv6 literals (e.g. `[::1]`) so that
/// [`Url::to_string`] round-trips.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    /// Scheme without the trailing `:` (lowercased), e.g. `https`.
    pub scheme: String,
    /// Username from the userinfo, if present.
    pub username: Option<String>,
    /// Password from the userinfo (the part after `:`), if present.
    pub password: Option<String>,
    /// Host — a registered name, IPv4, or bracketed IPv6 literal.
    pub host: String,
    /// Explicit port, if the authority carried one.
    pub port: Option<u16>,
    /// Path component; may be empty.
    pub path: String,
    /// Query without the leading `?`.
    pub query: Option<String>,
    /// Fragment without the leading `#`.
    pub fragment: Option<String>,
}

impl Url {
    /// Parse a string into a [`Url`].
    ///
    /// Fails if the string lacks a scheme or a host, or carries a malformed
    /// port — those are what separates a locator from a generic [`Uri`].
    pub fn parse(input: &str) -> Result<Url, UrlError> {
        let uri = Uri::parse(input)?;
        let scheme = uri.scheme.ok_or(UrlError::MissingScheme)?;
        let authority = uri.authority.ok_or(UrlError::MissingHost)?;

        // authority = [ userinfo "@" ] host [ ":" port ]
        let (userinfo, hostport) = match authority.rsplit_once('@') {
            Some((user, hp)) => (Some(user), hp),
            None => (None, authority.as_str()),
        };
        let (username, password) = match userinfo {
            Some(ui) => match ui.split_once(':') {
                Some((u, p)) => (Some(u.to_string()), Some(p.to_string())),
                None => (Some(ui.to_string()), None),
            },
            None => (None, None),
        };

        // Split host from port. IPv6 literals are bracketed, so only look for
        // the port-colon after the closing `]`.
        let (host, port) = if hostport.starts_with('[') {
            let close = hostport.find(']').ok_or(UrlError::MissingHost)?;
            let host = hostport[..=close].to_string();
            let port = parse_port(&hostport[close + 1..])?;
            (host, port)
        } else {
            match hostport.rsplit_once(':') {
                Some((h, p)) => (h.to_string(), parse_port(&format!(":{p}"))?),
                None => (hostport.to_string(), None),
            }
        };

        if host.is_empty() || host == "[]" {
            return Err(UrlError::MissingHost);
        }

        Ok(Url {
            scheme,
            username,
            password,
            host,
            port,
            path: uri.path,
            query: uri.query,
            fragment: uri.fragment,
        })
    }
}

/// Parse the trailing `":<port>"` (or empty string) into an optional port.
fn parse_port(s: &str) -> Result<Option<u16>, UrlError> {
    match s.strip_prefix(':') {
        None if s.is_empty() => Ok(None),
        None => Err(UrlError::InvalidPort(s.to_string())),
        Some(p) => p
            .parse::<u16>()
            .map(Some)
            .map_err(|_| UrlError::InvalidPort(p.to_string())),
    }
}

impl fmt::Display for Url {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}://", self.scheme)?;
        if let Some(user) = &self.username {
            f.write_str(user)?;
            if let Some(pass) = &self.password {
                write!(f, ":{pass}")?;
            }
            f.write_str("@")?;
        }
        f.write_str(&self.host)?;
        if let Some(port) = self.port {
            write!(f, ":{port}")?;
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

impl FromStr for Url {
    type Err = UrlError;
    fn from_str(s: &str) -> Result<Url, UrlError> {
        Url::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_url() {
        let url = Url::parse("https://user:pw@example.com:8443/a/b?q=1#f").unwrap();
        assert_eq!(url.scheme, "https");
        assert_eq!(url.username.as_deref(), Some("user"));
        assert_eq!(url.password.as_deref(), Some("pw"));
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, Some(8443));
        assert_eq!(url.path, "/a/b");
        assert_eq!(url.query.as_deref(), Some("q=1"));
        assert_eq!(url.fragment.as_deref(), Some("f"));
    }

    #[test]
    fn parses_minimal_url() {
        let url = Url::parse("https://example.com").unwrap();
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, None);
        assert_eq!(url.username, None);
        assert_eq!(url.path, "");
    }

    #[test]
    fn username_without_password() {
        let url = Url::parse("ftp://anon@host/pub").unwrap();
        assert_eq!(url.username.as_deref(), Some("anon"));
        assert_eq!(url.password, None);
    }

    #[test]
    fn ipv6_host_with_port() {
        let url = Url::parse("http://[::1]:8080/x").unwrap();
        assert_eq!(url.host, "[::1]");
        assert_eq!(url.port, Some(8080));
        assert_eq!(url.path, "/x");
    }

    #[test]
    fn rejects_missing_scheme() {
        assert_eq!(Url::parse("example.com/path"), Err(UrlError::MissingScheme));
    }

    #[test]
    fn rejects_missing_host() {
        assert_eq!(Url::parse("mailto:a@b.com"), Err(UrlError::MissingHost));
    }

    #[test]
    fn rejects_bad_port() {
        assert_eq!(
            Url::parse("http://host:notaport/"),
            Err(UrlError::InvalidPort("notaport".to_string()))
        );
    }

    #[test]
    fn round_trips() {
        for s in [
            "https://user:pw@example.com:8443/a/b?q=1#f",
            "https://example.com",
            "ftp://anon@host/pub",
            "http://[::1]:8080/x",
        ] {
            assert_eq!(Url::parse(s).unwrap().to_string(), s, "round-trip {s}");
        }
    }
}
