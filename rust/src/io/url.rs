/// URL parsing, normalization, and percent-encoding kernels.
///
/// Mirrors the helpers in `yggdrasil/io/url.py`:
///
/// * `parse_url`         — `urlsplit` + `_parse_netloc` + entity fix-up
///                         + Windows drive-letter fix-up + the
///                         "no //" authority recovery, returning a tuple
///                         `(scheme, userinfo, host, port, path, query, fragment)`.
/// * `normalize_components` — `_normalize_components` (lowercase scheme/host,
///                            strip trailing dot, drop default port,
///                            sorted-query, fragment lstrip).
/// * `encode_path` / `encode_query` / `encode_fragment` /
///   `encode_userinfo` — the four percent-encoders with the same `safe`
///                       sets that the Python module uses.
/// * `parse_netloc`      — exposed standalone for callers that already
///                         have a netloc (e.g. test paths).
/// * `parse_qsl_sorted`  — sorted ``urlencode(parse_qsl(...))`` to match
///                         the Python query normalizer.
///
/// Rust-side semantics are intentionally byte-for-byte equivalent to the
/// Python originals so the Python `URL` dataclass can call into either
/// implementation without observable differences. Keep the unit tests in
/// `python/tests/test_yggdrasil/test_io/test_url.py` green when changing
/// anything here.
use pyo3::prelude::*;

const SAFE_PATH: &str = "/:@-._~!$&'()*+,;=";
const SAFE_QUERY: &str = "-._~!$'()*+,;=:@/?&=";
const SAFE_FRAGMENT: &str = "-._~!$&'()*+,;=:@/?";
const SAFE_USERINFO: &str = ":!$&'()*+,;=";

/// Returns Some(default) if `scheme` has a well-known default port.
fn default_port(scheme: &str) -> Option<u32> {
    match scheme {
        "http" | "ws" => Some(80),
        "https" | "wss" => Some(443),
        _ => None,
    }
}

fn is_unreserved(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~')
}

fn percent_encode(input: &str, safe: &str) -> String {
    // Python's urllib `quote(..., safe=...)` keeps unreserved chars
    // (alphanum + "-._~"), keeps every char listed in `safe`, and
    // percent-encodes everything else byte-by-byte (UTF-8). Match that
    // exactly so encoded strings round-trip with the Python output.
    let safe_bytes = safe.as_bytes();
    let mut out = String::with_capacity(input.len());
    for &b in input.as_bytes() {
        if is_unreserved(b) || safe_bytes.contains(&b) {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(hex_upper(b >> 4));
            out.push(hex_upper(b & 0x0f));
        }
    }
    out
}

fn hex_upper(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'A' + (nibble - 10)) as char,
        _ => '0',
    }
}

fn percent_decode(input: &str) -> String {
    // Mirrors urllib.parse.unquote semantics: percent triplets are
    // decoded as UTF-8, malformed triplets pass through verbatim.
    let bytes = input.as_bytes();
    let mut out_bytes: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) =
                (decode_hex(bytes[i + 1]), decode_hex(bytes[i + 2]))
            {
                out_bytes.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out_bytes.push(bytes[i]);
        i += 1;
    }
    // urllib.parse.unquote is UTF-8-by-default and uses errors="replace"
    // for invalid sequences. Match that with from_utf8_lossy.
    String::from_utf8_lossy(&out_bytes).into_owned()
}

fn decode_hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(10 + b - b'a'),
        b'A'..=b'F' => Some(10 + b - b'A'),
        _ => None,
    }
}

fn maybe_decode(value: &str, decode: bool) -> String {
    if decode && !value.is_empty() {
        percent_decode(value)
    } else {
        value.to_string()
    }
}

fn lower_if(value: &str) -> String {
    if value.is_empty() {
        String::new()
    } else {
        value.to_ascii_lowercase()
    }
}

fn strip_trailing_dot(host: &str) -> &str {
    host.strip_suffix('.').unwrap_or(host)
}

fn parse_port_text(text: &str) -> Option<u32> {
    if text.is_empty() {
        return None;
    }
    if !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse::<u32>().ok().filter(|&p| p > 0)
}

/// Parse a `netloc` chunk into `(userinfo, host, port)`.
///
/// `port = 0` means "no port" — kept as a `u32` (rather than `Option`)
/// to match the Python side's _NO_PORT sentinel; the FFI layer maps the
/// 0 back to `None` when it crosses into Python.
fn parse_netloc_internal(netloc: &str, decode: bool) -> (String, String, u32) {
    if netloc.is_empty() {
        return (String::new(), String::new(), 0);
    }

    let (userinfo_raw, hostport) = match netloc.rfind('@') {
        Some(idx) => (&netloc[..idx], &netloc[idx + 1..]),
        None => ("", netloc),
    };
    let userinfo = maybe_decode(userinfo_raw, decode);

    if let Some(rest) = hostport.strip_prefix('[') {
        if let Some(rb) = rest.find(']') {
            let host = &rest[..rb];
            let after = &rest[rb + 1..];
            let port = if let Some(after_colon) = after.strip_prefix(':') {
                parse_port_text(after_colon).unwrap_or(0)
            } else {
                0
            };
            return (userinfo, host.to_string(), port);
        }
        return (userinfo, hostport.to_string(), 0);
    }

    if let Some(idx) = hostport.rfind(':') {
        let (host, after) = (&hostport[..idx], &hostport[idx + 1..]);
        let port = parse_port_text(after).unwrap_or(0);
        return (userinfo, host.to_string(), port);
    }

    (userinfo, hostport.to_string(), 0)
}

/// Strip leading "//" if present so the rest can be parsed as a netloc.
/// Returns (netloc, rest_after_authority).
fn split_authority(after_scheme: &str) -> (Option<&str>, &str) {
    if let Some(rest) = after_scheme.strip_prefix("//") {
        let cut = rest
            .bytes()
            .position(|b| matches!(b, b'/' | b'?' | b'#'))
            .unwrap_or(rest.len());
        return (Some(&rest[..cut]), &rest[cut..]);
    }
    (None, after_scheme)
}

fn split_query_fragment(rest: &str) -> (&str, &str, &str) {
    // path, query, fragment — using Python's urlsplit precedence
    // (fragment delimiter wins over query).
    let (head, fragment) = match rest.find('#') {
        Some(idx) => (&rest[..idx], &rest[idx + 1..]),
        None => (rest, ""),
    };
    let (path, query) = match head.find('?') {
        Some(idx) => (&head[..idx], &head[idx + 1..]),
        None => (head, ""),
    };
    (path, query, fragment)
}

fn split_scheme(raw: &str) -> (Option<&str>, &str) {
    // Match Python's urlsplit scheme rule: starts with ASCII letter,
    // followed by alphanum / '+' / '-' / '.', terminated by ':'. The
    // single-letter Windows-drive detection ("C:") is handled by the
    // caller because it needs the path component too.
    let bytes = raw.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_alphabetic() {
        return (None, raw);
    }
    for (i, &b) in bytes.iter().enumerate().skip(1) {
        if b == b':' {
            // Empty scheme isn't valid; we only emit Some when there is
            // at least one alpha char.
            return (Some(&raw[..i]), &raw[i + 1..]);
        }
        if !(b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.')) {
            return (None, raw);
        }
    }
    (None, raw)
}

/// Split a raw URL string into the seven canonical components, matching
/// Python's `urlsplit` + `_parse_netloc` + entity decoding + Windows
/// drive-letter fix-up + the "missing //" authority recovery.
fn split_url(
    raw: &str,
    default_scheme: Option<&str>,
    decode: bool,
) -> (String, String, String, u32, String, String, String) {
    let (scheme_opt, after_scheme) = split_scheme(raw);
    let mut scheme = scheme_opt.unwrap_or("").to_string();

    let (netloc_opt, rest) = split_authority(after_scheme);
    let (path_raw, query_raw, fragment_raw) = split_query_fragment(rest);

    let (userinfo, host, port) = match netloc_opt {
        Some(netloc) => parse_netloc_internal(netloc, decode),
        None => (String::new(), String::new(), 0),
    };

    let mut path = maybe_decode(path_raw, decode);

    // Single-letter "scheme" → Windows drive letter. Reattach drive,
    // normalize backslash separators, force the file scheme.
    if scheme.len() == 1
        && scheme.chars().next().map_or(false, |c| c.is_ascii_alphabetic())
    {
        let drive = scheme.to_ascii_uppercase();
        let mut rest_path = path.replace('\\', "/");
        if !rest_path.starts_with('/') {
            rest_path.insert(0, '/');
        }
        path = format!("/{drive}:{rest_path}");
        scheme = "file".to_string();
    }

    if let Some(default) = default_scheme {
        if !default.is_empty() {
            scheme = default.to_string();
        }
    }

    // XML/HTML entity fix-up: ``&amp;`` → ``&``.
    let query = maybe_decode(query_raw, decode).replace("&amp;", "&");
    let fragment = maybe_decode(fragment_raw, decode).replace("&amp;", "&");

    // "http:example.com/path" — recover host from the first path
    // segment when the authority "//" is missing. Skipped for `file`
    // and for schemaless input (which would mis-extract a real path).
    let mut final_host = host;
    let mut final_path = path;
    if !scheme.is_empty() && scheme != "file" && final_host.is_empty() && !final_path.is_empty() {
        if let Some(idx) = final_path.find('/') {
            let (h, p) = final_path.split_at(idx);
            final_host = h.to_string();
            let mut new_p = String::from("/");
            new_p.push_str(p.trim_start_matches('/'));
            final_path = new_p;
        } else {
            final_host = final_path.clone();
            final_path = "/".to_string();
        }
    }

    (
        scheme, userinfo, final_host, port, final_path, query, fragment,
    )
}

/// Sorted `urlencode(parse_qsl(...))` matching the Python normalizer.
fn normalize_query_internal(query: &str) -> String {
    let q = query.trim_start_matches('?');
    if q.is_empty() {
        return String::new();
    }
    let mut items: Vec<(String, String)> = Vec::new();
    for pair in q.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = match pair.find('=') {
            Some(idx) => (&pair[..idx], &pair[idx + 1..]),
            None => (pair, ""),
        };
        // parse_qsl decodes '+' to space then percent-decodes both sides.
        let key = percent_decode(&k.replace('+', " "));
        let val = percent_decode(&v.replace('+', " "));
        items.push((key, val));
    }
    items.sort();
    let mut out = String::new();
    let mut first = true;
    for (k, v) in items {
        if !first {
            out.push('&');
        }
        first = false;
        out.push_str(&urlencode_pair(&k));
        out.push('=');
        out.push_str(&urlencode_pair(&v));
    }
    out
}

fn urlencode_pair(value: &str) -> String {
    // urlencode uses quote_plus semantics: space -> '+', everything
    // else uses urllib.parse.quote with the default safe set.
    let mut out = String::with_capacity(value.len());
    for &b in value.as_bytes() {
        if is_unreserved(b) {
            out.push(b as char);
        } else if b == b' ' {
            out.push('+');
        } else {
            out.push('%');
            out.push(hex_upper(b >> 4));
            out.push(hex_upper(b & 0x0f));
        }
    }
    out
}

fn normalize_path_internal(path: &str) -> String {
    if path.is_empty() {
        return "/".to_string();
    }
    path.to_string()
}

fn remove_default_port(scheme: &str, host: &str, port: u32) -> u32 {
    if scheme.is_empty() || host.is_empty() || port == 0 {
        return 0;
    }
    if default_port(scheme) == Some(port) {
        0
    } else {
        port
    }
}

// ---------------------------------------------------------------------------
// Public PyO3 surface
// ---------------------------------------------------------------------------

/// Parse a URL into the seven canonical components.
///
/// Returns `(scheme, userinfo, host, port, path, query, fragment)`.
/// `userinfo`, `query`, `fragment` may be empty strings; the Python
/// caller normalizes empties to `None` to match the dataclass shape.
/// `port = 0` means "no port".
#[pyfunction]
#[pyo3(signature = (raw, *, default_scheme=None, decode=false, normalize=true))]
pub fn parse_url(
    raw: &str,
    default_scheme: Option<&str>,
    decode: bool,
    normalize: bool,
) -> (String, String, String, u32, String, String, String) {
    let (scheme, userinfo, host, port, path, query, fragment) =
        split_url(raw, default_scheme, decode);

    if !normalize {
        let q = query.trim_start_matches('?').to_string();
        let f = fragment.trim_start_matches('#').to_string();
        let p = if path.is_empty() {
            "/".to_string()
        } else {
            path
        };
        return (scheme, userinfo, host, port, p, q, f);
    }

    normalize_components_internal(scheme, userinfo, host, port, path, query, fragment, false)
}

/// Apply the cross-component normalization pipeline.
///
/// `os_find_file` is `true` only for `file://` URLs — it tells the
/// caller side to run `os.path.realpath` on the path. We do **not**
/// call `realpath` from Rust because the Python module's behavior on
/// Windows drive letters is platform-sensitive and lives in the Python
/// helper; instead we just preserve the path here and rely on the
/// Python wrapper to invoke `_normalize_path(...)` for `file` URLs.
#[pyfunction]
#[pyo3(signature = (scheme, userinfo, host, port, path, query, fragment, os_find_file=false))]
pub fn normalize_components(
    scheme: &str,
    userinfo: &str,
    host: &str,
    port: u32,
    path: &str,
    query: &str,
    fragment: &str,
    os_find_file: bool,
) -> (String, String, String, u32, String, String, String) {
    normalize_components_internal(
        scheme.to_string(),
        userinfo.to_string(),
        host.to_string(),
        port,
        path.to_string(),
        query.to_string(),
        fragment.to_string(),
        os_find_file,
    )
}

fn normalize_components_internal(
    scheme: String,
    userinfo: String,
    host: String,
    port: u32,
    path: String,
    query: String,
    fragment: String,
    _os_find_file: bool,
) -> (String, String, String, u32, String, String, String) {
    let scheme_n = lower_if(&scheme);
    let host_lower = lower_if(&host);
    let host_n = strip_trailing_dot(&host_lower).to_string();
    let port_n = remove_default_port(&scheme_n, &host_n, port);
    let path_n = normalize_path_internal(&path);
    let query_n = normalize_query_internal(&query);
    let fragment_n = fragment.trim_start_matches('#').to_string();
    (scheme_n, userinfo, host_n, port_n, path_n, query_n, fragment_n)
}

#[pyfunction]
pub fn encode_path(value: &str) -> String {
    // Match the Python helper: when the input has no space we extend the
    // safe set with '%' so pre-encoded triplets aren't double-encoded.
    let safe = if !value.contains(' ') {
        format!("{}%", SAFE_PATH)
    } else {
        SAFE_PATH.to_string()
    };
    percent_encode(value, &safe)
}

#[pyfunction]
pub fn encode_query(value: &str) -> String {
    let safe = format!("{}%", SAFE_QUERY);
    percent_encode(value, &safe)
}

#[pyfunction]
pub fn encode_fragment(value: &str) -> String {
    let safe = format!("{}%", SAFE_FRAGMENT);
    percent_encode(value, &safe)
}

#[pyfunction]
pub fn encode_userinfo(value: &str) -> String {
    if value.is_empty() {
        return String::new();
    }
    percent_encode(value, SAFE_USERINFO)
}

#[pyfunction]
#[pyo3(signature = (netloc, *, decode=false))]
pub fn parse_netloc(netloc: &str, decode: bool) -> (String, String, u32) {
    parse_netloc_internal(netloc, decode)
}

#[pyfunction]
pub fn normalize_query(query: &str) -> String {
    normalize_query_internal(query)
}

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(parse_url, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_components, module)?)?;
    module.add_function(wrap_pyfunction!(encode_path, module)?)?;
    module.add_function(wrap_pyfunction!(encode_query, module)?)?;
    module.add_function(wrap_pyfunction!(encode_fragment, module)?)?;
    module.add_function(wrap_pyfunction!(encode_userinfo, module)?)?;
    module.add_function(wrap_pyfunction!(parse_netloc, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_query, module)?)?;
    Ok(())
}
