//! WASM bindings for ygg — a thin layer over the `ygg` engine crate.
//!
//! These classes only marshal types across the boundary and expose a
//! JS-idiomatic surface (`camelCase`, `parse`, `toString`). All parsing
//! logic lives in the engine; parse errors are thrown as JS `Error`s.

use wasm_bindgen::prelude::*;
use ygg_engine::{Uri as CoreUri, Url as CoreUrl};

/// A URI split into its RFC 3986 components.
#[wasm_bindgen]
pub struct Uri {
    inner: CoreUri,
}

#[wasm_bindgen]
impl Uri {
    /// Parse a string into its URI components.
    pub fn parse(value: &str) -> Result<Uri, JsError> {
        CoreUri::parse(value)
            .map(|inner| Uri { inner })
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn scheme(&self) -> Option<String> {
        self.inner.scheme.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn authority(&self) -> Option<String> {
        self.inner.authority.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn path(&self) -> String {
        self.inner.path.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn query(&self) -> Option<String> {
        self.inner.query.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn fragment(&self) -> Option<String> {
        self.inner.fragment.clone()
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_js_string(&self) -> String {
        self.inner.to_string()
    }
}

/// A parsed URL: a located URI with its authority decomposed.
#[wasm_bindgen]
pub struct Url {
    inner: CoreUrl,
}

#[wasm_bindgen]
impl Url {
    /// Parse a string into a URL (requires a scheme and a host).
    pub fn parse(value: &str) -> Result<Url, JsError> {
        CoreUrl::parse(value)
            .map(|inner| Url { inner })
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn scheme(&self) -> String {
        self.inner.scheme.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn username(&self) -> Option<String> {
        self.inner.username.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn password(&self) -> Option<String> {
        self.inner.password.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn host(&self) -> String {
        self.inner.host.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn port(&self) -> Option<u16> {
        self.inner.port
    }

    #[wasm_bindgen(getter)]
    pub fn path(&self) -> String {
        self.inner.path.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn query(&self) -> Option<String> {
        self.inner.query.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn fragment(&self) -> Option<String> {
        self.inner.fragment.clone()
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_js_string(&self) -> String {
        self.inner.to_string()
    }
}
