# @platob/ygg

WASM bindings over the Rust [`ygg`](../rust/ygg) engine. All logic lives in
Rust (compiled to WebAssembly); this package only exposes a JS/TS surface.

## Install

```bash
npm install @platob/ygg
```

## Use

```js
const { Uri, Url } = require("@platob/ygg");

const u = Url.parse("https://user:pw@example.com:8443/a/b?q=1#f");
u.scheme;      // 'https'
u.host;        // 'example.com'
u.port;        // 8443
u.path;        // '/a/b'
u.toString();  // round-trips back to the original

Uri.parse("urn:isbn:0451450523").path; // 'isbn:0451450523'
```

Built from `js/` with `wasm-pack` and published by
`.github/workflows/publish-js.yml`.
