# Yggdrasil Python documentation

This directory is the landing page for all Python-facing guidance. Use the links
below to jump directly to the reference or template you need.

## Quick links

- [Module map](modules.md): brief descriptions of every package within
  `yggdrasil`.
- [Module index](modules/README.md): entry points to detailed docs for each
  submodule (Databricks helpers, schema utilities, HTTP/auth helpers, etc.).
- [Developer templates](developer-templates.md): copy/paste-ready snippets for
  common setup and integration tasks.
- [Python utility reference](pyutils.md): overview of cross-cutting helpers used
  throughout the codebase.
- [Serialization guide](ser.md): notes on how Yggdrasil handles structured data
  and Arrow schema conversion.

## Installation

Install the package into an existing environment:

```bash
pip install -e .[dev]
```

The optional `dev` extras supply linting and Databricks dependencies if you are
working in that environment.

## Quick start

For a minimal example that uses the enhanced dataclass decorator together with
Arrow schema helpers, see the **developer templates** or jump straight into the
`yggdrasil.dataclasses` documentation via the module index above.
