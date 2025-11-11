# Yggdrasil

Yggdrasil is a multi-language research playground. Every language-specific
implementation lives in its own top-level directory. The Python package resides
in `python/` and is designed to be installed directly from this repository using
`pip` and the `git+https` scheme.

## Installing the Python utilities

To install the Python helpers straight from the repository, point `pip` at the
Git URL:

```bash
pip install "yggdrasil @ git+https://github.com/example/yggdrasil.git"
```

Because the packaging metadata is provided at the repository root, no extra
`subdirectory` hints are required even though the importable code lives under
`python/`.

After installation the package exposes a small Arrow-centric helper that you can
try out from the command line:

```bash
python -m yggdrasil.example
```

This script prints a greeting and then displays a tiny `pyarrow.Table` carrying
metadata shared across language boundaries.

## Repository layout

- `python/` – Source for the distributable Python package.
- `cpp/`, `rust/`, … – Additional language implementations can be added beside
  `python/` as the project grows.

## Development

1. Create a virtual environment.
2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```
3. Run the Python unit tests (to be added) and lint checks before sending a PR.

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for
more details.
