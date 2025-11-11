# Yggdrasil

Yggdrasil is a multi-language research playground. Every language-specific
implementation lives in its own top-level directory. The Python package resides
in `python/src/` and is designed to be installed directly from this repository using
`pip` and the `git+https` scheme.

## Installing the Python utilities

Because the packaging metadata is provided at the repository root, the package
can be installed directly from Git without passing a `subdirectory` hint even
though the importable code lives under `python/src/`.

### Linux / macOS

Use the system `python3` interpreter and point `pip` at the repository URL:

```bash
python3 -m pip install "yggdrasil @ git+https://github.com/example/yggdrasil.git"
```

### Windows

Launch a terminal (PowerShell or Command Prompt) and run:

```powershell
py -m pip install "yggdrasil @ git+https://github.com/example/yggdrasil.git"
```

After installation the package exposes a small Arrow-centric CLI that you can
try out from the command line:

```bash
yggdrasil greet Freya
yggdrasil arrow-cast 1 2 3
```

The `greet` command prints a salutation while `arrow-cast` demonstrates the
Arrow casting registry by widening a list of integers. You can also inspect the
demo Arrow table through the module-level runner:

```bash
python -m yggdrasil.cli demo-table
```

The demo table command displays the tiny `pyarrow.Table` that carries metadata
shared across language boundaries.

## Repository layout

- `python/src/` – Source for the distributable Python package.
- `python/tests/` – In-repo pytest suite for the Python utilities.
- `cpp/`, `rust/`, … – Additional language implementations can be added beside
  `python/` as the project grows.

## Development

Profiles for common workflows are declared in `pyproject.toml`:

- `test` installs the pytest runner used by the in-tree suite.
- `dev` includes the testing tools plus linters for everyday iteration.

To work on the project:

1. Create and activate a virtual environment.
2. Install the editable package together with the development profile:
   ```bash
   python -m pip install -e ".[dev]"
   ```
3. Run the automated checks before sending a pull request:
   ```bash
   pytest
   ruff check python/src python/tests
   mypy python/src
   ```

## Contributing

We welcome issues and pull requests. Please open a discussion if you plan larger
changes so the direction can be aligned early. Contributions should include:

1. Clear descriptions of the motivation and behaviour changes.
2. Updates to documentation where it helps users follow along.
3. Passing tests (`pytest`) and lint checks (`ruff`, `mypy`) using the provided
   development profile.

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for
more details.
