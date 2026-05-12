# Development

## Local install

```bash
git clone https://github.com/Platob/Yggdrasil.git
cd Yggdrasil/python
uv venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -e .[dev]
```

## Tests

```bash
cd python
pytest                                                   # full suite
pytest tests/test_yggdrasil/test_data/                   # one area
pytest tests/test_yggdrasil/test_data/test_registry.py   # one file
pytest tests/test_yggdrasil/test_io/test_url.py::test_x  # one test
```

`pytest-asyncio` runs in `strict` mode — async tests need the explicit marker. Databricks live-integration tests are gated by the `integration` marker and skipped unless `DATABRICKS_HOST` is set.

### Engine TestCase bases

Any test that touches a dataframe, Arrow object, or engine-side type **must** subclass the matching base from `yggdrasil.<engine>.tests` (no top-level engine imports — they break base installs and defeat the skip-on-missing behavior).

| Engine | Base class | Module |
|---|---|---|
| Arrow | `ArrowTestCase` | `yggdrasil.arrow.tests` |
| Polars | `PolarsTestCase` | `yggdrasil.polars.tests` |
| pandas | `PandasTestCase` | `yggdrasil.pandas.tests` |
| Spark | `SparkTestCase` | `yggdrasil.spark.tests` |

```python
from yggdrasil.arrow.tests import ArrowTestCase

class TestX(ArrowTestCase):
    def test_table(self):
        t = self.table({"id": [1, 2]})
        self.assertSchemaEqual(t.schema, self.pa.schema([self.pa.field("id", self.pa.int64())]))
```

Use the helpers (`self.table(...)`, `self.df(...)`, `self.lazy(...)`, `self.record_batch(...)`, `self.tmp_path`, `self.write_parquet(...)`, `self.arrow_to_polars(...)`) and the built-in assertions (`assertFrameEqual`, `assertSchemaEqual`, `assertSeriesEqual`).

Cross-engine tests can multi-inherit (`class TestX(PolarsTestCase, ArrowTestCase): ...`) or split into sibling classes in the same file.

`SparkTestCase` shares a single process-wide `SparkSession`; don't call `SparkSession.builder` yourself, and don't stop the session in `tearDown`.

## Lint and format

```bash
ruff check
black .
```

## Docs locally

```bash
mkdocs serve     # http://127.0.0.1:8000
mkdocs build     # static site → python/site/
```

The site is automatically deployed to GitHub Pages by [`.github/workflows/docs.yml`](https://github.com/Platob/Yggdrasil/blob/main/.github/workflows/docs.yml) on every push to `main` that touches `python/docs/**`, `python/src/**`, `mkdocs.yml`, or the workflow itself. Manual runs are available via *workflow_dispatch*.

## Optional dependencies

Pull integrations in only as needed:

```bash
uv pip install -e .[data]         # pandas + numpy + sqlglot
uv pip install -e .[bigdata]      # pyspark + delta-spark
uv pip install -e .[delta]        # deltalake
uv pip install -e .[databricks]   # databricks-sdk
uv pip install -e .[api]          # fastapi + uvicorn + pydantic
uv pip install -e .[pickle]       # cloudpickle/dill/zstandard/xxhash/blake3
uv pip install -e .[http]         # urllib3 + xxhash
uv pip install -e .[mongo]        # mongoengine
uv pip install -e .[postgres]     # psycopg + adbc-driver-postgresql
uv pip install -e .[kafka]        # confluent-kafka
```

## Releasing

The version in [`python/pyproject.toml`](https://github.com/Platob/Yggdrasil/blob/main/python/pyproject.toml) is the single source of truth.

| Workflow | Builds |
|---|---|
| [`publish.yml`](https://github.com/Platob/Yggdrasil/blob/main/.github/workflows/publish.yml) | `ygg` sdist + pure-Python wheel → PyPI; tags `vX.Y.Z`; cuts a GitHub Release |
| [`docs.yml`](https://github.com/Platob/Yggdrasil/blob/main/.github/workflows/docs.yml) | MkDocs Material site → GitHub Pages |

Do not push to `main` from an agent session — develop on a branch and open a PR.

## Conventions worth knowing

- **Python 3.10 minimum.** No 3.11+ syntax (e.g. `typing.Self`) without a fallback.
- **Be forgiving on input, strict on meaning.** Accept the shapes a real caller has, but fail loudly on conflicting arguments.
- **Preserve schema intent across boundaries.** Field names, order, nullability, metadata, nested structure, precision/scale, timezone intent are part of the user contract.
- **Error messages must answer:** what you passed, what was expected, valid values, what to try next. See [`AGENTS.md`](https://github.com/Platob/Yggdrasil/blob/main/AGENTS.md) for tone and worked examples.
- **Comments describe weirdness, not syntax.** Version quirks, engine edge cases, schema invariants, compatibility hacks — yes; `# loop through fields` — no.
- **Type hints must match runtime.** If a method can return `None`, annotate `| None`.
- **Keyword-only arguments** are preferred for ambiguous options.
