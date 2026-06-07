"""Project scaffolding — a ready-to-push repo from scratch, any language.

Loki's "new project" path: lay down a clean, git-initialised repository with a
README, a composed ``.gitignore``, and one folder **per language** — each with
``src/`` + ``tests/`` and a pre-built manifest (``pyproject.toml`` for Python,
``package.json`` for TypeScript, ``Cargo.toml`` for Rust, ``go.mod`` for Go) —
then make the initial commit so it's ready to ``git push`` to GitHub.

Data over code: the per-language layout lives in :data:`LANGUAGES` (manifest +
``src``/``tests`` starter files + ignore lines); :func:`scaffold_project` just
renders the templates, writes the tree, and runs git. Add a language by adding a
row, not a branch.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

__all__ = ["LANGUAGES", "scaffold_project", "resolve_languages"]

_PYPROJECT = """\
[project]
name = "{name}"
version = "0.1.0"
description = "{description}"
readme = "../README.md"
requires-python = ">=3.9"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{pkg}"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

_PACKAGE_JSON = """\
{
  "name": "{name}",
  "version": "0.1.0",
  "description": "{description}",
  "type": "module",
  "main": "src/index.ts",
  "scripts": {
    "build": "tsc",
    "test": "vitest run"
  },
  "devDependencies": {
    "typescript": "^5.5.0",
    "vitest": "^2.0.0"
  }
}
"""

_TSCONFIG = """\
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "bundler",
    "rootDir": "src",
    "outDir": "dist",
    "strict": true,
    "esModuleInterop": true,
    "declaration": true
  },
  "include": ["src", "tests"]
}
"""

#: language → its in-folder layout. ``{name}``/``{pkg}``/``{description}`` are
#: substituted; paths are relative to the ``<language>/`` folder.
LANGUAGES: dict[str, dict[str, Any]] = {
    "python": {
        "files": {
            "pyproject.toml": _PYPROJECT,
            "src/{pkg}/__init__.py": '__all__: list[str] = []\n__version__ = "0.1.0"\n',
            "src/{pkg}/main.py": (
                'def main() -> None:\n    print("hello from {name}")\n\n\n'
                'if __name__ == "__main__":\n    main()\n'
            ),
            "tests/test_smoke.py": (
                "from {pkg} import __version__\n\n\n"
                "def test_version() -> None:\n    assert __version__\n"
            ),
        },
        "ignore": ("__pycache__/", "*.py[cod]", ".venv/", "venv/", "dist/",
                   "*.egg-info/", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/"),
    },
    "typescript": {
        "files": {
            "package.json": _PACKAGE_JSON,
            "tsconfig.json": _TSCONFIG,
            "src/index.ts": 'export const hello = (): string => "hello from {name}";\n',
            "tests/index.test.ts": (
                'import { describe, it, expect } from "vitest";\n'
                'import { hello } from "../src/index";\n\n'
                'describe("{name}", () => {\n'
                '  it("greets", () => expect(hello()).toContain("{name}"));\n'
                "});\n"
            ),
        },
        "ignore": ("node_modules/", "dist/", "*.tsbuildinfo", "coverage/"),
    },
    "rust": {
        "files": {
            "Cargo.toml": (
                '[package]\nname = "{pkg}"\nversion = "0.1.0"\nedition = "2021"\n'
                'description = "{description}"\n\n[dependencies]\n'
            ),
            "src/main.rs": (
                'fn greet() -> String {\n    "hello from {name}".to_string()\n}\n\n'
                'fn main() {\n    println!("{}", greet());\n}\n\n'
                "#[cfg(test)]\nmod tests {\n    use super::*;\n\n"
                '    #[test]\n    fn greets() {\n        assert!(greet().contains("{name}"));\n    }\n}\n'
            ),
            "tests/smoke.rs": "#[test]\nfn smoke() {\n    assert_eq!(2 + 2, 4);\n}\n",
        },
        "ignore": ("/target", "Cargo.lock", "**/*.rs.bk"),
    },
    "go": {
        "files": {
            "go.mod": "module {pkg}\n\ngo 1.21\n",
            "src/main.go": (
                'package main\n\nimport "fmt"\n\n'
                'func greet() string { return "hello from {name}" }\n\n'
                "func main() { fmt.Println(greet()) }\n"
            ),
            "tests/smoke_test.go": (
                'package main\n\nimport "testing"\n\n'
                'func TestSmoke(t *testing.T) {\n    if 2+2 != 4 {\n        t.Fatal("math")\n    }\n}\n'
            ),
        },
        "ignore": ("bin/", "*.exe", "*.test", "*.out"),
    },
}

#: NL aliases → canonical language key (for autonomous routing from a prompt).
_ALIASES: dict[str, str] = {
    "python": "python", "py": "python",
    "typescript": "typescript", "ts": "typescript", "javascript": "typescript",
    "js": "typescript", "node": "typescript", "nodejs": "typescript",
    "rust": "rust", "cargo": "rust", "go": "go", "golang": "go",
}

_COMMON_IGNORE = (".DS_Store", ".idea/", ".vscode/", "*.log", ".env")


def resolve_languages(text: str) -> list[str]:
    """The languages named in *text* (word-boundary match) → canonical keys.

    Empty when none are mentioned — the caller defaults (to Python). Used by the
    autonomous router so "scaffold a rust + python cli" picks both.
    """
    low = text.lower()
    found: list[str] = []
    for alias, canon in _ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", low) and canon not in found:
            found.append(canon)
    return found


def _pkg(name: str) -> str:
    """A project name → an identifier-safe package/module name."""
    pkg = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    return pkg or "app"


def _render(template: str, fields: dict[str, str]) -> str:
    """Substitute ``{name}`` / ``{pkg}`` / ``{description}`` only — leaving the
    literal braces in JSON / Rust / Go source untouched (so no ``str.format``)."""
    out = template
    for key, value in fields.items():
        out = out.replace("{" + key + "}", value)
    return out


def scaffold_project(
    name: str,
    languages: list[str],
    *,
    base_dir: Optional[str] = None,
    description: Optional[str] = None,
    git: bool = True,
) -> dict[str, Any]:
    """Create a ready-to-push project tree, one ``src``/``tests`` folder per language.

    Lays down ``README.md`` + a composed ``.gitignore`` at the root and a
    ``<language>/`` folder per requested language (manifest + starter ``src`` and
    ``tests``). With ``git`` it ``git init``s and makes the initial commit, so the
    repo only needs a remote + ``git push``. Returns the path, the file list, the
    languages used, and the push hint.
    """
    import tempfile

    langs = [lang for lang in dict.fromkeys(languages) if lang in LANGUAGES] or ["python"]
    unknown = [lang for lang in languages if lang not in LANGUAGES]
    description = description or f"{name} — scaffolded by Loki."
    pkg = _pkg(name)

    root = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix="ygg-scaffold-"))
    root = root / name
    root.mkdir(parents=True, exist_ok=True)

    fields = {"name": name, "pkg": pkg, "description": description}
    written: list[str] = []
    for lang in langs:
        for rel, template in LANGUAGES[lang]["files"].items():
            path = root / lang / _render(rel, fields)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_render(template, fields))
            written.append(str(path.relative_to(root)))

    # Composed .gitignore — common lines + each language's block.
    ignore = [f"# {name} — .gitignore", *_COMMON_IGNORE, ""]
    for lang in langs:
        ignore += [f"# {lang}", *LANGUAGES[lang]["ignore"], ""]
    (root / ".gitignore").write_text("\n".join(ignore).rstrip() + "\n")

    layout = "\n".join(
        f"- `{lang}/` — {lang} (`src/`, `tests/`, `{_manifest(lang)}`)" for lang in langs
    )
    (root / "README.md").write_text(
        f"# {name}\n\n{description}\n\n## Layout\n\n{layout}\n\n"
        f"## Develop\n\n```bash\n{_develop_hint(langs)}\n```\n\n"
        f"## Push\n\n```bash\ngit remote add origin <your-repo-url>\n"
        f"git push -u origin main\n```\n"
    )
    written += [".gitignore", "README.md"]

    committed = False
    if git:
        committed = _git_init(root)

    return {
        "name": name,
        "path": str(root),
        "languages": langs,
        "unknown_languages": unknown,
        "files": sorted(written),
        "git": committed,
        "push": "git remote add origin <url> && git push -u origin main",
    }


def _manifest(lang: str) -> str:
    return {"python": "pyproject.toml", "typescript": "package.json",
            "rust": "Cargo.toml", "go": "go.mod"}.get(lang, "manifest")


def _develop_hint(langs: list[str]) -> str:
    hints = {
        "python": "cd python && uv venv && uv pip install -e . && pytest",
        "typescript": "cd typescript && npm install && npm test",
        "rust": "cd rust && cargo test",
        "go": "cd go && go test ./...",
    }
    return "\n".join(hints[lang] for lang in langs if lang in hints)


def _git_init(root: Path) -> bool:
    """``git init`` + initial commit on ``main`` — best-effort (False if no git)."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    def run(*a: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(root), *a],
                              capture_output=True, text=True, env=env, timeout=30)

    try:
        if run("init", "-b", "main").returncode != 0:
            run("init")                                  # older git without -b
            run("checkout", "-b", "main")
        run("add", "-A")

        # Identity may be unset in a fresh container — pass it inline. Try a
        # normal commit first (honouring the user's signing config); if that
        # fails (e.g. a signing hook is unavailable), retry without signing so a
        # scaffold never dies on the initial commit.
        def _commit(*extra: str) -> int:
            return subprocess.run(
                ["git", "-C", str(root), "-c", "user.name=Loki",
                 "-c", "user.email=loki@yggdrasil.local", *extra,
                 "commit", "-m", "Initial scaffold"],
                capture_output=True, text=True, env=env, timeout=30,
            ).returncode

        return _commit() == 0 or _commit("-c", "commit.gpgsign=false") == 0
    except (OSError, subprocess.SubprocessError):
        return False
