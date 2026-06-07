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

__all__ = ["LANGUAGES", "PRESETS", "CLOUD", "scaffold_project",
           "resolve_languages", "resolve_preset", "resolve_cloud"]

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

_BACKEND_MAIN = '''\
"""{name} — real-time FastAPI backend (WebSocket + SSE)."""
from __future__ import annotations

import asyncio
import json
import random
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI(title="{name}")


def _tick() -> dict:
    """One real-time sample — swap for your Kafka/Kinesis/Delta stream source."""
    return {"ts": time.time(), "value": round(random.uniform(0, 100), 2)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "{name}"}


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    try:
        while True:
            await sock.send_json(_tick())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return


@app.get("/stream")
async def stream() -> StreamingResponse:
    async def gen():
        while True:
            yield f"data: {json.dumps(_tick())}\\n\\n"
            await asyncio.sleep(1.0)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse("<h1>{name}</h1><p>WS at /ws · SSE at /stream · health at /health</p>")
'''

_BACKEND_TEST = '''\
from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    r = TestClient(app).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ws_streams_a_tick() -> None:
    with TestClient(app).websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert "ts" in msg and "value" in msg
'''

_BACKEND_PYPROJECT = '''\
[project]
name = "{pkg}-backend"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.10"
dependencies = ["fastapi>=0.110", "uvicorn[standard]>=0.29", "websockets>=12"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/app"]

[tool.pytest.ini_options]
testpaths = ["tests"]
'''

_FRONTEND_HTML = '''\
<!doctype html>
<html><head><meta charset="utf-8"><title>{name}</title>
<style>body{font-family:system-ui;margin:2rem}#v{font-size:3rem}</style></head>
<body><h1>{name}</h1><div>live value: <span id="v">—</span></div>
<script src="app.js"></script></body></html>
'''

_FRONTEND_JS = '''\
const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
const ws = new WebSocket(url);
ws.onmessage = (e) => { document.getElementById("v").textContent = JSON.parse(e.data).value; };
ws.onclose = () => setTimeout(() => location.reload(), 2000);
'''

_DOCKERFILE = '''\
FROM python:3.11-slim
WORKDIR /app
COPY backend/ /app/backend/
RUN pip install --no-cache-dir -e /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--app-dir", "backend/src", "--host", "0.0.0.0", "--port", "8000"]
'''

_COMPOSE = '''\
services:
  {pkg}:
    build: .
    ports: ["8000:8000"]
    restart: unless-stopped
'''

_CI = '''\
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e backend && pip install pytest httpx
      - run: pytest backend/tests
'''

#: Full-app presets — a coherent runnable tree, not just a library skeleton.
PRESETS: dict[str, dict[str, Any]] = {
    "fullstack-realtime": {
        "summary": "real-time FastAPI backend (WebSocket + SSE) + live frontend + Docker/CI",
        "files": {
            "backend/pyproject.toml": _BACKEND_PYPROJECT,
            "backend/src/app/__init__.py": '__version__ = "0.1.0"\n',
            "backend/src/app/main.py": _BACKEND_MAIN,
            "backend/tests/test_app.py": _BACKEND_TEST,
            "frontend/index.html": _FRONTEND_HTML,
            "frontend/app.js": _FRONTEND_JS,
            "Dockerfile": _DOCKERFILE,
            "docker-compose.yml": _COMPOSE,
            ".github/workflows/ci.yml": _CI,
        },
        "ignore": ("__pycache__/", "*.py[cod]", ".venv/", "dist/", "*.egg-info/",
                   ".pytest_cache/", "node_modules/", ".env"),
    },
}

#: Cloud deploy add-ons — real-time-aware notes/specs per target.
CLOUD: dict[str, dict[str, str]] = {
    "aws": {
        "deploy/aws.md": (
            "# Deploy to AWS\n\n"
            "Real-time path: API on **ECS Fargate** (or App Runner) behind an ALB; "
            "stream source on **Kinesis** / **MSK (Kafka)**.\n\n"
            "```bash\n"
            "aws ecr create-repository --repository-name {pkg}\n"
            "docker build -t {pkg} . && docker tag {pkg} <acct>.dkr.ecr.<region>.amazonaws.com/{pkg}\n"
            "docker push <acct>.dkr.ecr.<region>.amazonaws.com/{pkg}\n"
            "# then: ECS service (Fargate) + ALB target group on :8000\n"
            "```\n\nLoki: `ygg loki run aws-ecs` / `aws-s3` to inspect the account.\n"
        ),
    },
    "databricks": {
        "deploy/databricks.md": (
            "# Deploy on Databricks\n\n"
            "Real-time path: **Structured Streaming** (Kafka/Auto Loader → Delta) on a "
            "job cluster; serve features/metrics via **Model Serving** or a SQL warehouse.\n\n"
            "```bash\n"
            "ygg databricks deploy            # build + upload the ygg wheel env\n"
            "ygg databricks jobs              # manage the streaming job\n"
            "```\n\nThe backend reads the live Delta table (or a serving endpoint) for {name}.\n"
        ),
        "databricks.yml": (
            "# Databricks Asset Bundle (skeleton)\n"
            "bundle:\n  name: {pkg}\n\nresources:\n  jobs:\n    {pkg}_stream:\n"
            "      name: {pkg}-stream\n      tasks:\n        - task_key: ingest\n"
            "          notebook_task:\n            notebook_path: ./ingest\n"
        ),
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


#: Phrasing → a full-app preset (else the per-language library layout).
_PRESET_SIGNALS: dict[str, tuple[str, ...]] = {
    "fullstack-realtime": ("real-time", "realtime", "real time", "full-stack", "full stack",
                           "fullstack", "web app", "webapp", "websocket", "streaming app",
                           "dashboard app", "live app", "api + frontend", "backend and frontend"),
}


def resolve_preset(text: str) -> str:
    """The app preset named in *text*, or ``"lib"`` (per-language skeleton)."""
    low = text.lower()
    for preset, signals in _PRESET_SIGNALS.items():
        if any(s in low for s in signals):
            return preset
    return "lib"


def resolve_cloud(text: str) -> list[str]:
    """Cloud deploy targets named in *text* (``aws`` / ``databricks``)."""
    low = text.lower()
    targets = []
    if re.search(r"\b(aws|amazon|ecs|fargate|lambda|kinesis|s3)\b", low):
        targets.append("aws")
    if re.search(r"\b(databricks|delta|unity catalog|spark|lakehouse)\b", low):
        targets.append("databricks")
    return targets


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
    languages: Optional[list[str]] = None,
    *,
    preset: str = "lib",
    cloud: Optional[list[str]] = None,
    base_dir: Optional[str] = None,
    description: Optional[str] = None,
    git: bool = True,
) -> dict[str, Any]:
    """Create a ready-to-push project tree, then ``git init`` + initial commit.

    ``preset="lib"`` (default) lays down one ``<language>/`` folder per language
    (``src``/``tests`` + manifest). A full-app preset
    (e.g. ``"fullstack-realtime"``) lays down a coherent runnable app instead —
    a real-time FastAPI backend (WebSocket + SSE), a live frontend, Docker + CI —
    and ``cloud=["aws","databricks"]`` adds the matching deploy add-ons. Returns
    the path, file list, languages/preset/cloud used, and the push hint.
    """
    import tempfile

    languages = languages or ["python"]
    cloud = [c for c in (cloud or []) if c in CLOUD]
    description = description or f"{name} — scaffolded by Loki."
    pkg = _pkg(name)
    fields = {"name": name, "pkg": pkg, "description": description}

    root = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix="ygg-scaffold-"))
    root = root / name
    root.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    langs: list[str] = []
    unknown: list[str] = []

    def _write(rel: str, template: str) -> None:
        path = root / _render(rel, fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render(template, fields))
        written.append(str(path.relative_to(root)))

    if preset in PRESETS:
        for rel, template in PRESETS[preset]["files"].items():
            _write(rel, template)
        ignore_lines = list(PRESETS[preset]["ignore"])
        layout = PRESETS[preset]["summary"]
        develop = "docker compose up --build   # or: uvicorn app.main:app --app-dir backend/src --reload"
    else:                                            # per-language library layout
        preset = "lib"
        langs = [lang for lang in dict.fromkeys(languages) if lang in LANGUAGES] or ["python"]
        unknown = [lang for lang in languages if lang not in LANGUAGES]
        for lang in langs:
            for rel, template in LANGUAGES[lang]["files"].items():
                _write(f"{lang}/{rel}", template)
        ignore_lines = []
        for lang in langs:
            ignore_lines += [f"# {lang}", *LANGUAGES[lang]["ignore"], ""]
        layout = "\n".join(f"- `{lang}/` — {lang} (`src/`, `tests/`, `{_manifest(lang)}`)"
                           for lang in langs)
        develop = _develop_hint(langs)

    for target in cloud:                             # cloud deploy add-ons
        for rel, template in CLOUD[target].items():
            _write(rel, template)

    ignore = [f"# {name} — .gitignore", *_COMMON_IGNORE, "", *ignore_lines]
    (root / ".gitignore").write_text("\n".join(ignore).rstrip() + "\n")

    deploy = ("\n## Deploy\n\n" + "\n".join(f"- `{target}/` — see `deploy/{target}.md`"
                                            for target in cloud)) if cloud else ""
    (root / "README.md").write_text(
        f"# {name}\n\n{description}\n\n## Layout\n\n{layout}\n\n"
        f"## Develop\n\n```bash\n{develop}\n```\n{deploy}\n"
        f"\n## Push\n\n```bash\ngit remote add origin <your-repo-url>\n"
        f"git push -u origin main\n```\n"
    )
    written += [".gitignore", "README.md"]

    committed = _git_init(root) if git else False
    return {
        "name": name,
        "path": str(root),
        "preset": preset,
        "languages": langs,
        "cloud": cloud,
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
