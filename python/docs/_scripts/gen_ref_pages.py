"""Generate API reference pages for mkdocstrings.

The literate-nav SUMMARY.md sits at ``reference/SUMMARY.md`` and its links
are resolved relative to that file. We therefore store nav entries with paths
that don't include the leading ``reference/`` prefix.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

PACKAGE_ROOT = Path("src/yggdrasil")
NAV_ROOT = "reference"

nav = mkdocs_gen_files.Nav()

for path in sorted(PACKAGE_ROOT.rglob("*.py")):
    module_path = path.relative_to("src").with_suffix("")
    parts = module_path.parts

    if parts[-1] == "__main__":
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        rel_doc_path = Path(*parts, "index.md")
    else:
        rel_doc_path = Path(*parts).with_suffix(".md")

    if not parts:
        continue

    full_doc_path = Path(NAV_ROOT) / rel_doc_path
    nav[parts] = rel_doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# `{ident}`\n\n::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, Path("..") / path)

with mkdocs_gen_files.open(f"{NAV_ROOT}/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
