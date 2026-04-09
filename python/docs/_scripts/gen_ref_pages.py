"""Generate API reference pages for mkdocstrings."""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

PACKAGE_ROOT = Path("src/yggdrasil")

nav = mkdocs_gen_files.Nav()

for path in sorted(PACKAGE_ROOT.rglob("*.py")):
    module_path = path.relative_to("src").with_suffix("")
    parts = module_path.parts

    if parts[-1] == "__main__":
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = Path("reference", *parts, "index.md")
    else:
        doc_path = Path("reference", *parts).with_suffix(".md")

    if not parts:
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(doc_path, Path("..") / path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
