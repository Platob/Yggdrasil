from __future__ import annotations

import ast
import logging
import re

from ..config import Settings
from ..schemas.ai import (
    CodeAnalysis,
    CodeAnalysisRequest,
    CodeAnalysisResponse,
    CodeIssue,
    RunAnalysisRequest,
    RunAnalysisResponse,
    RunInsight,
)

LOGGER = logging.getLogger(__name__)

# Known standard-library modules that don't need pip install
_STDLIB = frozenset({
    "abc", "ast", "asyncio", "base64", "builtins", "collections", "concurrent",
    "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal", "dis",
    "email", "enum", "errno", "functools", "gc", "glob", "hashlib", "heapq", "html",
    "http", "importlib", "inspect", "io", "itertools", "json", "logging", "math",
    "multiprocessing", "operator", "os", "pathlib", "pickle", "platform", "pprint",
    "queue", "random", "re", "shutil", "signal", "socket", "sqlite3", "ssl",
    "statistics", "string", "struct", "subprocess", "sys", "tempfile", "textwrap",
    "threading", "time", "timeit", "traceback", "types", "typing", "unicodedata",
    "unittest", "urllib", "uuid", "warnings", "weakref", "xml", "zipfile", "zlib",
    "__future__",
})


def _cyclomatic_complexity(tree: ast.AST) -> int:
    """Count branches to estimate cyclomatic complexity."""
    branch_nodes = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
    )
    count = 1
    for node in ast.walk(tree):
        if isinstance(node, branch_nodes):
            count += 1
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            count += len(node.values) - 1
    return count


def _extract_imports(tree: ast.AST) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    seen: set[str] = set()
    result = []
    for i in imports:
        if i not in seen:
            seen.add(i)
            result.append(i)
    return result


def _check_issues(tree: ast.AST, code: str) -> list[CodeIssue]:
    issues: list[CodeIssue] = []
    lines = code.splitlines()

    for node in ast.walk(tree):
        # Bare except catches everything including KeyboardInterrupt
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append(CodeIssue(
                line=node.lineno, severity="warning", category="style",
                message="Bare `except:` catches all exceptions including KeyboardInterrupt",
                suggestion="Use `except Exception:` or a specific exception type",
            ))

        # Global state mutation
        if isinstance(node, ast.Global):
            issues.append(CodeIssue(
                line=node.lineno, severity="info", category="style",
                message=f"Global variable(s): {', '.join(node.names)}",
                suggestion="Prefer returning values or using function parameters",
            ))

        # Very long functions are hard to maintain and test
        if isinstance(node, ast.FunctionDef):
            func_lines = (node.end_lineno or node.lineno) - node.lineno
            if func_lines > 100:
                issues.append(CodeIssue(
                    line=node.lineno, severity="info", category="style",
                    message=f"Function `{node.name}` is {func_lines} lines long",
                    suggestion="Consider breaking it into smaller functions",
                ))

        # Missing return type annotation on non-trivial functions
        if isinstance(node, ast.FunctionDef) and node.returns is None:
            if node.name not in ("__init__", "__str__", "__repr__"):
                body_size = len(node.body)
                if body_size > 3:
                    issues.append(CodeIssue(
                        line=node.lineno, severity="info", category="style",
                        message=f"Function `{node.name}` is missing a return type annotation",
                        suggestion=f"Add `-> ReturnType` to `def {node.name}(...)`",
                    ))

    # Check for hardcoded credentials patterns in source text
    cred_patterns = [
        (r'password\s*=\s*["\'][^"\']+["\']', "Possible hardcoded password"),
        (r'api_key\s*=\s*["\'][^"\']+["\']', "Possible hardcoded API key"),
        (r'secret\s*=\s*["\'][^"\']+["\']', "Possible hardcoded secret"),
        (r'token\s*=\s*["\'][A-Za-z0-9+/]{20,}', "Possible hardcoded token"),
    ]
    for i, line in enumerate(lines, 1):
        for pattern, msg in cred_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append(CodeIssue(
                    line=i, severity="warning", category="security",
                    message=msg,
                    suggestion="Use environment variables or a secrets manager instead",
                ))

    return issues


class AIService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def analyze_code(self, req: CodeAnalysisRequest) -> CodeAnalysisResponse:
        code = req.code
        language = req.language.lower()

        if language != "python":
            return CodeAnalysisResponse(analysis=CodeAnalysis(
                language=language,
                loc=len(code.splitlines()),
                complexity=1,
                summary=f"Static analysis only supports Python; got {language!r}.",
                score=100,
            ))

        loc = len([line for line in code.splitlines() if line.strip()])

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            issue = CodeIssue(
                line=exc.lineno or 0, severity="error", category="syntax",
                message=f"Syntax error: {exc.msg}",
                suggestion="Fix the syntax error before running this function",
            )
            return CodeAnalysisResponse(analysis=CodeAnalysis(
                language="python", loc=loc, complexity=1,
                issues=[issue], summary="Code has a syntax error.", score=0,
            ))

        complexity = _cyclomatic_complexity(tree)
        imports = _extract_imports(tree)
        third_party = [i for i in imports if i not in _STDLIB and not i.startswith("_")]
        issues = _check_issues(tree, code)

        has_main = any(
            (isinstance(n, ast.FunctionDef) and n.name == "main")
            or (
                isinstance(n, ast.If)
                and isinstance(getattr(n.test, 'comparators', [None])[0], ast.Constant)
                and getattr(n.test.comparators[0], 's', None) == "__main__"
            )
            for n in ast.walk(tree)
        )
        has_return = any(
            isinstance(n, ast.Return) and n.value is not None
            for n in ast.walk(tree)
        )

        # Score deductions based on issue severity
        score = 100
        score -= sum(
            5 if i.severity == "warning" else 10 if i.severity == "error" else 1
            for i in issues
        )
        if complexity > 10:
            score -= min(20, (complexity - 10) * 2)
        score = max(0, score)

        summary_parts = []
        if issues:
            errors = [i for i in issues if i.severity == "error"]
            warnings = [i for i in issues if i.severity == "warning"]
            if errors:
                summary_parts.append(f"{len(errors)} error(s)")
            if warnings:
                summary_parts.append(f"{len(warnings)} warning(s)")
        if complexity > 15:
            summary_parts.append(f"high complexity ({complexity})")
        if not summary_parts:
            summary_parts.append("looks clean")

        summary = f"Python · {loc} LOC · complexity {complexity} · {', '.join(summary_parts)}."

        return CodeAnalysisResponse(analysis=CodeAnalysis(
            language="python",
            loc=loc,
            complexity=complexity,
            imports=imports,
            suggested_deps=third_party,
            issues=issues,
            has_main=has_main,
            has_return=has_return,
            summary=summary,
            score=score,
        ))

    def analyze_run(self, req: RunAnalysisRequest) -> RunAnalysisResponse:
        status = req.status
        duration = req.duration
        stderr = req.stderr or ""
        stdout = req.stdout or ""
        returncode = req.returncode

        suggestions: list[str] = []

        if status == "completed":
            if duration and duration > 30:
                suggestions.append(
                    "This run took a long time -- consider adding caching "
                    "or breaking it into smaller steps"
                )
            if "MemoryError" in stdout or "MemoryError" in stderr:
                suggestions.append(
                    "Memory error detected in output -- consider processing data in chunks"
                )
            interpretation = "Function completed successfully"
            if duration:
                ms = duration * 1000
                duration_label = f"{ms:.0f}ms" if ms < 1000 else f"{duration:.2f}s"
                interpretation = f"Completed in {duration_label}"
            else:
                duration_label = "—"
        elif status == "failed":
            duration_label = f"{duration:.2f}s" if duration else "—"
            if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
                match = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr)
                if match:
                    mod = match.group(1)
                    suggestions.append(f"Install missing dependency: `pip install {mod}`")
                    suggestions.append(
                        "Consider adding this to the function's `dependencies` list"
                    )
                interpretation = "Failed due to missing dependency"
            elif "TimeoutError" in stderr or "timed out" in stderr.lower():
                suggestions.append(
                    "Increase the function timeout or optimize the hot path"
                )
                interpretation = "Function timed out"
            elif "MemoryError" in stderr:
                suggestions.append(
                    "Reduce memory usage -- process data in smaller batches"
                )
                interpretation = "Function ran out of memory"
            elif returncode and returncode != 0:
                suggestions.append("Check the stderr output for error details")
                interpretation = f"Process exited with code {returncode}"
            else:
                interpretation = "Function failed -- check stderr for details"
        else:
            duration_label = "—"
            interpretation = f"Run status: {status}"

        return RunAnalysisResponse(insight=RunInsight(
            status=status,
            duration_label=duration_label,
            interpretation=interpretation,
            suggestions=suggestions,
        ))
