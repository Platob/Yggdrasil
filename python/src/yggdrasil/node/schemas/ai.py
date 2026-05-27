from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class CodeIssue(StrictModel):
    line: int
    col: int = 0
    severity: str   # "error", "warning", "info"
    category: str   # "syntax", "style", "performance", "security", "deps"
    message: str
    suggestion: str = ""


class CodeAnalysis(StrictModel):
    language: str
    loc: int                            # lines of code
    complexity: int                     # cyclomatic complexity (branches + 1)
    imports: list[str] = Field(default_factory=list)
    suggested_deps: list[str] = Field(default_factory=list)
    issues: list[CodeIssue] = Field(default_factory=list)
    has_main: bool = False              # whether there's a main() or if __name__ == "__main__"
    has_return: bool = False
    summary: str = ""
    score: int = 100                    # 0-100 quality score


class CodeAnalysisRequest(StrictModel):
    code: str
    language: str = "python"
    function_name: str = ""


class CodeAnalysisResponse(StrictModel):
    analysis: CodeAnalysis


class RunInsight(StrictModel):
    status: str
    duration_label: str
    interpretation: str
    suggestions: list[str] = Field(default_factory=list)


class RunAnalysisRequest(StrictModel):
    status: str
    duration: float | None = None
    stdout: str | None = None
    stderr: str | None = None
    returncode: int | None = None


class RunAnalysisResponse(StrictModel):
    insight: RunInsight
