from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .models import ReviewResult
from .review import review_current_diff

mcp = FastMCP(
    name="gemini-review",
    instructions=(
        "Provides a Gemini-based second-opinion review for the current git diff in the "
        "workspace subtree where the server is launched."
    ),
)


@mcp.tool(
    name="gemini_review_diff",
    description=(
        "Review the current workspace git diff with Gemini and return normalized structured findings."
    ),
    structured_output=True,
)
def gemini_review_diff(
    review_focus: Annotated[str | None, Field(default=None)],
    max_input_chars: Annotated[int | None, Field(default=None, ge=1000)],
    path_filters: Annotated[list[str] | None, Field(default=None)],
    task_scope: Annotated[str | None, Field(default=None)],
    baseline_context: Annotated[str | None, Field(default=None)],
    uncommitted_diff: Annotated[str | None, Field(default=None)],
    critical_review_findings: Annotated[str | list[str] | None, Field(default=None)],
) -> ReviewResult:
    return review_current_diff(
        review_focus=review_focus,
        max_input_chars=max_input_chars,
        path_filters=path_filters,
        task_scope=task_scope,
        baseline_context=baseline_context,
        uncommitted_diff=uncommitted_diff,
        critical_review_findings=critical_review_findings,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
