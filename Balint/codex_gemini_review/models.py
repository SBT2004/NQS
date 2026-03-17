from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Status = Literal["ok", "no_changes", "unavailable", "invalid_response"]
Severity = Literal["high", "medium", "low"]
Category = Literal["logic", "correctness", "numerical", "performance", "maintainability"]


class ReviewFinding(BaseModel):
    severity: Severity
    category: Category
    file: str | None = None
    line_hint: str | None = None
    issue: str
    why_it_matters: str
    suggested_fix: str


class ReviewMeta(BaseModel):
    reviewed_files: list[str] = Field(default_factory=list)
    truncated: bool = False


class ReviewResult(BaseModel):
    status: Status
    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list)
    meta: ReviewMeta = Field(default_factory=ReviewMeta)
