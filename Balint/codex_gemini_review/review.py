from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, cast

from pydantic import ValidationError

from .models import Category, ReviewFinding, ReviewMeta, ReviewResult, Severity

DEFAULT_CLI_COMMAND = "gemini"
DEFAULT_MAX_INPUT_CHARS = 120_000
IGNORED_DIR_NAMES = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".ipynb_checkpoints",
}
IGNORED_SUFFIXES = {
    ".pb",
    ".prof",
    ".pyc",
    ".pyo",
}
IGNORED_FILE_NAMES = {
    ".DS_Store",
}
IGNORED_PATH_PART_SEQUENCES = [
    ("plugins", "profile"),
]
SYSTEM_PROMPT = """You are an independent code reviewer for a small Python/JAX scientific codebase.

Focus on:
- logic errors
- correctness issues
- numerical or scientific mistakes
- performance issues
- maintainability problems that could lead to bugs

Ignore:
- style nitpicks unless they affect correctness, clarity, or maintainability
- security unless explicitly requested

Return JSON only. Do not wrap the response in markdown fences."""


@dataclass(frozen=True)
class RepoScope:
    repo_root: Path
    scope_root: Path
    scope_prefix: PurePosixPath | None


@dataclass(frozen=True)
class ReviewPayload:
    content: str
    reviewed_files: list[str]
    truncated: bool


class ReviewUnavailableError(RuntimeError):
    """Raised when Gemini cannot be reached or initialized."""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def get_repo_scope(cwd: Path | None = None) -> RepoScope:
    scope_root = (cwd or Path.cwd()).resolve()
    repo_root = Path(_git_output(["rev-parse", "--show-toplevel"], cwd=scope_root).strip()).resolve()
    try:
        relative = scope_root.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError(f"{scope_root} is not inside git repo {repo_root}") from exc
    scope_prefix = None if relative == Path(".") else PurePosixPath(relative.as_posix())
    return RepoScope(repo_root=repo_root, scope_root=scope_root, scope_prefix=scope_prefix)


def collect_review_payload(
    cwd: Path | None = None,
    path_filters: list[str] | None = None,
    max_input_chars: int | None = None,
) -> ReviewPayload:
    scope = get_repo_scope(cwd)
    limit = max_input_chars if max_input_chars is not None else get_default_max_input_chars()
    pathspecs = _build_pathspecs(scope, path_filters)

    unstaged_files = _filter_reviewable_paths(
        _git_lines(["diff", "--name-only", "--", *pathspecs], cwd=scope.repo_root)
    )
    staged_files = _filter_reviewable_paths(
        _git_lines(["diff", "--cached", "--name-only", "--", *pathspecs], cwd=scope.repo_root)
    )
    untracked_files = _filter_reviewable_paths(
        _git_lines(["ls-files", "--others", "--exclude-standard", "--", *pathspecs], cwd=scope.repo_root)
    )

    unstaged_diff = _git_diff_text(scope.repo_root, staged=False, files=unstaged_files)
    staged_diff = _git_diff_text(scope.repo_root, staged=True, files=staged_files)

    sections: list[str] = []
    reviewed_files = sorted(
        {
            *unstaged_files,
            *staged_files,
            *untracked_files,
        }
    )

    if unstaged_diff:
        sections.append("### Unstaged tracked changes\n```diff\n" + unstaged_diff.rstrip() + "\n```")
    if staged_diff:
        sections.append("### Staged tracked changes\n```diff\n" + staged_diff.rstrip() + "\n```")
    if untracked_files:
        sections.append(_render_untracked_files(scope.repo_root, untracked_files))

    content = "\n\n".join(section for section in sections if section).strip()
    if not content:
        return ReviewPayload(content="", reviewed_files=[], truncated=False)

    truncated = len(content) > limit
    if truncated:
        content = content[: limit - len("\n\n[TRUNCATED]\n")] + "\n\n[TRUNCATED]\n"

    return ReviewPayload(content=content, reviewed_files=reviewed_files, truncated=truncated)


def build_review_prompt(payload: ReviewPayload, review_focus: str | None = None) -> str:
    requested_focus = review_focus or "logic, correctness, numerical issues, performance, maintainability"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Use this exact JSON shape:\n"
        "{\n"
        '  "summary": "string",\n'
        '  "findings": [\n'
        "    {\n"
        '      "severity": "high|medium|low",\n'
        '      "category": "logic|correctness|numerical|performance|maintainability",\n'
        '      "file": "string|null",\n'
        '      "line_hint": "string|null",\n'
        '      "issue": "string",\n'
        '      "why_it_matters": "string",\n'
        '      "suggested_fix": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Requested focus: {requested_focus}\n"
        f"Reviewed files: {', '.join(payload.reviewed_files) if payload.reviewed_files else 'none'}\n\n"
        "Changes to review:\n"
        f"{payload.content}"
    )


def review_current_diff(
    cwd: Path | None = None,
    review_focus: str | None = None,
    max_input_chars: int | None = None,
    path_filters: list[str] | None = None,
    command_runner: CommandRunner | None = None,
) -> ReviewResult:
    payload = collect_review_payload(cwd=cwd, path_filters=path_filters, max_input_chars=max_input_chars)
    if not payload.content:
        return ReviewResult(
            status="no_changes",
            summary="No tracked or untracked changes were found in the current workspace scope.",
            findings=[],
            meta=ReviewMeta(reviewed_files=[], truncated=False),
        )

    prompt = build_review_prompt(payload=payload, review_focus=review_focus)
    try:
        text = generate_review_text(
            prompt=prompt,
            cwd=(cwd or Path.cwd()),
            command_runner=command_runner,
        )
    except ReviewUnavailableError as exc:
        return ReviewResult(
            status="unavailable",
            summary=str(exc),
            findings=[],
            meta=ReviewMeta(reviewed_files=payload.reviewed_files, truncated=payload.truncated),
        )

    parsed = parse_review_response(
        text=text,
        reviewed_files=payload.reviewed_files,
        truncated=payload.truncated,
    )
    return parsed


def generate_review_text(
    prompt: str,
    cwd: Path | None = None,
    command_runner: CommandRunner | None = None,
) -> str:
    runner = command_runner or subprocess.run
    command = _build_cli_command()
    try:
        process = runner(
            command,
            cwd=(cwd or Path.cwd()),
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ReviewUnavailableError(
            "Gemini review unavailable: Gemini CLI is not installed or not on PATH."
        ) from exc
    except Exception as exc:  # pragma: no cover - exercised via mocks
        raise ReviewUnavailableError(f"Gemini review unavailable: {exc}") from exc

    payload = _parse_cli_output(process.stdout)
    if process.returncode != 0:
        error_message = _extract_cli_error(payload) or process.stderr.strip()
        raise ReviewUnavailableError(f"Gemini review unavailable: {error_message or 'CLI failed.'}")

    if payload is None:
        raise ReviewUnavailableError("Gemini review unavailable: CLI returned malformed JSON.")

    error_message = _extract_cli_error(payload)
    if error_message:
        raise ReviewUnavailableError(f"Gemini review unavailable: {error_message}")

    response = payload.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ReviewUnavailableError("Gemini review unavailable: empty response from CLI.")
    return response


def parse_review_response(text: str, reviewed_files: list[str], truncated: bool) -> ReviewResult:
    raw = _extract_json_object(text)
    if raw is None:
        return ReviewResult(
            status="invalid_response",
            summary="Gemini returned a non-JSON review payload.",
            findings=[],
            meta=ReviewMeta(reviewed_files=reviewed_files, truncated=truncated),
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ReviewResult(
            status="invalid_response",
            summary="Gemini returned malformed JSON.",
            findings=[],
            meta=ReviewMeta(reviewed_files=reviewed_files, truncated=truncated),
        )

    try:
        summary = str(data["summary"]).strip()
        raw_findings = data.get("findings", [])
        findings = [_normalize_finding(item) for item in raw_findings]
        return ReviewResult(
            status="ok",
            summary=summary,
            findings=findings,
            meta=ReviewMeta(reviewed_files=reviewed_files, truncated=truncated),
        )
    except (KeyError, TypeError, ValidationError, ValueError):
        return ReviewResult(
            status="invalid_response",
            summary="Gemini returned JSON that did not match the expected review schema.",
            findings=[],
            meta=ReviewMeta(reviewed_files=reviewed_files, truncated=truncated),
        )


def get_default_max_input_chars() -> int:
    value = os.getenv("GEMINI_REVIEW_MAX_INPUT_CHARS")
    if value is None:
        return DEFAULT_MAX_INPUT_CHARS
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_MAX_INPUT_CHARS
    return max(parsed, 1_000)


def _build_cli_command() -> list[str]:
    command = os.getenv("GEMINI_CLI_COMMAND", DEFAULT_CLI_COMMAND).strip() or DEFAULT_CLI_COMMAND
    args = [command, "--output-format", "json"]
    model = os.getenv("GEMINI_MODEL")
    if model:
        args.extend(["--model", model])
    return args


def _normalize_finding(item: Any) -> ReviewFinding:
    if not isinstance(item, dict):
        raise ValueError("Finding must be an object.")

    severity = _normalize_severity(item.get("severity"))
    category = _normalize_category(item.get("category"))
    file_value = item.get("file")
    line_hint = item.get("line_hint")
    return ReviewFinding(
        severity=severity,
        category=category,
        file=None if file_value in (None, "") else str(file_value),
        line_hint=None if line_hint in (None, "") else str(line_hint),
        issue=str(item["issue"]).strip(),
        why_it_matters=str(item["why_it_matters"]).strip(),
        suggested_fix=str(item["suggested_fix"]).strip(),
    )


def _normalize_enum(value: Any, allowed: set[str]) -> str:
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        raise ValueError(f"Unexpected enum value: {value}")
    return normalized


def _normalize_severity(value: Any) -> Severity:
    return cast(Severity, _normalize_enum(value, {"high", "medium", "low"}))


def _normalize_category(value: Any) -> Category:
    return cast(
        Category,
        _normalize_enum(
            value,
            {"logic", "correctness", "numerical", "performance", "maintainability"},
        ),
    )


def _extract_json_object(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return stripped[start : end + 1]


def _parse_cli_output(stdout: str) -> dict[str, Any] | None:
    raw = stdout.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_cli_error(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    error = payload.get("error")
    if error is None:
        return None
    if isinstance(error, str):
        return error.strip()
    if isinstance(error, dict):
        for key in ("message", "details", "code"):
            value = error.get(key)
            if value:
                return str(value).strip()
        return json.dumps(error)
    return str(error).strip()


def _render_untracked_files(repo_root: Path, files: Iterable[str]) -> str:
    blocks: list[str] = ["### Untracked files"]
    for rel_path in files:
        file_path = repo_root / Path(rel_path)
        if file_path.is_file():
            content = file_path.read_text(encoding="utf-8", errors="replace")
            blocks.append(f"File: {rel_path}\n```text\n{content.rstrip()}\n```")
    return "\n\n".join(blocks)


def _build_pathspecs(scope: RepoScope, path_filters: list[str] | None) -> list[str]:
    filters = path_filters or ["."]
    pathspecs: list[str] = []
    for item in filters:
        normalized = PurePosixPath(str(item).replace("\\", "/"))
        if str(normalized) in {"", "."}:
            if scope.scope_prefix is None:
                pathspecs.append(".")
            else:
                pathspecs.append(scope.scope_prefix.as_posix())
            continue
        if scope.scope_prefix is None:
            pathspecs.append(normalized.as_posix())
        else:
            pathspecs.append((scope.scope_prefix / normalized).as_posix())
    return pathspecs


def _git_lines(args: list[str], cwd: Path) -> list[str]:
    output = _git_output(args=args, cwd=cwd, allow_empty=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _git_diff_text(repo_root: Path, staged: bool, files: list[str]) -> str:
    if not files:
        return ""
    args = ["diff"]
    if staged:
        args.append("--cached")
    args.extend(["--no-ext-diff", "--", *files])
    return _git_output(args, cwd=repo_root, allow_empty=True)


def _filter_reviewable_paths(paths: Iterable[str]) -> list[str]:
    return [path for path in paths if _is_reviewable_path(path)]


def _is_reviewable_path(path: str) -> bool:
    pure_path = PurePosixPath(path.replace("\\", "/"))
    if any(part in IGNORED_DIR_NAMES for part in pure_path.parts):
        return False
    lower_parts = tuple(part.lower() for part in pure_path.parts)
    for sequence in IGNORED_PATH_PART_SEQUENCES:
        sequence_length = len(sequence)
        for index in range(len(lower_parts) - sequence_length + 1):
            if lower_parts[index : index + sequence_length] == sequence:
                return False
    if pure_path.name in IGNORED_FILE_NAMES:
        return False
    if pure_path.name.endswith(".trace.json.gz"):
        return False
    if any(pure_path.name.endswith(suffix) for suffix in IGNORED_SUFFIXES):
        return False
    return True


def _git_output(args: list[str], cwd: Path, allow_empty: bool = False) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.strip()
        if allow_empty and "no changes added to commit" in stderr.lower():
            return ""
        raise RuntimeError(stderr or f"git {' '.join(args)} failed with exit code {process.returncode}")
    return process.stdout
