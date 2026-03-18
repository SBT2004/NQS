from __future__ import annotations

import subprocess
from pathlib import Path

from codex_gemini_review.review import (
    ReviewPayload,
    SupplementalReviewContext,
    build_review_prompt_with_context,
    collect_review_payload,
    parse_review_response,
    review_current_diff,
)


def test_collect_review_payload_includes_staged_unstaged_and_untracked(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    _write(tmp_path / "tracked.py", "print('v1')\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-m", "initial")

    _write(tmp_path / "tracked.py", "print('v2')\n")
    _git(tmp_path, "add", "tracked.py")
    _write(tmp_path / "tracked.py", "print('v3')\n")
    _write(tmp_path / "notes.txt", "hello\n")

    payload = collect_review_payload(cwd=tmp_path, max_input_chars=10_000)

    assert "### Staged tracked changes" in payload.content
    assert "### Unstaged tracked changes" in payload.content
    assert "### Untracked files" in payload.content
    assert "tracked.py" in payload.reviewed_files
    assert "notes.txt" in payload.reviewed_files
    assert payload.truncated is False


def test_review_current_diff_reports_no_changes(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    _write(tmp_path / "tracked.py", "print('v1')\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-m", "initial")

    result = review_current_diff(cwd=tmp_path)

    assert result.status == "no_changes"
    assert result.findings == []
    assert result.meta.reviewed_files == []


def test_collect_review_payload_truncates_deterministically(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    _write(tmp_path / "tracked.py", "print('v1')\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-m", "initial")

    _write(tmp_path / "tracked.py", "x = '" + ("a" * 5000) + "'\n")
    payload = collect_review_payload(cwd=tmp_path, max_input_chars=500)

    assert payload.truncated is True
    assert payload.content.endswith("[TRUNCATED]\n")
    assert len(payload.content) <= 500


def test_collect_review_payload_ignores_generated_artifacts(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    _write(tmp_path / "tracked.py", "print('v1')\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-m", "initial")

    artifact_dir = tmp_path / "__pycache__"
    artifact_dir.mkdir()
    _write(artifact_dir / "tracked.cpython-311.pyc", "junk")
    _write(tmp_path / "profile.prof", "junk")
    trace_dir = tmp_path / "plugins" / "profile" / "run1"
    trace_dir.mkdir(parents=True)
    _write(trace_dir / "trace.trace.json.gz", "junk")
    _write(tmp_path / "notes.txt", "keep me\n")

    payload = collect_review_payload(cwd=tmp_path, max_input_chars=10_000)

    assert "notes.txt" in payload.reviewed_files
    assert "__pycache__/tracked.cpython-311.pyc" not in payload.reviewed_files
    assert "profile.prof" not in payload.reviewed_files
    assert "plugins/profile/run1/trace.trace.json.gz" not in payload.reviewed_files
    assert "notes.txt" in payload.content


def test_parse_review_response_accepts_valid_json() -> None:
    result = parse_review_response(
        text=(
            '{"summary":"One issue found","findings":[{"severity":"high","category":"logic",'
            '"file":"src/nqs/driver.py","line_hint":"L12","issue":"Bug","why_it_matters":"Wrong result",'
            '"suggested_fix":"Adjust condition"}]}'
        ),
        reviewed_files=["src/nqs/driver.py"],
        truncated=False,
    )

    assert result.status == "ok"
    assert result.summary == "One issue found"
    assert result.findings[0].severity == "high"
    assert result.findings[0].category == "logic"


def test_build_review_prompt_with_context_includes_all_context_sections() -> None:
    prompt = build_review_prompt_with_context(
        payload=ReviewPayload(
            content="### Supplied uncommitted diff\n```diff\n+    return a - b\n```",
            reviewed_files=["dummy_addition.py"],
            truncated=False,
        ),
        review_focus="logic only",
        context=SupplementalReviewContext(
            task_scope="Review only the arithmetic helper change.",
            baseline_context="This path is hot in profiling.",
            critical_review_findings=["Potential arithmetic regression in dummy_addition.py"],
        ),
    )

    assert "### Task scope" in prompt
    assert "Review only the arithmetic helper change." in prompt
    assert "### Relevant baseline / profiler results" in prompt
    assert "This path is hot in profiling." in prompt
    assert "### Existing critical review findings" in prompt
    assert "Potential arithmetic regression in dummy_addition.py" in prompt
    assert "### Supplied uncommitted diff" in prompt


def test_review_current_diff_accepts_supplied_diff_and_context_from_non_git_directory(tmp_path: Path) -> None:
    supplied_diff = (
        "diff --git a/dummy_addition.py b/dummy_addition.py\n"
        "--- a/dummy_addition.py\n"
        "+++ b/dummy_addition.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-    return a + b\n"
        "+    return a - b\n"
    )

    def fake_runner(*_: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        prompt = str(kwargs["input"])
        assert "### Task scope" in prompt
        assert "Check only the addition helper." in prompt
        assert "### Relevant baseline / profiler results" in prompt
        assert "hot path" in prompt
        assert "### Existing critical review findings" in prompt
        assert "Suspicious arithmetic regression" in prompt
        assert "return a - b" in prompt
        return subprocess.CompletedProcess(
            args=["gemini"],
            returncode=0,
            stdout=(
                '{"response":"{\\"summary\\":\\"Found bug\\",\\"findings\\":[{\\"severity\\":\\"high\\",'
                '\\"category\\":\\"logic\\",\\"file\\":\\"dummy_addition.py\\",\\"line_hint\\":\\"return a - b\\",'
                '\\"issue\\":\\"Addition subtracts instead of adds\\",\\"why_it_matters\\":\\"Wrong arithmetic\\",'
                '\\"suggested_fix\\":\\"Use a + b\\"}]}"}'
            ),
            stderr="",
        )

    result = review_current_diff(
        cwd=tmp_path,
        task_scope="Check only the addition helper.",
        baseline_context="This is a hot path in profiling.",
        uncommitted_diff=supplied_diff,
        critical_review_findings=["Suspicious arithmetic regression"],
        command_runner=fake_runner,
    )

    assert result.status == "ok"
    assert result.meta.reviewed_files == ["dummy_addition.py"]
    assert result.findings[0].line_hint == "return a - b"


def test_parse_review_response_rejects_invalid_json() -> None:
    result = parse_review_response(
        text="not json",
        reviewed_files=["src/nqs/driver.py"],
        truncated=False,
    )

    assert result.status == "invalid_response"
    assert result.findings == []


def test_review_current_diff_returns_invalid_response_on_bad_schema(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    _write(tmp_path / "tracked.py", "print('v1')\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-m", "initial")
    _write(tmp_path / "tracked.py", "print('v2')\n")

    def fake_runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["gemini"],
            returncode=0,
            stdout='{"response":"{\\"summary\\":\\"x\\",\\"findings\\":[{\\"severity\\":\\"urgent\\"}]}"}',
            stderr="",
        )

    result = review_current_diff(cwd=tmp_path, command_runner=fake_runner)

    assert result.status == "invalid_response"
    assert result.meta.reviewed_files == ["tracked.py"]


def test_review_current_diff_returns_unavailable_when_cli_missing(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    _write(tmp_path / "tracked.py", "print('v1')\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-m", "initial")
    _write(tmp_path / "tracked.py", "print('v2')\n")

    def missing_runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("gemini not found")

    result = review_current_diff(cwd=tmp_path, command_runner=missing_runner)

    assert result.status == "unavailable"
    assert "not installed or not on PATH" in result.summary


def _git(cwd: Path, *args: str) -> None:
    process = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
