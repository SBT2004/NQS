from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codex_gemini_review.review import (  # noqa: E402
    DEFAULT_CLI_COMMAND,
    ReviewPayload,
    collect_review_payload,
    review_current_diff,
)

REPORTS_ROOT = PROJECT_ROOT / "gemini_review_reports"


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    scenario_type: str
    diff_shape: str
    path_filters: str | list[str] | None
    use_real_cli: bool
    max_input_chars: int | None = None
    runner_factory: Callable[[], Callable[..., subprocess.CompletedProcess[str]] | None] | None = None


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _init_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    _git(repo_dir, "init")
    _git(repo_dir, "config", "user.email", "test@example.com")
    _git(repo_dir, "config", "user.name", "Test User")
    _write(repo_dir / "tracked.py", "print('v1')\n")
    _write(repo_dir / "keep.py", "print('keep-v1')\n")
    _write(repo_dir / "skip.py", "print('skip-v1')\n")
    _git(repo_dir, "add", "tracked.py", "keep.py", "skip.py")
    _git(repo_dir, "commit", "-m", "initial")


def _apply_scenario(repo_dir: Path, scenario_id: str) -> None:
    if scenario_id == "no_changes":
        return
    if scenario_id == "small_tracked_diff":
        _write(repo_dir / "tracked.py", "print('v2-small')\n")
        return
    if scenario_id == "staged_plus_unstaged":
        _write(repo_dir / "tracked.py", "print('v2-staged')\n")
        _git(repo_dir, "add", "tracked.py")
        _write(repo_dir / "tracked.py", "print('v3-unstaged')\n")
        return
    if scenario_id == "untracked_file":
        _write(repo_dir / "notes.txt", "new untracked notes\n")
        return
    if scenario_id == "path_filtered_subset":
        _write(repo_dir / "keep.py", "print('keep-v2')\n")
        _write(repo_dir / "skip.py", "print('skip-v2')\n")
        return
    if scenario_id == "large_payload":
        _write(repo_dir / "tracked.py", "blob = '" + ("A" * 20_000) + "'\n")
        return
    if scenario_id == "many_files":
        for index in range(80):
            _write(repo_dir / f"many_{index:03d}.py", f"print('file-{index}-v1')\n")
        _git(repo_dir, "add", ".")
        _git(repo_dir, "commit", "-m", "many files baseline")
        for index in range(80):
            _write(repo_dir / f"many_{index:03d}.py", f"print('file-{index}-v2')\n")
        return
    if scenario_id == "known_bad_cli_missing":
        _write(repo_dir / "tracked.py", "print('v2-bad-cli-missing')\n")
        return
    if scenario_id == "known_bad_cli_malformed":
        _write(repo_dir / "tracked.py", "print('v2-bad-cli-malformed')\n")
        return
    if scenario_id == "known_bad_cli_timeout":
        _write(repo_dir / "tracked.py", "print('v2-bad-cli-timeout')\n")
        return
    raise ValueError(f"Unknown scenario_id {scenario_id!r}")


def _missing_runner() -> Callable[..., subprocess.CompletedProcess[str]]:
    def runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("gemini not found")

    return runner


def _malformed_runner() -> Callable[..., subprocess.CompletedProcess[str]]:
    def runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["gemini"],
            returncode=0,
            stdout='{"response":"{\\"summary\\":\\"x\\",\\"findings\\":[{\\"severity\\":\\"urgent\\"}]}"}',
            stderr="",
        )

    return runner


def _timeout_runner() -> Callable[..., subprocess.CompletedProcess[str]]:
    def runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["gemini"], timeout=120)

    return runner


def _build_scenarios() -> list[ScenarioSpec]:
    return [
        ScenarioSpec("no_changes", "baseline", "clean_repo", None, use_real_cli=False),
        ScenarioSpec("small_tracked_diff", "real_cli", "single_tracked_edit", None, use_real_cli=True),
        ScenarioSpec("staged_plus_unstaged", "real_cli", "mixed_staged_unstaged", None, use_real_cli=True),
        ScenarioSpec("untracked_file", "static_check", "single_untracked_file", None, use_real_cli=False),
        ScenarioSpec("path_filtered_subset", "static_check", "filtered_subset", "keep.py", use_real_cli=False),
        ScenarioSpec("large_payload", "real_cli", "large_single_file", None, use_real_cli=True, max_input_chars=1200),
        ScenarioSpec("many_files", "static_check", "many_changed_files", None, use_real_cli=False, max_input_chars=40_000),
        ScenarioSpec(
            "known_bad_cli_missing",
            "simulated_failure",
            "single_tracked_edit",
            None,
            use_real_cli=False,
            runner_factory=_missing_runner,
        ),
        ScenarioSpec(
            "known_bad_cli_malformed",
            "simulated_failure",
            "single_tracked_edit",
            None,
            use_real_cli=False,
            runner_factory=_malformed_runner,
        ),
        ScenarioSpec(
            "known_bad_cli_timeout",
            "simulated_failure",
            "single_tracked_edit",
            None,
            use_real_cli=False,
            runner_factory=_timeout_runner,
        ),
    ]


def _payload_stats(payload: ReviewPayload) -> dict[str, Any]:
    return {
        "content_chars": len(payload.content),
        "reviewed_file_count": len(payload.reviewed_files),
        "reviewed_files": payload.reviewed_files,
        "truncated": payload.truncated,
    }


def _scenario_note(result_status: str, payload: ReviewPayload, use_real_cli: bool) -> list[str]:
    notes: list[str] = []
    if payload.truncated:
        notes.append("payload_truncated")
    if not use_real_cli:
        notes.append("cli_not_invoked")
    if result_status == "unavailable":
        notes.append("tool_unavailable")
    if result_status == "invalid_response":
        notes.append("response_schema_failed")
    if result_status == "no_changes":
        notes.append("no_changes_detected")
    return notes


def _write_text(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def _evaluate_scenario(run_dir: Path, spec: ScenarioSpec) -> dict[str, Any]:
    repo_dir = run_dir / "temp_repos" / spec.scenario_id
    _init_repo(repo_dir)
    _apply_scenario(repo_dir, spec.scenario_id)

    payload = collect_review_payload(
        cwd=repo_dir,
        path_filters=spec.path_filters,
        max_input_chars=spec.max_input_chars,
    )
    payload_path = Path(
        _write_text(run_dir / "raw_artifacts" / f"{spec.scenario_id}_payload.txt", payload.content or "[EMPTY]\n")
    )

    debug_log_path = run_dir / "raw_artifacts" / "gemini_debug.log"
    runner = None if spec.use_real_cli else (spec.runner_factory() if spec.runner_factory is not None else None)
    env_updates = {
        "GEMINI_REVIEW_DEBUG": "1",
        "GEMINI_REVIEW_LOG_PATH": str(debug_log_path),
    }

    old_values = {key: os.environ.get(key) for key in env_updates}
    for key, value in env_updates.items():
        os.environ[key] = value

    raw_response_path: str | None = None
    latency_ms: float
    try:
        started = perf_counter()
        result = review_current_diff(
            cwd=repo_dir,
            path_filters=spec.path_filters,
            max_input_chars=spec.max_input_chars,
            command_runner=runner,
        )
        latency_ms = (perf_counter() - started) * 1000.0
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

    raw_response = {
        "status": result.status,
        "summary": result.summary,
        "findings": [finding.model_dump() for finding in result.findings],
        "meta": result.meta.model_dump(),
    }
    raw_response_path = _write_text(
        run_dir / "raw_artifacts" / f"{spec.scenario_id}_response.json",
        json.dumps(raw_response, indent=2),
    )

    timed_out = "timed out" in result.summary.lower()
    response_parse_ok = result.status == "ok"
    return {
        "scenario_id": spec.scenario_id,
        "scenario_type": spec.scenario_type,
        "diff_shape": spec.diff_shape,
        "path_filters": spec.path_filters,
        "payload_stats": _payload_stats(payload),
        "gemini_result_status": result.status,
        "findings_count": len(result.findings),
        "latency_ms": round(latency_ms, 3),
        "timed_out": timed_out,
        "response_parse_ok": response_parse_ok,
        "notes": _scenario_note(result.status, payload, spec.use_real_cli),
        "result_summary": result.summary,
        "raw_payload_path": str(payload_path),
        "raw_response_path": raw_response_path,
    }


def _aggregate_metrics(scenario_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    real_latencies: list[float] = []
    all_latencies: list[float] = []
    for result in scenario_results:
        status = str(result["gemini_result_status"])
        by_status[status] = by_status.get(status, 0) + 1
        latency = float(result["latency_ms"])
        all_latencies.append(latency)
        if result["scenario_type"] == "real_cli":
            real_latencies.append(latency)
    return {
        "scenario_count": len(scenario_results),
        "status_counts": by_status,
        "average_latency_ms": round(sum(all_latencies) / len(all_latencies), 3) if all_latencies else 0.0,
        "average_real_cli_latency_ms": round(sum(real_latencies) / len(real_latencies), 3) if real_latencies else 0.0,
        "real_cli_scenarios": [result["scenario_id"] for result in scenario_results if result["scenario_type"] == "real_cli"],
        "static_checks": [result["scenario_id"] for result in scenario_results if result["scenario_type"] == "static_check"],
        "simulated_failures": [
            result["scenario_id"] for result in scenario_results if result["scenario_type"] == "simulated_failure"
        ],
    }


def _working_behaviors(scenario_results: list[dict[str, Any]]) -> list[str]:
    working: list[str] = []
    if any(result["gemini_result_status"] == "no_changes" for result in scenario_results):
        working.append("The tool correctly short-circuits with no_changes on a clean repo.")
    if any(result["scenario_id"] == "path_filtered_subset" and result["payload_stats"]["reviewed_files"] == ["keep.py"] for result in scenario_results):
        working.append("Path filtering narrows reviewed files deterministically.")
    if any(result["scenario_id"] == "large_payload" and result["payload_stats"]["truncated"] for result in scenario_results):
        working.append("Payload truncation works and is surfaced in scenario metadata.")
    if any(result["scenario_id"] == "untracked_file" and result["payload_stats"]["reviewed_file_count"] >= 1 for result in scenario_results):
        working.append("Untracked file content is included in the review payload.")
    return working


def _failure_modes(scenario_results: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    real_unavailable = [result for result in scenario_results if result["scenario_type"] == "real_cli" and result["gemini_result_status"] == "unavailable"]
    if real_unavailable:
        failures.append("Real CLI runs are currently unavailable because the local gemini command is not on PATH.")
    if any(result["scenario_id"] == "known_bad_cli_malformed" and result["gemini_result_status"] == "invalid_response" for result in scenario_results):
        failures.append("Malformed CLI JSON is caught, but it still collapses into a generic invalid_response status.")
    if any(result["scenario_id"] == "known_bad_cli_timeout" and result["timed_out"] for result in scenario_results):
        failures.append("Timeouts are surfaced as unavailable, but timeout-specific metadata is minimal.")
    return failures


def _optimization_opportunities(scenario_results: list[dict[str, Any]]) -> list[str]:
    optimizations: list[str] = []
    if any(result["scenario_type"] == "real_cli" and result["gemini_result_status"] == "unavailable" for result in scenario_results):
        optimizations.append("Add a startup preflight that checks gemini CLI availability and cached login before running reviews.")
    if any(result["payload_stats"]["truncated"] for result in scenario_results):
        optimizations.append("Improve large-diff prioritization so truncated payloads preserve the highest-signal files first.")
    if any(result["scenario_type"] == "simulated_failure" for result in scenario_results):
        optimizations.append("Differentiate malformed-response, missing-CLI, and timeout failures more explicitly in structured output.")
    return optimizations


def _recommended_changes() -> list[str]:
    return [
        "Add a dedicated preflight command that reports gemini CLI path, login state, and effective timeout before review.",
        "Add status-specific metadata for timeout and malformed-response failures.",
        "Persist per-scenario payload and response artifacts exactly as generated for easier debugging.",
        "Consider reducing review payload size earlier for many-file diffs to keep latency predictable.",
    ]


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Gemini Second-Opinion Evaluation",
        "",
        "## Overview",
        "",
        f"- Generated at: `{report['environment']['generated_at_utc']}`",
        f"- Gemini CLI command: `{report['tool_configuration']['cli_command']}`",
        f"- Gemini CLI available on PATH: `{report['environment']['gemini_cli_available']}`",
        f"- Scenario count: `{report['aggregate_metrics']['scenario_count']}`",
        "",
        "## Scenario Matrix",
        "",
        "| Scenario | Type | Status | Findings | Latency ms | Notes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in report["scenario_results"]:
        lines.append(
            f"| `{item['scenario_id']}` | `{item['scenario_type']}` | `{item['gemini_result_status']}` | "
            f"{item['findings_count']} | {item['latency_ms']} | {', '.join(item['notes']) or '-'} |"
        )
    lines.extend(["", "## What Works", ""])
    for line in report["working_behaviors"]:
        lines.append(f"- {line}")
    lines.extend(["", "## What Fails", ""])
    for line in report["failure_modes"]:
        lines.append(f"- {line}")
    lines.extend(["", "## Bottlenecks", ""])
    for line in report["optimization_opportunities"]:
        lines.append(f"- {line}")
    lines.extend(["", "## Recommended Fixes", ""])
    for line in report["recommended_changes"]:
        lines.append(f"- {line}")
    lines.append("")
    return "\n".join(lines)


def generate_report(run_name: str = "latest") -> dict[str, Any]:
    run_dir = REPORTS_ROOT / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "raw_artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "temp_repos").mkdir(parents=True, exist_ok=True)

    scenarios = _build_scenarios()
    scenario_results = [_evaluate_scenario(run_dir, spec) for spec in scenarios]

    report = {
        "environment": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_executable": sys.executable,
            "project_root": str(PROJECT_ROOT),
            "gemini_cli_available": shutil.which(DEFAULT_CLI_COMMAND) is not None,
        },
        "tool_configuration": {
            "cli_command": os.getenv("GEMINI_CLI_COMMAND", DEFAULT_CLI_COMMAND),
            "gemini_model": os.getenv("GEMINI_MODEL"),
            "default_timeout_sec": 120,
        },
        "scenario_results": scenario_results,
        "aggregate_metrics": _aggregate_metrics(scenario_results),
        "working_behaviors": _working_behaviors(scenario_results),
        "failure_modes": _failure_modes(scenario_results),
        "optimization_opportunities": _optimization_opportunities(scenario_results),
        "recommended_changes": _recommended_changes(),
        "raw_artifacts": {
            "run_dir": str(run_dir),
            "debug_log": str(run_dir / "raw_artifacts" / "gemini_debug.log"),
            "payload_files": {
                item["scenario_id"]: item["raw_payload_path"] for item in scenario_results
            },
            "response_files": {
                item["scenario_id"]: item["raw_response_path"] for item in scenario_results
            },
        },
    }

    report_json = run_dir / "report.json"
    report_md = run_dir / "report.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md.write_text(_report_markdown(report), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "report_json": str(report_json),
        "report_md": str(report_md),
        "report": report,
    }


if __name__ == "__main__":
    result = generate_report()
    print(result["report_json"])
    print(result["report_md"])
