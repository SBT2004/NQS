from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemini_review_tool.gemini_second_opinion_eval import _aggregate_metrics, _working_behaviors  # noqa: E402


def test_aggregate_metrics_groups_scenarios() -> None:
    scenario_results = [
        {"scenario_id": "a", "scenario_type": "real_cli", "gemini_result_status": "unavailable", "latency_ms": 10.0},
        {"scenario_id": "b", "scenario_type": "static_check", "gemini_result_status": "no_changes", "latency_ms": 5.0},
        {"scenario_id": "c", "scenario_type": "simulated_failure", "gemini_result_status": "invalid_response", "latency_ms": 8.0},
    ]
    result = _aggregate_metrics(scenario_results)
    assert result["scenario_count"] == 3
    assert result["status_counts"]["unavailable"] == 1
    assert result["real_cli_scenarios"] == ["a"]
    assert result["static_checks"] == ["b"]
    assert result["simulated_failures"] == ["c"]


def test_working_behaviors_reports_expected_successes() -> None:
    scenario_results = [
        {
            "scenario_id": "no_changes",
            "gemini_result_status": "no_changes",
            "payload_stats": {"reviewed_files": [], "truncated": False, "reviewed_file_count": 0},
        },
        {
            "scenario_id": "path_filtered_subset",
            "gemini_result_status": "unavailable",
            "payload_stats": {"reviewed_files": ["keep.py"], "truncated": False, "reviewed_file_count": 1},
        },
        {
            "scenario_id": "large_payload",
            "gemini_result_status": "unavailable",
            "payload_stats": {"reviewed_files": ["tracked.py"], "truncated": True, "reviewed_file_count": 1},
        },
        {
            "scenario_id": "untracked_file",
            "gemini_result_status": "unavailable",
            "payload_stats": {"reviewed_files": ["notes.txt"], "truncated": False, "reviewed_file_count": 1},
        },
    ]
    result = _working_behaviors(scenario_results)
    assert any("no_changes" in entry for entry in result)
    assert any("Path filtering" in entry for entry in result)
    assert any("Payload truncation" in entry for entry in result)
