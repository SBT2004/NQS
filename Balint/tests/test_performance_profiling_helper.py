from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demos.performance_profiling_helper import (  # noqa: E402
    cprofile_groundstate_demo_run,
    run_groundstate_search_model,
    summarize_perfetto_trace,
)


def test_summarize_perfetto_trace_reads_trace_and_classifies_compile_phase(tmp_path: Path) -> None:
    run_dir = tmp_path / "profile_run"
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(parents=True)
    trace_file = trace_dir / "trace.trace.json.gz"
    payload = {
        "traceEvents": [
            {"name": "xla_compile", "cat": "compiler", "dur": 8000},
            {"name": "gemm_kernel", "cat": "kernel", "dur": 2000},
            {"name": "instant", "ph": "i"},
        ]
    }
    with gzip.open(trace_file, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    summary = summarize_perfetto_trace(
        run_dir,
        metadata={"run_name": "demo", "model_name": "CNN"},
    )
    top_events = cast(list[dict[str, Any]], summary["top_events_by_duration_ms"])
    summary_metadata = cast(dict[str, Any], summary["metadata"])

    assert summary["dominant_phase"] == "compile_heavy"
    assert summary["trace_files"] == [str(trace_file)]
    assert summary["event_count"] == 3
    assert summary["duration_event_count"] == 2
    assert top_events[0]["name"] == "xla_compile"
    assert summary_metadata["run_name"] == "demo"


def test_cprofile_groundstate_demo_run_profiles_one_model_by_default() -> None:
    result = cprofile_groundstate_demo_run(model_name="RBM", top_n=5)

    result_payload = cast(dict[str, Any], result["result"])
    profiled = cast(list[dict[str, Any]], result_payload["results"])
    assert len(profiled) == 1
    assert profiled[0]["model"] == "RBM"
    assert cast(list[dict[str, Any]], result["rows"])
    assert "backend_compile_and_load" in cast(str, result["summary"])


@pytest.mark.parametrize(
    ("model_name", "expected_steps"),
    [
        ("RBM", 48),
        ("FFNN", 160),
        ("CNN", 256),
    ],
)
def test_run_groundstate_search_model_fast_profile_converges_per_model(
    model_name: str,
    expected_steps: int,
) -> None:
    result = run_groundstate_search_model(model_name=model_name, fast_profile=True)

    profiled = cast(list[dict[str, Any]], result["results"])[0]
    assert len(profiled["trace"]) == expected_steps
    assert abs(float(profiled["delta_to_exact"])) < 0.1
