from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demos.performance_profiling_helper import summarize_perfetto_trace  # noqa: E402


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

    assert summary["dominant_phase"] == "compile_heavy"
    assert summary["trace_files"] == [str(trace_file)]
    assert summary["event_count"] == 3
    assert summary["duration_event_count"] == 2
    assert summary["top_events_by_duration_ms"][0]["name"] == "xla_compile"
    assert summary["metadata"]["run_name"] == "demo"
