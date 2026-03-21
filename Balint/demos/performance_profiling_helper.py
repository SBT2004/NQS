"""Helpers for profiling the 1D TFIM ground-state search demo.

This file is intentionally demo-oriented. It keeps the profiling setup in one
place so notebooks can focus on showing the workflow instead of repeating
boilerplate.
"""

from __future__ import annotations

import cProfile
import gzip
import io
import json
import pstats
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from demos.ising1d_ed_vs_vmc_helper import (  # noqa: E402
    build_ising_operator,
    exact_ground_state_energy_netket,
    run_demo,
    run_model_demo,
)
from src.nqs import CNN, FFNN, RBM  # noqa: E402


@dataclass(frozen=True)
class ProfileExperiment:
    model_factory: Callable[[int], Any]
    learning_rate: float
    seed: int
    n_samples: int
    n_discard_per_chain: int
    n_chains: int
    full_iterations: int
    profile_iterations: int


PROFILE_EXPERIMENTS: dict[str, ProfileExperiment] = {
    "RBM": ProfileExperiment(
        model_factory=lambda _length: RBM(alpha=2),
        learning_rate=2e-2,
        seed=0,
        n_samples=256,
        n_discard_per_chain=32,
        n_chains=16,
        full_iterations=128,
        profile_iterations=48,
    ),
    "FFNN": ProfileExperiment(
        model_factory=lambda _length: FFNN(hidden_dims=(32, 16)),
        learning_rate=1e-2,
        seed=1,
        n_samples=256,
        n_discard_per_chain=32,
        n_chains=16,
        full_iterations=160,
        profile_iterations=160,
    ),
    "CNN": ProfileExperiment(
        model_factory=lambda length: CNN(spatial_shape=(length, 1), channels=(16, 8), kernel_size=(5, 1)),
        learning_rate=5e-3,
        seed=2,
        n_samples=256,
        n_discard_per_chain=32,
        n_chains=16,
        full_iterations=256,
        profile_iterations=256,
    ),
}


def _format_function_name(function_key: tuple[str, int, str]) -> str:
    """Create a short readable label for one cProfile function entry."""

    filename, line_number, function_name = function_key
    return f"{Path(filename).name}:{line_number} ({function_name})"


def _build_profile_table(stats: pstats.Stats, top_n: int) -> tuple[list[dict[str, Any]], str]:
    """Return structured rows plus a padded plain-text table."""

    function_list = cast(list[tuple[str, int, str]], getattr(stats, "fcn_list"))
    stats_map = cast(
        dict[tuple[str, int, str], tuple[int, int, float, float, object]],
        getattr(stats, "stats"),
    )
    rows: list[dict[str, Any]] = []
    for rank, function_key in enumerate(function_list[:top_n], start=1):
        primitive_calls, total_calls, total_time, cumulative_time, _ = stats_map[function_key]
        per_call_ms = 1000.0 * total_time / max(total_calls, 1)
        row = {
            "rank": rank,
            "function": _format_function_name(function_key),
            "primitive_calls": primitive_calls,
            "total_calls": total_calls,
            "total_time_s": total_time,
            "cumulative_time_s": cumulative_time,
            "per_call_ms": per_call_ms,
        }
        rows.append(row)

    rank_width = max(len("#"), *(len(str(row["rank"])) for row in rows))
    function_width = max(len("Function"), *(len(str(row["function"])) for row in rows))
    calls_width = max(len("Calls"), *(len(str(row["total_calls"])) for row in rows))
    total_width = max(len("Total s"), *(len(f"{row['total_time_s']:.4f}") for row in rows))
    cumulative_width = max(len("Cum. s"), *(len(f"{row['cumulative_time_s']:.4f}") for row in rows))
    per_call_width = max(len("ms/call"), *(len(f"{row['per_call_ms']:.3f}") for row in rows))

    header = (
        f"{'#':>{rank_width}}  "
        f"{'Function':<{function_width}}  "
        f"{'Calls':>{calls_width}}  "
        f"{'Total s':>{total_width}}  "
        f"{'Cum. s':>{cumulative_width}}  "
        f"{'ms/call':>{per_call_width}}"
    )
    separator = (
        f"{'-' * rank_width}  "
        f"{'-' * function_width}  "
        f"{'-' * calls_width}  "
        f"{'-' * total_width}  "
        f"{'-' * cumulative_width}  "
        f"{'-' * per_call_width}"
    )
    body = "\n".join(
        (
            f"{row['rank']:>{rank_width}}  "
            f"{str(row['function']):<{function_width}}  "
            f"{row['total_calls']:>{calls_width}}  "
            f"{row['total_time_s']:.4f}".rjust(total_width) + "  "
            f"{row['cumulative_time_s']:.4f}".rjust(cumulative_width) + "  "
            f"{row['per_call_ms']:.3f}".rjust(per_call_width)
        )
        for row in rows
    )
    table = header if not body else f"{header}\n{separator}\n{body}"
    return rows, table


def run_groundstate_search_demo(
    length: int = 5,
    transverse_field: float = 1.0,
) -> dict[str, object]:
    """Run the full 1D TFIM demo used elsewhere in the repository."""

    return run_demo(length=length, transverse_field=transverse_field)


def _run_profile_model(
    model_name: str,
    *,
    length: int,
    transverse_field: float,
    n_iter: int,
    eval_samples: int,
    eval_repeats: int,
) -> dict[str, object]:
    """Run one model with an explicit budget override for profiling."""

    experiment = PROFILE_EXPERIMENTS[model_name]
    model = experiment.model_factory(length)
    _, operator = build_ising_operator(length, transverse_field)
    exact_energy = exact_ground_state_energy_netket(operator)
    energy_trace, final_energy = run_model_demo(
        model_name=model_name,
        model=model,
        operator=operator,
        length=length,
        seed=experiment.seed,
        learning_rate=experiment.learning_rate,
        n_samples=experiment.n_samples,
        n_discard_per_chain=experiment.n_discard_per_chain,
        n_chains=experiment.n_chains,
        n_iter=n_iter,
        eval_samples=eval_samples,
        eval_repeats=eval_repeats,
    )
    return {
        "length": length,
        "transverse_field": transverse_field,
        "exact_energy": exact_energy,
        "results": [
            {
                "model": model_name,
                "trace": energy_trace,
                "final_energy": final_energy,
                "delta_to_exact": final_energy - exact_energy,
            }
        ],
    }


def run_groundstate_search_model(
    model_name: str = "CNN",
    *,
    length: int = 5,
    transverse_field: float = 1.0,
    fast_profile: bool = True,
) -> dict[str, object]:
    """Run one model from the ground-state demo, optionally with a shorter budget."""

    if model_name not in PROFILE_EXPERIMENTS:
        raise ValueError(f"Unknown model_name {model_name!r}. Expected one of {sorted(PROFILE_EXPERIMENTS)}.")

    experiment = PROFILE_EXPERIMENTS[model_name]
    return _run_profile_model(
        model_name=model_name,
        length=length,
        transverse_field=transverse_field,
        n_iter=experiment.profile_iterations if fast_profile else experiment.full_iterations,
        eval_samples=256 if fast_profile else 4096,
        eval_repeats=1 if fast_profile else 4,
    )


def _warmup_trace_target(
    model_name: str | None,
    *,
    length: int,
    transverse_field: float,
    fast_profile: bool,
) -> None:
    """Warm up JAX/NetKet compilation before starting the trace.

    The earlier version traced the very first compile-heavy execution, which
    produced very large trace files and took a long time to finish. Warming up
    once outside the trace keeps the recorded profile focused on steady-state
    runtime bottlenecks instead of one-time compilation setup.
    """

    if model_name is None:
        run_groundstate_search_model(
            model_name="RBM",
            length=length,
            transverse_field=transverse_field,
            fast_profile=True,
        )
        return

    _run_profile_model(
        model_name=model_name,
        length=length,
        transverse_field=transverse_field,
        n_iter=1 if fast_profile else 2,
        eval_samples=64,
        eval_repeats=1,
    )


def cprofile_groundstate_demo_run(
    *,
    model_name: str | None = "RBM",
    length: int = 5,
    transverse_field: float = 1.0,
    sort_by: str = "tottime",
    top_n: int = 25,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Profile the ground-state search demo with cProfile.

    The returned dictionary contains:
    - `result`: the original demo result
    - `table`: padded human-readable summary
    - `summary`: raw `pstats` text for logs/AI inspection
    - `rows`: structured top-entry data
    """

    profiler = cProfile.Profile()
    profiler.enable()
    if model_name is None:
        result = run_groundstate_search_demo(length=length, transverse_field=transverse_field)
    else:
        run_groundstate_search_model(
            model_name=model_name,
            length=length,
            transverse_field=transverse_field,
            fast_profile=True,
        )
        result = run_groundstate_search_model(
            model_name=model_name,
            length=length,
            transverse_field=transverse_field,
            fast_profile=True,
        )
    profiler.disable()

    if output_path is not None:
        profiler.dump_stats(str(output_path))

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(sort_by)
    stats.print_stats(top_n)
    rows, table = _build_profile_table(stats, top_n)
    return {
        "result": result,
        "rows": rows,
        "table": table,
        "summary": stream.getvalue(),
        "stats_path": None if output_path is None else str(output_path),
    }


def trace_groundstate_demo_with_jax_profiler(
    log_dir: str | Path,
    *,
    model_name: str | None = "CNN",
    length: int = 5,
    transverse_field: float = 1.0,
    fast_profile: bool = True,
    create_perfetto_link: bool = False,
) -> dict[str, object]:
    """Write a JAX profiler trace for the ground-state search demo.

    By default this traces one model with a shorter profiling budget so the
    report finishes quickly. Pass `model_name=None` and `fast_profile=False`
    only if you really want the full demo trace.
    """

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    import jax

    _warmup_trace_target(
        model_name=model_name,
        length=length,
        transverse_field=transverse_field,
        fast_profile=fast_profile,
    )

    jax.profiler.start_trace(
        str(log_dir),
        create_perfetto_link=create_perfetto_link,
    )
    try:
        if model_name is None:
            result = run_groundstate_search_model(
                model_name="RBM",
                length=length,
                transverse_field=transverse_field,
                fast_profile=True,
            )
        else:
            result = _run_profile_model(
                model_name=model_name,
                length=length,
                transverse_field=transverse_field,
                n_iter=PROFILE_EXPERIMENTS[model_name].profile_iterations if fast_profile else PROFILE_EXPERIMENTS[model_name].full_iterations,
                eval_samples=128 if fast_profile else 4096,
                eval_repeats=1 if fast_profile else 4,
            )
    finally:
        jax.profiler.stop_trace()
    return result


def tensorboard_command(log_dir: str | Path) -> str:
    """Return the command needed to open the JAX trace in TensorBoard."""

    return f"python -m tensorboard.main --logdir {Path(log_dir)}"


def perfetto_trace_files(log_dir: str | Path) -> list[str]:
    """Return the generated Perfetto-compatible trace files.

    On this Windows setup the TensorBoard profile plugin can fail to load its
    native extension even when the trace itself was generated correctly. The
    `.trace.json.gz` files can still be opened directly in Perfetto at
    https://ui.perfetto.dev, which is the most reliable viewer path here.
    """

    log_dir = Path(log_dir)
    return [str(path) for path in sorted(log_dir.rglob("*.trace.json.gz"))]


def _is_compile_event(name: str, category: str) -> bool:
    label = f"{name} {category}".lower()
    return "compile" in label or "xla" in label


def _event_duration_us(event: dict[str, object]) -> float:
    value = event.get("dur", 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def summarize_perfetto_trace(
    log_dir: str | Path,
    *,
    metadata: dict[str, object] | None = None,
    top_n: int = 5,
) -> dict[str, object]:
    """Summarize Perfetto-compatible trace files produced by JAX profiling."""

    trace_paths = perfetto_trace_files(log_dir)
    trace_events: list[dict[str, object]] = []
    for trace_path in trace_paths:
        with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
            payload = cast(dict[str, object], json.load(handle))
        trace_events.extend(cast(list[dict[str, object]], payload.get("traceEvents", [])))

    duration_events: list[dict[str, object]] = []
    compile_duration_us = 0.0
    runtime_duration_us = 0.0
    for event in trace_events:
        duration_us = _event_duration_us(event)
        if duration_us <= 0:
            continue
        duration_events.append(event)
        name = str(event.get("name", ""))
        category = str(event.get("cat", ""))
        if _is_compile_event(name, category):
            compile_duration_us += duration_us
        else:
            runtime_duration_us += duration_us

    top_events = sorted(duration_events, key=_event_duration_us, reverse=True)[:top_n]
    return {
        "trace_files": trace_paths,
        "event_count": len(trace_events),
        "duration_event_count": len(duration_events),
        "top_events_by_duration_ms": [
            {
                "name": str(event.get("name", "")),
                "category": str(event.get("cat", "")),
                "duration_ms": _event_duration_us(event) / 1000.0,
            }
            for event in top_events
        ],
        "dominant_phase": "compile_heavy" if compile_duration_us >= runtime_duration_us else "runtime_heavy",
        "metadata": {} if metadata is None else metadata,
    }


def scalene_command(
    target: str = "Balint/demos/performance_profiling_helper.py",
    *,
    length: int = 5,
    transverse_field: float = 1.0,
) -> str:
    """Return a ready-to-run Scalene command for the ground-state demo path.

    Scalene is a CLI profiler, so the notebook usually just prints the command
    to run in a terminal.
    """

    return (
        f"python -m scalene {target} -- "
        f"--length {length} --transverse-field {transverse_field}"
    )


if __name__ == "__main__":
    length = 5
    transverse_field = 1.0
    model_name: str | None = "RBM"
    for index, argument in enumerate(sys.argv):
        if argument == "--length" and index + 1 < len(sys.argv):
            length = int(sys.argv[index + 1])
        if argument == "--transverse-field" and index + 1 < len(sys.argv):
            transverse_field = float(sys.argv[index + 1])
        if argument == "--model-name" and index + 1 < len(sys.argv):
            model_name = sys.argv[index + 1]
        if argument == "--all-models":
            model_name = None
    result = cprofile_groundstate_demo_run(
        model_name=model_name,
        length=length,
        transverse_field=transverse_field,
        top_n=20,
    )
    print(result["table"])
    print()
    print(result["summary"])
