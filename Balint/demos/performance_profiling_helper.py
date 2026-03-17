"""Helpers for profiling the 1D TFIM ground-state search demo.

This file is intentionally demo-oriented. It keeps the profiling setup in one
place so notebooks can focus on showing the workflow instead of repeating
boilerplate.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from demos.ising1d_ed_vs_vmc_helper import (  # noqa: E402
    build_ising_operator,
    exact_ground_state_energy,
    run_demo,
    run_model_demo,
)
from nqs import CNN, FFNN, RBM  # noqa: E402


PROFILE_EXPERIMENTS: dict[str, tuple[Any, float, int, int, int, int, int]] = {
    "RBM": (RBM(alpha=2), 2e-2, 0, 256, 32, 16, 128),
    "FFNN": (FFNN(hidden_dims=(32, 16)), 1e-2, 1, 256, 32, 16, 128),
    "CNN": (CNN(spatial_shape=(5, 1), channels=(16, 8), kernel_size=(5, 1)), 5e-3, 2, 256, 32, 16, 256),
}

FAST_PROFILE_ITERATIONS = {
    "RBM": 8,
    "FFNN": 8,
    "CNN": 12,
}

TRACE_PROFILE_ITERATIONS = {
    "RBM": 2,
    "FFNN": 2,
    "CNN": 3,
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

    model, learning_rate, seed, n_samples, n_discard_per_chain, n_chains, _ = PROFILE_EXPERIMENTS[model_name]
    _, operator = build_ising_operator(length, transverse_field)
    exact_energy = exact_ground_state_energy(operator)
    energy_trace, final_energy = run_model_demo(
        model_name=model_name,
        model=model,
        operator=operator,
        length=length,
        seed=seed,
        learning_rate=learning_rate,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
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

    _, _, _, _, _, _, n_iter = PROFILE_EXPERIMENTS[model_name]
    return _run_profile_model(
        model_name=model_name,
        length=length,
        transverse_field=transverse_field,
        n_iter=FAST_PROFILE_ITERATIONS[model_name] if fast_profile else n_iter,
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
    result = run_groundstate_search_demo(length=length, transverse_field=transverse_field)
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
                n_iter=TRACE_PROFILE_ITERATIONS[model_name] if fast_profile else PROFILE_EXPERIMENTS[model_name][6],
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
    for index, argument in enumerate(sys.argv):
        if argument == "--length" and index + 1 < len(sys.argv):
            length = int(sys.argv[index + 1])
        if argument == "--transverse-field" and index + 1 < len(sys.argv):
            transverse_field = float(sys.argv[index + 1])
    result = cprofile_groundstate_demo_run(
        length=length,
        transverse_field=transverse_field,
        top_n=20,
    )
    print(result["table"])
    print()
    print(result["summary"])
