"""Helper functions for benchmarking project-built and NetKet J1-J2 operators."""

from __future__ import annotations

import cProfile
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

import jax.numpy as jnp
import netket as nk
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nqs import RBM, SpinHilbert, SquareLattice, VMC, build_variational_state, build_vmc_driver, j1_j2  # noqa: E402


BenchmarkMode = Literal["comparison", "speed"]
DEFAULT_SIZE_PROGRESSION = ((2, 2), (2, 3), (3, 3), (3, 4), (4, 4))
DEFAULT_CHECKPOINTS = frozenset({(2, 2), (4, 4)})


def _heisenberg_matrix() -> np.ndarray:
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)


def build_j1j2_operators(
    *,
    Lx: int = 2,
    Ly: int = 2,
    J1: float = 1.0,
    J2: float = 0.5,
    pbc: bool = True,
) -> tuple[SpinHilbert, SquareLattice, nk.operator.LocalOperator, nk.operator.LocalOperator]:
    """Build the same small J1-J2 Hamiltonian through both operator paths."""

    graph = SquareLattice(Lx, Ly, pbc=pbc)
    hilbert = SpinHilbert(graph.n_nodes)
    nk_hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)

    heisenberg_matrix = _heisenberg_matrix()
    j1_edges = list(graph.iter_edges("J1", n=1))
    j2_edges = list(graph.iter_edges("J2", n=2))
    native_terms = cast(
        list[Any],
        [heisenberg_matrix * J1 for _ in j1_edges] + [heisenberg_matrix * J2 for _ in j2_edges],
    )
    native_sites = [[edge.i, edge.j] for edge in j1_edges] + [[edge.i, edge.j] for edge in j2_edges]
    netket_operator = nk.operator.LocalOperator(nk_hilbert, operators=native_terms, acting_on=native_sites)
    project_operator = j1_j2(hilbert, graph, J1=J1, J2=J2).to_netket()
    return hilbert, graph, netket_operator, project_operator


def _run_driver_with_timing(driver: VMC, n_iter: int) -> tuple[list[dict[str, object]], list[dict[str, float]]]:
    history: list[dict[str, object]] = []
    iteration_timing: list[dict[str, float]] = []
    cumulative_elapsed = 0.0

    for iteration_index in range(n_iter):
        start = time.perf_counter()
        step = driver.step()
        step_elapsed = time.perf_counter() - start
        cumulative_elapsed += step_elapsed
        history.append(step)
        iteration_timing.append(
            {
                "iteration_index": float(iteration_index),
                "step_time_s": step_elapsed,
                "cumulative_time_s": cumulative_elapsed,
                "energy": float(jnp.asarray(step["energy"])),
            }
        )

    return history, iteration_timing


def _evaluate_final_energy(
    *,
    hilbert: SpinHilbert,
    model: RBM,
    params: Any,
    operator: nk.operator.AbstractOperator,
    n_chains: int,
    eval_samples: int,
    eval_repeats: int,
    n_discard_per_chain: int,
    seed: int,
) -> float:
    eval_state = build_variational_state(
        model=model,
        hilbert=hilbert,
        params=params,
        seed=seed + 101,
        n_samples=eval_samples,
        n_discard_per_chain=max(n_discard_per_chain, 32),
        n_chains=n_chains,
    )
    estimates = [float(jnp.asarray(eval_state.energy(operator))) for _ in range(eval_repeats)]
    return sum(estimates) / len(estimates)


def _run_single_operator_path(
    *,
    label: str,
    operator: nk.operator.AbstractOperator,
    hilbert: SpinHilbert,
    n_iter: int,
    n_samples: int,
    n_discard_per_chain: int,
    n_chains: int,
    eval_samples: int,
    eval_repeats: int,
    seed: int,
) -> dict[str, object]:
    total_start = time.perf_counter()
    model = RBM(alpha=2)
    state, driver = build_vmc_driver(
        model=model,
        hilbert=hilbert,
        operator=operator,
        learning_rate=1e-2,
        seed=seed,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
    )
    history, iteration_timing = _run_driver_with_timing(driver, n_iter)
    energy_trace = [float(jnp.asarray(step["energy"])) for step in history]
    final_energy = _evaluate_final_energy(
        hilbert=hilbert,
        model=model,
        params=state.parameters,
        operator=operator,
        n_chains=n_chains,
        eval_samples=eval_samples,
        eval_repeats=eval_repeats,
        n_discard_per_chain=n_discard_per_chain,
        seed=seed,
    )
    total_runtime_s = time.perf_counter() - total_start
    return {
        "label": label,
        "trace": energy_trace,
        "final_energy": final_energy,
        "iteration_timing": iteration_timing,
        "training_runtime_s": iteration_timing[-1]["cumulative_time_s"] if iteration_timing else 0.0,
        "total_runtime_s": total_runtime_s,
    }


def run_j1j2_rbm_demo(
    *,
    Lx: int = 2,
    Ly: int = 2,
    J1: float = 1.0,
    J2: float = 0.5,
    pbc: bool = True,
    n_iter: int = 32,
    n_samples: int = 256,
    n_discard_per_chain: int = 32,
    n_chains: int = 16,
    eval_samples: int = 2048,
    eval_repeats: int = 3,
    mode: BenchmarkMode = "comparison",
) -> dict[str, object]:
    """Run either a project-only speed benchmark or a comparison benchmark."""

    hilbert, _, netket_operator, project_operator = build_j1j2_operators(Lx=Lx, Ly=Ly, J1=J1, J2=J2, pbc=pbc)
    runs: list[tuple[str, nk.operator.AbstractOperator, int]]
    if mode == "speed":
        runs = [("project_operator", project_operator, 0)]
    else:
        runs = [
            ("netket_native", netket_operator, 0),
            ("project_operator", project_operator, 0),
        ]

    results = [
        _run_single_operator_path(
            label=label,
            operator=operator,
            hilbert=hilbert,
            n_iter=n_iter,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            n_chains=n_chains,
            eval_samples=eval_samples,
            eval_repeats=eval_repeats,
            seed=seed,
        )
        for label, operator, seed in runs
    ]
    project_result = next(result for result in results if cast(str, result["label"]) == "project_operator")
    netket_result = next(
        (result for result in results if cast(str, result["label"]) == "netket_native"),
        None,
    )
    energy_gap = (
        None
        if netket_result is None
        else abs(cast(float, netket_result["final_energy"]) - cast(float, project_result["final_energy"]))
    )
    return {
        "shape": (Lx, Ly),
        "J1": J1,
        "J2": J2,
        "mode": mode,
        "results": results,
        "project_final_energy": cast(float, project_result["final_energy"]),
        "netket_final_energy": None if netket_result is None else cast(float, netket_result["final_energy"]),
        "energy_gap": energy_gap,
    }


def profile_j1j2_speed_run(
    *,
    Lx: int,
    Ly: int,
    J1: float,
    J2: float,
    pbc: bool,
    n_iter: int,
    n_samples: int,
    n_discard_per_chain: int,
    n_chains: int,
    eval_samples: int,
    eval_repeats: int,
) -> dict[str, object]:
    profiler = cProfile.Profile()
    profiler.enable()
    run_j1j2_rbm_demo(
        Lx=Lx,
        Ly=Ly,
        J1=J1,
        J2=J2,
        pbc=pbc,
        n_iter=n_iter,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        eval_samples=eval_samples,
        eval_repeats=eval_repeats,
        mode="speed",
    )
    profiler.disable()

    dominant_entry: tuple[str, int, str] | None = None
    dominant_cumulative_time = 0.0
    for key, stats in profiler.getstats() if False else []:
        pass

    import pstats

    stats = pstats.Stats(profiler)
    stats_map = cast(
        dict[tuple[str, int, str], tuple[int, int, float, float, dict[tuple[str, int, str], Any]]],
        cast(Any, stats).stats,
    )
    ignored_functions = {
        "<listcomp>",
        "_run_driver_with_timing",
        "_run_single_operator_path",
        "profile_j1j2_speed_run",
        "run_j1j2_rbm_demo",
        "reraise_with_filtered_traceback",
        "cache_miss",
        "_python_pjit_helper",
        "wrapper",
        "timed_function",
    }
    for (filename, lineno, funcname), stat in sorted(stats_map.items(), key=lambda item: item[1][3], reverse=True):
        if funcname in ignored_functions:
            continue
        dominant_entry = (filename, lineno, funcname)
        dominant_cumulative_time = stat[3]
        break

    if dominant_entry is None:
        dominant_entry = ("", 0, "unknown")

    filename, lineno, funcname = dominant_entry
    return {
        "function": funcname,
        "filename": filename,
        "line": lineno,
        "cumulative_time_s": dominant_cumulative_time,
        "label": f"{funcname} ({Path(filename).name}:{lineno})" if filename else funcname,
    }


def _classify_iteration_shape(iteration_timing: list[dict[str, float]]) -> dict[str, object]:
    if len(iteration_timing) < 2:
        return {
            "compile_dominated_early": False,
            "later_iterations_flatten": False,
            "growth_character": "insufficient_data",
        }

    first_step = iteration_timing[0]["step_time_s"]
    later_steps = [entry["step_time_s"] for entry in iteration_timing[1:]]
    mean_later = statistics.fmean(later_steps)
    compile_dominated_early = first_step > 1.5 * mean_later if mean_later > 0 else False
    later_iterations_flatten = max(later_steps) <= 1.25 * min(later_steps) if min(later_steps) > 0 else False
    growth_character = "fixed_overhead" if compile_dominated_early and later_iterations_flatten else "size_driven"
    return {
        "compile_dominated_early": compile_dominated_early,
        "later_iterations_flatten": later_iterations_flatten,
        "growth_character": growth_character,
    }


def generate_j1j2_runtime_report(
    *,
    sizes: tuple[tuple[int, int], ...] = DEFAULT_SIZE_PROGRESSION,
    comparison_checkpoints: frozenset[tuple[int, int]] = DEFAULT_CHECKPOINTS,
    J1: float = 1.0,
    J2: float = 0.5,
    pbc: bool = False,
    n_iter: int = 4,
    n_samples: int = 32,
    n_discard_per_chain: int = 8,
    n_chains: int = 4,
    eval_samples: int = 64,
    eval_repeats: int = 1,
    warm_runs: int = 3,
) -> dict[str, object]:
    size_scaling: list[dict[str, object]] = []
    iteration_timing: dict[str, list[dict[str, float]]] = {}
    analysis_notes: list[str] = []

    for Lx, Ly in sizes:
        cold_result = run_j1j2_rbm_demo(
            Lx=Lx,
            Ly=Ly,
            J1=J1,
            J2=J2,
            pbc=pbc,
            n_iter=n_iter,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            n_chains=n_chains,
            eval_samples=eval_samples,
            eval_repeats=eval_repeats,
            mode="speed",
        )
        warm_results = [
            run_j1j2_rbm_demo(
                Lx=Lx,
                Ly=Ly,
                J1=J1,
                J2=J2,
                pbc=pbc,
                n_iter=n_iter,
                n_samples=n_samples,
                n_discard_per_chain=n_discard_per_chain,
                n_chains=n_chains,
                eval_samples=eval_samples,
                eval_repeats=eval_repeats,
                mode="speed",
            )
            for _ in range(warm_runs)
        ]
        warm_runtimes = [
            cast(float, cast(list[dict[str, object]], result["results"])[0]["total_runtime_s"])
            for result in warm_results
        ]
        representative_warm = min(warm_results, key=lambda result: abs(
            cast(float, cast(list[dict[str, object]], result["results"])[0]["total_runtime_s"]) - statistics.median(warm_runtimes)
        ))
        project_result = cast(list[dict[str, object]], representative_warm["results"])[0]
        iteration_rows = cast(list[dict[str, float]], project_result["iteration_timing"])
        iteration_summary = _classify_iteration_shape(iteration_rows)
        hotspot = profile_j1j2_speed_run(
            Lx=Lx,
            Ly=Ly,
            J1=J1,
            J2=J2,
            pbc=pbc,
            n_iter=n_iter,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            n_chains=n_chains,
            eval_samples=eval_samples,
            eval_repeats=eval_repeats,
        )
        comparison_result = (
            None
            if (Lx, Ly) not in comparison_checkpoints
            else run_j1j2_rbm_demo(
                Lx=Lx,
                Ly=Ly,
                J1=J1,
                J2=J2,
                pbc=pbc,
                n_iter=n_iter,
                n_samples=n_samples,
                n_discard_per_chain=n_discard_per_chain,
                n_chains=n_chains,
                eval_samples=eval_samples,
                eval_repeats=eval_repeats,
                mode="comparison",
            )
        )
        size_key = f"{Lx}x{Ly}"
        iteration_timing[size_key] = iteration_rows
        size_scaling.append(
            {
                "shape": (Lx, Ly),
                "cold_runtime_s": cast(float, cast(list[dict[str, object]], cold_result["results"])[0]["total_runtime_s"]),
                "warm_runtime_s": cast(float, project_result["total_runtime_s"]),
                "median_warm_runtime_s": statistics.median(warm_runtimes),
                "project_final_energy": cast(float, representative_warm["project_final_energy"]),
                "netket_final_energy": None if comparison_result is None else cast(float, comparison_result["netket_final_energy"]),
                "energy_gap": None if comparison_result is None else cast(float, comparison_result["energy_gap"]),
                "dominant_hotspot": cast(str, hotspot["label"]),
                "hotspot_cumulative_time_s": cast(float, hotspot["cumulative_time_s"]),
                **iteration_summary,
            }
        )
        analysis_notes.append(
            f"{size_key}: early compile dominated={iteration_summary['compile_dominated_early']}, "
            f"later flatten={iteration_summary['later_iterations_flatten']}, "
            f"growth={iteration_summary['growth_character']}"
        )

    return {
        "config": {
            "sizes": list(sizes),
            "comparison_checkpoints": list(comparison_checkpoints),
            "J1": J1,
            "J2": J2,
            "pbc": pbc,
            "n_iter": n_iter,
            "n_samples": n_samples,
            "n_discard_per_chain": n_discard_per_chain,
            "n_chains": n_chains,
            "eval_samples": eval_samples,
            "eval_repeats": eval_repeats,
            "warm_runs": warm_runs,
        },
        "size_scaling": size_scaling,
        "iteration_timing": iteration_timing,
        "analysis_notes": analysis_notes,
    }


def format_runtime_change_report(report: dict[str, object]) -> str:
    size_rows = cast(list[dict[str, object]], report["size_scaling"])
    iteration_rows = cast(dict[str, list[dict[str, float]]], report["iteration_timing"])

    lines = [
        "Size scaling",
        "| Size | Cold runtime (s) | Warm runtime (s) | Median warm runtime (s) | Energy gap | Dominant hotspot |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in size_rows:
        gap = row["energy_gap"]
        gap_text = "n/a" if gap is None else f"{cast(float, gap):.6f}"
        lines.append(
            f"| {cast(tuple[int, int], row['shape'])[0]}x{cast(tuple[int, int], row['shape'])[1]} "
            f"| {cast(float, row['cold_runtime_s']):.3f} "
            f"| {cast(float, row['warm_runtime_s']):.3f} "
            f"| {cast(float, row['median_warm_runtime_s']):.3f} "
            f"| {gap_text} "
            f"| {cast(str, row['dominant_hotspot'])} |"
        )

    lines.extend(
        [
            "",
            "Iteration timing",
            "| Size | Iteration | Step time (s) | Cumulative time (s) | Energy |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for size_key, rows in iteration_rows.items():
        for row in rows:
            lines.append(
                f"| {size_key} | {int(row['iteration_index'])} | {row['step_time_s']:.3f} "
                f"| {row['cumulative_time_s']:.3f} | {row['energy']:.6f} |"
            )

    lines.extend(["", "Analysis"])
    lines.extend(f"- {note}" for note in cast(list[str], report["analysis_notes"]))
    return "\n".join(lines)
