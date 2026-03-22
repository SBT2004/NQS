"""Shared experiment helpers for notebook-facing workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .. import observables
from ..exact_diag import exact_ground_state, solve_sparse_ground_state, sparse_operator_matrix
from ..graph import SquareLattice
from ..hilbert import SpinHilbert
from ..operator import j1_j2, tfim
from ..sampler import MetropolisLocal, _metropolis_step
from ..vmc_setup import build_model, build_variational_state, build_vmc_experiment

if TYPE_CHECKING:
    from ..observables import SupportsSamplingAndLogValue

_EXACT_STATEVECTOR_UNAVAILABLE_MESSAGE = "exact_statevector is only available for small full-summation states."


@dataclass
class _TFIM5x5VMCBenchmarkContext:
    system: dict[str, Any]
    model: Any
    variational_state: Any
    vmc_driver: Any
    benchmark_states: jax.Array | None = None
    benchmark_sample_batch: Any | None = None


def half_subsystem(n_sites: int) -> tuple[int, ...]:
    return tuple(range(max(1, n_sites // 2)))


def build_system(
    lattice_shape: tuple[int, int] = (2, 2),
    *,
    pbc: bool = True,
    hamiltonian: str = "tfim",
    J: float = 1.0,
    h: float = 0.8,
    J1: float = 1.0,
    J2: float = 0.4,
) -> dict[str, Any]:
    lattice = SquareLattice(lattice_shape[0], lattice_shape[1], pbc=pbc)
    spin_space = SpinHilbert(lattice.n_nodes)

    if hamiltonian == "tfim":
        hamiltonian_operator = tfim(spin_space, lattice, J=J, h=h)
    elif hamiltonian == "j1_j2":
        hamiltonian_operator = j1_j2(spin_space, lattice, J1=J1, J2=J2)
    else:
        raise ValueError(f"Unsupported hamiltonian: {hamiltonian}")

    return {
        "graph": lattice,
        "hilbert": spin_space,
        "operator": hamiltonian_operator,
        "hamiltonian": hamiltonian,
        "parameters": {"J": J, "h": h, "J1": J1, "J2": J2, "pbc": pbc},
    }


def tfim_config(
    *,
    lattice_shape: tuple[int, int] = (2, 2),
    J: float = 1.0,
    h: float,
    pbc: bool,
) -> dict[str, Any]:
    return {
        "lattice_shape": lattice_shape,
        "hamiltonian": "tfim",
        "J": J,
        "h": h,
        "pbc": pbc,
    }


def tfim_proxy_sweep_points(
    lengths: Sequence[int],
    *,
    J: float = 1.0,
    h: float,
    pbc: bool,
) -> list[dict[str, Any]]:
    if not lengths:
        raise ValueError("lengths must contain at least one system size.")
    return [
        {
            "label": f"tfim_1d_{int(length)}x1",
            **tfim_config(lattice_shape=(int(length), 1), J=J, h=h, pbc=pbc),
        }
        for length in lengths
    ]


def edge_table(graph: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for neighbor_order, bond_name in ((1, "J1"), (2, "J2")):
        try:
            for edge in graph.iter_edges(bond_name, n=neighbor_order):
                rows.append(
                    {
                        "bond": bond_name,
                        "sites": (edge.i, edge.j),
                        "coords": (graph.index_to_coord(edge.i), graph.index_to_coord(edge.j)),
                    }
                )
        except ValueError:
            continue
    return pd.DataFrame(rows)


def exact_observables_summary(
    operator: Any,
    subsystem: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    hilbert = operator.hilbert
    subsystem_sites = half_subsystem(hilbert.size) if subsystem is None else subsystem
    exact = exact_ground_state(operator)
    ground_energy = float(exact["ground_energy"])
    ground_state = np.asarray(exact["ground_state"], dtype=np.complex128)
    states = hilbert.all_states()
    probabilities = np.abs(ground_state) ** 2
    signed_states = hilbert.states_to_pm1(states).astype(np.float64, copy=False)
    correlation_matrix = pd.DataFrame(
        np.einsum("bi,bj,b->ij", signed_states, signed_states, probabilities, optimize=True),
        index=np.arange(hilbert.size),
        columns=np.arange(hilbert.size),
    )

    spectrum_table = pd.DataFrame(
        {
            "level": np.array([0], dtype=np.int64),
            "energy": np.array([ground_energy], dtype=np.float64),
        }
    )

    entropy_rows: list[dict[str, Any]] = []
    for subsystem_size in range(1, (hilbert.size // 2) + 1):
        current_subsystem = tuple(range(subsystem_size))
        entropy_rows.append(
            {
                "subsystem_size": subsystem_size,
                "von_neumann": observables.von_neumann_entropy(ground_state, current_subsystem),
                "renyi2": observables.renyi_entropy_from_statevector(
                    ground_state,
                    current_subsystem,
                    alpha=2.0,
                ),
            }
        )
    entropy_table = pd.DataFrame(entropy_rows)
    scaling_fit = (
        observables.fit_log_entropy_scaling(
            entropy_table["subsystem_size"].to_numpy(),
            entropy_table["renyi2"].to_numpy(),
        )
        if len(entropy_table) >= 2
        else None
    )

    return {
        "ground_energy": ground_energy,
        "ground_state": ground_state,
        "spectrum_table": spectrum_table,
        "entropy_table": entropy_table,
        "scaling_fit": scaling_fit,
        "half_partition_von_neumann": observables.von_neumann_entropy(ground_state, subsystem_sites),
        "half_partition_renyi2": observables.renyi_entropy_from_statevector(
            ground_state,
            subsystem_sites,
            alpha=2.0,
        ),
        "correlation_matrix": correlation_matrix,
    }


def run_incremental_exercise_1_ed_benchmark(
    lengths: Sequence[int],
    *,
    J: float = 1.0,
    h: float = 1.0,
    pbc: bool = False,
) -> pd.DataFrame:
    if not lengths:
        raise ValueError("lengths must contain at least one system size.")

    rows: list[dict[str, Any]] = []
    for raw_length in lengths:
        length = int(raw_length)
        benchmark_start = perf_counter()
        try:
            system = build_system(
                lattice_shape=(length, 1),
                pbc=pbc,
                hamiltonian="tfim",
                J=J,
                h=h,
            )
            assembly_start = perf_counter()
            sparse_matrix = sparse_operator_matrix(system["operator"])
            assembly_seconds = perf_counter() - assembly_start
            solve_start = perf_counter()
            exact = solve_sparse_ground_state(sparse_matrix)
            solve_seconds = perf_counter() - solve_start
        except Exception as exc:
            rows.append(
                {
                    "length": length,
                    "label": f"tfim_1d_{length}x1",
                    "assembly_seconds": np.nan,
                    "solve_seconds": np.nan,
                    "runtime_seconds": perf_counter() - benchmark_start,
                    "completed": False,
                    "ground_energy": np.nan,
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                }
            )
            continue

        rows.append(
            {
                "length": length,
                "label": f"tfim_1d_{length}x1",
                "assembly_seconds": assembly_seconds,
                "solve_seconds": solve_seconds,
                "runtime_seconds": perf_counter() - benchmark_start,
                "completed": True,
                "ground_energy": float(exact["ground_energy"]),
                "failure_type": "",
                "failure_message": "",
            }
        )

    return pd.DataFrame(rows)


def _block_until_ready(value: Any) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves and hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _timed_ms(fn: Any) -> tuple[Any, float]:
    start = perf_counter()
    result = fn()
    _block_until_ready(result)
    return result, (perf_counter() - start) * 1_000.0


def _build_tfim_5x5_vmc_benchmark_context(
    *,
    seed: int,
    model_kwargs: dict[str, Any],
    learning_rate: float,
    n_samples: int,
    n_discard_per_chain: int,
    n_chains: int,
    h: float,
) -> _TFIM5x5VMCBenchmarkContext:
    system = build_system(
        lattice_shape=(5, 5),
        pbc=False,
        hamiltonian="tfim",
        h=h,
    )
    model, variational_state, vmc_driver = build_vmc_experiment(
        hilbert=system["hilbert"],
        operator=system["operator"],
        learning_rate=learning_rate,
        seed=seed,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        model_name="RBM",
        model_kwargs=model_kwargs,
        lattice_shape=(5, 5),
        exact_backend_max_states=0,
    )
    return _TFIM5x5VMCBenchmarkContext(
        system=system,
        model=model,
        variational_state=variational_state,
        vmc_driver=vmc_driver,
    )


def _prepare_model_eval_states(
    context: _TFIM5x5VMCBenchmarkContext,
    *,
    seed: int,
    n_states: int,
) -> None:
    if context.benchmark_states is not None:
        return
    key = jax.random.PRNGKey(seed)
    context.benchmark_states = jax.random.bernoulli(
        key,
        p=0.5,
        shape=(n_states, context.system["hilbert"].size),
    ).astype(jnp.uint8)


def _prepare_fixed_sample_batch(
    context: _TFIM5x5VMCBenchmarkContext,
    *,
    seed_offset: int,
) -> None:
    if context.benchmark_sample_batch is not None:
        return
    # Use an independent batch so local-energy diagnostics can be timed
    # without mutating the stateful sampler benchmark path.
    context.benchmark_sample_batch = context.variational_state.independent_sample_with_log_values(
        seed_offset=seed_offset
    )


def _fixed_batch_local_energy_gradient(
    context: _TFIM5x5VMCBenchmarkContext,
) -> dict[str, Any]:
    backend = cast(Any, context.variational_state)._expectation_backend
    sample_batch = context.benchmark_sample_batch
    params = context.variational_state.parameters
    operator = context.system["operator"]
    if sample_batch is None:
        raise ValueError("benchmark_sample_batch must be prepared before gradient timing.")

    local_energies = jax.lax.stop_gradient(
        backend._local_energies(
            operator,
            params,
            sample_batch.states,
            sample_log_values=sample_batch.log_values,
        )
    )

    def surrogate_loss(current_params: Any) -> jax.Array:
        log_values = jnp.asarray(context.model.log_psi(current_params, sample_batch.states))
        centered_local_energies = local_energies - jnp.mean(local_energies)
        return 2.0 * jnp.real(jnp.mean(jnp.conj(log_values) * centered_local_energies))

    grads = jax.grad(surrogate_loss)(params)
    return {"local_energies": local_energies, "grads": grads}


def _measure_tfim_5x5_vmc_stage(
    *,
    stage: str,
    scope: str,
    prepare_context: Any,
    run_stage: Any,
    build_context: Any,
    warmed_repeats: int,
) -> dict[str, Any]:
    context = build_context()
    prepare_context(context)
    _, cold_start_ms = _timed_ms(lambda: run_stage(context))

    warmed_runs_ms: list[float] = []
    for _ in range(warmed_repeats):
        _, warmed_ms = _timed_ms(lambda: run_stage(context))
        warmed_runs_ms.append(warmed_ms)

    warmed_mean_ms = float(np.mean(warmed_runs_ms))
    warmed_min_ms = float(np.min(warmed_runs_ms))
    return {
        "stage": stage,
        "scope": scope,
        "cold_start_ms": float(cold_start_ms),
        "warmed_mean_ms": warmed_mean_ms,
        "warmed_min_ms": warmed_min_ms,
        "cold_to_warm_ratio": float(cold_start_ms / warmed_mean_ms) if warmed_mean_ms > 0.0 else np.nan,
        "warmed_repeats": warmed_repeats,
        "regression_metric": "warmed_mean_ms",
        "suggested_max_regression_ratio": 1.25,
    }


def run_tfim_5x5_vmc_performance_benchmark(
    *,
    rbm_alpha: int = 2,
    learning_rate: float = 1e-2,
    n_samples: int = 256,
    n_discard_per_chain: int = 16,
    n_chains: int = 8,
    h: float = 1.0,
    seed: int = 0,
    warmed_repeats: int = 3,
    model_eval_batch_size: int = 256,
) -> dict[str, Any]:
    if warmed_repeats <= 0:
        raise ValueError("warmed_repeats must be positive.")
    if model_eval_batch_size <= 0:
        raise ValueError("model_eval_batch_size must be positive.")

    model_kwargs = {"alpha": int(rbm_alpha)}

    def build_context() -> _TFIM5x5VMCBenchmarkContext:
        return _build_tfim_5x5_vmc_benchmark_context(
            seed=seed,
            model_kwargs=model_kwargs,
            learning_rate=learning_rate,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            n_chains=n_chains,
            h=h,
        )

    stage_rows = [
        _measure_tfim_5x5_vmc_stage(
            stage="model_log_psi",
            scope="fixed state batch",
            build_context=build_context,
            prepare_context=lambda context: _prepare_model_eval_states(
                context,
                seed=seed + 1_001,
                n_states=model_eval_batch_size,
            ),
            run_stage=lambda context: context.model.log_psi(
                context.variational_state.parameters,
                context.benchmark_states,
            ),
            warmed_repeats=warmed_repeats,
        ),
        _measure_tfim_5x5_vmc_stage(
            stage="sampler_draw",
            scope="stateful sampled batch",
            build_context=build_context,
            prepare_context=lambda context: None,
            run_stage=lambda context: context.variational_state.sample_with_log_values(),
            warmed_repeats=warmed_repeats,
        ),
        _measure_tfim_5x5_vmc_stage(
            stage="local_energy",
            scope="fixed independent sample batch",
            build_context=build_context,
            prepare_context=lambda context: _prepare_fixed_sample_batch(
                context,
                seed_offset=30_000,
            ),
            run_stage=lambda context: cast(Any, context.variational_state)._expectation_backend._local_energies(
                context.system["operator"],
                context.variational_state.parameters,
                context.benchmark_sample_batch.states,
                sample_log_values=context.benchmark_sample_batch.log_values,
            ),
            warmed_repeats=warmed_repeats,
        ),
        _measure_tfim_5x5_vmc_stage(
            stage="local_energy_gradient",
            scope="fixed independent sample batch",
            build_context=build_context,
            prepare_context=lambda context: _prepare_fixed_sample_batch(
                context,
                seed_offset=30_000,
            ),
            run_stage=_fixed_batch_local_energy_gradient,
            warmed_repeats=warmed_repeats,
        ),
        _measure_tfim_5x5_vmc_stage(
            stage="vmc_step",
            scope="stateful end-to-end sampled step",
            build_context=build_context,
            prepare_context=lambda context: None,
            run_stage=lambda context: context.vmc_driver.step(),
            warmed_repeats=warmed_repeats,
        ),
    ]

    timing_table = pd.DataFrame(stage_rows)
    regression_gate_table = timing_table[
        [
            "stage",
            "scope",
            "cold_start_ms",
            "warmed_mean_ms",
            "regression_metric",
            "suggested_max_regression_ratio",
        ]
    ].copy()
    regression_gate_table["gate_description"] = regression_gate_table.apply(
        lambda row: (
            f"Flag regressions when future {row['regression_metric']} exceeds "
            f"{float(row['warmed_mean_ms']):.3f} ms by more than "
            f"{float(row['suggested_max_regression_ratio']):.2f}x; "
            "track cold_start_ms separately for compile/tracing drift."
        ),
        axis=1,
    )

    return {
        "benchmark_label": "tfim_5x5_open_rbm_vmc",
        "system_label": "tfim_5x5_open",
        "model_name": "RBM",
        "model_kwargs": model_kwargs,
        "lattice_shape": (5, 5),
        "hamiltonian": "tfim",
        "pbc": False,
        "seed": seed,
        "n_sites": 25,
        "n_samples": n_samples,
        "n_discard_per_chain": n_discard_per_chain,
        "n_chains": n_chains,
        "learning_rate": learning_rate,
        "h": h,
        "model_eval_batch_size": model_eval_batch_size,
        "warmed_repeats": warmed_repeats,
        "timing_table": timing_table,
        "regression_gate_table": regression_gate_table,
        "notes": [
            "cold_start_ms is measured on a fresh benchmark context and includes first-call tracing/compilation for that stage.",
            "warmed_mean_ms and warmed_min_ms are measured by rerunning the same stage on the warmed context.",
            "local_energy and local_energy_gradient use a fixed independently sampled batch so expectation cost can be compared separately from sampler cost.",
            "vmc_step measures the full sampled driver.step() path, including sampling, local-energy evaluation, gradient construction, and parameter update.",
        ],
    }


def format_tfim_5x5_vmc_performance_report(benchmark: dict[str, Any]) -> str:
    timing_table = cast(pd.DataFrame, benchmark["timing_table"])
    regression_gate_table = cast(pd.DataFrame, benchmark["regression_gate_table"])
    timing_lines = timing_table.to_string(
        index=False,
        formatters={
            "cold_start_ms": lambda value: f"{float(value):.3f}",
            "warmed_mean_ms": lambda value: f"{float(value):.3f}",
            "warmed_min_ms": lambda value: f"{float(value):.3f}",
            "cold_to_warm_ratio": lambda value: f"{float(value):.2f}",
            "suggested_max_regression_ratio": lambda value: f"{float(value):.2f}",
        },
    )
    gate_lines = regression_gate_table.to_string(
        index=False,
        formatters={
            "cold_start_ms": lambda value: f"{float(value):.3f}",
            "warmed_mean_ms": lambda value: f"{float(value):.3f}",
            "suggested_max_regression_ratio": lambda value: f"{float(value):.2f}",
        },
    )
    notes = "\n".join(f"- {note}" for note in benchmark["notes"])
    return "\n".join(
        [
            (
                f"Benchmark: {benchmark['benchmark_label']} "
                f"(system={benchmark['system_label']}, model={benchmark['model_name']}, "
                f"alpha={benchmark['model_kwargs']['alpha']}, samples={benchmark['n_samples']}, "
                f"chains={benchmark['n_chains']}, warmed_repeats={benchmark['warmed_repeats']})"
            ),
            "",
            "Stage timings (ms):",
            timing_lines,
            "",
            "Regression gates:",
            gate_lines,
            "",
            "Notes:",
            notes,
        ]
    )


def history_table(history: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step_entry in history:
        row = {
            "step": int(step_entry["step"]),
            "energy": float(np.asarray(step_entry["energy"])),
        }
        observable_log = dict(step_entry.get("observables", {}))
        for key, value in observable_log.items():
            if key in row:
                raise ValueError(f"Observable key {key!r} collides with a core history field.")
            row[key] = float(np.asarray(value))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if "renyi2_entropy" in frame.columns:
        frame["renyi2_entropy"] = frame["renyi2_entropy"].astype(float)
    return frame


def _independent_sample_batches_with_log_values(
    variational_state: SupportsSamplingAndLogValue,
    *,
    n_independent_runs: int,
) -> list[Any]:
    if n_independent_runs <= 0:
        raise ValueError("n_independent_runs must be positive.")
    sampling_state = cast(Any, variational_state)
    if hasattr(sampling_state, "independent_sample_with_log_values"):
        return [
            sampling_state.independent_sample_with_log_values(seed_offset=run_index)
            for run_index in range(n_independent_runs)
        ]
    return [
        np.asarray(variational_state.independent_sample(seed_offset=run_index), dtype=np.uint8)
        for run_index in range(n_independent_runs)
    ]


def _renyi2_entropy_statistics_from_sample_batches(
    variational_state: SupportsSamplingAndLogValue,
    *,
    subsystem: tuple[int, ...],
    sample_batches: Sequence[Any],
    cutoff: float = 1e-12,
) -> dict[str, float]:
    if not sample_batches:
        raise ValueError("sample_batches must not be empty.")

    estimates: list[float] = []
    for sample_batch in sample_batches:
        if hasattr(sample_batch, "states") and hasattr(sample_batch, "log_values"):
            states = sample_batch.states
            original_log_values = np.asarray(sample_batch.log_values, dtype=np.complex128)
        else:
            states = sample_batch
            original_log_values = None
        estimates.append(
            observables._renyi2_entropy_from_samples(
                variational_state.log_value,
                np.asarray(states, dtype=np.uint8),
                subsystem=subsystem,
                cutoff=cutoff,
                original_log_values=original_log_values,
            )
        )

    estimate_array = np.asarray(estimates, dtype=np.float64)
    return {
        "mean": float(np.mean(estimate_array)),
        "std": float(np.std(estimate_array, ddof=0)),
        "n_repeats": float(len(estimates)),
    }


def _renyi2_entropy_scaling_from_sample_batch(
    variational_state: SupportsSamplingAndLogValue,
    *,
    sample_batch: np.ndarray,
    subsystem_limit: int,
    original_log_values: np.ndarray | None = None,
    cutoff: float = 1e-12,
) -> list[float]:
    flattened_samples = np.asarray(sample_batch, dtype=np.uint8).reshape(-1, sample_batch.shape[-1])
    if flattened_samples.shape[0] < 2:
        raise ValueError("at least two samples are required for the SWAP estimator.")

    pair_count = flattened_samples.shape[0] // 2
    paired_samples = flattened_samples[: 2 * pair_count].reshape(pair_count, 2, flattened_samples.shape[1])
    original_left = paired_samples[:, 0, :]
    original_right = paired_samples[:, 1, :]
    if original_log_values is None:
        original_log = np.asarray(
            variational_state.log_value(flattened_samples),
            dtype=np.complex128,
        ).reshape(-1)
    else:
        original_log = np.asarray(original_log_values, dtype=np.complex128).reshape(-1)

    swapped_batches: list[np.ndarray] = []
    for subsystem_size in range(1, subsystem_limit + 1):
        subsystem_sites = np.arange(subsystem_size, dtype=np.intp)
        swapped_left = original_left.copy()
        swapped_right = original_right.copy()
        swapped_left[:, subsystem_sites] = original_right[:, subsystem_sites]
        swapped_right[:, subsystem_sites] = original_left[:, subsystem_sites]
        swapped_batches.append(np.concatenate([swapped_left, swapped_right], axis=0))

    swapped_log_values = np.asarray(
        variational_state.log_value(np.concatenate(swapped_batches, axis=0)),
        dtype=np.complex128,
    ).reshape(subsystem_limit, 2 * pair_count)
    left_log = original_log[0 : 2 * pair_count : 2]
    right_log = original_log[1 : 2 * pair_count : 2]

    entropies: list[float] = []
    for subsystem_index in range(subsystem_limit):
        swapped_log = swapped_log_values[subsystem_index]
        swapped_left_log = swapped_log[:pair_count]
        swapped_right_log = swapped_log[pair_count:]
        estimator = np.exp(swapped_left_log + swapped_right_log - left_log - right_log)
        swap_value = np.real_if_close(np.mean(estimator))
        if np.iscomplexobj(swap_value):
            if abs(np.imag(swap_value)) > cutoff:
                entropies.append(np.nan)
                continue
            swap_real = float(np.real(swap_value))
        else:
            swap_real = float(swap_value)
        if swap_real <= 0:
            entropies.append(np.nan)
            continue
        entropies.append(float(-np.log(swap_real)))
    return entropies


def sampled_entropy_scaling_summary(
    variational_state: SupportsSamplingAndLogValue,
    n_sites: int,
    max_subsystem_size: int | None = None,
    n_independent_runs: int = 1,
    sample_batches: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if n_independent_runs <= 0:
        raise ValueError("n_independent_runs must be positive.")
    if sample_batches is not None and len(sample_batches) < n_independent_runs:
        raise ValueError("sample_batches must provide at least n_independent_runs entries.")
    subsystem_limit = max(1, n_sites // 2) if max_subsystem_size is None else max_subsystem_size
    run_tables: list[pd.DataFrame] = []
    sampling_state = cast(Any, variational_state)
    for run_index in range(n_independent_runs):
        original_log_values: np.ndarray | None = None
        if sample_batches is not None:
            sample_with_values = sample_batches[run_index]
            if hasattr(sample_with_values, "states") and hasattr(sample_with_values, "log_values"):
                sample_batch = np.asarray(sample_with_values.states, dtype=np.uint8)
                original_log_values = np.asarray(sample_with_values.log_values, dtype=np.complex128)
            else:
                sample_batch = np.asarray(sample_with_values, dtype=np.uint8)
        elif hasattr(sampling_state, "independent_sample_with_log_values"):
            sample_with_values = sampling_state.independent_sample_with_log_values(seed_offset=run_index)
            sample_batch = np.asarray(sample_with_values.states, dtype=np.uint8)
            original_log_values = np.asarray(sample_with_values.log_values, dtype=np.complex128)
        else:
            sample_batch = np.asarray(variational_state.independent_sample(seed_offset=run_index))
        renyi2_by_size = _renyi2_entropy_scaling_from_sample_batch(
            variational_state,
            sample_batch=sample_batch,
            subsystem_limit=subsystem_limit,
            original_log_values=original_log_values,
        )
        entropy_rows: list[dict[str, Any]] = []
        for subsystem_size in range(1, subsystem_limit + 1):
            entropy_rows.append(
                {
                    "subsystem_size": subsystem_size,
                    "run_index": run_index,
                    "renyi2": renyi2_by_size[subsystem_size - 1],
                }
            )
        run_tables.append(pd.DataFrame(entropy_rows))

    entropy_samples = pd.concat(run_tables, ignore_index=True)
    entropy_table = (
        entropy_samples.groupby("subsystem_size", as_index=False)
        .agg(
            renyi2=("renyi2", "mean"),
            renyi2_std=("renyi2", "std"),
        )
        .fillna({"renyi2_std": 0.0})
    )

    subsystem_sizes = np.asarray(entropy_table["subsystem_size"], dtype=np.float64)
    renyi_values = np.asarray(entropy_table["renyi2"], dtype=np.float64)
    valid_mask = np.isfinite(renyi_values)
    scaling_fit = (
        observables.fit_log_entropy_scaling(
            subsystem_sizes[valid_mask],
            renyi_values[valid_mask],
        )
        if np.count_nonzero(valid_mask) >= 2
        else None
    )
    return {
        "entropy_table": entropy_table,
        "entropy_samples": entropy_samples,
        "scaling_fit": scaling_fit,
    }


def _energy_trend_metrics(history_df: pd.DataFrame, *, window: int = 5) -> dict[str, float]:
    energy_values = np.asarray(history_df["energy"], dtype=np.float64)
    if energy_values.size == 0:
        raise ValueError("history_df must contain at least one energy value.")

    tail_size = min(window, energy_values.size)
    tail = energy_values[-tail_size:]
    if tail_size >= 2:
        tail_steps = np.arange(tail_size, dtype=np.float64)
        tail_slope = float(np.polyfit(tail_steps, tail, deg=1)[0])
    else:
        tail_slope = 0.0

    return {
        "initial_energy": float(energy_values[0]),
        "best_energy": float(np.min(energy_values)),
        "energy_drop": float(energy_values[0] - energy_values[-1]),
        "tail_window_energy_std": float(np.std(tail, ddof=0)),
        "tail_window_energy_slope": tail_slope,
    }


def _mean_nn_correlation_from_spins(
    spins_pm1: np.ndarray,
    nn_edges: np.ndarray,
) -> float:
    if nn_edges.size == 0:
        return float("nan")
    correlations = spins_pm1[:, nn_edges[:, 0]] * spins_pm1[:, nn_edges[:, 1]]
    return float(np.mean(correlations))


def _sampled_final_observable_summary(
    variational_state: SupportsSamplingAndLogValue,
    graph: SquareLattice,
    *,
    entropy_n_independent_runs: int,
    observable_n_independent_runs: int = 3,
    sample_batches: Sequence[Any] | None = None,
) -> dict[str, float]:
    if observable_n_independent_runs <= 0:
        raise ValueError("observable_n_independent_runs must be positive.")
    if sample_batches is not None:
        required_batches = max(observable_n_independent_runs, entropy_n_independent_runs)
        if len(sample_batches) < required_batches:
            raise ValueError("sample_batches must cover both observable and entropy runs.")

    abs_magnetization_values: list[float] = []
    nn_correlation_values: list[float] = []
    nn_edges = np.asarray(tuple(graph.iter_neighbor_pairs(1)), dtype=np.intp)
    for run_index in range(observable_n_independent_runs):
        if sample_batches is not None:
            current_sample_batch = sample_batches[run_index]
            if hasattr(current_sample_batch, "states"):
                sample_batch = np.asarray(current_sample_batch.states, dtype=np.uint8)
            else:
                sample_batch = np.asarray(current_sample_batch, dtype=np.uint8)
        else:
            sample_batch = np.asarray(
                variational_state.independent_sample(seed_offset=1_000 + run_index),
                dtype=np.uint8,
            )
        spins_pm1 = 2.0 * sample_batch.astype(np.float64) - 1.0
        per_sample_magnetization = np.mean(spins_pm1, axis=1)
        abs_magnetization_values.append(float(np.mean(np.abs(per_sample_magnetization))))
        nn_correlation_values.append(_mean_nn_correlation_from_spins(spins_pm1, nn_edges))

    subsystem = half_subsystem(graph.n_nodes)
    try:
        if sample_batches is not None:
            entropy_stats = _renyi2_entropy_statistics_from_sample_batches(
                variational_state,
                subsystem=subsystem,
                sample_batches=sample_batches[:entropy_n_independent_runs],
            )
        else:
            entropy_stats = observables.renyi2_entropy_statistics(
                variational_state,
                subsystem=subsystem,
                n_repeats=entropy_n_independent_runs,
                force_sampled=True,
            )
        half_partition_renyi2 = float(entropy_stats["mean"])
        half_partition_renyi2_std = float(entropy_stats["std"])
        entropy_success = 1.0
    except ValueError:
        half_partition_renyi2 = np.nan
        half_partition_renyi2_std = np.nan
        entropy_success = 0.0

    return {
        "final_half_partition_renyi2": half_partition_renyi2,
        "final_half_partition_renyi2_std": half_partition_renyi2_std,
        "entropy_estimator_success": entropy_success,
        "final_abs_magnetization": float(np.mean(abs_magnetization_values)),
        "final_abs_magnetization_std": float(np.std(abs_magnetization_values, ddof=0)),
        "final_nn_correlation": float(np.mean(nn_correlation_values)),
        "final_nn_correlation_std": float(np.std(nn_correlation_values, ddof=0)),
    }


def _non_ed_training_observables(
    variational_state: SupportsSamplingAndLogValue,
    graph: SquareLattice,
    *,
    step: int,
) -> dict[str, object]:
    sample_batch = np.asarray(variational_state.independent_sample(seed_offset=10_000 + step), dtype=np.uint8)
    spins_pm1 = 2.0 * sample_batch.astype(np.float64) - 1.0
    per_sample_magnetization = np.mean(spins_pm1, axis=1)
    nn_edges = np.asarray(tuple(graph.iter_neighbor_pairs(1)), dtype=np.intp)
    return {
        "abs_magnetization": float(np.mean(np.abs(per_sample_magnetization))),
        "nn_correlation": _mean_nn_correlation_from_spins(spins_pm1, nn_edges),
    }


def run_vmc_experiment(
    *,
    model_name: str,
    model_kwargs: dict[str, Any],
    lattice_shape: tuple[int, int] = (2, 2),
    pbc: bool = True,
    hamiltonian: str = "tfim",
    J: float = 1.0,
    h: float = 0.8,
    J1: float = 1.0,
    J2: float = 0.4,
    learning_rate: float = 1e-2,
    n_samples: int = 128,
    n_discard_per_chain: int = 16,
    n_chains: int = 8,
    n_iter: int = 20,
    callback_every: int = 5,
    entropy_n_independent_runs: int | None = None,
    entropy_force_sampled: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    system = build_system(
        lattice_shape=lattice_shape,
        pbc=pbc,
        hamiltonian=hamiltonian,
        J=J,
        h=h,
        J1=J1,
        J2=J2,
    )
    hilbert = system["hilbert"]
    model, variational_state, vmc_driver = build_vmc_experiment(
        hilbert=hilbert,
        operator=system["operator"],
        learning_rate=learning_rate,
        seed=seed,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        model_name=model_name,
        model_kwargs=model_kwargs,
        lattice_shape=lattice_shape,
    )
    entropy_logger = observables.entropy_callback(
        subsystem=half_subsystem(hilbert.size),
        force_sampled=entropy_force_sampled,
    )
    history = vmc_driver.run(
        n_iter,
        callback=entropy_logger,
        callback_every=callback_every,
    )
    history_df = history_table(history)
    exact = exact_observables_summary(system["operator"])
    final_energy = float(np.asarray(variational_state.energy(system["operator"])))
    independent_run_count = entropy_n_independent_runs
    if independent_run_count is None:
        independent_run_count = 5 if model_name.upper() == "RBM" else 1
    final_entropy = float(
        observables.renyi2_entropy(
            variational_state,
            subsystem=half_subsystem(hilbert.size),
            n_repeats=independent_run_count,
            force_sampled=entropy_force_sampled,
        )
    )
    if hamiltonian == "tfim" and not entropy_force_sampled:
        entropy_scan = renyi2_subsystem_scan_summary(
            variational_state,
            hilbert.size,
            n_independent_runs=independent_run_count,
            force_sampled=False,
        )
        sampled_entropy = entropy_scan
    else:
        sampled_entropy = sampled_entropy_scaling_summary(
            variational_state,
            hilbert.size,
            n_independent_runs=independent_run_count,
        )
        entropy_scan = sampled_entropy

    return {
        "model_name": model_name,
        "model_kwargs": dict(model_kwargs),
        "system": system,
        "history": history,
        "history_df": history_df,
        "exact": exact,
        "parameter_count": _count_model_parameters(variational_state.parameters),
        "final_energy": final_energy,
        "final_entropy": final_entropy,
        "energy_error": final_energy - exact["ground_energy"],
        "entropy_scan_table": entropy_scan["entropy_table"],
        "entropy_scan_samples": entropy_scan["entropy_samples"],
        "entropy_scan_fit": entropy_scan["scaling_fit"],
        "sampled_entropy_table": sampled_entropy["entropy_table"],
        "sampled_entropy_samples": sampled_entropy["entropy_samples"],
        "sampled_scaling_fit": sampled_entropy["scaling_fit"],
        "entropy_n_independent_runs": independent_run_count,
        "entropy_force_sampled": entropy_force_sampled,
    }


def _comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "model": result["model_name"],
            "final_energy": result["final_energy"],
            "exact_ground_energy": result["exact"]["ground_energy"],
            "energy_error": result["energy_error"],
            "final_renyi2_entropy": result["final_entropy"],
        }
        for result in results
    ]
    return pd.DataFrame(rows).sort_values("energy_error", key=np.abs).reset_index(drop=True)


def _count_model_parameters(params: Any) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params)))


def _scale_parameter_tree(tree: Any, factor: float) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf) * factor, tree)


def _transform_initial_parameters(
    params: Any,
    *,
    parameter_scale: float = 1.0,
    phase_scale: float = 1.0,
) -> Any:
    if isinstance(params, Mapping):
        updated: dict[str, Any] = {}
        for key, value in params.items():
            lower_key = key.lower()
            if "phase" in lower_key:
                updated[key] = _scale_parameter_tree(value, float(parameter_scale) * float(phase_scale))
                continue
            if key.startswith("Dense_") and isinstance(value, dict):
                dense_update: dict[str, Any] = {}
                for leaf_name, leaf in value.items():
                    leaf_array = jnp.asarray(leaf)
                    if leaf_name == "bias" and leaf_array.shape == (2,):
                        scaled_leaf = leaf_array * float(parameter_scale)
                        dense_update[leaf_name] = scaled_leaf.at[1].set(
                            leaf_array[1] * float(parameter_scale) * float(phase_scale)
                        )
                    elif leaf_name == "kernel" and leaf_array.shape[-1] == 2:
                        scaled_leaf = leaf_array * float(parameter_scale)
                        dense_update[leaf_name] = scaled_leaf.at[..., 1].set(
                            leaf_array[..., 1] * float(parameter_scale) * float(phase_scale)
                        )
                    else:
                        dense_update[leaf_name] = _transform_initial_parameters(
                            leaf,
                            parameter_scale=parameter_scale,
                            phase_scale=phase_scale,
                        )
                updated[key] = dense_update
                continue
            updated[key] = _transform_initial_parameters(
                value,
                parameter_scale=parameter_scale,
                phase_scale=phase_scale,
            )
        return updated
    return jnp.asarray(params) * float(parameter_scale)


def _zero_phase_parameters(params: Any) -> Any:
    return _transform_initial_parameters(params, phase_scale=0.0)


def initialize_random_parameters(
    model: Any,
    hilbert: SpinHilbert,
    *,
    seed: int,
    parameter_scale: float = 1.0,
    phase_scale: float = 1.0,
) -> Any:
    base_params = model.init(jax.random.PRNGKey(seed), hilbert)
    return _transform_initial_parameters(
        base_params,
        parameter_scale=parameter_scale,
        phase_scale=phase_scale,
    )


def _initialization_label(
    *,
    parameter_scale: float,
    phase_scale: float,
    explicit_label: str | None = None,
) -> str:
    if explicit_label is not None:
        return explicit_label
    labels: list[str] = []
    if not np.isclose(parameter_scale, 1.0):
        labels.append(f"scale={parameter_scale:g}")
    if np.isclose(phase_scale, 0.0):
        labels.append("real-amplitude")
    elif not np.isclose(phase_scale, 1.0):
        labels.append(f"phase_scale={phase_scale:g}")
    if not labels:
        return "default"
    return ", ".join(labels)


def _normalize_random_architecture_entry(
    model_label: str,
    config: Mapping[str, Any],
    *,
    real_amplitude_only: bool,
) -> dict[str, Any]:
    if "model_name" in config:
        model_name = str(config["model_name"])
        model_kwargs = dict(config.get("model_kwargs", {}))
        init_config = dict(config.get("initialization", {}))
        normalized_label = str(config.get("label", model_label))
    else:
        model_name = model_label
        model_kwargs = dict(config)
        init_config = {}
        normalized_label = model_label

    if bool(init_config.pop("real_amplitude_only", False)):
        init_config.setdefault("phase_scale", 0.0)
    if real_amplitude_only and "phase_scale" not in init_config:
        init_config["phase_scale"] = 0.0

    parameter_scale = float(init_config.get("parameter_scale", 1.0))
    phase_scale = float(init_config.get("phase_scale", 1.0))
    initialization_label = _initialization_label(
        parameter_scale=parameter_scale,
        phase_scale=phase_scale,
        explicit_label=cast(str | None, init_config.get("label")),
    )
    return {
        "model": normalized_label,
        "architecture_family": model_name,
        "model_kwargs": model_kwargs,
        "parameter_scale": parameter_scale,
        "phase_scale": phase_scale,
        "initialization_label": initialization_label,
    }


def _is_exact_entropy_unavailable_error(exc: ValueError) -> bool:
    return str(exc) == _EXACT_STATEVECTOR_UNAVAILABLE_MESSAGE


def sampler_acceptance_diagnostics(
    variational_state: SupportsSamplingAndLogValue,
    *,
    n_steps: int | None = None,
    seed_offset: int = 0,
) -> dict[str, Any]:
    sampling_state = cast(Any, variational_state)
    if not hasattr(sampling_state, "sampler") or not hasattr(sampling_state, "model") or not hasattr(
        sampling_state,
        "parameters",
    ):
        raise TypeError("variational_state must expose sampler, model, and parameters for diagnostics.")
    sampler = sampling_state.sampler
    if not isinstance(sampler, MetropolisLocal):
        raise TypeError("sampler diagnostics require a MetropolisLocal sampler.")

    total_steps = sampler.n_discard_per_chain + sampler._steps_per_chain() if n_steps is None else int(n_steps)
    if total_steps <= 0:
        raise ValueError("n_steps must be positive.")

    rng_key = jax.random.PRNGKey(sampler.seed + seed_offset + 1)
    init_key, step_key = jax.random.split(rng_key)
    states = sampler._random_states(init_key)
    log_values = jnp.asarray(sampling_state.model.log_psi(sampling_state.parameters, states))

    rows: list[dict[str, Any]] = []
    for step_index in range(total_steps):
        previous_states = states
        states, log_values, step_key = _metropolis_step(
            sampling_state.model.log_psi,
            sampling_state.parameters,
            states=states,
            log_values=log_values,
            rng_key=step_key,
            n_chains=sampler.n_chains,
            n_sites=sampler.hilbert.size,
        )
        accepted = np.any(np.asarray(states) != np.asarray(previous_states), axis=1)
        rows.append(
            {
                "step": step_index + 1,
                "phase": "burn_in" if step_index < sampler.n_discard_per_chain else "sampling",
                "acceptance_rate": float(np.mean(accepted)),
            }
        )

    acceptance_table = pd.DataFrame(rows)
    summary_table = cast(
        pd.DataFrame,
        cast(Any, acceptance_table)
        .groupby("phase", as_index=False)
        .agg(
            steps=("step", "count"),
            mean_acceptance=("acceptance_rate", "mean"),
            min_acceptance=("acceptance_rate", "min"),
            max_acceptance=("acceptance_rate", "max"),
        )
        .sort_values(by="phase")
        .reset_index(drop=True),
    )
    config_table = pd.DataFrame(
        [
            {"parameter": "n_sites", "value": sampler.hilbert.size},
            {"parameter": "n_chains", "value": sampler.n_chains},
            {"parameter": "n_samples", "value": sampler.n_samples},
            {"parameter": "n_discard_per_chain", "value": sampler.n_discard_per_chain},
            {"parameter": "steps_per_chain", "value": sampler._steps_per_chain()},
            {"parameter": "diagnostic_steps", "value": total_steps},
        ]
    )
    return {
        "config_table": config_table,
        "acceptance_table": acceptance_table,
        "summary_table": summary_table,
        "overall_acceptance": float(acceptance_table["acceptance_rate"].mean()),
    }


def renyi2_subsystem_scan_summary(
    variational_state: SupportsSamplingAndLogValue,
    n_sites: int,
    max_subsystem_size: int | None = None,
    n_independent_runs: int = 1,
    force_sampled: bool = False,
) -> dict[str, Any]:
    if n_independent_runs <= 0:
        raise ValueError("n_independent_runs must be positive.")
    if force_sampled:
        return sampled_entropy_scaling_summary(
            variational_state,
            n_sites=n_sites,
            max_subsystem_size=max_subsystem_size,
            n_independent_runs=n_independent_runs,
        )

    subsystem_limit = max(1, n_sites // 2) if max_subsystem_size is None else max_subsystem_size
    entropy_rows: list[dict[str, Any]] = []
    for subsystem_size in range(1, subsystem_limit + 1):
        subsystem = tuple(range(subsystem_size))
        entropy_rows.append(
            {
                "subsystem_size": subsystem_size,
                "renyi2": observables.renyi2_entropy(variational_state, subsystem=subsystem),
                "renyi2_std": 0.0,
            }
        )

    entropy_table = pd.DataFrame(entropy_rows)
    scaling_fit = (
        observables.fit_log_entropy_scaling(
            entropy_table["subsystem_size"].to_numpy(),
            entropy_table["renyi2"].to_numpy(),
        )
        if len(entropy_table) >= 2
        else None
    )
    return {
        "entropy_table": entropy_table,
        "entropy_samples": None,
        "scaling_fit": scaling_fit,
    }


def run_architecture_comparison(
    architecture_configs: dict[str, dict[str, Any]],
    *,
    seeds: Sequence[int],
    lattice_shape: tuple[int, int] = (2, 2),
    pbc: bool = True,
    hamiltonian: str = "tfim",
    J: float = 1.0,
    h: float = 0.8,
    J1: float = 1.0,
    J2: float = 0.4,
    n_samples: int = 128,
    n_discard_per_chain: int = 16,
    n_chains: int = 8,
    learning_rate: float = 1e-2,
    n_iter: int = 20,
    callback_every: int = 5,
    entropy_n_independent_runs: int = 1,
    max_subsystem_size: int | None = None,
    entropy_force_sampled: bool = False,
) -> dict[str, Any]:
    if not architecture_configs:
        raise ValueError("architecture_configs must not be empty.")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("seeds must contain at least one value.")

    system = build_system(
        lattice_shape=lattice_shape,
        pbc=pbc,
        hamiltonian=hamiltonian,
        J=J,
        h=h,
        J1=J1,
        J2=J2,
    )

    trial_rows: list[dict[str, Any]] = []
    entropy_scan_rows: list[pd.DataFrame] = []
    trial_results: list[dict[str, Any]] = []
    for model_name, model_kwargs in architecture_configs.items():
        for seed in seed_values:
            result = run_vmc_experiment(
                model_name=model_name,
                model_kwargs=model_kwargs,
                lattice_shape=lattice_shape,
                pbc=pbc,
                hamiltonian=hamiltonian,
                J=J,
                h=h,
                J1=J1,
                J2=J2,
                learning_rate=learning_rate,
                n_samples=n_samples,
                n_discard_per_chain=n_discard_per_chain,
                n_chains=n_chains,
                n_iter=n_iter,
                callback_every=callback_every,
                entropy_n_independent_runs=entropy_n_independent_runs,
                entropy_force_sampled=entropy_force_sampled,
                seed=seed,
            )
            entropy_table = result["entropy_scan_table"].copy()
            entropy_table["model"] = model_name
            entropy_table["seed"] = seed
            entropy_table["parameter_count"] = result["parameter_count"]
            entropy_scan_rows.append(entropy_table)

            max_partition = int(entropy_table["subsystem_size"].max())
            half_partition_row = entropy_table.loc[entropy_table["subsystem_size"] == max_partition].iloc[0]
            scaling_fit = result["entropy_scan_fit"]
            trial_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "parameter_count": result["parameter_count"],
                    "max_subsystem_size": max_partition,
                    "final_energy": result["final_energy"],
                    "exact_ground_energy": result["exact"]["ground_energy"],
                    "energy_error": result["energy_error"],
                    "half_partition_renyi2": float(half_partition_row["renyi2"]),
                    "scaling_slope": float(scaling_fit["slope"]) if scaling_fit is not None else np.nan,
                    "scaling_r_squared": float(scaling_fit["r_squared"]) if scaling_fit is not None else np.nan,
                }
            )
            trial_results.append(
                {
                    **result,
                    "seed": seed,
                    "parameter_count": result["parameter_count"],
                }
            )

    trial_table = pd.DataFrame(trial_rows).sort_values(["model", "seed"]).reset_index(drop=True)
    entropy_scan_table = cast(
        Any,
        pd.concat(entropy_scan_rows, ignore_index=True)
        .groupby(["model", "subsystem_size"], as_index=False)
        .agg(
            parameter_count=("parameter_count", "first"),
            n_trials=("seed", "nunique"),
            renyi2=("renyi2", "mean"),
            renyi2_std=("renyi2", "std"),
        )
        .fillna({"renyi2_std": 0.0})
    ).sort_values(by=["model", "subsystem_size"]).reset_index(drop=True)
    summary_table = cast(
        Any,
        trial_table.groupby("model", as_index=False)
        .agg(
            parameter_count=("parameter_count", "first"),
            n_trials=("seed", "nunique"),
            final_energy=("final_energy", "mean"),
            exact_ground_energy=("exact_ground_energy", "first"),
            energy_error=("energy_error", "mean"),
            half_partition_renyi2=("half_partition_renyi2", "mean"),
            half_partition_renyi2_std=("half_partition_renyi2", "std"),
            scaling_slope=("scaling_slope", "mean"),
            scaling_r_squared=("scaling_r_squared", "mean"),
        )
        .fillna({"half_partition_renyi2_std": 0.0})
    ).sort_values(by="model").reset_index(drop=True)
    return {
        "system": system,
        "trial_table": trial_table,
        "summary_table": summary_table,
        "entropy_scan_table": entropy_scan_table,
        "trial_results": trial_results,
        "entropy_n_independent_runs": entropy_n_independent_runs,
        "entropy_force_sampled": entropy_force_sampled,
    }


def run_architecture_disorder_comparison(
    architecture_configs: dict[str, dict[str, Any]],
    *,
    seeds: Sequence[int],
    lattice_shape: tuple[int, int] = (2, 2),
    pbc: bool = True,
    hamiltonian: str = "tfim",
    J: float = 1.0,
    h: float = 0.8,
    J1: float = 1.0,
    J2: float = 0.4,
    n_samples: int = 128,
    n_discard_per_chain: int = 16,
    n_chains: int = 8,
    learning_rate: float = 1e-2,
    n_iter: int = 20,
    callback_every: int = 5,
    entropy_n_independent_runs: int = 1,
    max_subsystem_size: int | None = None,
    entropy_force_sampled: bool = False,
) -> dict[str, Any]:
    return run_architecture_comparison(
        architecture_configs,
        seeds=seeds,
        lattice_shape=lattice_shape,
        pbc=pbc,
        hamiltonian=hamiltonian,
        J=J,
        h=h,
        J1=J1,
        J2=J2,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        learning_rate=learning_rate,
        n_iter=n_iter,
        callback_every=callback_every,
        entropy_n_independent_runs=entropy_n_independent_runs,
        max_subsystem_size=max_subsystem_size,
        entropy_force_sampled=entropy_force_sampled,
    )


def run_architecture_benchmark(
    *,
    architecture_configs: dict[str, dict[str, Any]],
    lattice_shape: tuple[int, int] = (2, 2),
    pbc: bool = True,
    hamiltonian: str = "tfim",
    J: float = 1.0,
    h: float = 0.8,
    J1: float = 1.0,
    J2: float = 0.4,
    learning_rate: float = 1e-2,
    n_samples: int = 128,
    n_discard_per_chain: int = 16,
    n_chains: int = 8,
    n_iter: int = 20,
    callback_every: int = 5,
    entropy_force_sampled: bool = False,
    base_seed: int = 0,
    netket_reference_energy: float | None = None,
) -> dict[str, Any]:
    results = run_architecture_sweep(
        architecture_configs,
        lattice_shape=lattice_shape,
        pbc=pbc,
        hamiltonian=hamiltonian,
        J=J,
        h=h,
        J1=J1,
        J2=J2,
        learning_rate=learning_rate,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        n_iter=n_iter,
        callback_every=callback_every,
        entropy_force_sampled=entropy_force_sampled,
        base_seed=base_seed,
    )
    summary_table = _comparison_table(results)
    summary_table = summary_table.sort_values("model").reset_index(drop=True)
    summary_table["netket_reference_energy"] = netket_reference_energy
    summary_table["netket_gap"] = (
        np.nan
        if netket_reference_energy is None
        else summary_table["final_energy"] - float(netket_reference_energy)
    )
    return {
        "results": results,
        "summary_table": summary_table,
        "netket_reference_energy": netket_reference_energy,
    }


def run_random_architecture_study(
    architecture_configs: dict[str, dict[str, Any]],
    *,
    seeds: Sequence[int],
    lattice_shape: tuple[int, int] = (2, 2),
    pbc: bool = True,
    hamiltonian: str = "tfim",
    J: float = 1.0,
    h: float = 0.8,
    J1: float = 1.0,
    J2: float = 0.4,
    n_samples: int = 128,
    n_discard_per_chain: int = 16,
    n_chains: int = 8,
    entropy_n_independent_runs: int = 4,
    max_subsystem_size: int | None = None,
    real_amplitude_only: bool = False,
) -> dict[str, Any]:
    if not architecture_configs:
        raise ValueError("architecture_configs must not be empty.")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("seeds must contain at least one value.")

    system = build_system(
        lattice_shape=lattice_shape,
        pbc=pbc,
        hamiltonian=hamiltonian,
        J=J,
        h=h,
        J1=J1,
        J2=J2,
    )
    hilbert = system["hilbert"]
    subsystem_limit = max(1, hilbert.size // 2) if max_subsystem_size is None else max_subsystem_size

    trial_rows: list[dict[str, Any]] = []
    entropy_scan_rows: list[pd.DataFrame] = []
    trial_results: list[dict[str, Any]] = []
    for raw_model_label, raw_config in architecture_configs.items():
        entry = _normalize_random_architecture_entry(
            raw_model_label,
            raw_config,
            real_amplitude_only=real_amplitude_only,
        )
        model_name = str(entry["architecture_family"])
        model_label = str(entry["model"])
        model_kwargs = dict(cast(dict[str, Any], entry["model_kwargs"]))
        model = build_model(
            model_name=model_name,
            model_kwargs=model_kwargs,
            lattice_shape=lattice_shape,
        )
        for seed in seed_values:
            params = initialize_random_parameters(
                model,
                hilbert,
                seed=seed,
                parameter_scale=float(entry["parameter_scale"]),
                phase_scale=float(entry["phase_scale"]),
            )
            variational_state = build_variational_state(
                model=model,
                hilbert=hilbert,
                seed=seed,
                n_samples=n_samples,
                n_discard_per_chain=n_discard_per_chain,
                n_chains=n_chains,
                params=params,
            )
            exact_entropy_scan: dict[str, Any] | None
            try:
                exact_entropy_scan = renyi2_subsystem_scan_summary(
                    variational_state,
                    hilbert.size,
                    max_subsystem_size=subsystem_limit,
                    n_independent_runs=1,
                    force_sampled=False,
                )
                exact_available = True
            except ValueError as exc:
                if not _is_exact_entropy_unavailable_error(exc):
                    raise
                exact_entropy_scan = None
                exact_available = False
            sampled_entropy_scan = sampled_entropy_scaling_summary(
                variational_state,
                hilbert.size,
                max_subsystem_size=subsystem_limit,
                n_independent_runs=entropy_n_independent_runs,
            )
            parameter_count = _count_model_parameters(variational_state.parameters)

            sampled_table = sampled_entropy_scan["entropy_table"].copy()
            sampled_table["model"] = model_label
            sampled_table["architecture_family"] = model_name
            sampled_table["seed"] = seed
            sampled_table["parameter_count"] = parameter_count
            sampled_table["parameter_scale"] = float(entry["parameter_scale"])
            sampled_table["phase_scale"] = float(entry["phase_scale"])
            sampled_table["initialization_label"] = str(entry["initialization_label"])
            sampled_table["exact_available"] = exact_available
            if exact_entropy_scan is None:
                exact_table = pd.DataFrame(
                    {
                        "subsystem_size": sampled_table["subsystem_size"].to_numpy(),
                        "exact_renyi2": np.full(len(sampled_table), np.nan, dtype=np.float64),
                        "exact_renyi2_std": np.full(len(sampled_table), np.nan, dtype=np.float64),
                    }
                )
            else:
                exact_table = exact_entropy_scan["entropy_table"].copy().rename(
                    columns={"renyi2": "exact_renyi2", "renyi2_std": "exact_renyi2_std"}
                )
            sampled_table = sampled_table.merge(
                exact_table[["subsystem_size", "exact_renyi2", "exact_renyi2_std"]],
                on="subsystem_size",
                how="left",
            )
            entropy_scan_rows.append(sampled_table)

            max_partition = int(sampled_table["subsystem_size"].max())
            half_partition_row = sampled_table.loc[sampled_table["subsystem_size"] == max_partition].iloc[0]
            valid_entropy_points = int(np.count_nonzero(np.isfinite(sampled_table["renyi2"])))
            trial_rows.append(
                {
                    "model": model_label,
                    "architecture_family": model_name,
                    "seed": seed,
                    "parameter_count": parameter_count,
                    "parameter_scale": float(entry["parameter_scale"]),
                    "phase_scale": float(entry["phase_scale"]),
                    "initialization_label": str(entry["initialization_label"]),
                    "exact_available": exact_available,
                    "max_subsystem_size": max_partition,
                    "valid_entropy_points": valid_entropy_points,
                    "valid_entropy_fraction": valid_entropy_points / max_partition if max_partition > 0 else np.nan,
                    "half_partition_exact_renyi2": float(half_partition_row["exact_renyi2"]),
                    "half_partition_sampled_renyi2": float(half_partition_row["renyi2"]),
                    "half_partition_sampled_std": float(half_partition_row["renyi2_std"]),
                    "sampled_minus_exact": float(half_partition_row["renyi2"] - half_partition_row["exact_renyi2"]),
                }
            )
            trial_results.append(
                {
                    "model_name": model_name,
                    "model_label": model_label,
                    "model_kwargs": dict(model_kwargs),
                    "seed": seed,
                    "parameter_count": parameter_count,
                    "parameter_scale": float(entry["parameter_scale"]),
                    "phase_scale": float(entry["phase_scale"]),
                    "initialization_label": str(entry["initialization_label"]),
                    "system": system,
                    "exact_available": exact_available,
                    "exact_entropy_scan_table": None if exact_entropy_scan is None else exact_entropy_scan["entropy_table"],
                    "sampled_entropy_scan_table": sampled_entropy_scan["entropy_table"],
                    "sampled_entropy_scan_samples": sampled_entropy_scan["entropy_samples"],
                    "real_amplitude_only": real_amplitude_only,
                }
            )

    trial_table = pd.DataFrame(trial_rows).sort_values(["model", "seed"]).reset_index(drop=True)
    entropy_scan_table = cast(
        Any,
        pd.concat(entropy_scan_rows, ignore_index=True)
        .groupby(["model", "architecture_family", "initialization_label", "subsystem_size"], as_index=False)
        .agg(
            parameter_count=("parameter_count", "first"),
            parameter_scale=("parameter_scale", "first"),
            phase_scale=("phase_scale", "first"),
            exact_available=("exact_available", "max"),
            n_trials=("seed", "nunique"),
            exact_renyi2=("exact_renyi2", "mean"),
            sampled_renyi2=("renyi2", "mean"),
            sampled_renyi2_std=("renyi2", "std"),
            estimator_std=("renyi2_std", "mean"),
        )
        .fillna({"sampled_renyi2_std": 0.0, "estimator_std": 0.0})
    ).sort_values(by=["model", "subsystem_size"]).reset_index(drop=True)
    summary_table = cast(
        Any,
        trial_table.groupby(["model", "architecture_family", "initialization_label"], as_index=False)
        .agg(
            parameter_count=("parameter_count", "first"),
            parameter_scale=("parameter_scale", "first"),
            phase_scale=("phase_scale", "first"),
            exact_available=("exact_available", "max"),
            n_trials=("seed", "nunique"),
            valid_entropy_points=("valid_entropy_points", "mean"),
            valid_entropy_fraction=("valid_entropy_fraction", "mean"),
            half_partition_exact_renyi2=("half_partition_exact_renyi2", "mean"),
            half_partition_sampled_renyi2=("half_partition_sampled_renyi2", "mean"),
            half_partition_sampled_std=("half_partition_sampled_renyi2", "std"),
            estimator_std=("half_partition_sampled_std", "mean"),
            sampled_minus_exact=("sampled_minus_exact", "mean"),
        )
        .fillna({"half_partition_sampled_std": 0.0, "estimator_std": 0.0})
    ).sort_values(by="model").reset_index(drop=True)
    return {
        "system": system,
        "trial_table": trial_table,
        "summary_table": summary_table,
        "entropy_scan_table": entropy_scan_table,
        "trial_results": trial_results,
        "entropy_n_independent_runs": entropy_n_independent_runs,
        "real_amplitude_only": real_amplitude_only,
    }


def _default_sweep_label(config: dict[str, Any]) -> str:
    hamiltonian = str(config["hamiltonian"])
    lattice_shape = tuple(config.get("lattice_shape", (2, 2)))
    lattice_label = "x".join(str(length) for length in lattice_shape)
    if hamiltonian == "tfim":
        return f"tfim_L{lattice_label}_h{config.get('h', 0.8)}"
    if hamiltonian == "j1_j2":
        return f"j1j2_L{lattice_label}_J2{config.get('J2', 0.4)}"
    return f"{hamiltonian}_L{lattice_label}"


def run_non_ed_vmc_benchmark(
    benchmark_configs: dict[str, dict[str, Any]],
    *,
    sweep_points: Sequence[dict[str, Any]],
    learning_rate: float = 1e-2,
    n_samples: int = 128,
    n_discard_per_chain: int = 16,
    n_chains: int = 8,
    n_iter: int = 20,
    callback_every: int = 5,
    entropy_n_independent_runs: int = 4,
    max_entropy_subsystem_size: int | None = 4,
    base_seed: int = 0,
) -> dict[str, Any]:
    if not benchmark_configs:
        raise ValueError("benchmark_configs must not be empty.")
    if not sweep_points:
        raise ValueError("sweep_points must not be empty.")

    summary_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    entropy_frames: list[pd.DataFrame] = []
    benchmark_results: list[dict[str, Any]] = []
    for config_offset, (benchmark_label, benchmark_config) in enumerate(benchmark_configs.items()):
        if "model_name" not in benchmark_config:
            raise ValueError("each benchmark config must define model_name.")
        model_name = str(benchmark_config["model_name"])
        model_kwargs = dict(benchmark_config.get("model_kwargs", {}))

        for sweep_offset, raw_point in enumerate(sweep_points):
            point = dict(raw_point)
            if "hamiltonian" not in point:
                raise ValueError("each sweep point must define a hamiltonian.")

            system_label = str(point.pop("label", _default_sweep_label(point)))
            raw_lattice_shape = tuple(point.pop("lattice_shape", (2, 2)))
            if len(raw_lattice_shape) != 2:
                raise ValueError("lattice_shape must contain exactly two dimensions.")
            lattice_shape = (int(raw_lattice_shape[0]), int(raw_lattice_shape[1]))
            pbc = bool(point.pop("pbc", True))
            hamiltonian = str(point.pop("hamiltonian"))
            seed = base_seed + (config_offset * len(sweep_points)) + sweep_offset

            system = build_system(
                lattice_shape=lattice_shape,
                pbc=pbc,
                hamiltonian=hamiltonian,
                **point,
            )
            hilbert = system["hilbert"]
            model, variational_state, vmc_driver = build_vmc_experiment(
                hilbert=hilbert,
                operator=system["operator"],
                learning_rate=learning_rate,
                seed=seed,
                n_samples=n_samples,
                n_discard_per_chain=n_discard_per_chain,
                n_chains=n_chains,
                model_name=model_name,
                model_kwargs=model_kwargs,
                lattice_shape=lattice_shape,
                exact_backend_max_states=0,
            )
            parameter_count = _count_model_parameters(variational_state.parameters)
            callback_runtime_seconds = 0.0

            def benchmark_callback(step: int, driver: Any, graph: SquareLattice = system["graph"]) -> dict[str, object]:
                nonlocal callback_runtime_seconds
                callback_start = perf_counter()
                result = _non_ed_training_observables(
                    driver.variational_state,
                    graph,
                    step=step,
                )
                callback_runtime_seconds += perf_counter() - callback_start
                return result

            run_start = perf_counter()
            history = vmc_driver.run(
                n_iter,
                callback=benchmark_callback,
                callback_every=callback_every,
            )
            run_runtime_seconds = perf_counter() - run_start
            training_runtime_seconds = max(0.0, run_runtime_seconds - callback_runtime_seconds)
            history_df = history_table(history)
            postprocessing_start = perf_counter()
            observable_summary_runs = 3
            shared_diagnostic_samples = _independent_sample_batches_with_log_values(
                variational_state,
                n_independent_runs=max(observable_summary_runs, entropy_n_independent_runs),
            )
            if hasattr(variational_state, "energy_on_sample_batch"):
                final_energy = float(
                    np.asarray(
                        cast(Any, variational_state).energy_on_sample_batch(
                            system["operator"],
                            shared_diagnostic_samples[0],
                        )
                    )
                )
            else:
                final_energy = float(np.asarray(variational_state.energy(system["operator"])))
            final_row: dict[str, Any] = {
                column: np.nan
                for column in history_df.columns
                if column not in {"step", "energy"}
            }
            final_row.update(
                {
                    "step": n_iter,
                    "energy": final_energy,
                    "is_post_update": True,
                }
            )
            history_df["is_post_update"] = False
            history_df = pd.concat(
                [history_df, pd.DataFrame([final_row])],
                ignore_index=True,
            )
            history_df["benchmark_label"] = benchmark_label
            history_df["model"] = model_name
            history_df["system_label"] = system_label
            history_df["lattice_shape"] = [lattice_shape] * len(history_df)
            history_df["n_sites"] = hilbert.size
            history_df["parameter_count"] = parameter_count
            history_df["seed"] = seed
            history_frames.append(history_df)

            entropy_scan_start = perf_counter()
            entropy_scan = sampled_entropy_scaling_summary(
                variational_state,
                hilbert.size,
                max_subsystem_size=max_entropy_subsystem_size,
                n_independent_runs=entropy_n_independent_runs,
                sample_batches=shared_diagnostic_samples[:entropy_n_independent_runs],
            )
            entropy_scan_runtime_seconds = perf_counter() - entropy_scan_start
            entropy_table = entropy_scan["entropy_table"].copy()
            entropy_table["benchmark_label"] = benchmark_label
            entropy_table["model"] = model_name
            entropy_table["system_label"] = system_label
            entropy_table["lattice_shape"] = [lattice_shape] * len(entropy_table)
            entropy_table["n_sites"] = hilbert.size
            entropy_table["parameter_count"] = parameter_count
            entropy_frames.append(entropy_table)
            valid_entropy_points = int(np.count_nonzero(np.isfinite(entropy_table["renyi2"])))

            trend_metrics = _energy_trend_metrics(history_df)
            observable_metrics = _sampled_final_observable_summary(
                variational_state,
                system["graph"],
                entropy_n_independent_runs=entropy_n_independent_runs,
                observable_n_independent_runs=observable_summary_runs,
                sample_batches=shared_diagnostic_samples,
            )
            postprocessing_runtime_seconds = max(
                0.0,
                perf_counter() - postprocessing_start - entropy_scan_runtime_seconds,
            )
            report_runtime_seconds = (
                callback_runtime_seconds
                + postprocessing_runtime_seconds
                + entropy_scan_runtime_seconds
            )
            total_runtime_seconds = training_runtime_seconds + report_runtime_seconds

            summary_rows.append(
                {
                    "benchmark_label": benchmark_label,
                    "benchmark_mode": "sampled",
                    "model": model_name,
                    "model_kwargs": model_kwargs,
                    "system_label": system_label,
                    "hamiltonian": hamiltonian,
                    "lattice_shape": lattice_shape,
                    "n_sites": hilbert.size,
                    "seed": seed,
                    "parameter_count": parameter_count,
                    "final_energy": final_energy,
                    "final_energy_per_site": final_energy / hilbert.size,
                    "training_runtime_seconds": training_runtime_seconds,
                    "mean_training_step_time_seconds": training_runtime_seconds / max(1, n_iter),
                    "callback_runtime_seconds": callback_runtime_seconds,
                    "postprocessing_runtime_seconds": postprocessing_runtime_seconds,
                    "entropy_scan_runtime_seconds": entropy_scan_runtime_seconds,
                    "report_runtime_seconds": report_runtime_seconds,
                    "total_runtime_seconds": total_runtime_seconds,
                    "history_points": len(history_df),
                    "valid_entropy_points": valid_entropy_points,
                    **trend_metrics,
                    **observable_metrics,
                }
            )
            benchmark_results.append(
                {
                    "benchmark_label": benchmark_label,
                    "benchmark_mode": "sampled",
                    "model_name": model_name,
                    "model_kwargs": model_kwargs,
                    "system_label": system_label,
                    "seed": seed,
                    "model": model,
                    "system": system,
                    "history": history,
                    "history_df": history_df,
                    "entropy_scan_table": entropy_table,
                    "parameter_count": parameter_count,
                    "final_energy": final_energy,
                    "training_runtime_seconds": training_runtime_seconds,
                    "callback_runtime_seconds": callback_runtime_seconds,
                    "postprocessing_runtime_seconds": postprocessing_runtime_seconds,
                    "entropy_scan_runtime_seconds": entropy_scan_runtime_seconds,
                    "report_runtime_seconds": report_runtime_seconds,
                    "total_runtime_seconds": total_runtime_seconds,
                    "trend_metrics": trend_metrics,
                    "observable_metrics": observable_metrics,
                }
            )

    summary_table = pd.DataFrame(summary_rows).sort_values(
        by=["n_sites", "model", "parameter_count", "benchmark_label"]
    ).reset_index(drop=True)
    training_history_table = pd.concat(history_frames, ignore_index=True)
    entropy_scan_table = pd.concat(entropy_frames, ignore_index=True)
    return {
        "summary_table": summary_table,
        "training_history_table": training_history_table,
        "entropy_scan_table": entropy_scan_table,
        "benchmark_results": benchmark_results,
        "entropy_n_independent_runs": entropy_n_independent_runs,
        "max_entropy_subsystem_size": max_entropy_subsystem_size,
    }


def run_hamiltonian_system_size_sweep(
    sweep_points: Sequence[dict[str, Any]],
    *,
    model_name: str,
    model_kwargs: dict[str, Any],
    learning_rate: float = 1e-2,
    n_samples: int = 128,
    n_discard_per_chain: int = 16,
    n_chains: int = 8,
    n_iter: int = 20,
    callback_every: int = 5,
    entropy_n_independent_runs: int | None = None,
    entropy_force_sampled: bool = False,
    base_seed: int = 0,
) -> dict[str, Any]:
    if not sweep_points:
        raise ValueError("sweep_points must not be empty.")

    summary_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    sweep_results: list[dict[str, Any]] = []
    for offset, raw_point in enumerate(sweep_points):
        point = dict(raw_point)
        if "hamiltonian" not in point:
            raise ValueError("each sweep point must define a hamiltonian.")

        label = str(point.pop("label", _default_sweep_label(point)))
        raw_lattice_shape = tuple(point.pop("lattice_shape", (2, 2)))
        if len(raw_lattice_shape) != 2:
            raise ValueError("lattice_shape must contain exactly two dimensions.")
        lattice_shape = (int(raw_lattice_shape[0]), int(raw_lattice_shape[1]))
        pbc = bool(point.pop("pbc", True))
        hamiltonian = str(point.pop("hamiltonian"))
        seed = base_seed + offset
        coupling_row = {
            "J": float(point.get("J", np.nan)),
            "h": float(point.get("h", np.nan)),
            "J1": float(point.get("J1", np.nan)),
            "J2": float(point.get("J2", np.nan)),
        }
        result = run_vmc_experiment(
            model_name=model_name,
            model_kwargs=model_kwargs,
            lattice_shape=lattice_shape,
            pbc=pbc,
            hamiltonian=hamiltonian,
            learning_rate=learning_rate,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            n_chains=n_chains,
            n_iter=n_iter,
            callback_every=callback_every,
            entropy_n_independent_runs=entropy_n_independent_runs,
            entropy_force_sampled=entropy_force_sampled,
            seed=seed,
            **point,
        )
        history_frame = result["history_df"].copy()
        history_frame["sweep_label"] = label
        history_frame["hamiltonian"] = hamiltonian
        history_frame["lattice_shape"] = [lattice_shape] * len(history_frame)
        history_frame["seed"] = seed
        for key, value in coupling_row.items():
            history_frame[key] = value
        history_frames.append(history_frame)

        summary_rows.append(
            {
                "sweep_label": label,
                "hamiltonian": hamiltonian,
                "lattice_shape": lattice_shape,
                "n_sites": int(np.prod(lattice_shape)),
                "seed": seed,
                "final_energy": result["final_energy"],
                "exact_ground_energy": result["exact"]["ground_energy"],
                "energy_error": result["energy_error"],
                "final_renyi2_entropy": result["final_entropy"],
                "history_points": len(history_frame),
                **coupling_row,
            }
        )
        sweep_results.append(
            {
                "label": label,
                "hamiltonian": hamiltonian,
                "lattice_shape": lattice_shape,
                "seed": seed,
                "couplings": coupling_row,
                **result,
            }
        )

    summary_table = pd.DataFrame(summary_rows).sort_values(by=["hamiltonian", "n_sites", "seed"]).reset_index(drop=True)
    training_history_table = pd.concat(history_frames, ignore_index=True)
    return {
        "summary_table": summary_table,
        "training_history_table": training_history_table,
        "sweep_results": sweep_results,
        "model_name": model_name,
        "model_kwargs": dict(model_kwargs),
        "entropy_force_sampled": entropy_force_sampled,
    }


def _ghz_statevector(n_sites: int) -> np.ndarray:
    if n_sites <= 0:
        raise ValueError("n_sites must be positive.")
    state = np.zeros(1 << n_sites, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2.0)
    state[-1] = 1.0 / np.sqrt(2.0)
    return state


def _ghz_state_metrics(
    statevector: np.ndarray | Sequence[complex],
    subsystem: tuple[int, ...] | None = None,
) -> dict[str, float]:
    flat_state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    n_sites = int(np.log2(flat_state.size))
    if (1 << n_sites) != flat_state.size:
        raise ValueError("statevector length must be a power of two.")
    norm = np.linalg.norm(flat_state)
    if norm == 0:
        raise ValueError("statevector must have non-zero norm.")

    normalized_state = flat_state / norm
    target = _ghz_statevector(n_sites)
    subsystem_sites = half_subsystem(n_sites) if subsystem is None else subsystem
    return {
        "ghz_fidelity": float(np.abs(np.vdot(target, normalized_state)) ** 2),
        "cat_sector_weight": float(np.abs(normalized_state[0]) ** 2 + np.abs(normalized_state[-1]) ** 2),
        "half_partition_renyi2": observables.renyi_entropy_from_statevector(
            normalized_state,
            subsystem=subsystem_sites,
            alpha=2.0,
        ),
    }


def run_ghz_bonus_workflow(
    *,
    model_name: str,
    model_kwargs: dict[str, Any],
    lattice_shape: tuple[int, int] = (2, 2),
    learning_rate: float = 1e-2,
    n_samples: int = 128,
    n_discard_per_chain: int = 16,
    n_chains: int = 8,
    n_iter: int = 20,
    callback_every: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    system = build_system(
        lattice_shape=lattice_shape,
        pbc=True,
        hamiltonian="tfim",
        J=1.0,
        h=0.0,
    )
    hilbert = system["hilbert"]
    model, variational_state, vmc_driver = build_vmc_experiment(
        hilbert=hilbert,
        operator=system["operator"],
        learning_rate=learning_rate,
        seed=seed,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        model_name=model_name,
        model_kwargs=model_kwargs,
        lattice_shape=lattice_shape,
    )
    entropy_logger = observables.entropy_callback(subsystem=half_subsystem(hilbert.size))
    history = vmc_driver.run(
        n_iter,
        callback=entropy_logger,
        callback_every=callback_every,
    )
    history_df = history_table(history)
    final_statevector = np.asarray(variational_state.exact_statevector(), dtype=np.complex128)
    metrics = _ghz_state_metrics(final_statevector)
    return {
        "system": system,
        "history": history,
        "history_df": history_df,
        "final_energy": float(np.asarray(variational_state.energy(system["operator"]))),
        "target_statevector": _ghz_statevector(hilbert.size),
        "final_statevector": final_statevector,
        "ghz_metrics": metrics,
    }


def run_architecture_sweep(
    architecture_configs: dict[str, dict[str, Any]],
    *,
    lattice_shape: tuple[int, int] = (2, 2),
    pbc: bool = True,
    hamiltonian: str = "tfim",
    J: float = 1.0,
    h: float = 0.8,
    J1: float = 1.0,
    J2: float = 0.4,
    learning_rate: float = 1e-2,
    n_samples: int = 128,
    n_discard_per_chain: int = 16,
    n_chains: int = 8,
    n_iter: int = 20,
    callback_every: int = 5,
    entropy_force_sampled: bool = False,
    base_seed: int = 0,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for offset, (model_name, model_kwargs) in enumerate(architecture_configs.items()):
        results.append(
            run_vmc_experiment(
                model_name=model_name,
                model_kwargs=model_kwargs,
                lattice_shape=lattice_shape,
                pbc=pbc,
                hamiltonian=hamiltonian,
                J=J,
                h=h,
                J1=J1,
                J2=J2,
                learning_rate=learning_rate,
                n_samples=n_samples,
                n_discard_per_chain=n_discard_per_chain,
                n_chains=n_chains,
                n_iter=n_iter,
                callback_every=callback_every,
                entropy_force_sampled=entropy_force_sampled,
                seed=base_seed + offset,
            )
        )
    return results


def parameter_table(
    *,
    lattice_shape: tuple[int, int],
    hamiltonian: str,
    training_config: dict[str, Any],
    model_name: str,
    model_kwargs: dict[str, Any],
    coupling_config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"section": "system", "parameter": "lattice_shape", "value": lattice_shape})
    rows.append({"section": "system", "parameter": "hamiltonian", "value": hamiltonian})
    for key, value in coupling_config.items():
        rows.append({"section": "couplings", "parameter": key, "value": value})
    rows.append({"section": "model", "parameter": "model_name", "value": model_name})
    for key, value in model_kwargs.items():
        rows.append({"section": "model", "parameter": key, "value": value})
    for key, value in training_config.items():
        rows.append({"section": "training", "parameter": key, "value": value})
    return pd.DataFrame(rows)
