"""Helper functions for the high-level NQS showcase notebook."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast
import jax
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nqs import CNN  # noqa: E402
from src.nqs import FFNN  # noqa: E402
from src.nqs import RBM  # noqa: E402
from src.nqs import SpinHilbert  # noqa: E402
from src.nqs import SquareLattice  # noqa: E402
from src.nqs import build_variational_state  # noqa: E402
from src.nqs import build_vmc_driver  # noqa: E402
from src.nqs import j1_j2  # noqa: E402
from src.nqs.exact_diag import exact_ground_state  # noqa: E402
from src.nqs.operator import tfim  # noqa: E402
import src.nqs.observables as observables  # noqa: E402

if TYPE_CHECKING:
    from src.nqs.observables import SupportsSamplingAndLogValue


def half_subsystem(n_sites: int) -> tuple[int, ...]:
    return tuple(range(max(1, n_sites // 2)))


def make_model(model_name: str, lattice_shape: tuple[int, int], **model_kwargs: Any):
    normalized_name = model_name.upper()
    if normalized_name == "RBM":
        return RBM(**model_kwargs)
    if normalized_name == "FFNN":
        return FFNN(**model_kwargs)
    if normalized_name == "CNN":
        kwargs = dict(model_kwargs)
        kwargs.setdefault("spatial_shape", lattice_shape)
        return CNN(**kwargs)
    raise ValueError(f"Unsupported model_name: {model_name}")


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
        "netket_operator": hamiltonian_operator.to_netket(),
        "hamiltonian": hamiltonian,
        "parameters": {"J": J, "h": h, "J1": J1, "J2": J2, "pbc": pbc},
    }


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
    exact = exact_ground_state(operator)
    hilbert = operator.hilbert
    subsystem_sites = half_subsystem(hilbert.size) if subsystem is None else subsystem
    states = hilbert.all_states()
    probabilities = np.abs(exact["ground_state"]) ** 2

    correlation_rows: list[dict[str, Any]] = []
    for site_i in range(hilbert.size):
        for site_j in range(hilbert.size):
            correlation_rows.append(
                {
                    "site_i": site_i,
                    "site_j": site_j,
                    "correlation": observables.spin_spin_correlation(
                        states,
                        site_i=site_i,
                        site_j=site_j,
                        weights=probabilities,
                    ),
                }
            )

    spectrum_table = pd.DataFrame(
        {
            "level": np.arange(min(8, hilbert.n_states)),
            "energy": exact["eigenvalues"][: min(8, hilbert.n_states)],
        }
    )

    entropy_rows: list[dict[str, Any]] = []
    for subsystem_size in range(1, (hilbert.size // 2) + 1):
        current_subsystem = tuple(range(subsystem_size))
        entropy_rows.append(
            {
                "subsystem_size": subsystem_size,
                "von_neumann": observables.von_neumann_entropy(exact["ground_state"], current_subsystem),
                "renyi2": observables.renyi_entropy_from_statevector(
                    exact["ground_state"],
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
        "ground_energy": exact["ground_energy"],
        "ground_state": exact["ground_state"],
        "spectrum_table": spectrum_table,
        "entropy_table": entropy_table,
        "scaling_fit": scaling_fit,
        "half_partition_von_neumann": observables.von_neumann_entropy(exact["ground_state"], subsystem_sites),
        "half_partition_renyi2": observables.renyi_entropy_from_statevector(
            exact["ground_state"],
            subsystem_sites,
            alpha=2.0,
        ),
        "correlation_matrix": pd.DataFrame(correlation_rows).pivot(
            index="site_i",
            columns="site_j",
            values="correlation",
        ),
    }


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


def sampled_entropy_scaling_summary(
    variational_state: SupportsSamplingAndLogValue,
    n_sites: int,
    max_subsystem_size: int | None = None,
    n_independent_runs: int = 1,
) -> dict[str, Any]:
    if n_independent_runs <= 0:
        raise ValueError("n_independent_runs must be positive.")
    subsystem_limit = max(1, n_sites // 2) if max_subsystem_size is None else max_subsystem_size
    run_tables: list[pd.DataFrame] = []
    for run_index in range(n_independent_runs):
        sample_batch = np.asarray(variational_state.independent_sample(seed_offset=run_index))
        entropy_rows: list[dict[str, Any]] = []
        for subsystem_size in range(1, subsystem_limit + 1):
            subsystem = tuple(range(subsystem_size))
            try:
                renyi2 = observables.renyi2_entropy_from_samples(
                    variational_state.log_value,
                    sample_batch,
                    subsystem=subsystem,
                )
            except ValueError:
                renyi2 = np.nan
            entropy_rows.append(
                {
                    "subsystem_size": subsystem_size,
                    "run_index": run_index,
                    "renyi2": renyi2,
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
    model = make_model(model_name, lattice_shape, **model_kwargs)
    variational_state, vmc_driver = build_vmc_driver(
        model=model,
        hilbert=hilbert,
        operator=system["netket_operator"],
        learning_rate=learning_rate,
        seed=seed,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
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
    final_energy = float(np.asarray(variational_state.energy(system["netket_operator"])))
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
    sampled_entropy = sampled_entropy_scaling_summary(
        variational_state,
        hilbert.size,
        n_independent_runs=independent_run_count,
    )

    return {
        "model_name": model_name,
        "model_kwargs": dict(model_kwargs),
        "system": system,
        "history": history,
        "history_df": history_df,
        "exact": exact,
        "final_energy": final_energy,
        "final_entropy": final_entropy,
        "energy_error": final_energy - exact["ground_energy"],
        "sampled_entropy_table": sampled_entropy["entropy_table"],
        "sampled_entropy_samples": sampled_entropy["entropy_samples"],
        "sampled_scaling_fit": sampled_entropy["scaling_fit"],
        "entropy_n_independent_runs": independent_run_count,
        "entropy_force_sampled": entropy_force_sampled,
    }


def comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
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


def count_model_parameters(params: Any) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params)))


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
    hilbert = system["hilbert"]

    trial_rows: list[dict[str, Any]] = []
    entropy_scan_rows: list[pd.DataFrame] = []
    trial_results: list[dict[str, Any]] = []
    for model_name, model_kwargs in architecture_configs.items():
        for seed in seed_values:
            model = make_model(model_name, lattice_shape, **model_kwargs)
            variational_state = build_variational_state(
                model=model,
                hilbert=hilbert,
                seed=seed,
                n_samples=n_samples,
                n_discard_per_chain=n_discard_per_chain,
                n_chains=n_chains,
            )
            entropy_summary = renyi2_subsystem_scan_summary(
                variational_state,
                n_sites=hilbert.size,
                max_subsystem_size=max_subsystem_size,
                n_independent_runs=entropy_n_independent_runs,
                force_sampled=entropy_force_sampled,
            )
            parameter_count = count_model_parameters(variational_state.parameters)
            entropy_table = entropy_summary["entropy_table"].copy()
            entropy_table["model"] = model_name
            entropy_table["seed"] = seed
            entropy_table["parameter_count"] = parameter_count
            entropy_scan_rows.append(entropy_table)

            max_partition = int(entropy_table["subsystem_size"].max())
            half_partition_row = entropy_table.loc[entropy_table["subsystem_size"] == max_partition].iloc[0]
            scaling_fit = entropy_summary["scaling_fit"]
            trial_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "parameter_count": parameter_count,
                    "max_subsystem_size": max_partition,
                    "half_partition_renyi2": float(half_partition_row["renyi2"]),
                    "scaling_slope": float(scaling_fit["slope"]) if scaling_fit is not None else np.nan,
                    "scaling_r_squared": float(scaling_fit["r_squared"]) if scaling_fit is not None else np.nan,
                }
            )
            trial_results.append(
                {
                    "model_name": model_name,
                    "model_kwargs": dict(model_kwargs),
                    "seed": seed,
                    "parameter_count": parameter_count,
                    "entropy_table": entropy_summary["entropy_table"],
                    "entropy_samples": entropy_summary["entropy_samples"],
                    "scaling_fit": scaling_fit,
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


def _default_sweep_label(config: dict[str, Any]) -> str:
    hamiltonian = str(config["hamiltonian"])
    lattice_shape = tuple(config.get("lattice_shape", (2, 2)))
    lattice_label = "x".join(str(length) for length in lattice_shape)
    if hamiltonian == "tfim":
        return f"tfim_L{lattice_label}_h{config.get('h', 0.8)}"
    if hamiltonian == "j1_j2":
        return f"j1j2_L{lattice_label}_J2{config.get('J2', 0.4)}"
    return f"{hamiltonian}_L{lattice_label}"


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


def ghz_statevector(n_sites: int) -> np.ndarray:
    if n_sites <= 0:
        raise ValueError("n_sites must be positive.")
    state = np.zeros(1 << n_sites, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2.0)
    state[-1] = 1.0 / np.sqrt(2.0)
    return state


def ghz_state_metrics(
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
    target = ghz_statevector(n_sites)
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
    model = make_model(model_name, lattice_shape, **model_kwargs)
    variational_state, vmc_driver = build_vmc_driver(
        model=model,
        hilbert=hilbert,
        operator=system["netket_operator"],
        learning_rate=learning_rate,
        seed=seed,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
    )
    entropy_logger = observables.entropy_callback(subsystem=half_subsystem(hilbert.size))
    history = vmc_driver.run(
        n_iter,
        callback=entropy_logger,
        callback_every=callback_every,
    )
    history_df = history_table(history)
    final_statevector = np.asarray(variational_state.exact_statevector(), dtype=np.complex128)
    metrics = ghz_state_metrics(final_statevector)
    return {
        "system": system,
        "history": history,
        "history_df": history_df,
        "final_energy": float(np.asarray(variational_state.energy(system["netket_operator"]))),
        "target_statevector": ghz_statevector(hilbert.size),
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
