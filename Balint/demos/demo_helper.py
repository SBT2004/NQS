"""Helper functions for the high-level NQS showcase notebook."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nqs import Adam  # noqa: E402
from nqs import CNN  # noqa: E402
from nqs import FFNN  # noqa: E402
from nqs import NetKetSampler  # noqa: E402
from nqs import Operator  # noqa: E402
from nqs import RBM  # noqa: E402
from nqs import SpinHilbert  # noqa: E402
from nqs import SquareLattice  # noqa: E402
from nqs import VMC  # noqa: E402
from nqs import VariationalState  # noqa: E402
from nqs import collect_terms  # noqa: E402
from nqs import j1_j2  # noqa: E402
from nqs import sx_term  # noqa: E402
from nqs import szsz_term  # noqa: E402
import nqs.observables as observables  # noqa: E402


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


def build_tfim_operator(spin_space: SpinHilbert, lattice: Any, J: float, h: float) -> Operator:
    interaction_terms = [szsz_term(edge.i, edge.j, coefficient=-J) for edge in lattice.iter_edges("J1", n=1)]
    field_terms = [sx_term(site, coefficient=-h) for site in range(spin_space.size)]
    return Operator(spin_space, collect_terms(interaction_terms, field_terms))


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
        hamiltonian_operator = build_tfim_operator(spin_space, lattice, J=J, h=h)
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


def operator_matrix(operator: Any) -> np.ndarray:
    hilbert = operator.hilbert
    dimension = hilbert.n_states
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for column_index, state in enumerate(hilbert.all_states()):
        for connected_state, value in operator.connected_elements(state):
            row_index = hilbert.state_to_index(connected_state)
            matrix[row_index, column_index] += value
    return matrix


def exact_ground_state(operator: Any) -> dict[str, Any]:
    matrix = operator_matrix(operator)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues.real)
    ordered_values = eigenvalues[order]
    ordered_vectors = eigenvectors[:, order]
    ground_state = ordered_vectors[:, 0]
    return {
        "matrix": matrix,
        "eigenvalues": ordered_values.real,
        "ground_state": ground_state,
        "ground_energy": float(ordered_values[0].real),
    }


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
            row[key] = float(np.asarray(value))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if "renyi2_entropy" in frame.columns:
        frame["renyi2_entropy"] = frame["renyi2_entropy"].astype(float)
    return frame


def sampled_entropy_scaling_summary(
    variational_state: VariationalState,
    n_sites: int,
    max_subsystem_size: int | None = None,
) -> dict[str, Any]:
    subsystem_limit = max(1, n_sites // 2) if max_subsystem_size is None else max_subsystem_size
    sample_batch = np.asarray(variational_state.sample())
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
                "renyi2": renyi2,
            }
        )

    entropy_table = pd.DataFrame(entropy_rows)
    valid_rows = entropy_table.dropna(subset=["renyi2"])
    scaling_fit = (
        observables.fit_log_entropy_scaling(
            valid_rows["subsystem_size"].to_numpy(),
            valid_rows["renyi2"].to_numpy(),
        )
        if len(valid_rows) >= 2
        else None
    )
    return {
        "entropy_table": entropy_table,
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
    params = model.init(jax.random.PRNGKey(seed), hilbert)
    state_sampler = NetKetSampler(
        hilbert=hilbert,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        seed=seed,
    )
    variational_state = VariationalState(
        model=model,
        params=params,
        sampler=state_sampler,
    )
    vmc_driver = VMC(
        operator=system["netket_operator"],
        variational_state=variational_state,
        optimizer=Adam(learning_rate=learning_rate),
    )
    history: list[dict[str, Any]] = []
    entropy_logger = observables.entropy_callback(subsystem=half_subsystem(hilbert.size))
    for step in range(n_iter):
        step_result = dict(vmc_driver.step())
        step_result["step"] = step
        if step % callback_every == 0:
            step_result["observables"] = entropy_logger(step, vmc_driver)
        history.append(step_result)
    history_df = history_table(history)
    exact = exact_observables_summary(system["operator"])
    final_energy = float(np.asarray(variational_state.energy(system["netket_operator"])))
    final_entropy = float(
        observables.renyi2_entropy_from_samples(
            variational_state.log_value,
            np.asarray(variational_state.sample()),
            subsystem=half_subsystem(hilbert.size),
        )
    )
    sampled_entropy = sampled_entropy_scaling_summary(variational_state, hilbert.size)

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
        "sampled_scaling_fit": sampled_entropy["scaling_fit"],
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
