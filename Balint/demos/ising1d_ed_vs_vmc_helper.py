"""Helper functions for the 1D Ising ED versus VMC demo notebook."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import netket as nk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nqs import Adam, CNN, FFNN, RBM, NetKetSampler, SpinHilbert, VMC, VariationalState  # noqa: E402


class SupportsDemoModel(Protocol):
    """Minimal model contract used by this demo helper."""

    def init(self, rng_key: jax.Array, hilbert: SpinHilbert) -> Any:
        ...

    def log_psi(self, params: Any, states: jax.Array) -> jax.Array:
        ...


def build_ising_operator(length: int, field_strength: float) -> tuple[Any, nk.operator.AbstractOperator]:
    """Create the NetKet Hilbert space and 1D Ising Hamiltonian used in the demo."""

    netket_hilbert = nk.hilbert.Spin(s=0.5, N=length)
    graph = nk.graph.Chain(length=length, pbc=False)
    operator = nk.operator.IsingJax(  # pyright: ignore[reportCallIssue]
        hilbert=netket_hilbert,
        graph=graph,
        h=field_strength,
    )
    return netket_hilbert, operator


def exact_ground_state_energy(operator: nk.operator.AbstractOperator) -> float:
    """Compute the exact ground-state energy with NetKet's Lanczos ED helper."""

    return float(nk.exact.lanczos_ed(operator, k=1, compute_eigenvectors=False)[0])


def run_model_demo(
    model_name: str,
    model: SupportsDemoModel,
    operator: nk.operator.AbstractOperator,
    length: int,
    seed: int,
    learning_rate: float,
    n_samples: int = 16,
    n_discard_per_chain: int = 2,
    n_chains: int = 4,
    n_iter: int = 3,
) -> tuple[list[float], float]:
    """Run a short VMC optimization for one model and return its energy trace."""

    hilbert = SpinHilbert(length)
    params = model.init(jax.random.PRNGKey(seed), hilbert)
    sampler = NetKetSampler(
        hilbert=hilbert,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        seed=seed,
    )
    state = VariationalState(model=model, params=params, sampler=sampler)
    driver = VMC(
        operator=operator,
        variational_state=state,
        optimizer=Adam(learning_rate=learning_rate),
    )
    history = driver.run(n_iter)

    energy_trace = [float(jnp.asarray(step["energy"])) for step in history]
    final_energy = energy_trace[-1]
    print(f"{model_name:>4} energies: {[round(value, 6) for value in energy_trace]}")
    return energy_trace, final_energy


def run_demo(
    length: int = 5,
    transverse_field: float = 1.0,
) -> dict[str, object]:
    """Run the full benchmark and return exact energy plus model results."""

    _, operator = build_ising_operator(length, transverse_field)
    exact_energy = exact_ground_state_energy(operator)

    experiments = [
        ("RBM", RBM(alpha=1), 1e-2, 0),
        ("FFNN", FFNN(hidden_dims=(10, 5)), 5e-3, 1),
        ("CNN", CNN(spatial_shape=(length, 1), channels=(2,), kernel_size=(3, 1)), 5e-3, 2),
    ]

    results: list[dict[str, float | list[float] | str]] = []
    for model_name, model, learning_rate, seed in experiments:
        energy_trace, final_energy = run_model_demo(
            model_name=model_name,
            model=model,
            operator=operator,
            length=length,
            seed=seed,
            learning_rate=learning_rate,
        )
        results.append(
            {
                "model": model_name,
                "trace": energy_trace,
                "final_energy": final_energy,
                "delta_to_exact": final_energy - exact_energy,
            }
        )

    return {
        "length": length,
        "transverse_field": transverse_field,
        "exact_energy": exact_energy,
        "results": results,
    }
