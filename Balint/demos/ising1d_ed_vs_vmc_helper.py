"""Helper functions for the 1D Ising ED versus VMC demo notebook."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import netket as nk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
    n_samples: int = 256,
    n_discard_per_chain: int = 32,
    n_chains: int = 16,
    n_iter: int = 64,
    eval_samples: int = 4096,
    eval_repeats: int = 4,
) -> tuple[list[float], float]:
    """Optimize one ansatz with sampled VMC and return its energy trace."""

    hilbert = SpinHilbert(length)
    params = model.init(jax.random.PRNGKey(seed), hilbert)
    train_sampler = NetKetSampler(
        hilbert=hilbert,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_chains=n_chains,
        seed=seed,
    )
    state = VariationalState(model=model, params=params, sampler=train_sampler)
    driver = VMC(
        operator=operator,
        variational_state=state,
        optimizer=Adam(learning_rate=learning_rate),
    )
    history = driver.run(n_iter)

    energy_trace = [float(jnp.asarray(step["energy"])) for step in history]

    # Use a separate higher-statistics state for the final benchmark estimate so
    # the comparison to ED is less noisy than the training trace.
    eval_sampler = NetKetSampler(
        hilbert=hilbert,
        n_samples=eval_samples,
        n_discard_per_chain=max(n_discard_per_chain, 32),
        n_chains=n_chains,
        seed=seed + 101,
    )
    eval_state = VariationalState(model=model, params=state.parameters, sampler=eval_sampler)
    estimates = [float(jnp.asarray(eval_state.energy(operator))) for _ in range(eval_repeats)]
    final_energy = sum(estimates) / len(estimates)
    preview = [round(value, 6) for value in energy_trace[:5]]
    tail = [round(value, 6) for value in energy_trace[-5:]]
    print(f"{model_name:>4} start: {preview} ... end: {tail}")
    return energy_trace, final_energy


def run_demo(
    length: int = 5,
    transverse_field: float = 1.0,
) -> dict[str, object]:
    """Run the full benchmark and return exact energy plus model results."""

    _, operator = build_ising_operator(length, transverse_field)
    exact_energy = exact_ground_state_energy(operator)

    experiments = [
        ("RBM", RBM(alpha=2), 2e-2, 0, 256, 32, 16, 128),
        ("FFNN", FFNN(hidden_dims=(32, 16)), 1e-2, 1, 256, 32, 16, 128),
        ("CNN", CNN(spatial_shape=(length, 1), channels=(16, 8), kernel_size=(5, 1)), 5e-3, 2, 256, 32, 16, 256),
    ]

    results: list[dict[str, float | list[float] | str]] = []
    for model_name, model, learning_rate, seed, n_samples, n_discard_per_chain, n_chains, n_iter in experiments:
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
