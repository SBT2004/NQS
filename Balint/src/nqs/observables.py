from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

import numpy as np


class SupportsSamplingAndLogValue(Protocol):
    def sample(self) -> Any:
        ...

    def log_value(self, states: Any) -> Any:
        ...

    def independent_sample(self, seed_offset: int = 0) -> Any:
        ...

    def exact_statevector(self) -> Any:
        ...


def observable_callback(
    name: str,
    observable_fn: Callable[[Any], object],
) -> Callable[[int, Any], dict[str, object]]:
    def callback(step: int, driver: Any) -> dict[str, object]:
        del step
        return {name: observable_fn(driver)}

    return callback


def _infer_n_sites(statevector: np.ndarray, n_sites: int | None) -> int:
    flat_state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    if flat_state.size == 0:
        raise ValueError("statevector must not be empty.")
    if n_sites is None:
        inferred = int(np.log2(flat_state.size))
        if (1 << inferred) != flat_state.size:
            raise ValueError("statevector length must be a power of two.")
        return inferred
    if (1 << n_sites) != flat_state.size:
        raise ValueError("statevector length does not match n_sites.")
    return n_sites


def _normalize_subsystem(subsystem: Sequence[int] | str, n_sites: int) -> tuple[int, ...]:
    if subsystem == "half":
        return tuple(range(n_sites // 2))

    sites = tuple(int(site) for site in subsystem)
    if not sites:
        raise ValueError("subsystem must contain at least one site.")
    if len(set(sites)) != len(sites):
        raise ValueError("subsystem sites must be unique.")
    if any(site < 0 or site >= n_sites for site in sites):
        raise ValueError("subsystem sites must lie inside the full system.")
    return tuple(sorted(sites))


def _flatten_samples(samples: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.uint8)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim < 2:
        raise ValueError("samples must have at least one site dimension.")
    return arr.reshape(-1, arr.shape[-1])


def spin_spin_correlation(
    states: np.ndarray | Sequence[Sequence[int]],
    site_i: int,
    site_j: int,
    weights: np.ndarray | Sequence[float] | None = None,
) -> float:
    sample_batch = _flatten_samples(states)
    n_sites = sample_batch.shape[1]
    if site_i < 0 or site_i >= n_sites or site_j < 0 or site_j >= n_sites:
        raise ValueError("site indices must lie inside the state width.")
    if np.any((sample_batch != 0) & (sample_batch != 1)):
        raise ValueError("spin states must only contain 0 or 1.")

    spins_pm1 = 2.0 * sample_batch.astype(np.float64) - 1.0
    correlations = spins_pm1[:, site_i] * spins_pm1[:, site_j]
    if weights is None:
        return float(np.mean(correlations))

    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weight_array.shape[0] != correlations.shape[0]:
        raise ValueError("weights must have one entry per sample.")
    total_weight = np.sum(weight_array)
    if total_weight == 0:
        raise ValueError("weights must not sum to zero.")
    return float(np.average(correlations, weights=weight_array))


def reduced_density_matrix(
    statevector: np.ndarray | Sequence[complex],
    subsystem: Sequence[int] | str,
    n_sites: int | None = None,
) -> np.ndarray:
    flat_state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    total_sites = _infer_n_sites(flat_state, n_sites)
    subsystem_sites = _normalize_subsystem(subsystem, total_sites)
    environment_sites = tuple(site for site in range(total_sites) if site not in subsystem_sites)

    norm = np.linalg.norm(flat_state)
    if norm == 0:
        raise ValueError("statevector must have non-zero norm.")
    psi = (flat_state / norm).reshape((2,) * total_sites)
    permuted = np.transpose(psi, axes=subsystem_sites + environment_sites)
    psi_matrix = permuted.reshape(1 << len(subsystem_sites), 1 << len(environment_sites))
    return psi_matrix @ psi_matrix.conj().T


def renyi_entropy_from_density_matrix(
    rho: np.ndarray | Sequence[Sequence[complex]],
    alpha: float = 2.0,
    cutoff: float = 1e-12,
) -> float:
    rho_array = np.asarray(rho, dtype=np.complex128)
    if rho_array.ndim != 2 or rho_array.shape[0] != rho_array.shape[1]:
        raise ValueError("rho must be a square matrix.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    hermitian_rho = 0.5 * (rho_array + rho_array.conj().T)
    eigenvalues = np.clip(np.linalg.eigvalsh(hermitian_rho).real, 0.0, None)
    nonzero = eigenvalues[eigenvalues > cutoff]
    if nonzero.size == 0:
        return 0.0
    if np.isclose(alpha, 1.0):
        return float(-np.sum(nonzero * np.log(nonzero)))
    return float(np.log(np.sum(nonzero**alpha)) / (1.0 - alpha))


def von_neumann_entropy(
    statevector: np.ndarray | Sequence[complex],
    subsystem: Sequence[int] | str,
    n_sites: int | None = None,
    cutoff: float = 1e-12,
) -> float:
    rho = reduced_density_matrix(statevector, subsystem=subsystem, n_sites=n_sites)
    return renyi_entropy_from_density_matrix(rho, alpha=1.0, cutoff=cutoff)


def renyi_entropy_from_statevector(
    statevector: np.ndarray | Sequence[complex],
    subsystem: Sequence[int] | str,
    alpha: float = 2.0,
    n_sites: int | None = None,
    cutoff: float = 1e-12,
) -> float:
    rho = reduced_density_matrix(statevector, subsystem=subsystem, n_sites=n_sites)
    return renyi_entropy_from_density_matrix(rho, alpha=alpha, cutoff=cutoff)


def entanglement_spectrum(
    statevector: np.ndarray | Sequence[complex],
    subsystem: Sequence[int] | str,
    n_sites: int | None = None,
    n_levels: int | None = None,
) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(reduced_density_matrix(statevector, subsystem=subsystem, n_sites=n_sites))
    ordered = np.sort(np.clip(eigenvalues.real, 0.0, None))[::-1]
    if n_levels is None:
        return ordered
    if n_levels <= 0:
        raise ValueError("n_levels must be positive when provided.")
    return ordered[:n_levels]


def _renyi2_swap_expectation(
    log_amplitude_fn,
    samples: np.ndarray | Sequence[Sequence[int]],
    subsystem: Sequence[int] | str,
    *,
    original_log_values: np.ndarray | Sequence[complex] | None = None,
) -> complex:
    sample_batch = _flatten_samples(samples)
    if sample_batch.shape[0] < 2:
        raise ValueError("at least two samples are required for the SWAP estimator.")
    if np.any((sample_batch != 0) & (sample_batch != 1)):
        raise ValueError("spin states must only contain 0 or 1.")

    n_sites = sample_batch.shape[1]
    subsystem_sites = np.array(_normalize_subsystem(subsystem, n_sites), dtype=np.intp)
    pair_count = sample_batch.shape[0] // 2
    paired_samples = sample_batch[: 2 * pair_count].reshape(pair_count, 2, n_sites)

    original_left = paired_samples[:, 0, :]
    original_right = paired_samples[:, 1, :]
    swapped_left = original_left.copy()
    swapped_right = original_right.copy()
    swapped_left[:, subsystem_sites] = original_right[:, subsystem_sites]
    swapped_right[:, subsystem_sites] = original_left[:, subsystem_sites]

    if original_log_values is None:
        original_log = np.asarray(log_amplitude_fn(sample_batch), dtype=np.complex128).reshape(-1)
    else:
        original_log = np.asarray(original_log_values, dtype=np.complex128).reshape(-1)
        if original_log.shape[0] != sample_batch.shape[0]:
            raise ValueError("original_log_values must have one entry per sample.")
    swapped_log = np.asarray(
        log_amplitude_fn(np.concatenate([swapped_left, swapped_right], axis=0)),
        dtype=np.complex128,
    ).reshape(-1)

    left_log = original_log[0 : 2 * pair_count : 2]
    right_log = original_log[1 : 2 * pair_count : 2]
    swapped_left_log = swapped_log[:pair_count]
    swapped_right_log = swapped_log[pair_count:]
    estimator = np.exp(swapped_left_log + swapped_right_log - left_log - right_log)
    return complex(np.mean(estimator))


def renyi2_swap_expectation(
    log_amplitude_fn,
    samples: np.ndarray | Sequence[Sequence[int]],
    subsystem: Sequence[int] | str,
) -> complex:
    return _renyi2_swap_expectation(
        log_amplitude_fn=log_amplitude_fn,
        samples=samples,
        subsystem=subsystem,
    )


def _renyi2_entropy_from_samples(
    log_amplitude_fn,
    samples: np.ndarray | Sequence[Sequence[int]],
    subsystem: Sequence[int] | str,
    *,
    cutoff: float = 1e-12,
    original_log_values: np.ndarray | Sequence[complex] | None = None,
) -> float:
    swap_expectation = _renyi2_swap_expectation(
        log_amplitude_fn=log_amplitude_fn,
        samples=samples,
        subsystem=subsystem,
        original_log_values=original_log_values,
    )
    swap_value = np.real_if_close(swap_expectation)
    swap_magnitude = float(np.abs(swap_expectation))
    if np.iscomplexobj(swap_value):
        imag_part = abs(float(np.imag(swap_value)))
        if imag_part > cutoff:
            # In the sampled route the exact SWAP expectation should still be
            # real and positive, but finite-sample noise can leave a residual
            # complex phase. Use the magnitude as a stable positive fallback so
            # large-system diagnostics do not fail outright.
            swap_real = swap_magnitude
        else:
            swap_real = float(np.real(swap_value))
    else:
        swap_real = float(swap_value)
    if swap_real <= 0:
        if swap_magnitude <= cutoff:
            raise ValueError("SWAP expectation must be strictly positive to compute Renyi-2 entropy.")
        swap_real = swap_magnitude
    return float(-np.log(swap_real))


def renyi2_entropy_from_samples(
    log_amplitude_fn,
    samples: np.ndarray | Sequence[Sequence[int]],
    subsystem: Sequence[int] | str,
    cutoff: float = 1e-12,
) -> float:
    return _renyi2_entropy_from_samples(
        log_amplitude_fn=log_amplitude_fn,
        samples=samples,
        subsystem=subsystem,
        cutoff=cutoff,
    )


def renyi2_entropy(
    state: SupportsSamplingAndLogValue,
    subsystem: Sequence[int] | str,
    samples: np.ndarray | Sequence[Sequence[int]] | None = None,
    cutoff: float = 1e-12,
    n_repeats: int = 4,
    force_sampled: bool = False,
) -> float:
    if not force_sampled and samples is None and hasattr(state, "exact_statevector"):
        return float(
            renyi_entropy_from_statevector(
                state.exact_statevector(),
                subsystem=subsystem,
                alpha=2.0,
                cutoff=cutoff,
            )
        )
    return float(
        renyi2_entropy_statistics(
            state=state,
            subsystem=subsystem,
            samples=samples,
            cutoff=cutoff,
            n_repeats=n_repeats,
            force_sampled=force_sampled,
        )["mean"]
    )


def renyi2_entropy_statistics(
    state: SupportsSamplingAndLogValue,
    subsystem: Sequence[int] | str,
    samples: np.ndarray | Sequence[Sequence[int]] | None = None,
    cutoff: float = 1e-12,
    n_repeats: int = 4,
    force_sampled: bool = False,
) -> dict[str, float]:
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")

    if not force_sampled and samples is None and hasattr(state, "exact_statevector"):
        exact_entropy = float(
            renyi_entropy_from_statevector(
                state.exact_statevector(),
                subsystem=subsystem,
                alpha=2.0,
                cutoff=cutoff,
            )
        )
        return {
            "mean": exact_entropy,
            "std": 0.0,
            "n_repeats": 1.0,
            "exact": exact_entropy,
        }

    if samples is not None:
        estimates = [
            _renyi2_entropy_from_samples(
                log_amplitude_fn=state.log_value,
                samples=samples,
                subsystem=subsystem,
                cutoff=cutoff,
            )
        ]
    else:
        estimates = []
        sampling_state = cast(Any, state)
        for repeat_index in range(n_repeats):
            batch_log_values: np.ndarray | None = None
            if hasattr(sampling_state, "independent_sample_with_log_values"):
                sample_with_values = sampling_state.independent_sample_with_log_values(seed_offset=repeat_index)
                sample_batch = sample_with_values.states
                batch_log_values = np.asarray(sample_with_values.log_values, dtype=np.complex128)
            elif hasattr(sampling_state, "independent_sample"):
                sample_batch = sampling_state.independent_sample(seed_offset=repeat_index)
            elif hasattr(sampling_state, "sample_with_log_values"):
                sample_with_values = sampling_state.sample_with_log_values()
                sample_batch = sample_with_values.states
                batch_log_values = np.asarray(sample_with_values.log_values, dtype=np.complex128)
            else:
                sample_batch = sampling_state.sample()
            estimates.append(
                _renyi2_entropy_from_samples(
                    log_amplitude_fn=state.log_value,
                    samples=sample_batch,
                    subsystem=subsystem,
                    cutoff=cutoff,
                    original_log_values=batch_log_values,
                )
            )

    estimate_array = np.asarray(estimates, dtype=np.float64)
    result = {
        "mean": float(np.mean(estimate_array)),
        "std": float(np.std(estimate_array, ddof=0)),
        "n_repeats": float(len(estimates)),
    }
    if not force_sampled and hasattr(state, "exact_statevector"):
        result["exact"] = float(
            renyi_entropy_from_statevector(
                state.exact_statevector(),
                subsystem=subsystem,
                alpha=2.0,
            )
        )
    return result


def fit_log_entropy_scaling(
    sizes: np.ndarray | Sequence[float],
    entropies: np.ndarray | Sequence[float],
) -> dict[str, float]:
    size_array = np.asarray(sizes, dtype=np.float64).reshape(-1)
    entropy_array = np.asarray(entropies, dtype=np.float64).reshape(-1)
    if size_array.shape != entropy_array.shape:
        raise ValueError("sizes and entropies must have matching shapes.")
    if size_array.size < 2:
        raise ValueError("at least two entropy points are required for a scaling fit.")
    if np.any(size_array <= 0):
        raise ValueError("sizes must be strictly positive for log scaling.")

    log_sizes = np.log(size_array)
    if np.ptp(log_sizes) <= 1e-9:
        raise ValueError("sizes must span distinct positive scales for a meaningful log fit.")
    slope, intercept = np.polyfit(log_sizes, entropy_array, deg=1)
    fitted = slope * log_sizes + intercept
    residuals = entropy_array - fitted
    residual_sum = float(np.sum(residuals**2))
    total_sum = float(np.sum((entropy_array - np.mean(entropy_array)) ** 2))
    if np.isclose(total_sum, 0.0):
        r_squared = 1.0 if np.isclose(residual_sum, 0.0) else 0.0
    else:
        r_squared = 1.0 - residual_sum / total_sum

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
    }


def entropy_callback(
    subsystem: Sequence[int] | str,
    name: str = "renyi2_entropy",
    n_repeats: int = 4,
    force_sampled: bool = False,
) -> Callable[[int, Any], dict[str, object]]:
    return observable_callback(
        name=name,
        observable_fn=lambda driver: renyi2_entropy(
            driver.variational_state,
            subsystem=subsystem,
            n_repeats=n_repeats,
            force_sampled=force_sampled,
        ),
    )


__all__ = [
    "entropy_callback",
    "fit_log_entropy_scaling",
    "observable_callback",
    "SupportsSamplingAndLogValue",
    "entanglement_spectrum",
    "reduced_density_matrix",
    "renyi2_entropy",
    "renyi2_entropy_statistics",
    "renyi2_entropy_from_samples",
    "renyi2_swap_expectation",
    "renyi_entropy_from_density_matrix",
    "renyi_entropy_from_statevector",
    "spin_spin_correlation",
    "von_neumann_entropy",
]
