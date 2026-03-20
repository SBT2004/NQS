import sys
import unittest
from pathlib import Path
from typing import cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nqs.observables as observables


def _bell_state() -> np.ndarray:
    state = np.zeros(4, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2.0)
    state[3] = 1.0 / np.sqrt(2.0)
    return state


def _log_amplitude_from_statevector(statevector: np.ndarray):
    amplitudes = np.asarray(statevector, dtype=np.complex128).reshape(-1)

    def evaluate(states: np.ndarray) -> np.ndarray:
        state_batch = np.asarray(states, dtype=np.uint8).reshape(-1, states.shape[-1])
        indices = np.sum(state_batch.astype(np.int64) << np.arange(state_batch.shape[1], dtype=np.int64), axis=1)
        selected = amplitudes[indices]
        result = np.full(indices.shape, -np.inf + 0.0j, dtype=np.complex128)
        nonzero = np.abs(selected) > 0
        result[nonzero] = np.log(selected[nonzero])
        return result

    return evaluate


def _phased_statevector(statevector: np.ndarray, phase: float) -> np.ndarray:
    return np.asarray(statevector, dtype=np.complex128) * np.exp(1j * phase)


class ObservableTests(unittest.TestCase):
    def test_spin_spin_correlation_matches_pm1_average(self) -> None:
        states = np.array([[0, 0], [1, 1], [0, 1]], dtype=np.uint8)
        correlation = observables.spin_spin_correlation(states, site_i=0, site_j=1)
        self.assertAlmostEqual(correlation, 1.0 / 3.0)

    def test_exact_entropies_for_bell_state(self) -> None:
        bell = _bell_state()

        von_neumann = observables.von_neumann_entropy(bell, subsystem=(0,))
        renyi2 = observables.renyi_entropy_from_statevector(bell, subsystem=(0,), alpha=2.0)

        self.assertAlmostEqual(von_neumann, np.log(2.0))
        self.assertAlmostEqual(renyi2, np.log(2.0))

    def test_swap_renyi2_entropy_for_controlled_bell_samples(self) -> None:
        bell = _bell_state()
        log_amplitude = _log_amplitude_from_statevector(bell)
        samples = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
            ],
            dtype=np.uint8,
        )

        swap_expectation = observables.renyi2_swap_expectation(
            log_amplitude_fn=log_amplitude,
            samples=samples,
            subsystem=(0,),
        )
        entropy = observables.renyi2_entropy_from_samples(
            log_amplitude_fn=log_amplitude,
            samples=samples,
            subsystem=(0,),
        )

        self.assertAlmostEqual(float(np.real(swap_expectation)), 0.5)
        self.assertAlmostEqual(entropy, np.log(2.0))

    def test_swap_renyi2_entropy_accepts_complex_phase_wavefunction(self) -> None:
        phased_bell = _phased_statevector(_bell_state(), phase=np.pi / 4.0)
        log_amplitude = _log_amplitude_from_statevector(phased_bell)
        samples = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
            ],
            dtype=np.uint8,
        )

        entropy = observables.renyi2_entropy_from_samples(
            log_amplitude_fn=log_amplitude,
            samples=samples,
            subsystem=(0,),
        )

        self.assertAlmostEqual(entropy, np.log(2.0))

    def test_log_entropy_scaling_fit_recovers_known_parameters(self) -> None:
        sizes = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        entropies = 1.25 + 0.5 * np.log(sizes)

        fit = observables.fit_log_entropy_scaling(sizes, entropies)

        self.assertAlmostEqual(fit["slope"], 0.5)
        self.assertAlmostEqual(fit["intercept"], 1.25)
        self.assertAlmostEqual(fit["r_squared"], 1.0)
        self.assertAlmostEqual(fit["rmse"], 0.0)

    def test_log_entropy_scaling_fit_rejects_degenerate_sizes(self) -> None:
        sizes = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        entropies = np.array([0.1, 0.2, 0.3], dtype=np.float64)

        with self.assertRaises(ValueError):
            observables.fit_log_entropy_scaling(sizes, entropies)

    def test_renyi2_entropy_statistics_averages_independent_runs_and_reports_exact_reference(self) -> None:
        bell = _bell_state()
        log_amplitude = _log_amplitude_from_statevector(bell)
        controlled_samples = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
            ],
            dtype=np.uint8,
        )

        class FakeState:
            def __init__(self) -> None:
                self.sample_calls: list[int] = []

            def sample(self) -> np.ndarray:
                raise AssertionError("independent_sample should be used when available")

            def independent_sample(self, seed_offset: int = 0) -> np.ndarray:
                self.sample_calls.append(seed_offset)
                return controlled_samples

            def log_value(self, states: np.ndarray) -> np.ndarray:
                return log_amplitude(states)

            def exact_statevector(self) -> np.ndarray:
                return bell

        state = FakeState()
        stats = observables.renyi2_entropy_statistics(state, subsystem=(0,), n_repeats=3)

        self.assertEqual(state.sample_calls, [0, 1, 2])
        self.assertAlmostEqual(stats["mean"], np.log(2.0))
        self.assertAlmostEqual(stats["std"], 0.0)
        self.assertAlmostEqual(stats["exact"], np.log(2.0))

    def test_entropy_callback_uses_repeated_entropy_average(self) -> None:
        bell = _bell_state()
        log_amplitude = _log_amplitude_from_statevector(bell)
        controlled_samples = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
            ],
            dtype=np.uint8,
        )

        class FakeState:
            def independent_sample(self, seed_offset: int = 0) -> np.ndarray:
                return controlled_samples

            def log_value(self, states: np.ndarray) -> np.ndarray:
                return log_amplitude(states)

            def exact_statevector(self) -> np.ndarray:
                return bell

        class FakeDriver:
            def __init__(self) -> None:
                self.variational_state = FakeState()

        callback = observables.entropy_callback(subsystem=(0,), n_repeats=2)
        result = callback(0, FakeDriver())
        self.assertAlmostEqual(float(cast(float, result["renyi2_entropy"])), np.log(2.0))


if __name__ == "__main__":
    unittest.main()
