import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demos.demo_helper import run_vmc_experiment, sampled_entropy_scaling_summary  # noqa: E402


def _bell_log_amplitude(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.uint8).reshape(-1, 2)
    result = np.full(states.shape[0], -np.inf + 0.0j, dtype=np.complex128)
    for index, state in enumerate(states):
        if np.array_equal(state, np.array([0, 0], dtype=np.uint8)) or np.array_equal(
            state, np.array([1, 1], dtype=np.uint8)
        ):
            result[index] = -0.5 * np.log(2.0)
    return result


class DemoHelperTests(unittest.TestCase):
    def test_sampled_entropy_scaling_summary_averages_independent_runs(self) -> None:
        class FakeState:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def sample(self) -> np.ndarray:
                return self.independent_sample(0)

            def independent_sample(self, seed_offset: int = 0) -> np.ndarray:
                self.calls.append(seed_offset)
                return np.array(
                    [
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [1, 1],
                    ],
                    dtype=np.uint8,
                )

            def log_value(self, states: np.ndarray) -> np.ndarray:
                return _bell_log_amplitude(states)

            def exact_statevector(self) -> np.ndarray:
                return np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=np.complex128)

        state = FakeState()
        summary = sampled_entropy_scaling_summary(state, n_sites=2, n_independent_runs=3)

        self.assertEqual(state.calls, [0, 1, 2])
        self.assertEqual(summary["entropy_table"]["subsystem_size"].tolist(), [1])
        self.assertAlmostEqual(float(summary["entropy_table"]["renyi2"].iloc[0]), np.log(2.0))
        self.assertAlmostEqual(float(summary["entropy_table"]["renyi2_std"].iloc[0]), 0.0)
        self.assertEqual(len(summary["entropy_samples"]), 3)

    def test_run_vmc_experiment_records_all_steps_and_rbm_entropy_repeat_default(self) -> None:
        result = run_vmc_experiment(
            model_name="RBM",
            model_kwargs={"alpha": 1},
            lattice_shape=(2, 2),
            hamiltonian="tfim",
            n_iter=3,
            n_samples=32,
            n_discard_per_chain=4,
            n_chains=4,
            callback_every=1,
            seed=0,
        )

        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(result["history_df"]["step"].tolist(), [0, 1, 2])
        self.assertEqual(result["entropy_n_independent_runs"], 5)


if __name__ == "__main__":
    unittest.main()
