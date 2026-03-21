import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demos.demo_helper import ghz_state_metrics, ghz_statevector, history_table, run_architecture_disorder_comparison, run_ghz_bonus_workflow, run_hamiltonian_system_size_sweep, run_vmc_experiment, sampled_entropy_scaling_summary  # noqa: E402


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

    def test_run_architecture_disorder_comparison_returns_exam_ready_tables(self) -> None:
        result = run_architecture_disorder_comparison(
            architecture_configs={
                "RBM": {"alpha": 1},
                "FFNN": {"hidden_dims": (4,)},
                "CNN": {"channels": (2,), "kernel_size": (1, 1)},
            },
            seeds=(0, 1),
            lattice_shape=(2, 2),
            hamiltonian="tfim",
            n_samples=4,
            n_discard_per_chain=1,
            n_chains=2,
            entropy_n_independent_runs=1,
        )

        self.assertEqual(result["summary_table"]["model"].tolist(), ["CNN", "FFNN", "RBM"])
        self.assertTrue((result["summary_table"]["parameter_count"] > 0).all())
        self.assertEqual(result["trial_table"]["seed"].tolist(), [0, 1, 0, 1, 0, 1])
        self.assertEqual(
            result["entropy_scan_table"]["subsystem_size"].drop_duplicates().tolist(),
            [1, 2],
        )

    def test_run_hamiltonian_system_size_sweep_tracks_training_entropy(self) -> None:
        result = run_hamiltonian_system_size_sweep(
            sweep_points=[
                {
                    "label": "tfim_critical",
                    "hamiltonian": "tfim",
                    "lattice_shape": (2, 2),
                    "h": 1.0,
                },
                {
                    "label": "j1j2_frustrated",
                    "hamiltonian": "j1_j2",
                    "lattice_shape": (2, 2),
                    "J1": 1.0,
                    "J2": 0.5,
                },
            ],
            model_name="RBM",
            model_kwargs={"alpha": 1},
            n_samples=16,
            n_discard_per_chain=2,
            n_chains=4,
            n_iter=2,
            callback_every=1,
        )

        self.assertEqual(
            result["summary_table"]["sweep_label"].tolist(),
            ["j1j2_frustrated", "tfim_critical"],
        )
        self.assertIn("renyi2_entropy", result["training_history_table"].columns)
        self.assertEqual(
            set(result["training_history_table"]["sweep_label"].tolist()),
            {"tfim_critical", "j1j2_frustrated"},
        )

    def test_ghz_state_metrics_match_ideal_ghz_state(self) -> None:
        metrics = ghz_state_metrics(ghz_statevector(4))

        self.assertAlmostEqual(metrics["ghz_fidelity"], 1.0)
        self.assertAlmostEqual(metrics["cat_sector_weight"], 1.0)
        self.assertAlmostEqual(metrics["half_partition_renyi2"], np.log(2.0))

    def test_run_ghz_bonus_workflow_returns_training_outputs(self) -> None:
        result = run_ghz_bonus_workflow(
            model_name="RBM",
            model_kwargs={"alpha": 1},
            lattice_shape=(2, 2),
            n_samples=16,
            n_discard_per_chain=2,
            n_chains=4,
            n_iter=1,
            callback_every=1,
            seed=0,
        )

        self.assertEqual(result["system"]["hamiltonian"], "tfim")
        self.assertEqual(result["system"]["parameters"]["h"], 0.0)
        self.assertIn("renyi2_entropy", result["history_df"].columns)
        self.assertIn("ghz_fidelity", result["ghz_metrics"])

    def test_history_table_rejects_observable_key_collisions(self) -> None:
        with self.assertRaises(ValueError):
            history_table(
                [
                    {
                        "step": 0,
                        "energy": 1.0,
                        "observables": {"energy": 2.0},
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
