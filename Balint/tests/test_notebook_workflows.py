import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nqs.workflows import (  # noqa: E402
    history_table,
    run_architecture_benchmark,
    run_architecture_comparison,
    run_ghz_bonus_workflow,
    run_hamiltonian_system_size_sweep,
    run_non_ed_vmc_benchmark,
    run_random_architecture_study,
    run_vmc_experiment,
    sampled_entropy_scaling_summary,
    tfim_config,
    tfim_proxy_sweep_points,
)


def _bell_log_amplitude(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.uint8).reshape(-1, 2)
    result = np.full(states.shape[0], -np.inf + 0.0j, dtype=np.complex128)
    for index, state in enumerate(states):
        if np.array_equal(state, np.array([0, 0], dtype=np.uint8)) or np.array_equal(
            state, np.array([1, 1], dtype=np.uint8)
        ):
            result[index] = -0.5 * np.log(2.0)
    return result


class NotebookWorkflowTests(unittest.TestCase):
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

    def test_tfim_helpers_return_notebook_ready_configurations(self) -> None:
        config = tfim_config(lattice_shape=(4, 1), h=1.0, pbc=False)
        sweep_points = tfim_proxy_sweep_points([4, 6], h=1.0, pbc=False)

        self.assertEqual(config["hamiltonian"], "tfim")
        self.assertEqual(config["lattice_shape"], (4, 1))
        self.assertEqual([point["label"] for point in sweep_points], ["tfim_1d_4x1", "tfim_1d_6x1"])
        self.assertEqual([point["lattice_shape"] for point in sweep_points], [(4, 1), (6, 1)])

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
        self.assertTrue(np.isfinite(result["entropy_scan_table"]["renyi2"]).all())

    def test_run_architecture_comparison_returns_exam_ready_tables(self) -> None:
        result = run_architecture_comparison(
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
            n_iter=1,
            callback_every=1,
            entropy_n_independent_runs=1,
        )

        self.assertEqual(result["summary_table"]["model"].tolist(), ["CNN", "FFNN", "RBM"])
        self.assertTrue((result["summary_table"]["parameter_count"] > 0).all())
        self.assertIn("final_energy", result["summary_table"].columns)
        self.assertIn("energy_error", result["summary_table"].columns)
        self.assertEqual(result["trial_table"]["seed"].tolist(), [0, 1, 0, 1, 0, 1])
        self.assertEqual(
            result["entropy_scan_table"]["subsystem_size"].drop_duplicates().tolist(),
            [1, 2],
        )
        self.assertTrue(np.isfinite(result["entropy_scan_table"]["renyi2"]).all())

    def test_run_architecture_benchmark_reports_netket_gap_column(self) -> None:
        result = run_architecture_benchmark(
            architecture_configs={
                "RBM": {"alpha": 1},
                "FFNN": {"hidden_dims": (4,)},
                "CNN": {"channels": (2,), "kernel_size": (1, 1)},
            },
            lattice_shape=(2, 2),
            hamiltonian="tfim",
            n_iter=1,
            n_samples=16,
            n_discard_per_chain=2,
            n_chains=4,
            netket_reference_energy=-5.0,
        )

        self.assertEqual(result["summary_table"]["model"].tolist(), ["CNN", "FFNN", "RBM"])
        self.assertTrue((result["summary_table"]["netket_reference_energy"] == -5.0).all())
        self.assertIn("netket_gap", result["summary_table"].columns)

    def test_run_random_architecture_study_reports_sampled_and_exact_entropy(self) -> None:
        result = run_random_architecture_study(
            architecture_configs={
                "FFNN": {"hidden_dims": (4,)},
                "CNN": {"channels": (2,), "kernel_size": (1, 1)},
            },
            seeds=(0, 1),
            lattice_shape=(2, 2),
            hamiltonian="tfim",
            n_samples=16,
            n_discard_per_chain=2,
            n_chains=4,
            entropy_n_independent_runs=2,
            real_amplitude_only=True,
        )

        self.assertEqual(result["summary_table"]["model"].tolist(), ["CNN", "FFNN"])
        self.assertTrue((result["summary_table"]["parameter_count"] > 0).all())
        self.assertIn("half_partition_exact_renyi2", result["summary_table"].columns)
        self.assertIn("half_partition_sampled_renyi2", result["summary_table"].columns)
        self.assertEqual(result["trial_table"]["seed"].tolist(), [0, 1, 0, 1])
        self.assertEqual(
            result["entropy_scan_table"]["subsystem_size"].drop_duplicates().tolist(),
            [1, 2],
        )
        self.assertTrue(np.isfinite(result["entropy_scan_table"]["exact_renyi2"]).all())
        self.assertTrue(np.isfinite(result["entropy_scan_table"]["sampled_renyi2"]).all())

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

    def test_run_non_ed_vmc_benchmark_reports_non_exact_training_metrics(self) -> None:
        result = run_non_ed_vmc_benchmark(
            benchmark_configs={
                "rbm_small": {
                    "model_name": "RBM",
                    "model_kwargs": {"alpha": 1},
                }
            },
            sweep_points=[
                {
                    "label": "tfim_4x4_non_ed",
                    "hamiltonian": "tfim",
                    "lattice_shape": (4, 4),
                    "pbc": False,
                    "h": 1.0,
                }
            ],
            n_samples=8,
            n_discard_per_chain=1,
            n_chains=2,
            n_iter=2,
            callback_every=1,
            entropy_n_independent_runs=1,
            max_entropy_subsystem_size=2,
        )

        summary = result["summary_table"]
        self.assertEqual(summary["benchmark_label"].tolist(), ["rbm_small"])
        self.assertEqual(summary["n_sites"].tolist(), [16])
        self.assertTrue((summary["parameter_count"] > 0).all())
        self.assertTrue(np.isfinite(summary["final_energy"]).all())
        self.assertIn("runtime_seconds", summary.columns)
        self.assertIn("callback_runtime_seconds", summary.columns)
        self.assertIn("total_runtime_seconds", summary.columns)
        self.assertIn("tail_window_energy_std", summary.columns)
        self.assertIn("final_half_partition_renyi2", summary.columns)
        self.assertIn("final_nn_correlation", summary.columns)
        self.assertIn("valid_entropy_points", summary.columns)
        self.assertEqual(
            result["training_history_table"]["system_label"].tolist(),
            ["tfim_4x4_non_ed"] * 3,
        )
        self.assertIn("abs_magnetization", result["training_history_table"].columns)
        self.assertIn("nn_correlation", result["training_history_table"].columns)
        self.assertIn("is_post_update", result["training_history_table"].columns)
        self.assertEqual(
            result["entropy_scan_table"]["subsystem_size"].drop_duplicates().tolist(),
            [1, 2],
        )
        self.assertTrue((summary["valid_entropy_points"] >= 0).all())
        self.assertTrue((summary["valid_entropy_points"] <= 2).all())
        self.assertTrue((summary["total_runtime_seconds"] >= summary["runtime_seconds"]).all())
        history = result["training_history_table"]
        post_update_rows = history.loc[history["is_post_update"]]
        self.assertEqual(post_update_rows["step"].tolist(), [2])
        self.assertTrue(np.isclose(post_update_rows["energy"].iloc[0], summary["final_energy"].iloc[0]))
        self.assertTrue(np.isclose(summary["best_energy"].iloc[0], np.min(history["energy"])))
        self.assertTrue(
            np.isclose(
                summary["energy_drop"].iloc[0],
                history["energy"].iloc[0] - summary["final_energy"].iloc[0],
            )
        )

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
