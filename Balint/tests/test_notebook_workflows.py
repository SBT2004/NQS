import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nqs.workflows import (  # noqa: E402
    build_system,
    exact_observables_summary,
    history_table,
    initialize_random_parameters,
    run_architecture_benchmark,
    run_architecture_comparison,
    run_ghz_bonus_workflow,
    run_incremental_exercise_1_ed_benchmark,
    run_hamiltonian_system_size_sweep,
    run_non_ed_vmc_benchmark,
    run_random_architecture_study,
    run_vmc_experiment,
    sampler_acceptance_diagnostics,
    sampled_entropy_scaling_summary,
    tfim_config,
    tfim_proxy_sweep_points,
)
import nqs.observables as observables  # noqa: E402
from nqs.exact_diag_debug import dense_debug_operator_matrix  # noqa: E402
from nqs.sampler import SampleBatch  # noqa: E402
from nqs.vmc_setup import build_model, build_variational_state  # noqa: E402


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
    def test_exact_observables_summary_preserves_statevector_observables_from_sparse_ground_state(self) -> None:
        class SparseGroundStateResult(dict[str, object]):
            def __getitem__(self, key: str) -> object:
                if key not in {"ground_energy", "ground_state"}:
                    raise AssertionError(f"unexpected dense ED dependency: {key}")
                return super().__getitem__(key)

        bell = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=np.complex128)
        system = build_system(lattice_shape=(2, 1), pbc=False, hamiltonian="tfim", h=1.0)

        with patch(
            "nqs.workflows._core.exact_ground_state",
            return_value=SparseGroundStateResult(
                ground_energy=-1.25,
                ground_state=bell,
            ),
        ) as exact_ground_state_mock:
            summary = exact_observables_summary(system["operator"], subsystem=(0,))

        exact_ground_state_mock.assert_called_once_with(system["operator"])
        self.assertAlmostEqual(summary["ground_energy"], -1.25)
        np.testing.assert_allclose(summary["ground_state"], bell)
        np.testing.assert_allclose(summary["spectrum_table"]["energy"].to_numpy(), np.array([-1.25]))
        self.assertAlmostEqual(summary["half_partition_von_neumann"], np.log(2.0))
        self.assertAlmostEqual(summary["half_partition_renyi2"], np.log(2.0))
        np.testing.assert_allclose(
            observables.entanglement_spectrum(summary["ground_state"], subsystem=(0,), n_levels=2),
            np.array([0.5, 0.5]),
        )
        np.testing.assert_allclose(summary["correlation_matrix"].to_numpy(), np.ones((2, 2)))

    def test_exact_observables_summary_matches_dense_reference_on_small_chain(self) -> None:
        system = build_system(lattice_shape=(4, 1), pbc=False, hamiltonian="tfim", h=1.0)
        dense_eigenvalues, dense_eigenvectors = np.linalg.eigh(dense_debug_operator_matrix(system["operator"]))
        dense_ground_state = np.asarray(dense_eigenvectors[:, 0], dtype=np.complex128)
        dominant_amplitude = dense_ground_state[np.argmax(np.abs(dense_ground_state))]
        if dominant_amplitude != 0:
            dense_ground_state *= np.exp(-1j * np.angle(dominant_amplitude))
        dense_probabilities = np.abs(dense_ground_state) ** 2
        dense_states = system["hilbert"].all_states()
        dense_correlations = np.einsum(
            "bi,bj,b->ij",
            system["hilbert"].states_to_pm1(dense_states),
            system["hilbert"].states_to_pm1(dense_states),
            dense_probabilities,
            optimize=True,
        )

        summary = exact_observables_summary(system["operator"], subsystem=(0, 1))

        self.assertAlmostEqual(summary["ground_energy"], float(dense_eigenvalues[0].real), places=10)
        overlap = np.vdot(dense_ground_state, summary["ground_state"])
        self.assertAlmostEqual(float(np.abs(overlap)), 1.0, places=10)
        self.assertAlmostEqual(
            summary["half_partition_von_neumann"],
            observables.von_neumann_entropy(dense_ground_state, subsystem=(0, 1)),
            places=10,
        )
        self.assertAlmostEqual(
            summary["half_partition_renyi2"],
            observables.renyi_entropy_from_statevector(dense_ground_state, subsystem=(0, 1), alpha=2.0),
            places=10,
        )
        np.testing.assert_allclose(summary["correlation_matrix"].to_numpy(), dense_correlations)

    def test_exact_observables_summary_avoids_dense_hamiltonian_allocation(self) -> None:
        system = build_system(lattice_shape=(4, 1), pbc=False, hamiltonian="tfim", h=1.0)

        with patch(
            "scipy.sparse._csr.csr_array.toarray",
            side_effect=AssertionError("dense Hamiltonian allocation is not allowed on the main ED path"),
        ):
            summary = exact_observables_summary(system["operator"], subsystem=(0, 1))

        self.assertTrue(np.isfinite(summary["ground_energy"]))

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

    def test_sampled_entropy_scaling_summary_reuses_original_sample_log_values(self) -> None:
        controlled_samples = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
            ],
            dtype=np.uint8,
        )
        original_log_values = _bell_log_amplitude(controlled_samples)

        class FakeState:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def sample(self) -> np.ndarray:
                raise AssertionError("sample should not be used when independent_sample_with_log_values is available")

            def independent_sample(self, seed_offset: int = 0) -> np.ndarray:
                raise AssertionError(
                    "independent_sample should not be used when independent_sample_with_log_values is available"
                )

            def independent_sample_with_log_values(self, seed_offset: int = 0) -> SampleBatch:
                self.calls.append(seed_offset)
                return SampleBatch(
                    states=jax.numpy.asarray(controlled_samples),
                    log_values=jax.numpy.asarray(original_log_values),
                )

            def log_value(self, states: np.ndarray) -> np.ndarray:
                sample_array = np.asarray(states, dtype=np.uint8)
                if np.array_equal(sample_array, controlled_samples):
                    raise AssertionError("original samples should reuse provided log values")
                return _bell_log_amplitude(sample_array)

            def exact_statevector(self) -> np.ndarray:
                return np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=np.complex128)

        state = FakeState()
        summary = sampled_entropy_scaling_summary(state, n_sites=2, n_independent_runs=2)

        self.assertEqual(state.calls, [0, 1])
        self.assertAlmostEqual(float(summary["entropy_table"]["renyi2"].iloc[0]), np.log(2.0))

    def test_sampled_entropy_scaling_summary_uses_provided_sample_batches_without_resampling(self) -> None:
        controlled_samples = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 1],
            ],
            dtype=np.uint8,
        )
        provided_batches = [
            SampleBatch(
                states=jax.numpy.asarray(controlled_samples),
                log_values=jax.numpy.asarray(_bell_log_amplitude(controlled_samples)),
            )
            for _ in range(2)
        ]

        class FakeState:
            def sample(self) -> np.ndarray:
                raise AssertionError("provided sample batches should bypass sample")

            def independent_sample(self, seed_offset: int = 0) -> np.ndarray:
                raise AssertionError("provided sample batches should bypass independent_sample")

            def independent_sample_with_log_values(self, seed_offset: int = 0) -> SampleBatch:
                raise AssertionError("provided sample batches should bypass independent_sample_with_log_values")

            def log_value(self, states: np.ndarray) -> np.ndarray:
                sample_array = np.asarray(states, dtype=np.uint8)
                if np.array_equal(sample_array, controlled_samples):
                    raise AssertionError("provided original log values should be reused for original samples")
                return _bell_log_amplitude(sample_array)

            def exact_statevector(self) -> np.ndarray:
                return np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=np.complex128)

        summary = sampled_entropy_scaling_summary(
            FakeState(),
            n_sites=2,
            n_independent_runs=2,
            sample_batches=provided_batches,
        )

        self.assertAlmostEqual(float(summary["entropy_table"]["renyi2"].iloc[0]), np.log(2.0))

    def test_sampled_entropy_scaling_summary_keeps_valid_subsystems_when_one_swap_estimate_fails(self) -> None:
        controlled_samples = np.array(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        provided_batches = [
            SampleBatch(
                states=jax.numpy.asarray(controlled_samples),
                log_values=jax.numpy.asarray(np.zeros(controlled_samples.shape[0], dtype=np.complex128)),
            )
        ]

        class FakeState:
            def sample(self) -> np.ndarray:
                raise AssertionError("sample should not be used when sample_batches are provided")

            def independent_sample(self, seed_offset: int = 0) -> np.ndarray:
                raise AssertionError("independent_sample should not be used when sample_batches are provided")

            def log_value(self, states: np.ndarray) -> np.ndarray:
                lookup: dict[tuple[int, ...], complex] = {
                    (1, 0, 0, 0): 0.0j,
                    (0, 1, 1, 1): 0.0j,
                    (1, 1, 0, 0): 0.5j * np.pi,
                    (0, 0, 1, 1): 0.5j * np.pi,
                }
                result = []
                for state in np.asarray(states, dtype=np.uint8).reshape(-1, 4):
                    key = tuple(int(value) for value in state.tolist())
                    if key not in lookup:
                        raise AssertionError(f"unexpected state queried: {key}")
                    result.append(lookup[key])
                return np.asarray(result, dtype=np.complex128)

            def exact_statevector(self) -> np.ndarray:
                raise AssertionError("exact_statevector should not be used on the sampled path")

        summary = sampled_entropy_scaling_summary(
            FakeState(),
            n_sites=4,
            n_independent_runs=1,
            sample_batches=provided_batches,
        )

        self.assertEqual(summary["entropy_table"]["subsystem_size"].tolist(), [1, 2])
        self.assertAlmostEqual(float(summary["entropy_table"]["renyi2"].iloc[0]), 0.0)
        self.assertTrue(np.isnan(float(summary["entropy_table"]["renyi2"].iloc[1])))

    def test_tfim_helpers_return_notebook_ready_configurations(self) -> None:
        config = tfim_config(lattice_shape=(4, 1), h=1.0, pbc=False)
        sweep_points = tfim_proxy_sweep_points([4, 6], h=1.0, pbc=False)

        self.assertEqual(config["hamiltonian"], "tfim")
        self.assertEqual(config["lattice_shape"], (4, 1))
        self.assertEqual([point["label"] for point in sweep_points], ["tfim_1d_4x1", "tfim_1d_6x1"])
        self.assertEqual([point["lattice_shape"] for point in sweep_points], [(4, 1), (6, 1)])

    def test_incremental_exercise_1_ed_benchmark_records_per_size_completion(self) -> None:
        benchmark = run_incremental_exercise_1_ed_benchmark([4, 6], h=1.0, pbc=False)

        self.assertEqual(benchmark["label"].tolist(), ["tfim_1d_4x1", "tfim_1d_6x1"])
        self.assertTrue(benchmark["completed"].all())
        self.assertTrue((benchmark["runtime_seconds"] > 0.0).all())
        self.assertTrue((benchmark["assembly_seconds"] > 0.0).all())
        self.assertTrue((benchmark["solve_seconds"] > 0.0).all())
        self.assertTrue((benchmark["failure_message"] == "").all())
        self.assertTrue(np.isfinite(benchmark["ground_energy"]).all())

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

    def test_run_random_architecture_study_supports_labeled_init_variants_without_exact_backend(self) -> None:
        result = run_random_architecture_study(
            architecture_configs={
                "rbm_default": {
                    "model_name": "RBM",
                    "model_kwargs": {"alpha": 1},
                },
                "rbm_real_scaled": {
                    "model_name": "RBM",
                    "model_kwargs": {"alpha": 1},
                    "initialization": {
                        "parameter_scale": 0.25,
                        "phase_scale": 0.0,
                        "label": "scale=0.25, real-amplitude",
                    },
                },
            },
            seeds=(0,),
            lattice_shape=(4, 4),
            hamiltonian="tfim",
            pbc=False,
            h=2.5,
            n_samples=32,
            n_discard_per_chain=2,
            n_chains=4,
            entropy_n_independent_runs=1,
        )

        summary = result["summary_table"]
        self.assertEqual(summary["model"].tolist(), ["rbm_default", "rbm_real_scaled"])
        self.assertEqual(summary["architecture_family"].tolist(), ["RBM", "RBM"])
        self.assertEqual(summary["exact_available"].tolist(), [False, False])
        self.assertTrue(np.isnan(summary["half_partition_exact_renyi2"]).all())
        self.assertEqual(summary["initialization_label"].tolist(), ["default", "scale=0.25, real-amplitude"])
        self.assertTrue((summary["valid_entropy_points"] >= 0).all())
        self.assertTrue((summary["valid_entropy_fraction"] >= 0.0).all())
        self.assertTrue((summary["valid_entropy_fraction"] <= 1.0).all())
        self.assertEqual(
            result["entropy_scan_table"]["initialization_label"].drop_duplicates().tolist(),
            ["default", "scale=0.25, real-amplitude"],
        )
        self.assertIsNone(result["trial_results"][0]["exact_entropy_scan_table"])

    def test_run_random_architecture_study_propagates_unexpected_exact_entropy_errors(self) -> None:
        with patch(
            "nqs.workflows._core.renyi2_subsystem_scan_summary",
            side_effect=ValueError("unexpected exact failure"),
        ):
            with self.assertRaisesRegex(ValueError, "unexpected exact failure"):
                run_random_architecture_study(
                    architecture_configs={"RBM": {"alpha": 1}},
                    seeds=(0,),
                    lattice_shape=(2, 2),
                    hamiltonian="tfim",
                    n_samples=16,
                    n_discard_per_chain=2,
                    n_chains=4,
                    entropy_n_independent_runs=1,
                    real_amplitude_only=True,
                )

    def test_initialize_random_parameters_can_zero_phase_and_rescale(self) -> None:
        system = build_system(lattice_shape=(2, 2), pbc=False, hamiltonian="tfim", h=1.0)
        model = build_model(model_name="RBM", model_kwargs={"alpha": 1}, lattice_shape=(2, 2))

        default_params = initialize_random_parameters(model, system["hilbert"], seed=0)
        scaled_real_params = initialize_random_parameters(
            model,
            system["hilbert"],
            seed=0,
            parameter_scale=0.5,
            phase_scale=0.0,
        )

        np.testing.assert_allclose(
            np.asarray(scaled_real_params["visible_bias"]),
            0.5 * np.asarray(default_params["visible_bias"]),
        )
        np.testing.assert_allclose(
            np.asarray(scaled_real_params["phase_bias"]),
            np.zeros_like(np.asarray(default_params["phase_bias"])),
        )

    def test_sampler_acceptance_diagnostics_reports_burn_in_and_sampling_phases(self) -> None:
        system = build_system(lattice_shape=(2, 2), pbc=False, hamiltonian="tfim", h=1.0)
        model = build_model(model_name="FFNN", model_kwargs={"hidden_dims": (4,)}, lattice_shape=(2, 2))
        params = initialize_random_parameters(
            model,
            system["hilbert"],
            seed=0,
            phase_scale=0.0,
        )
        variational_state = build_variational_state(
            model=model,
            hilbert=system["hilbert"],
            seed=0,
            n_samples=8,
            n_discard_per_chain=2,
            n_chains=4,
            params=params,
        )

        diagnostics = sampler_acceptance_diagnostics(variational_state)

        self.assertEqual(diagnostics["summary_table"]["phase"].tolist(), ["burn_in", "sampling"])
        self.assertTrue((diagnostics["summary_table"]["mean_acceptance"] >= 0.0).all())
        self.assertTrue((diagnostics["summary_table"]["mean_acceptance"] <= 1.0).all())
        self.assertIn("steps_per_chain", diagnostics["config_table"]["parameter"].tolist())
        self.assertGreaterEqual(diagnostics["overall_acceptance"], 0.0)
        self.assertLessEqual(diagnostics["overall_acceptance"], 1.0)

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
        self.assertIn("benchmark_mode", summary.columns)
        self.assertIn("training_runtime_seconds", summary.columns)
        self.assertIn("callback_runtime_seconds", summary.columns)
        self.assertIn("postprocessing_runtime_seconds", summary.columns)
        self.assertIn("entropy_scan_runtime_seconds", summary.columns)
        self.assertIn("report_runtime_seconds", summary.columns)
        self.assertIn("total_runtime_seconds", summary.columns)
        self.assertIn("tail_window_energy_std", summary.columns)
        self.assertIn("final_half_partition_renyi2", summary.columns)
        self.assertIn("final_nn_correlation", summary.columns)
        self.assertIn("valid_entropy_points", summary.columns)
        self.assertEqual(summary["benchmark_mode"].tolist(), ["sampled"])
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
        self.assertTrue((summary["training_runtime_seconds"] > 0.0).all())
        self.assertTrue((summary["callback_runtime_seconds"] >= 0.0).all())
        self.assertTrue((summary["postprocessing_runtime_seconds"] >= 0.0).all())
        self.assertTrue((summary["entropy_scan_runtime_seconds"] >= 0.0).all())
        self.assertTrue((summary["report_runtime_seconds"] >= summary["callback_runtime_seconds"]).all())
        self.assertTrue(
            np.allclose(
                summary["report_runtime_seconds"],
                summary["callback_runtime_seconds"]
                + summary["postprocessing_runtime_seconds"]
                + summary["entropy_scan_runtime_seconds"],
            )
        )
        self.assertTrue(
            np.allclose(
                summary["total_runtime_seconds"],
                summary["training_runtime_seconds"] + summary["report_runtime_seconds"],
            )
        )
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
